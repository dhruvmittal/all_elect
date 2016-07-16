# ALL OF THE HELPER FUNCTIONS
import pandas as pd
import numpy as np
import random as rm
import datetime
from sklearn.linear_model import LinearRegression
import dill

# f = open('/home/vagrant/all_elect/pollster_df.dill','rb')
# f = open('pollster_df.dill','rb')
f = open('pollster_df_som.dill','rb')
aggg = dill.load(f)
f.close()

def map_to_rank(pollster):
    pollster = pollster.strip('*')
    to_rank = {0:'B', 1:'D', 2:'C', 3:'D', 4:'A', 'etc':'C'}
    to_cluster = aggg['cluster'].to_dict()
    try:
        c = to_cluster[pollster]
    except KeyError:
        try:
            renamed = {'Rasmussen Reports':'Rasmussen Reports/Pulse Opinion Research',
                       'PPP (D)':'Public Policy Polling'
                       ,'PPP (D)':'Public Policy Polling'}
            c = to_cluster[renamed[pollster]]
        except KeyError:
            c = 'etc'
    return to_rank[c]
    
# Custom Average Weighted Poll Aggregate, built out of my
# Assigned letter grades
def CustomAveragePoll(polls, contestant):
    # Step 0: Prepare Data
    # Map the pollster grades onto the polls.
    polls['rank'] = polls['Poll'].map(map_to_rank)
    # Operate on poll data to make it a time indexed series
    ts = polls[['FixedDate', contestant, 'rank']]
    # set a criteria for the first set -> Grade A
    grade = 'A'


    # temp code to drop the duplicate time indices
    ts = ts.drop_duplicates('FixedDate').set_index('FixedDate')

    # create a holder time series matched to the dates in the original
    holder = pd.Series(index=ts.index)
    holder.iloc[0] = ts.iloc[0].loc[contestant]
    
    
    # Fill the holder with the best of the polls.
    # These points should remain in the model at all times.
    # print polls.loc[:,contestant]
    for date in holder.index:
#         print ts.loc[date,:]
#         print ts.loc[date, 'rank'], ts.loc[date, contestant]
        
        if ts.loc[date, 'rank'] == grade:
            holder.loc[date] = ts.loc[date, contestant]

    # time-based interpolation
    time_fill = holder.copy()
    time_fill = time_fill.interpolate(method='time')

    # go through the rest, adding points to the holder based on quality
    # for each poll, add a point halfway between the line connecting
    # the existing time series and the polled point.
    ordered_grades = 'BCDEF'
    for grade in ordered_grades:
        for date in holder.index:
            # print ts.loc[date, 'rank']
            if ts.loc[date, 'rank'] == grade:
              val_new = ts.loc[date, contestant]
              val_old = time_fill.loc[date]
              final_val = (val_new + val_old) * 1./2
              holder.loc[date] = final_val
              time_fill = holder.copy()
              time_fill = time_fill.interpolate(method='time')

    # print '2'
    return holder
    
def next_time_step(time_series, next_time):
    
    linreg = LinearRegression()
    
    time_deltas = np.asarray(map(np.float32, [(next_time - ts_day).days for ts_day in time_series.index]))
    y_data = np.asarray(time_series.values)
    l = len(time_series)
    X = time_deltas.reshape(l,1)
    y = y_data.reshape(l,1)
    w = np.array(map(lambda x:1./x, X))
    linreg.fit(X,y,w)
    
    p_x = np.array([0]).reshape(1,1)
    p_y = linreg.predict(p_x)
    return appropriateRandomStep(time_series, next_time, p_y[0])

def appropriateRandomStep(time_series, next_time, next_time_value):
    # The Random Step should be a Gaussian noise element that attempts to
    # revert towards the mean with some probability, but provides the option
    # to runaway if the momentum is high.
    # The size of the random step should be dictated by the scale of the
    # variance of the actual polling (reduced actual, to be precise)

    typical_var = np.std(time_series.values)
    typical_mean = np.mean(time_series.values)
    if next_time_value > typical_mean:
        return next_time_value -  abs(rm.gauss(0, typical_var))
    elif next_time_value < typical_mean:
        return next_time_value + abs(rm.gauss(0, typical_var))
    else:
        return rm.gauss(next_time_value, typical_var)
        
def all_steps(time_series, enddate):
    dates = pd.date_range(start=time_series.index[-1], freq='w', end=enddate)
    added_points = pd.Series(index=dates[1:])
    i = 0
    for da in added_points.index:
        added_points.loc[da] = next_time_step(time_series.append(added_points.dropna()), da)
	# print 'step', i
    return added_points
    
def scrape_to_predict(url, year, how_predict, last_poll_date=datetime.datetime.today(), 
        first_poll_date=datetime.datetime(1772,11,15), tabnum=1):
    df = pd.read_html(url,
                      header=0,
                     parse_dates=True,
                     skiprows=[1,2],
                     tupleize_cols=True)
    df = df[tabnum]
    # print '0'
    df['StartDate'] = df['Date'].apply(lambda x:pd.to_datetime(x.split(' - ')[0] + ' ' + str(year)))
    df['EndDate'] = df['Date'].apply(lambda x:pd.to_datetime(x.split(' - ')[1] + ' ' + str(year)))
    # print 'stp',df
    fixYears(df)
    tmpdates = sorted(df.loc[:,'FixedDate'].values)
    first_allowed = tmpdates[0].astype('M8[ms]').astype('O')
    last_allowed = tmpdates[-1].astype('M8[ms]').astype('O')
    # print df
    df = df[df['FixedDate'] < last_poll_date]
    df = df[df['FixedDate'] > first_poll_date]
    # print '00'
    last_poll_used_date = sorted(df['FixedDate'].values)[-1].astype('M8[ms]').astype('O')
    # print type(last_poll_used_date)
    cands = []
    out=pd.DataFrame()
    for e in list(df):
        if e.strip() not in ['Poll', 'Date', 'Sample', 'Spread', 'StartDate', 'EndDate', 'FixedDate', 'MoE']:
            cands.append(e)
    for c in cands:
        cap = CustomAveragePoll(df.sort_values('FixedDate'), c)
        predictions = all_steps(cap,how_predict)
        out[c] = cap.append(predictions)
#         plt.plot(cap)
#         plt.plot(predictions)
#     plt.plot(out)
    return out, last_poll_used_date, first_allowed, last_allowed
    
def normalize_df(df):
    for d in df.index:
        s = sum(df.loc[d])
        for col in list(df):
            # Note: randint to somewhat account for 'undecided voters'
            df.loc[d,col] = df.loc[d,col] *rm.randint(87,97)*1./s
        
def fixYears(df):
    old_dates = df.loc[:,'EndDate'].values
    new_dates = np.empty(old_dates.shape, dtype=datetime.datetime)
    new_dates[0] = old_dates[0].astype('M8[ms]').astype('O')
    current_year = old_dates[0].astype('M8[ms]').astype('O').year
    # print old_dates, new_dates, current_year
    for i in range(1, len(old_dates)):
        last_month = new_dates[i-1].month
        now_date = old_dates[i].astype('M8[ms]').astype('O')
        # print now_date
        if now_date.month == 12 and last_month == 1:
            current_year -=1
        new_dates[i] = datetime.datetime(current_year, now_date.month, now_date.day)
    df['FixedDate'] = new_dates
