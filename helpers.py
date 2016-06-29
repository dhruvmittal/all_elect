# ALL OF THE HELPER FUNCTIONS
import pandas as pd
import numpy as np
import seaborn as sns
import random as rm
import datetime
from sklearn.linear_model import LinearRegression
import dill

f = open('/home/vagrant/all_elect/pollster_df.dill','rb')
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
    ts = polls[['EndDate', contestant, 'rank']]
    # set a criteria for the first set -> Grade A
    grade = 'A'


    # temp code to drop the duplicate time indices
    ts = ts.drop_duplicates('EndDate').set_index('EndDate')

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
    if next_time_value > time_series.values[-1]:
        return next_time_value -  abs(rm.gauss(0, typical_var))
    elif next_time_value < time_series.values[-1]:
        return next_time_value + abs(rm.gauss(0, typical_var))
    else:
        return rm.gauss(next_time_value, typical_var)
        
def all_steps(time_series, enddate):
    dates = pd.date_range(start=time_series.index[-1], freq='w', end=enddate)
    added_points = pd.Series(index=dates[1:])
    for da in added_points.index:
        added_points.loc[da] = next_time_step(time_series.append(added_points.dropna()), da)
    return added_points
    
def scrape_to_predict(url, year, how_predict, last_poll_date=datetime.datetime.today()):
    df = pd.read_html(url,
                      header=0,
                     parse_dates=True,
                     skiprows=[1,2],
                     tupleize_cols=True)
    df = df[1]
    df['StartDate'] = df['Date'].apply(lambda x:pd.to_datetime(x.split(' - ')[0] + ' ' + str(year)))
    df['EndDate'] = df['Date'].apply(lambda x:pd.to_datetime(x.split(' - ')[1] + ' ' + str(year)))
    df = df[df['EndDate'] < last_poll_date]
    last_poll_used_date = sorted(df['EndDate'].values)[-1].astype('M8[ms]').astype('O')
    print type(last_poll_used_date)
    cands = []
    out=pd.DataFrame()
    for e in list(df):
        if e.strip() not in ['Poll', 'Date', 'Sample', 'Spread', 'StartDate', 'EndDate', 'MoE']:
            cands.append(e)
    for c in cands:
        cap = CustomAveragePoll(df.sort_values('EndDate'), c)
        predictions = all_steps(cap,how_predict)
        out[c] = cap.append(predictions)
#         plt.plot(cap)
#         plt.plot(predictions)
#     plt.plot(out)
    return out, last_poll_used_date
    
def normalize_df(df):
    for d in df.index:
        s = sum(df.loc[d])
        for col in list(df):
            df.loc[d,col] = df.loc[d,col] * 100./s
        
