from flask import Flask, render_template, request, redirect
import pandas as pd
import datetime
import time

# from bokeh.charts import Line, show, output_notebook
from bokeh.embed import components
from bokeh.plotting import figure, output_file, show
from bokeh.models import Span

import helpers




data_settings = {
'TXSEN2014':{"url":"http://www.realclearpolitics.com/epolls/2014/senate/tx/texas_senate_cornyn_vs_alameel-5011.html",'year':2014, 'name':'Texas Senate 2014', 'tab':1, 'election_date':'2014-11-04'},
'NCGOV2012':{"url":"http://www.realclearpolitics.com/epolls/2012/governor/nc/north_carolina_governor_mccrory_vs_dalton-2103.html", "year":2012, 'name':'NC Governor 2012','tab':1, 'election_date':'2012-11-6'},
'CAGOV2014':{"url":"http://www.realclearpolitics.com/epolls/2014/governor/ca/california_governor_kashkari_vs_brown-5080.html", "year":2014, 'name': 'CA Governor 2014', 'tab':4,'election_date':'2014-11-04'},
'AKGOV2014':{"url":"http://www.realclearpolitics.com/epolls/2014/governor/ak/alaska_governor_parnell_vs_walker-5215.html", 'year':2014, 'name': 'AK Governor 2014', 'tab':4,'election_date':'2014-11-04'},
'NYGOV2014':{"url":"http://www.realclearpolitics.com/epolls/2014/governor/ny/new_york_governor_astorino_vs_cuomo-4177.html", 'year':2014, 'name': 'NY Governor 2014', 'tab':4,'election_date':'2014-11-04'},
'HWGOV2014':{"url":"http://www.realclearpolitics.com/epolls/2014/governor/hi/hawaii_governor_aiona_vs_ige-4310.html", 'year':2014, 'name': 'HW Governor 2014', 'tab':4,'election_date':'2014-11-04'},
'MSSEN2012':{'url':'http://www.realclearpolitics.com/epolls/2012/senate/ma/massachusetts_senate_brown_vs_warren-2093.html', 'year':2012, 'name':'MS Senate 2012', 'tab':5,'election_date':'2012-11-06'},
'PRES2012':{'url':'http://www.realclearpolitics.com/epolls/2012/president/us/general_election_romney_vs_obama-1171.html', 'year':2012, 'name':'2012 General Election', 'tab':5,'election_date':'2012-11-06'},
'PRES2016':{'url':'http://www.realclearpolitics.com/epolls/2016/president/us/general_election_trump_vs_clinton-5491.html', 'year':2016, 'name':'2016 General Election', 'tab':2,'election_date':'2016-11-08'}
        }

app = Flask(__name__)


@app.route('/')
def main():
  return redirect('/index')

@app.route('/index')
def index():
  return render_template('index.html')

@app.route('/result', methods=['POST'])
def doSomething():
	if request.form['election']:
		if request.method == 'POST':
			print 'Got a POST request that looks like ' + request.form['election']
			print request.form
			temp_set = data_settings[request.form['election']]
			try:
				print "date range:",request.form['daterange']
				last_date = datetime.datetime.strptime(request.form['daterange'].split(' - ')[1], "%a %b %d %Y")
				first_date = datetime.datetime.strptime(request.form['daterange'].split(' - ')[0], "%a %b %d %Y")
				df,span_date,first_a,last_a = helpers.scrape_to_predict(temp_set['url'],
					temp_set['year'],
					datetime.datetime.strptime(temp_set['election_date'], '%Y-%m-%d'),
					tabnum=temp_set['tab'],
					last_poll_date=last_date,
					first_poll_date=first_date
				)
			except:
				print 'else time'
				last_date=None
				first_date=None
				df,span_date,first_a,last_a = helpers.scrape_to_predict(temp_set['url'],
					temp_set['year'],
					datetime.datetime.strptime(temp_set['election_date'], '%Y-%m-%d'),
					tabnum=temp_set['tab']
				)
		
			helpers.normalize_df(df)
			p = figure(width=800, height=600, x_axis_type="datetime", title=temp_set['name'])
			p.xaxis.axis_label="Date"
			p.yaxis.axis_label="Points"
			for name in list(df):
				if "(R)" in name:
					color = "red"
				elif "(D)" in name:
					color = "blue"
				else:
					color = "green"
				p.line(df.index, df[name], line_color=color, line_width=3,legend=name) 
			my_end_span = Span(location=time.mktime(span_date.timetuple())*1000, dimension='height',line_color='purple',
			line_dash='dashed',line_width=3)
			p.add_layout(my_end_span)
			script, div = components(p)
			first_a = first_a.strftime('%Y.%m.%d')
			last_a = last_a.strftime('%Y.%m.%d')
			return result(request.form['election'], script, div,
				first_a, last_a, current_last=last_date, current_first=first_date)
		else:
			print 'not a post'
			return 'what even' 

@app.route('/err')
def incorrect_code():
    return render_template('err.html')

@app.route('/about')
def about():
    return render_template('about.html')

# @app.route('/result/<election>/<polldate>')
@app.route('/result')
def result(election, bk_script, bk_div, first_allowed, last_allowed,
		last_poll_date=datetime.date.today(), current_last=None, current_first=None):
	if current_last==None:
		current_last = last_allowed
	if current_first==None:
		current_first = first_allowed
        election_len = len(election)
	return render_template('result.html', election=election, b_s_=bk_script, b_d_=bk_div,
		    f_a_ = first_allowed, l_a_=last_allowed, c_l_=current_last,c_f_=current_first,
                    elect_len=election_len)



if __name__ == '__main__':
  app.run(host='0.0.0.0',port=33507)
