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
'TXSEN2014':{"url":"http://www.realclearpolitics.com/epolls/2014/senate/tx/texas_senate_cornyn_vs_alameel-5011.html",'year':2014, 'name':'Texas Senate 2014', 'tab':1},
'NCGOV2012':{"url":"http://www.realclearpolitics.com/epolls/2012/governor/nc/north_carolina_governor_mccrory_vs_dalton-2103.html", "year":2012, 'name':'NC Governor 2012','tab':1},
'CAGOV2014':{"url":"http://www.realclearpolitics.com/epolls/2014/governor/ca/california_governor_kashkari_vs_brown-5080.html", "year":2014, 'name': 'CA Governor 2014', 'tab':4},
'AKGOV2014':{"url":"http://www.realclearpolitics.com/epolls/2014/governor/ak/alaska_governor_parnell_vs_walker-5215.html", 'year':2014, 'name': 'AK Governor 2014', 'tab':4},
'NYGOV2014':{"url":"http://www.realclearpolitics.com/epolls/2014/governor/ny/new_york_governor_astorino_vs_cuomo-4177.html", 'year':2014, 'name': 'NY Governor 2014', 'tab':4},
'HWGOV2014':{"url":"http://www.realclearpolitics.com/epolls/2014/governor/hi/hawaii_governor_aiona_vs_ige-4310.html", 'year':2014, 'name': 'HW Governor 2014', 'tab':4},
'MSSEN2012':{'url':'http://www.realclearpolitics.com/epolls/2012/senate/ma/massachusetts_senate_brown_vs_warren-2093.html', 'year':2012, 'name':'MS Senate 2012', 'tab':5},
'PRES2012':{'url':'http://www.realclearpolitics.com/epolls/2012/president/us/general_election_romney_vs_obama-1171.html', 'year':2012, 'name':'2012 General Election', 'tab':5},
'PRES2016':{'url':'http://www.realclearpolitics.com/epolls/2016/president/us/general_election_trump_vs_clinton-5491.html', 'year':2016, 'name':'2016 General Election', 'tab':2}
        }

app = Flask(__name__)


@app.route('/')
def main():
  return redirect('/index')

@app.route('/index')
def index():
  return render_template('index.html')

@app.route('/index', methods=['POST'])
def doSomething():
    if request.form['election']:
        if request.method == 'POST':
            # print 'Got a POST request that looks like ' + request.form['election']
            temp_set = data_settings[request.form['election']]
            df,span_date = helpers.scrape_to_predict(temp_set['url'],
                    temp_set['year'],
                    datetime.datetime(temp_set['year'],12,31),
                    tabnum=temp_set['tab']
                )
            # print 'spandate', span_date
            # if request.form['normalize']:
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
            # print script
	    # print 'end of doSomething', df
	    # print script, div
            return result(request.form['election'], script, div)
        else:
            print 'not a post'
            return 'what even' 

@app.route('/err')
def incorrect_code():
    return render_template('err.html')

# @app.route('/result/<election>/<polldate>')
@app.route('/result')
def result(election, bk_script, bk_div, last_poll_date=datetime.date.today()):
    return render_template('result.html', election=election, b_s_=bk_script, b_d_=bk_div)



if __name__ == '__main__':
  app.run(host='0.0.0.0',port=33507)
