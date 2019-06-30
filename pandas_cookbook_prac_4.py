# -*- coding: utf-8 -*-
"""
Created on Mon Jul  9 15:44:25 2018

@author: dsc
"""

from os import getcwd, chdir
wd = getcwd()
chdir(wd)
getcwd()

import numpy as np
import datetime
import matplotlib.pyplot as plt
#m pandas option setting to see more columns 
import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

import os
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = 'C:/Anaconda3/Library/plugins/platforms'

date = datetime.date(year=2013, month=6, day=7)
time = datetime.time(hour=12, minute=30, second=19, microsecond=463198)
dt=datetime.datetime(year=2013, month=6, day=7, hour=12, minute=30, second=19,
                     microsecond=463198)

print('date is ', date)
print('time is ', time)
print('datetime is ', dt)

td = datetime.timedelta(weeks=2, days=5, hours=10, minutes=20, seconds=6.73,
                        milliseconds=99, microseconds=8)
print(td)

print('new date is', date+td)
print('new datetime is', dt+td)

pd.Timestamp(year=2012, month=12, day=21, hour=5, minute=10, second=8, 
             microsecond=99)

pd.Timestamp('2016/1/10')
pd.Timestamp('2014-5/10')
pd.Timestamp('Jan 3, 2019 20:45.56')
pd.Timestamp('2016-01-05T05:34:43.123456789')
pd.Timestamp(500)
pd.Timestamp(5000, unit='D')

pd.to_datetime('2015-5-13')
pd.to_datetime('2015-13-5', dayfirst=True)
pd.to_datetime('Start Date: Sep 30, 2017 Start Time: 1:30 pm', 
               format='Start Date: %b %d, %Y Start Time: %I:%M %p')
pd.to_datetime('2017-04-11 00:00:00')
s = pd.Series([10, 100, 1000, 10000])
pd.to_datetime(s, unit='D')

s= pd.Series(['12-5-2015', '14-1-2013', '20/12/2017', '40/23/2017'])
pd.to_datetime(s, dayfirst=True, errors='coerce')
pd.to_datetime(['Aug 3 1999 3:45:56', '10/31/2017'])

# Timedelta
pd.Timedelta('12 days 5 hours 3 minutes 123456789 nanoseconds')
pd.Timedelta(days=5, minutes=7.34)
pd.Timedelta(100, unit='W')
pd.to_timedelta('67:15:45.454')
s = pd.Series([10, 100])
pd.to_timedelta(s, unit='s')
time_strings = ['2 days 24 minutes 89.67 seconds', '00:45:23.6']
pd.to_timedelta(time_strings)
pd.Timedelta('12 days 5 hours 3 minutes')*2
pd.Timestamp('1/1/2017') + \
pd.Timedelta('12 days 5 hours 3 minutes')*2

td1 = pd.to_timedelta([10, 100], unit='s')
td2 = pd.to_timedelta(['3 hours', '4 hours'])
td1 + td2
pd.Timedelta('12 days') / pd.Timedelta('3 days')
ts = pd.Timestamp('2016-10-1 4:23:23.9')
ts.ceil('h')
ts.year, ts.month, ts.day, ts.hour, ts.minute, ts.second
ts.dayofweek, ts.dayofyear, ts.daysinmonth
ts.to_pydatetime()
td = pd.Timedelta(125.8723, unit='h')
td
td.round('min')
td.components
td.total_seconds()
date_string_list=['Sep 30 1984']*10000
# pd.to_datetime(date_string_list, format='%b %d %Y')
pd.to_datetime(date_string_list)

crime = pd.read_hdf('./data/crime.h5', 'crime')
crime.dtypes

crime = crime.set_index('REPORTED_DATE')
crime.head()
crime.loc['2016-05-12 16:45:00']
crime.loc['2016-05-12']
crime.loc['2015-05'].shape
crime.loc['2016'].shape
crime.loc['2016-05-12 03']
crime.loc['Dec 2015'].sort_index()
crime.loc['2016 Sep, 15'].shape
crime.loc['21st October 2014 05'].shape
crime.loc['2015-3-4':'2016-1-1'].sort_index()
crime.loc['2015-3-4 22' : '2016-1-1 11:45:00'].sort_index()

crime.loc['2015-3-4':'2016-1-1']
crime_sort = crime.sort_index()
crime_sort.loc['2015-3-4':'2016-1-1']

crime=pd.read_hdf('./data/crime.h5', 'crime').set_index('REPORTED_DATE')
print(type(crime.index))
crime.between_time('2:00', '5:00', include_end=False).head()
crime.at_time('5:47').head()
crime_sort = crime.sort_index()
crime_sort.first(pd.offsets.MonthBegin(6))
## 1 row :: 2012-07-01 :: why?? 6 min?
crime_sort.first(pd.offsets.MonthEnd(6))
crime_sort.first(pd.offsets.MonthBegin(6, normalize=True))
crime_sort.loc[:'2012-06']

crime_sort.first('5D')
crime_sort.first('5B')
crime_sort.first('7W')
crime_sort.first('3QS')
crime_sort.first('A')

crime_sort = pd.read_hdf('./data/crime.h5', 'crime')\
.set_index('REPORTED_DATE')\
.sort_index()

crime_quarterly = crime_sort.resample('Q')['IS_CRIME', 'IS_TRAFFIC'].sum()
crime_quarterly.head()

crime_sort.resample('QS')['IS_CRIME', 'IS_TRAFFIC'].sum().head()
crime_sort.loc['2012-4-1':'2012-6-30', ['IS_CRIME', 'IS_TRAFFIC']].sum()
crime_quarterly2 = crime_sort.groupby(pd.Grouper(freq='Q'))\
['IS_CRIME', 'IS_TRAFFIC'].sum()

crime_quarterly2.equals(crime_quarterly)

plot_kwargs = dict(figsize=(16, 4), 
                   color = ['black', 'lightgrey'],
                   title='Denver Crimes and Traffic Accidents')
crime_quarterly.plot(**plot_kwargs)

crime_sort.resample('QS-MAR')['IS_CRIME', 'IS_TRAFFIC']\
.sum().head()

crime_begin = crime_quarterly.iloc[0]
crime_begin

crime_quarterly.div(crime_begin).sub(1).round(2).plot(**plot_kwargs)

crime=pd.read_hdf('./data/crime.h5', 'crime')
crime.head()
wd_counts=crime['REPORTED_DATE'].dt.weekday_name.value_counts()
wd_counts

days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 
        'Sunday']
title = 'Denver Crimes and Traffic Accidents per Weekday'
wd_counts.reindex(days).plot(kind='barh', title=title)

title='Denver Crimes and Traffic Accidents per Year'
crime['REPORTED_DATE'].dt.year.value_counts().sort_index()\
.plot(kind='barh', title=title)

weekday = crime['REPORTED_DATE'].dt.weekday_name
year = crime['REPORTED_DATE'].dt.year

crime.groupby(weekday)
crime.groupby(year)
crime.groupby([year, weekday]) # no error 
# crime.groupby([year, weekday]).size() # error

# linear extrapolation
criteria = crime['REPORTED_DATE'].dt.year == 2017
crime.loc[criteria, 'REPORTED_DATE'].dt.dayofyear.max()
crime_pct = crime['REPORTED_DATE'].dt.dayofyear.le(272).groupby(year).mean().round(3)
crime_pct.loc[2012:2016].median()

crime_sort = pd.read_hdf('./data/crime.h5', 'crime')\
.set_index('REPORTED_DATE').sort_index()

common_attrs = set(dir(crime_sort.index)) & set(dir(pd.Timestamp))
common_attrs
print([attr for attr in common_attrs if attr[0] != '_'])
crime_sort.index.weekday_name.value_counts()
crime_sort.groupby(lambda x : x.weekday_name)['IS_CRIME', 'IS_TRAFFIC'].sum()
funcs = [lambda x: x.round('2h').hour, lambda x : x.year]
cr_group = crime_sort.groupby(funcs)['IS_CRIME', 'IS_TRAFFIC'].sum()
cr_group.unstack()
cr_final = cr_group.unstack()
cr_final
cr_final.style.highlight_max(color='lightgrey')
cr_final.xs('IS_TRAFFIC', axis='columns', level=0)
cr_final.xs(2016, axis='columns', level=1).head()

employee = pd.read_csv('./data/employee.csv', 
                       parse_dates=['JOB_DATE', 'HIRE_DATE'],
                       index_col='HIRE_DATE')
employee.head()
employee.groupby('GENDER')['BASE_SALARY'].mean().round(-2)
employee.resample('10AS')['BASE_SALARY'].mean().round(-2)
sal_avg = employee.groupby('GENDER').resample('10AS')['BASE_SALARY']\
.mean().round(-2)

sal_avg.unstack('GENDER')
employee[employee['GENDER'] == 'Male'].index.min()
employee[employee['GENDER'] == 'Female'].index.min()

sal_avg2 = employee.groupby(['GENDER', pd.Grouper(freq='10AS')])\
['BASE_SALARY'].mean().round(-2)

sal_avg2
sal_final = sal_avg2.unstack('GENDER')

'resample' in dir(employee.groupby('GENDER'))
'groupby' in dir(employee.resample('10AS'))

years = sal_final.index.year
years_right = years + 9
sal_final.index = years.astype(str) + '-' + years_right.astype(str)
sal_final

cuts = pd.cut(employee.index.year, bins=5, precision=0)
cuts
cuts.categories.values
employee.groupby([cuts, 'GENDER'])['BASE_SALARY'].mean().unstack('GENDER').round(-2)

# .merge_asof()
crime_sort = pd.read_hdf('./data/crime.h5', 'crime')\
.set_index('REPORTED_DATE').sort_index()

crime_sort.index.max()
crime_sort = crime_sort[:'2017-8']
crime_sort.index.max()
all_data = crime_sort.groupby([pd.Grouper(freq='M'), 'OFFENSE_CATEGORY_ID']).size()
all_data.head()
all_data = all_data.sort_values().reset_index(name='Total')
all_data.head()

goal = all_data[all_data['REPORTED_DATE'] == '2017-8-31'].reset_index(drop=True)
goal['Total_Goal'] = goal['Total'].mul(.8).astype(int)
goal.head()

#pd.merge_asof(goal, all_data, left_on = 'Total_Goal', 
#              right_on = 'Total', by='OFFENSE_CATEGORY_ID', 
#              suffixes=('_Current', '_Last')) # error
# error :: MergeError: incompatible merge keys [1] int32 and int64, must be the same type

goal.info()
all_data.info()
all_data['Total'] = all_data['Total'].astype('int32')
all_data.info()

pd.merge_asof(goal, all_data, left_on = 'Total_Goal', 
              right_on = 'Total', by='OFFENSE_CATEGORY_ID', 
              suffixes=('_Current', '_Last'))

pd.Period(year=2012, month=5, day=17, hour=14, minute=20, freq='T')
ad_period = crime_sort.groupby([lambda x : x.to_period('M'), 
                                'OFFENSE_CATEGORY_ID']).size()
ad_period = ad_period.sort_values()\
.reset_index(name='Total')\
.rename(columns={'level_0':'REPORTED_DATE'})

ad_period.head()
cols = ['OFFENSE_CATEGORY_ID', 'Total']
all_data[cols].equals(ad_period[cols])

aug_2018 = pd.Period('2017-8', freq='M')
goal_period = ad_period[ad_period['REPORTED_DATE'] == aug_2018].reset_index(drop=True)
goal_period.info()
goal['Total_Goal'] = goal['Total_Goal'].astype(int)
goal_period['Total_Goal'] = goal_period['Total'].mul(.8).astype(int)

#pd.merge_asof(goal_period, ad_period, left_on = 'Total_Goal', 
#              right_on = 'Total' , by = 'OFFENSE_CATEGORY_ID', 
#              suffixes = ('_Current', '_Last')).head()
# error :: incompatible merge keys [1] int32 and int64, must be the same type

x = [-3, 5, 7]
y = [10, 2, 5]
plt.figure(figsize=(15, 3))
plt.plot(x, y)
plt.xlim(0, 10)
plt.ylim(-3, 8)
plt.xlabel('X Axis')
plt.ylabel('Y axis')
plt.title('Line Plot')
plt.suptitle('Figure Title', size=20, y=1.03)

fig, ax = plt.subplots(figsize=(15, 3))
ax.plot(x, y)
ax.set_xlim(0, 10)
ax.set_ylim(-3, 8)
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_title('Line Plot')
fig.suptitle('Figure Title', size=20, y=1.03)

fig, ax = plt.subplots(nrows=1, ncols=1)
# fig.get_size_inches()
fig.set_size_inches(14, 4)
# fig
# fig.axes
# fig.axes[0] is ax
fig.set_facecolor('.9')
ax.set_facecolor('.7')

ax_children = ax.get_children()
ax_children #  4 spines, 2 aixs

spines = ax.spines
spine_left = spines['left']
spine_left.set_position(('outward', -100))
spine_left.set_linewidth(5)

spine_bottom = spines['bottom']
spine_bottom.set_visible(False)

ax.xaxis.grid(True, which='major', linewidth=2, color='black', linestyle='--')
ax.xaxis.set_ticks([.2, .4, .55, .93])
ax.xaxis.set_label_text('X Axis', family='Verdana', fontsize=15)

ax.set_ylabel('Y lablel', family='Calibri', fontsize=20)
ax.set_yticks([.1, .9])
ax.set_yticklabels(['point 1', 'point 9'], rotation=46)

plot_objects = plt.subplots(nrows=1, ncols=1)
type(plot_objects)
fig = plot_objects[0]
ax = plot_objects[1]

plot_objects = plt.subplots(2, 4)
plot_objects[1]

fig.axes == fig.get_axes()
ax.xaxis == ax.get_xaxis()
ax.yaxis == ax.get_yaxis()
ax.xaxis.properties()

import matplotlib.pyplot as plt

movie = pd.read_csv('./data/movie.csv')
med_budget = movie.groupby('title_year')['budget'].median()/1e6
med_budget_roll = med_budget.rolling(5, min_periods=1).mean()
med_budget_roll.tail()

years = med_budget_roll.index.values
# years[-5:]
budget = med_budget_roll.values
# budget[-5:]
fig, ax = plt.subplots(figsize=(14,4), linewidth=5, edgecolor='.5')
ax.plot(years, budget, linestyle='--', linewidth=3, color='.2', label='All Movies')
text_kwargs=dict(fontsize=20, family='cursive')
ax.set_title('Median Movie Bduget', **text_kwargs)
ax.set_ylabel('Millions of Dollars', **text_kwargs)

movie_count = movie.groupby('title_year')['budget'].count()
movie_count.tail()
ct = movie_count.values
budget.max()
ct_norm = ct/ct.max() * budget.max()

fifth_year= (years % 5 == 0) & (years >= 1970)
years_5 = years[fifth_year]
ct_5 = ct[fifth_year]
ct_norm_5 = ct_norm[fifth_year]

ax.bar(years_5, ct_norm_5, 3, facecolor = '.5', alpha=.3, label='Movies per Year')
ax.set_xlim(1968, 2017)
for x, y, v, in  zip(years_5, ct_norm_5, ct_5):
    ax.text(x, y + .5, str(v), ha='center')
ax.legend()
fig

top10 = movie.sort_values('budget', ascending=False)\
.groupby('title_year')['budget']\
.apply(lambda x : x.iloc[:10].median() / 1e6)

top10_roll = top10.rolling(5, min_periods=1).mean()
top10_roll.tail()

fig2, ax_array = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
ax1 = ax_array[0]
ax2 = ax_array[1]

ax1.plot(years, budget, linestyle='--', linewidth=3, color='.2', label='All Movies')
ax1.bar(years_5, ct_norm_5, 3, facecolor='.5', alpha=.3, label='Movie per Year')
ax1.legend(loc='upper left')
ax1.set_xlim(1968, 2017)
plt.setp(ax1.get_xticklines(), visible=False)
for x, y, v in zip(years_5, ct_norm_5, ct_5):
    ax1.text(x, y+.5, str(v), ha='center')

ax2.plot(years, top10_roll.values, color='.2', label = 'Top 10 Movies')
ax2.legend(loc='upper left')
fig2.tight_layout()
fig2.suptitle('Median Movie Budget', y=1.02, **text_kwargs)
fig2.text(0, .6, 'Million of Dollars', rotation='vertical', ha='center', **text_kwargs)

import os
path = os.path.expanduser('~/Desktop/movie_budget.png')
fig2.savefig(path, bbox_inches='tight')

# rolling
med_budget = movie.groupby('title_year')['budget'].median()/1e6
med_budget.head()
med_budget.loc[2012:2016].mean()
med_budget.loc[2011:2015].mean()
med_budget.loc[2010:2014].mean()
med_budget_roll = med_budget.rolling(5, min_periods=1).mean()
med_budget_roll.tail()

movie = pd.read_csv('./data/movie.csv')
cols = ['budget', 'title_year', 'imdb_score', 'movie_title']
m = movie[cols].dropna()
m['budget2'] = m['budget']/ 1e6
np.random.seed(0)
movie_samp = m.query('title_year >= 2000').sample(100)
fig, ax = plt.subplots(figsize=(14, 6))
ax.scatter(x='title_year', y='imdb_score', s='budget2', data=movie_samp)
ax.scatter(x='title_year', y='imdb_score', s='budget2', data=movie_samp)
idx_min = movie_samp['imdb_score'].idxmin()
idx_max = movie_samp['imdb_score'].idxmax()

# annotating
for idx, offset in zip([idx_min, idx_max], [.5, -.5]):
    year = movie_samp.loc[idx, 'title_year']
    score = movie_samp.loc[idx, 'imdb_score']
    title = movie_samp.loc[idx, 'movie_title']
    ax.annotate(xy = (year, score), # data locaition 
                xytext=(year+1, score + offset), # comment string locaition 
                s = title + ' ({})'.format(score),
                ha = 'center',
                size=16,
                arrowprops= dict(arrowstyle='fancy'))
    
ax.set_title('IMDB Score by Year', size=25)
ax.grid(True)
fig

# pandas plot() method is kind of the wrapper of matplotlib

# plotting one var, two var
df = pd.DataFrame(index=['Atiya', 'Abbas', 'Cornelia', 'Stephanie', 'Monte'],
                  data={'Apples':[20, 10, 40, 20, 50],
                        'Oranges':[35, 40, 25, 19, 33]})
color = ['.2', '.7']
df.plot(kind='bar', color=color, figsize=(16, 4))
df.plot(kind='kde', color=color, figsize=(16, 4))

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 4))
fig.suptitle('Two Varialbe Plots', size=20, y=1.02)
df.plot(kind='line', color=color, ax=ax1, title='Line Plot')
df.plot(x='Apples', y='Oranges', kind='scatter', color=color, ax=ax2, title='Scatterplot')
df.plot(kind='bar', color=color, ax=ax3, title='Bar plot')


fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 4))
fig.suptitle('One Variable Plots', size=20, y=1.02)
df.plot(kind='kde', color = color, ax=ax1, title='KDE plot')
df.plot(kind='box', ax=ax2, title='Boxplot')
df.plot(kind='hist', color=color, ax=ax3, title='Histogram')

# flights dataset plotting
flights = pd.read_csv('./data/flights.csv')
flights.head()
flights.columns
flights.info()
flights.describe()
flights['DELAYED'] = flights['ARR_DELAY'].ge(15).astype(int)
flights.head()
cols = ['DIVERTED', 'CANCELLED', 'DELAYED']
flights[cols]
flights['ON_TIME'] = 1-flights[cols].any(axis=1)  
flights.head()     
cols.append('ON_TIME')
status = flights[cols].sum()
status

fig, ax_array = plt.subplots(2, 3, figsize=(18, 8))
(ax1, ax2, ax3), (ax4, ax5, ax6) = ax_array
fig.suptitle('2015 US Flights - Univariate Summary', size=20)
ac = flights['AIRLINE'].value_counts()
ac.plot(kind='barh', ax=ax1, title='Airline')
oc = flights['ORG_AIR'].value_counts()
oc.plot(kind='bar', ax=ax2, rot=0, title='Origin City')
dc = flights['DEST_AIR'].value_counts().head(10)
dc.plot(kind='bar', ax=ax3, rot=0, title='Destination City')
status.plot(kind='bar', ax=ax4, rot=0, log=True, title='Flight Status')
flights['DIST'].plot(kind='kde', ax=ax5, xlim=(0, 3000), title='Distance KDE')
flights['ARR_DELAY'].plot(kind='hist', ax=ax6, title='Arrival Delay', range=(0,200))

flights['SCHED_DEP'].plot(kind='box')
hour = flights['SCHED_DEP'] // 100
minute = flights['SCHED_DEP'] % 100
df_date = flights[['MONTH', 'DAY']].assign(YEAR=2015, HOUR=hour, MINUTE=minute)
df_date.head()
flight_dep = pd.to_datetime(df_date)
flight_dep.head()
flights.index = flight_dep
fc = flights.resample('W').size()
fc.plot(figsize=(12, 3), title='Flights per Week', grid=True)

fc_miss = fc.where(fc>1000)
fc_intp = fc_miss.interpolate(limit_direction='both')
fc_intp
ax = fc_intp.plot(color = 'black', figsize=(16, 4))
fc_intp[fc<500].plot(linewidth=10, grid=True, color='.8', ax=ax)
ax.annotate(xy=(.8, .55), xytext=(.8, .77), 
            xycoords='axes fraction', s='missing data',
            ha='center', size=20, arrowprops=dict())
ax.set_title('Flights per Week (Interpolated Missing Data)')

flights.groupby('DEST_AIR')['DIST']\
.agg(['mean', 'count'])\
.query('count > 100')\
.sort_values('mean')\
.tail(10).plot(kind='bar', y='mean', rot=0, legend=False, 
     title='Average Distance per Destination')

fs = flights.reset_index(drop=True)[['DIST', 'AIR_TIME']]\
.query('DIST <= 2000').dropna()

fs.plot(x='DIST', y='AIR_TIME', kind='scatter', s=1, figsize=(20, 15))

fs['DIST_GROUP'] = pd.cut(fs['DIST'], bins=range(0, 2001, 250))
fs['DIST_GROUP'].value_counts().sort_index()
normalize = lambda x : (x - x.mean()) / x.std()
fs['TIME_SCORE'] = fs.groupby('DIST_GROUP')['AIR_TIME']\
.transform(normalize)
fs.head()
ax = fs.boxplot(by='DIST_GROUP', column='TIME_SCORE', figsize = (16,4))
ax.set_title('Z_scores for Distance Groups')
ax.figure.suptitle('')

fs[fs['TIME_SCORE'] > 6].index
flights.iloc[fs[fs['TIME_SCORE'] > 6].index]
outliers = flights.iloc[fs[fs['TIME_SCORE'] > 6].index]
outliers = outliers[['AIRLINE', 'ORG_AIR', 'DEST_AIR', 'AIR_TIME', 'DIST', 'ARR_DELAY', 'DIVERTED']]
outliers['PLOT_NUM'] = range(1, len(outliers) + 1)
outliers

ax = fs.plot(x='DIST', y='AIR_TIME', kind='scatter', s=1, figsize=(16, 4), table=outliers)
outliers.plot(x='DIST', y='AIR_TIME', kind='scatter', s=25, ax=ax, grid=True)
outs = outliers[['AIR_TIME', 'DIST', 'PLOT_NUM']]
for t, d, n in outs.itertuples(index=False):
    ax.text(d+5, t+5, str(n))
plt.setp(ax.get_xticklabels(), y=.1)
plt.setp(ax.get_xticklines(), visible=False)
ax.set_xlabel('')
ax.set_title('Flight Time vs Distance with Outliers')

# meetup data
meetup = pd.read_csv('./data/meetup_groups.csv', parse_dates=['join_date'],
                     index_col='join_date')
meetup.head()

group_count = meetup.groupby([pd.Grouper(freq='W'), 'group']).size()
group_count.head()
gc2 = group_count.unstack('group', fill_value=0)
gc2.tail()
group_total = gc2.cumsum()
group_total.tail()
row_total = group_total.sum(axis='columns')
group_cum_pct = group_total.div(row_total, axis='index')
group_cum_pct.tail()

ax = group_cum_pct.plot(kind='area', figsize=(18, 4), cmap='Greys', 
                        xlim=('2013-6', None), 
                        ylim=(0, 1), legend=False)

ax.figure.suptitle('Houston Meetup Groups', size=25)
ax.set_xlabel('')
ax.yaxis.tick_right()
plot_kwargs = dict(xycoords='axes fraction',  size=15)
ax.annotate(xy=(.1, .7), s='R users', color = 'w', **plot_kwargs)
ax.annotate(xy=(.25, .16), s='Data Visualization', color = 'k', **plot_kwargs)
ax.annotate(xy=(.5, .55), s='Energy Data Science', color = 'k', **plot_kwargs)
ax.annotate(xy=(.83, .07), s='Data Science', color = 'k', **plot_kwargs)
ax.annotate(xy=(.86, .78), s='Machine Learning', color = 'w', **plot_kwargs)

pie_data = group_cum_pct.asfreq('3MS', method='bfill').tail(6).to_period('M').T
pie_data

from matplotlib.cm import Greys
greys = Greys(np.arange(50, 250, 40))
# np.arange(50,250,40)
# greys.shape
ax_array = pie_data.plot(kind='pie', subplots=True, layout=(2, 3), labels=None,
                         autopct='%1.0f%%', pctdistance=1.22, colors=greys)

ax1 = ax_array[0,0]
ax1.figure.legend(ax1.patches, pie_data.index, ncol=3)
for ax in ax_array.flatten():
    ax.xaxis.label.set_visible(True)
    ax.set_xlabel(ax.get_ylabel())
    ax.set_ylabel('')
ax1.figure.subplots_adjust(hspace=.3)

employee = pd.read_csv('./data/employee.csv', parse_dates=['HIRE_DATE', 'JOB_DATE'])
employee.head()

import seaborn as sns
sns.countplot(y= 'DEPARTMENT', data=employee)
employee['DEPARTMENT'].value_counts().plot('barh')

ax = sns.barplot(x='RACE', y='BASE_SALARY', data=employee)
ax.figure.set_size_inches(16, 4)

avg_sal = employee.groupby('RACE', sort=False)['BASE_SALARY'].mean()
ax = avg_sal.plot(kind='bar', rot=0, figsize=(16,4), width=.8)
ax.set_xlim(-.5, 5.5)
ax.setylabel('Mean Salary')

ax=sns.barplot(x='RACE', y='BASE_SALARY', hue='GENDER', data=employee, palette='Greys')
ax.figure.set_size_inches(16, 4)

employee.groupby(['RACE', 'GENDER'], sort=False)['BASE_SALARY'].mean().unstack('GENDER')\
.plot(kind='bar', figsize=(16,4), rot=0, width=.8, cmap='Greys')

sns.boxplot(x='GENDER', y='BASE_SALARY', data=employee, hue='RACE', palette='Greys')
ax.figure.set_size_inches(16, 4)

fig, ax_array = plt.subplots(1, 2, figsize=(14,4), sharey=True)
for g, ax in zip(['Female', 'Male'], ax_array):
    employee.query('GENDER==@g')\
    .boxplot(by='RACE', column='BASE_SALARY', 
             ax=ax, rot=20)
    ax.set_title(g + 'Salary')
    ax.set_xlabel('')
fig.suptitle('')

ax = employee.boxplot(by=['GENDER', 'RACE'], column='BASE_SALARY', figsize=(16, 4), 
                      rot=15)
ax.figure.suptitle('')

employee = pd.read_csv('./data/employee.csv')
employee.head()
employee = pd.read_csv('./data/employee.csv', parse_dates = ['HIRE_DATE', 'JOB_DATE'])
employee.head()
employee.dtypes
days_hired = pd.to_datetime('12-1-2016') - employee['HIRE_DATE']
one_year = pd.Timedelta(1, unit='Y')
one_year
employee['YEARS_EXPERIENCE'] = days_hired/one_year
employee[['HIRE_DATE', 'YEARS_EXPERIENCE']].head()
ax = sns.regplot(x = 'YEARS_EXPERIENCE', y='BASE_SALARY', data=employee)
ax.figure.set_size_inches(14,4)

g = sns.lmplot('YEARS_EXPERIENCE', 'BASE_SALARY', hue='GENDER', palette='Greys',
               scatter_kws={'s' : 10}, data=employee)
g.fig.set_size_inches(14,4)
type(g)

grid = sns.lmplot(x = 'YEARS_EXPERIENCE', y = 'BASE_SALARY', hue='GENDER', 
                  col='RACE', col_wrap=3, 
                  palette='Greys', sharex=False,
                  line_kws = {'linewidth' : 5}, data=employee)
grid.set(ylim = (20000, 120000))

deps = employee['DEPARTMENT'].value_counts().index[:2]
races = employee['RACE'].value_counts().index[:3]
is_dep = employee['DEPARTMENT'].isin(deps)
is_race = employee['RACE'].isin(races)
emp2 = employee[is_dep & is_race].copy()
emp2['DEPARTMENT'] = emp2['DEPARTMENT'].str.extract('(HPD|HFD)', expand=True)
emp2.head()
emp2.shape
emp2['DEPARTMENT'].value_counts()
emp2['RACE'].value_counts()


sns.factorplot(x = 'YEARS_EXPERIENCE', y='GENDER', col='RACE', row='DEPARTMENT',
               size=3, aspect=2, data=emp2, kind='violin')

diamonds = pd.read_csv('./data/diamonds.csv')
diamonds.head()
diamonds['cut'].unique()
cut_cats = ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal']
diamonds['color'].unique()
color_cats = ['J', 'I', 'H', 'G', 'F', 'E', 'D']
diamonds['clarity'].unique()
clarity_cats = ['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF']
diamonds['cut'] = pd.Categorical(diamonds['cut'], categories=cut_cats, ordered=True)
diamonds['color'] = pd.Categorical(diamonds['color'], categories=color_cats, ordered=True)
diamonds['clarity'] = pd.Categorical(diamonds['clarity'], categories=clarity_cats, ordered=True)

import seaborn as sns
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14,4))
sns.barplot(x='color', y='price', data=diamonds, ax=ax1)
sns.barplot(x='cut', y='price', data=diamonds, ax=ax2)
sns.barplot(x='clarity', y='price', data=diamonds, ax=ax3)
fig.suptitle('Price Decreasing with Increasing Quality?')

sns.factorplot(x='color', y='price', col='clarity', col_wrap=4, 
               data=diamonds, kind='bar')

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 4))
sns.barplot(x='color', y='carat', data=diamonds, ax=ax1)
sns.barplot(x='cut', y='carat', data=diamonds, ax=ax2)
sns.barplot(x='clarity', y='carat', data=diamonds, ax=ax3)
fig.suptitle('Diamond size decrease with quality')

diamonds['carat_category'] = pd.qcut(diamonds.carat, 5)
from matplotlib.cm import Greys
greys = Greys(np.arange(50,250,40))
g = sns.factorplot(x='clarity', y='price', data=diamonds, hue='carat_category', 
                   col='color', col_wrap=4, kind='point', palette=greys)
g.fig.suptitle('Diamond price by size, color and clarity', y=1.02, size=20)

g =sns.PairGrid(diamonds, size=5, 
                x_vars=['color', 'cut', 'clarity'],
                y_vars=['price'])
g.map(sns.barplot)
g.fig.suptitle('Replication with PairGrid', y=1.02)


























