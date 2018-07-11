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
import pandas as pd
import datetime

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
pd.to_datetime(date_string_list, format='%b %d %Y')
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
crime.groupby([year, weekday]).size() # error

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

pd.merge_asof(goal, all_data, left_on = 'Total_Goal', 
              right_on = 'Total', by='OFFENSE_CATEGORY_ID', 
              suffixes=('_Current', '_Last'))

goal.info()
all_data.info()
all_data['Total'] = all_data['Total'].astype(int32)
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

import matplotlib.pyplot as plt
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
.groupby('title_year')['budget'].apply(lambda x : x.iloc[:10].median() / 1e6)

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
















































