# -*- coding: utf-8 -*-
"""
Created on Mon Jul  9 15:44:25 2018

@author: dsc
"""
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

crime.groupb(weekday)
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


