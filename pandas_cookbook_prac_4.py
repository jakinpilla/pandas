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
