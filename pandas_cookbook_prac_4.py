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

