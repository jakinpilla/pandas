# -*- coding: utf-8 -*-
"""
Created on Tue Jul  3 10:49:49 2018

@author: dsc
"""

from os import getcwd
getcwd()
import numpy as np
import pandas as pd
employee = pd.read_csv('data/employee.csv')

employee.DEPARTMENT.value_counts().head()
employee.BASE_SALARY.describe().astype(int)

depts = ['Houston Police Department-HPD', 'Houston Fire Department (HFD)']
criteria_dept = employee.DEPARTMENT.isin(depts)
criteria_gender = employee.GENDER == 'Female'
criteria_sal = (employee.BASE_SALARY >= 80000) & (employee.BASE_SALARY <= 120000)
criteria_final= (criteria_dept & criteria_gender & criteria_sal)

select_columns = ['UNIQUE_ID', 'DEPARTMENT', 'GENDER', 'BASE_SALARY']
employee.loc[criteria_final, select_columns].head()

criteria_sal = employee.BASE_SALARY.between(80000, 120000)
top_5_depts = employee.DEPARTMENT.value_counts().index[:5]
criteria = ~employee.DEPARTMENT.isin(top_5_depts)
employee[criteria].head()

## validate stock margin normality

amzn = pd.read_csv('./data/amzn_stock.csv', index_col = 'Date', parse_dates=['Date'])
amzn.head()

amzn_daily_return = amzn.Close.pct_change()
amzn_daily_return.head()

amzn_daily_return= amzn_daily_return.dropna()
amzn_daily_return.hist(bins=20)


mean = amzn_daily_return.mean()
std = amzn_daily_return.std()

abs_z_score = amzn_daily_return.sub(mean).abs().div(std)

abs_z_score.lt(1)

pcts = [abs_z_score.lt(i).mean() for i in range(1, 4)]
pcts

def test_return_normality(stock_data):
    close = stock_data['Close']
    daily_return = close.pct_change().dropna()
    daily_return.hist(bins=20)
    mean = daily_return.mean()
    std = daily_return.std()
    
    abs_z_score = abs(daily_return - mean) / std
    pcts = [abs_z_score.lt(i).mean() for i in range(1, 4)]
    
    print('{:.3f} fall within 1 standard deviation. '
          '{:.3f} within and {:.3f} within 3'.format(*pcts))
    
slb = pd.read_csv('./data/slb_stock.csv', index_col = 'Date', parse_dates = ['Date'])
test_return_normality(slb)

qs = "DEPARTMENT in @depts and GENDER == 'Female' and 80000 <= BASE_SALARY <= 120000"

emp_filtered = employee.query(qs)
emp_filtered[select_columns].head()

top10_depts = employee.DEPARTMENT.value_counts().index[:10].tolist()
qs = "DEPARTMENT not in @top10_depts and GENDER == 'Female'"
employee_filtered2 = employee.query(qs)
employee_filtered2.head()

# presercve series with .where()
movie = pd.read_csv('./data/movie.csv', index_col = 'movie_title')
fb_likes = movie['actor_1_facebook_likes'].dropna()
fb_likes.head()
fb_likes.describe(percentiles=[.1, .25, .5, .75, .9]).astype(int)
fb_likes.hist()

criteria_high= fb_likes < 20000
criteria_high.mean().round(2)
fb_likes.where(criteria_high).head()
fb_likes.where(criteria_high, other=20000).head()
criteria_low = fb_likes > 300
fb_likes_cap = fb_likes.where(criteria_high, other=20000).where(criteria_low,300)
fb_likes_cap.head()
len(fb_likes), len(fb_likes_cap)
fb_likes_cap.hist()
fb_likes_cap2 = fb_likes.clip(lower=300, upper=20000)
fb_likes_cap2.equals(fb_likes_cap)

# dataframe row mask
movie = pd.read_csv('./data/movie.csv', index_col = 'movie_title')
c1 = movie['title_year'] >= 2010
c2 = movie['title_year'].isnull()
criteria = c1 | c2
movie.mask(criteria).head()
movie_mask = movie.mask(criteria).dropna(how='all')
movie_mask.head()
movie_boolean = movie[movie['title_year'] < 2010]
movie_mask.equals(movie_boolean)
movie_boolean = movie[movie['title_year'] < 2010]
movie_mask.equals(movie_boolean)
movie_mask.shape == movie_boolean.shape
movie_mask.dtypes == movie_boolean.dtypes
from pandas.testing import assert_frame_equal
assert_frame_equal(movie_boolean, movie_mask, check_dtype=False) ## no printing if not False

# boolean and .loc, .iloc

