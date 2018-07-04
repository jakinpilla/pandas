# -*- coding: utf-8 -*-
"""
Created on Tue Jul  3 10:49:49 2018

@author: dsc
"""

from os import getcwd, chdir
getcwd()
chdir("C:/Users/dsc/pandas")
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

top10_depts = employee.DEPARTMENT.value_counts().index[:10].tolist()
qs = "DEPARTMENT not in @top10_depts and GENDER == 'Female'"
employee_filtered2 = employee.query(qs)
employee_filtered2.head()

movie = pd.read_csv('./data/movie.csv', index_col = 'movie_title')
fb_likes = movie['actor_1_facebook_likes'].dropna()
fb_likes.head()

fb_likes.describe(percentiles=[.1, .25, .5, .75, .9])
fb_likes.hist()
criteria_high = fb_likes < 20000
criteria_high.mean().round(2)

# selecting data with boolea and loc, iloc
movie=pd.read_csv('./data/movie.csv', index_col = 'movie_title')
c1 = movie['content_rating'] == 'G'
c2 = movie['imdb_score'] < 4
criteria = c1 & c2
movie_loc = movie.loc[criteria]
movie_loc.head()
ovie_loc.equals(movie[criteria])

# we cannot use boolean series in iloc, we need to use boolean ndarray
movie_iloc = movie.iloc[criteria.values]
movie_iloc.equals(movie_loc)
criteria_col = movie.dtypes == np.int64
criteria_col.head()
movie.loc[:, criteria_col].head()
movie.iloc[:, criteria_col.values].head()
cols=['content_rating', 'imdb_score', 'title_year', 'gross']
movie.loc[criteria, cols].sort_values('imdb_score')
col_index = [movie.columns.get_loc(col) for col in cols]
col_index
movie.iloc[criteria.values, col_index]

a = criteria.values
a[:5]
len(a), len(criteria)
movie.loc[[True, False, True], [True, False, False, True]]

# Index arranging
college = pd.read_csv('./data/college.csv')
columns = college.columns
columns
columns.values
columns[5]
columns[[1, 8, 10]]
columns[-7:-4]
columns.min(), columns.max(), columns.isnull().sum()
columns + '_A'
columns > 'G'






