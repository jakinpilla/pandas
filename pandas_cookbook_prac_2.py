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
movie_loc.equals(movie[criteria])

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

c1 = columns[:4] ; c1
c2 = columns[2:6] ; c2
c1.union(c2)
c1.symmetric_difference(c2)

# cartesian product
s1 = pd.Series(index=list('aaab'), data=np.arange(4))
s1
s2 = pd.Series(index=list('cababb'), data = np.arange(6))
s2
s1+s2
s1 = pd.Series(index=list('aaabb'), data=np.arange(5))
s2 = pd.Series(index=list('aaabb'), data=np.arange(5))
s1+s2
s1 = pd.Series(index=list('aaabb'), data=np.arange(5))
s2 = pd.Series(index=list('bbaaa'), data=np.arange(5))
s1+s2

# explode index
employee = pd.read_csv('./data/employee.csv', index_col='RACE')
employee.head()
salary1 = employee['BASE_SALARY']
salary2 = employee['BASE_SALARY']
salary1 is salary2
salary1 = employee['BASE_SALARY'].copy()
salary2 = employee['BASE_SALARY'].copy()
salary1 is salary2

salary1 = salary1.sort_index()
salary1.head()
salary2.head()

salary_add = salary1 + salary2
salary_add.head()

salary_add1 = salary1 + salary1

len(salary1), len(salary2), len(salary_add), len(salary_add1)

index_vc = salary1.index.value_counts(dropna = False)
index_vc.pow(2).sum()

# fill_value()

baseball_14 = pd.read_csv('./data/baseball14.csv', index_col ='playerID')
baseball_15 = pd.read_csv('./data/baseball15.csv', index_col ='playerID')
baseball_16 = pd.read_csv('./data/baseball16.csv', index_col ='playerID')

baseball_14.head()
baseball_14.index.difference(baseball_15.index)
baseball_14.index.difference(baseball_16.index)

hits_14 = baseball_14['H']
hits_15 = baseball_15['H']
hits_16 = baseball_16['H']
hits_14.head()
(hits_14 + hits_15).head() ## occur NaN
hits_14.add(hits_15, fill_value=0).head()
hits_total  = hits_14.add(hits_15, fill_value=0).add(hits_16, fill_value=0)
hits_total.head()
hits_total.hasnans

# if all elements are 'nan', adding Series's result is also 'nan' even though fill_values() method is used.
s = pd.Series(index=['a', 'b', 'c', 'd'],
              data = [np.nan, 3, np.nan, 1])
s
s1 = pd.Series(index=['a', 'b', 'c'], data=[np.nan, 6, 10])
s1
s.add(s1, fill_value=5)

df_14 = baseball_14[['G', 'AB', 'R', 'H']]
df_14.head()
df_15 = baseball_15[['AB', 'R', 'H', 'HR']]
df_15.head()
(df_14 + df_15).head(10).style.highlight_null('yellow') # how can I see this result in spyder IDE...
df_14.add(df_15, fill_value=0).head(10).style.highlight_null('yellow')
