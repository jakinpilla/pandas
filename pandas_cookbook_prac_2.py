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

# extracting persons who recieve max salary in each department
employee = pd.read_csv('./data/employee.csv')
dept_sal =employee[['DEPARTMENT', 'BASE_SALARY']]
dept_sal = dept_sal.sort_values(['DEPARTMENT', 'BASE_SALARY'], ascending=[True, False])
max_dept_sal = dept_sal.drop_duplicates(subset = 'DEPARTMENT')
len(max_dept_sal)
max_dept_sal
max_dept_sal.columns
max_dept_sal = max_dept_sal.set_index('DEPARTMENT')
max_dept_sal
len(max_dept_sal)
employee = employee.set_index('DEPARTMENT')
len(employee)
employee.columns
employee['MAX_DEPT_SALARY'] = max_dept_sal['BASE_SALARY'] # we can use this code because employee row index is exactly matched with max_deptsal index one by one
employee.head()
len(employee)
employee.query('BASE_SALARY > MAX_DEPT_SALARY')

np.random.seed(1234)
random_salary = dept_sal.sample(n=10).set_index('DEPARTMENT')
random_salary

employee['MAX_SALARY2'] = max_dept_sal['BASE_SALARY'].head(3)
employee
max_dept_sal['BASE_SALARY'].head(3)
employee.MAX_SALARY2.value_counts()
employee.MAX_SALARY2.isnull().mean()
college = pd.read_csv('./data/college.csv', index_col = 'INSTNM')
college.dtypes
college.MD_EARN_WNE_P10.iloc[0]
college.GRAD_DEBT_MDN_SUPP.iloc[0]

college.MD_EARN_WNE_P10.sort_values(ascending=False).head()
cols = ['MD_EARN_WNE_P10', 'GRAD_DEBT_MDN_SUPP']
for col in cols:
    college[col] = pd.to_numeric(college[col], errors ='coerce')

college.dtypes.loc[cols]
college_n = college.select_dtypes(include=[np.number])
college_n.head()
criteria = college_n.nunique() == 2
criteria
binary_cols = college_n.columns[criteria].tolist()
binary_cols
college_n2 = college_n.drop(labels=binary_cols, axis = 'columns')
college_n2.head()
max_cols = college_n2.idxmax()
max_cols
unique_max_cols = max_cols.unique()
unique_max_cols[:5]
college_n2.loc[unique_max_cols].style.highlight_max()
college = pd.read_csv('./data/college.csv', index_col='INSTNM')
college_ugds = college.filter(like='UGDS_').head()
college_ugds.style.highlight_max(axis='columns')


college = pd.read_csv('./data/college.csv', index_col = 'INSTNM')
cols = ['MD_EARN_WNE_P10', 'GRAD_DEBT_MDN_SUPP']
for col in cols:
    college[col] = pd.to_numeric(college[col], errors ='coerce')
college_n = college.select_dtypes(include=[np.number])
criteria = college_n.nunique() == 2
binary_cols = college_n.columns[criteria].tolist()
college_n = college_n.drop(labels=binary_cols, axis = 'columns')
college_n.max().head()
college_n.eq(college_n.max()).head()
has_row_max = college_n.eq(college_n.max()).any(axis='columns')
has_row_max.head()
college_n.shape
has_row_max.sum()
college_n.eq(college_n.max()).cumsum().cumsum()
has_row_max2 = college_n.eq(college_n.max()).cumsum().cumsum().eq(1).any(axis='columns')
has_row_max2.head()
has_row_max2.sum()
idxmax_cols= has_row_max2[has_row_max2].index
idxmax_cols
set(college_n.idxmax().unique()) == set(idxmax_cols)

#m most frequent max values
college_ugds = college.filter(like='UGDS_')
highest_percentage_race = college_ugds.idxmax(axis='columns')
highest_percentage_race.value_counts(normalize =True)
college_black = college_ugds[highest_percentage_race == 'UGDS_BLACK']
college_black = college_black.drop('UGDS_BLACK', axis='columns')
college_black.idxmax(axis='columns').value_counts(normalize=True)

# group, filter, transform for aggregation
# split-apply-combine
flights = pd.read_csv('./data/flights.csv')
flights.head()
flights.groupby('AIRLINE').agg({'ARR_DELAY' : 'mean'}).head()
flights.groupby('AIRLINE')['ARR_DELAY'].agg('mean').head()
flights.groupby('AIRLINE')['ARR_DELAY'].agg(np.mean).head()
flights.groupby('AIRLINE')['ARR_DELAY'].mean().head()
grouped = flights.groupby('AIRLINE')
type(grouped)

flights.groupby(['AIRLINE', 'WEEKDAY'])['CANCELLED'].agg('sum').head(20)
flights.groupby(['AIRLINE', 'WEEKDAY'])['CANCELLED', 'DIVERTED'].agg('sum').head(20)

group_cols = ['ORG_AIR', 'DEST_AIR']
agg_dict = {'CANCELLED' : ['sum', 'mean', 'size'], 
            'AIR_TIME' : ['mean', 'var']}
flights.groupby(group_cols).agg(agg_dict).head()

# remove multi-index after groupby()
flights = pd.read_csv('./data/flights.csv')
airline_info = flights.groupby(['AIRLINE', 'WEEKDAY'])\
.agg({'DIST' : ['sum', 'mean'], 'ARR_DELAY' : ['min', 'max']}).astype(int)

airline_info.head(7)
level0 = airline_info.columns.get_level_values(0)
level0
level1 = airline_info.columns.get_level_values(1)
level1
airline_info.columns = level0 + '_' + level1
airline_info.head()
airline_info.reset_index().head() # row multiindex can be single index with reset_index() method easily
flights.groupby(['AIRLINE'], as_index=False)['DIST'].agg('mean').round(0)
college = pd.read_csv('./data/college.csv')
college.groupby('STABBR')['UGDS'].agg(['mean', 'std']).round(0).head()

def max_deviation(s):
    std_score = (s - s.mean()) / s.std()
    return std_score.abs().max()

college.groupby('STABBR')['UGDS'].agg(max_deviation).round(1).head()
college.groupby('STABBR')['UGDS', 'SATVRMID', 'SATMTMID'].agg(max_deviation).round(1).head()
college.groupby(['STABBR', 'RELAFFIL'])['UGDS', 'SATVRMID', 'SATMTMID'].agg([max_deviation, 'mean', 'std'])\
.round(1).head()

max_deviation.__name__
max_deviation.__name__ = 'Max Deviation'
college.groupby(['STABBR', 'RELAFFIL'])\
['UGDS', 'SATVRMID', 'SATMTMID'].agg([max_deviation, 'mean', 'std']).round(1).head()

# *arg, **kwargs / arbitrary argument list

college = pd.read_csv('./data/college.csv')
grouped = college.groupby(['STABBR', 'RELAFFIL'])
import inspect
inspect.signature(grouped.agg)

def pct_between_1_3k(s):
    return s.between(1000, 3000).mean()

college.groupby(['STABBR', 'RELAFFIL'])['UGDS'].agg(pct_between_1_3k).head(9)

def pct_between(s, low, high):
    return s.between(low, high).mean()

college.groupby(['STABBR', 'RELAFFIL'])['UGDS'].agg(pct_between, 1000, 10000).head(9)
college.groupby(['STABBR', 'RELAFFIL'])['UGDS'].agg(pct_between, high=10000, low=1000).head(9)
college.groupby(['STABBR', 'RELAFFIL'])['UGDS'].agg(pct_between, 1000, high=10000).head(9)

# *arg --> tuples / **kwargs --> dictionary

def make_agg_func(func, name, *args, **kwargs):
    def wrapper(x):
        return func(x, *args, **kwargs)
    wrapper.__name__ = name
    return wrapper

my_agg1 = make_agg_func(pct_between, 'pct_1_3k', low=1000, high=3000)
my_agg2 = make_agg_func(pct_between, 'pct_10_30k', 10000, 30000)
college.groupby(['STABBR', 'RELAFFIL'])['UGDS'].agg(['mean', my_agg1, my_agg2]).head()

# groupby() objects
college = pd.read_csv('./data/college.csv')
grouped = college.groupby(['STABBR', 'RELAFFIL'])
type(grouped)
print([attr for attr in dir(grouped) if not attr.startswith('_')])
grouped.ngroups
groups = list(grouped.groups.keys())
groups[:5]
grouped.get_group(('FL', 1)).head()


