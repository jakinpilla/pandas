# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from os import getcwd, chdir
wd = getcwd()
import pandas as pd
import numpy as np
# chdir("E:/pandas")
chdir(wd)
getcwd()
movie = pd.read_csv("./data/movie.csv")
index = movie.index
movie.head()
columns = movie.columns
movie.info()
data = movie.values
index
columns
data
type(index)
type(columns)
type(data)

## issubclass
issubclass(pd.RangeIndex, pd.Index)

## pandas.core.indexes.base.Index  : package name, modules path, type
## pandas index object

## RangeIndex is similer to python's range object. 
## because it is exactly defined only by start value, last value and step value, 
## so only less memory is needed

movie.dtypes
movie.get_dtype_counts()
movie['director_name']
movie.director_name
type(movie['director_name'])

director = movie['director_name']
director.name ## return "director_name"
director.to_frame()

s_attr_methods = set(dir(pd.Series))
len(s_attr_methods)
df_attr_methods = set(dir(pd.DataFrame))
len(df_attr_methods)

len(s_attr_methods & df_attr_methods)

movie = pd.read_csv('./data/movie.csv')
director = movie['director_name']
actor_1_fb_likes = movie['actor_1_facebook_likes']
director.head()
actor_1_fb_likes.head()

director.value_counts()
actor_1_fb_likes.value_counts()
director.size
director.shape
len(director)

director.count() ## not counting "nan"
actor_1_fb_likes.count()
actor_1_fb_likes.min(), actor_1_fb_likes.max(), actor_1_fb_likes.mean(), actor_1_fb_likes.median(), \
actor_1_fb_likes.std(), actor_1_fb_likes.sum()

actor_1_fb_likes.describe()
actor_1_fb_likes.quantile([.1, .2, .3, .4, .5, .6, .7, .8, .9])
director.isnull()

director.value_counts().head(3)
actor_1_fb_likes.isnull().sum()
actor_1_fb_likes.dtype
actor_1_fb_likes.fillna(0)\
.astype(int)\
.head()

actor_1_fb_likes.isnull
(actor_1_fb_likes.fillna(0)
.astype(int)
.head())

movie = pd.read_csv("./data/movie.csv")
movie2 = movie.set_index("movie_title")
movie2.head()

movie = pd.read_csv("./data/movie.csv", index_col = "movie_title")
movie2 = movie2.reset_index()
movie2.head()

idx_rename = {"Avatar" : "Ravatar", "Spectre" : "Ertceps"}
col_rename = {"director_name" : "Director Name", 
              "num_critic_for_reviews" : "Critical Reviews"}

movie_renamed = movie.rename(index = idx_rename, 
                             columns = col_rename)

movie_renamed.head()

movie = pd.read_csv("./data/movie.csv", index_col = "movie_title")
index = movie.index
columns = movie.columns
index_list =index.tolist()
column_list = columns.tolist()

# change columns labels by assigning list index
index_list[0] = 'Ravata'
index_list[2] = 'Ertceps'
column_list[1] = 'Director Name'
column_list[2] = 'Critical Reviews'

print(index_list)
print(column_list)

movie.index = index_list
movie.columns = column_list

movie = pd.read_csv("./data/movie.csv")
movie["has_seen"] = 0

movie["actor_director_facebook_likes"] =\
(movie["actor_1_facebook_likes"] +
 movie["actor_2_facebook_likes"] +
 movie["actor_3_facebook_likes"] +
 movie["director_facebook_likes"])

movie["actor_director_facebook_likes"].isnull().sum()
movie["actor_director_facebook_likes"] = \
movie["actor_director_facebook_likes"].fillna(0)

movie["is_cast_likes_more"] = \
(movie["cast_total_facebook_likes"] >= 
 movie["actor_director_facebook_likes"])

movie["is_cast_likes_more"].all()
movie = movie.drop("actor_director_facebook_likes", axis = "columns")

movie["actor_total_facebook_likes"] = \
(movie["actor_1_facebook_likes"] +
 movie["actor_2_facebook_likes"] +
 movie["actor_3_facebook_likes"])

movie["actor_total_facebook_likes"] = movie["actor_total_facebook_likes"].fillna(0) 

movie["is_cast_likes_more"] = \
(movie["cast_total_facebook_likes"] >=
 movie["actor_total_facebook_likes"])

movie["is_cast_likes_more"].all()

movie["pct_actor_cast_like"]  = \
(movie["actor_total_facebook_likes"] / movie["cast_total_facebook_likes"])

(movie["pct_actor_cast_like"].min(), movie["pct_actor_cast_like"].max())
movie.set_index("movie_title")["pct_actor_cast_like"].head()

## insert()

profit_index = movie.columns.get_loc('gross') + 1
profit_index
movie.insert(loc = profit_index, 
             column = 'profit',
             value = movie['gross'] - movie['budget'])

## del

movie["actor_director_facebook_likes"] =\
(movie["actor_1_facebook_likes"] +
 movie["actor_2_facebook_likes"] +
 movie["actor_3_facebook_likes"] +
 movie["director_facebook_likes"])

movie["actor_director_facebook_likes"].isnull().sum()
movie["actor_director_facebook_likes"] = \
movie["actor_director_facebook_likes"].fillna(0)
del movie['actor_director_facebook_likes']

## selecting multiful columns

movie_actor_director = movie[['actor_1_name', 'actor_2_name', 
                              'actor_3_name', 'director_name']]
movie_actor_director.head()
movie[['director_name']].head()
cols = ['actor_1_name', 'actor_2_name', 'actor_3_name', 'director_name']
movie_actor_director = movie[cols]

tuple1 = 1, 2, 3, 'a', 'b'
tuple2 = (1, 2, 3, 'a', 'b')
tuple1 == tuple2

## selecting multiful columns with methods

movie = pd.read_csv("./data/movie.csv", index_col = "movie_title")
movie.get_dtype_counts()
movie.select_dtypes(include = ['int64']).head()
movie.select_dtypes(include = ['number']).head()
movie.filter(like='facebook').head()
movie.filter(regex = '\d').head()
movie.filter(like='facebook').columns
movie.filter(regex = '\d').columns
movie.filter(items = ['actor_1_name', 'asdf']).head() # not raising KeyError

## arranging columns

## seperate discrete and continuous columns
## grouping commnon columns
## arrange the most important column first within grouped columns

movie = pd.read_csv("./data/movie.csv")
movie.head()
movie.columns
movie.info()

## seperate discrete columns
movie.select_dtypes(include = ['object']).columns
movie['content_rating'][:5]
disc_core = ['movie_title', 'title_year', 'content_rating', 'genres']

disc_people = ['director_name', 'actor_1_name', 'actor_2_name', 
               'actor_3_name']

disc_other = ['color', 'country', 'language', 'plot_keywords', 'movie_imdb_link']

cont_fb = ['director_facebook_likes', 'actor_1_facebook_likes', 
           'actor_2_facebook_likes', 'actor_3_facebook_likes', 
           'cast_total_facebook_likes', 'movie_facebook_likes']

cont_finance = ['budget', 'gross']

cont_num_reviews = ['num_voted_users', 'num_user_for_reviews', 
                    'num_critic_for_reviews']

cont_other = ['imdb_score', 'duration', 'aspect_ratio', 'facenumber_in_poster']

new_col_order = disc_core + disc_people + \
disc_other + cont_fb + cont_finance + cont_num_reviews + cont_other
set(movie.columns) == set(new_col_order)

movie2 = movie[new_col_order]
movie2.head()
movie2.columns

## operation for all dataframe

movie = pd.read_csv("./data/movie.csv")
movie.shape
movie.size
movie.ndim
len(movie)
movie.count()
movie.describe()
movie.describe(percentiles = [.01, .3, .99])
movie.min(skipna = False)
movie.min()
movie.isnull().sum()
movie.isnull().sum().sum()
movie.isnull().any()
movie.isnull().any().any()
movie.isnull().get_dtype_counts()
movie.select_dtypes(['object']).fillna('').min()
movie.select_dtypes(['object'])\
.fillna('')\
.min()

college = pd.read_csv("./data/college.csv")
# college + 5
collge = pd.read_csv("./data/college.csv", index_col = "INSTNM")
college_ugds_ = college.filter(like = "UGDS_")
college_ugds_.head()
college_ugds_ + 0.00501
(college_ugds_ + 0.00501) // 0.01
college_ugds_op_round = (college_ugds_ + 0.00501) // 0.01 / 100
college_ugds_op_round.head()
college_ugds_round = (college_ugds_ + .00001).round(2)
college_ugds_op_round.equals(college_ugds_round)
college_ugds_op_round_methods = college_ugds_.add(.00501)\
.floordiv(.01)\
.div(100)

college_ugds_op_round_methods.equals(college_ugds_op_round)

# np.nan == np. ## return False
None == None

college = pd.read_csv("./data/college.csv", index_col = "INSTNM")
college_ugds_ = college.filter(like = "UGDS_")
college_ugds_ == 0.0019
college_self_compare = college_ugds_ == college_ugds_
college_self_compare.head()
college_self_compare.all() ## return False because nan is not considered as identical each other
college_ugds_.isnull().sum()
college_ugds_.equals(college_ugds_)

college_ugds_.eq(.0019) # college_ugds_ == .0019

from pandas.testing import assert_frame_equal
assert_frame_equal(college_ugds_, college_ugds_) # there is no return why?

## dataframe axis

college = pd.read_csv("./data/college.csv", index_col = "INSTNM")
college_ugds_ = college.filter(like = 'UGDS_')
college_ugds_.head()
college_ugds_.count()
college_ugds_.count(axis = 'columns')
college_ugds_.sum(axis = 'columns').head()
college_ugds_.median(axis = 'index')
college_ugds_cumsum = college_ugds_.cumsum(axis = 1)
college_ugds_cumsum.head()

## diversity index

pd.read_csv('./data/college_diversity.csv', index_col = 'School')

college = pd.read_csv('./data/college.csv', index_col = 'INSTNM')
college_ugds_ = college.filter(like = 'UGDS_')
college_ugds_.isnull()\
.sum(axis = 1)\
.sort_values(ascending=False)\
.head()

college_ugds_ = college_ugds_.dropna(how = 'all')
college_ugds_.isnull().sum()
college_ugds_.ge(.15)

diversity_metric = college_ugds_.ge(.15).sum(axis = 'columns')
diversity_metric
diversity_metric.value_counts()
diversity_metric.sort_values(ascending = False).head()


college_ugds_.loc[['Regency Beauty Institute-Austin', 
                   'Central Texas Beauty College-Temple']]

us_news_top = ['Rutgers University-Newark',
               'Andrews University',
               'Stanford University',
               'University of Houston',
               'University of Nevada-Las Vegas']


diversity_metric.loc[us_news_top]
college_ugds_.max(axis = 1).sort_values(ascending=False).head(10)
(college_ugds_ > .01).all(axis = 1).any()

## EDA routine
college = pd.read_csv('./data/college.csv')
college.head()
college.shape
college.info()
college.describe().T
college.describe(include = [np.number]).T
college.describe(include = [np.object, pd.Categorical]).T
college.describe(include = [np.number] ,
                 percentiles = [.01, .05, .10, .25, .5,
                                .75, .9, .95, .99]).T

## data dictionary
data_dic = pd.read_csv('./data/college_data_dictionary.csv')

## memory saving
college = pd.read_csv('./data/college.csv')
different_cols = ['RELAFFIL', 'SATMTMID', 'CURROPER', 'INSTNM', 'STABBR']
col2 = college.loc[:, different_cols]
col2.info()
col2.head()
col2.dtypes
original_mem = col2.memory_usage(deep = True)
original_mem
col2['RELAFFIL'] = col2['RELAFFIL'].astype(np.int8)
col2.dtypes

college[different_cols].memory_usage(deep=True)
col2.select_dtypes(include = ['object']).nunique()
col2['STABBR'] = col2['STABBR'].astype('category')
col2.dtypes
new_mem = col2.memory_usage(deep = True)
new_mem

new_mem / original_mem

college.loc[0, 'CURROPER'] = 10000000
college.loc[0, 'INSTNM'] = college.loc[0, 'INSTNM'] + 'a'
college[['CURROPER', 'INSTNM']].memory_usage(deep = True)
college.describe(include = ['int8', 'float64']).T
college.describe(include = [np.int64, np.float64]).T
college.describe(include = ['int', 'float']).T
college.describe(include = ['number']).T


college['MENONLY'] = college['MENONLY'].astype('float16')
college['RELAFFIL'] = college['RELAFFIL'].astype('int8')

college.index = pd.Int64Index(college.index)
college.index.memory_usage()

movie = pd.read_csv('./data/movie.csv')
movie2 = movie[['movie_title', 'imdb_score', 'budget']]
movie2.head()
movie2.nlargest(100, 'imdb_score').head()
movie2.nlargest(100, 'imdb_score').nsmallest(5, 'budget')

# select max one of certain group by arranging(drop_duplicates())
movie = pd.read_csv('./data/movie.csv')
movie2 = movie[['movie_title', 'title_year', 'imdb_score']]
movie2.sort_values('title_year', ascending = False).head()
movie3 = movie2.sort_values(['title_year', 'imdb_score'], ascending = False)
movie3.head()
movie_top_year = movie3.drop_duplicates(subset = 'title_year')
movie_top_year
movie4 = movie[['movie_title', 'title_year', 'content_rating', 'budget']]
movie4.info()
movie4.content_rating.unique()
movie4_sorted = movie4.sort_values(['title_year', 'content_rating', 'budget'],
                                   ascending = [False, False, True])
movie4_sorted.drop_duplicates(subset = ['title_year', 'content_rating']).head()

# duplicate nlargest with sort_value()
movie = pd.read_csv('./data/movie.csv')
movie2 = movie[['movie_title', 'imdb_score', 'budget']]
movie_smallest_largest = movie2.nlargest(100, 'imdb_score')\
.nsmallest(5, 'budget')
movie_smallest_largest
movie2.sort_values('imdb_score', ascending = False).head()
movie2.sort_values('imdb_score', ascending = False).head(100)\
.sort_values('budget').head()

# pip install pandas_datareader
# This is due to the fact that is_list_like has been moved from pandas.core.common 
# to pandas.api.types in Pandas 0.23.0.

import pandas as pd
pd.core.common.is_list_like = pd.api.types.is_list_like

from pandas_datareader import data, wb
import fix_yahoo_finance as yf
yf.pdr_override()
import numpy as np
import datetime

#To get data:

start = datetime.datetime(2017, 1, 1)
end = datetime.datetime(2017, 12, 31)
tsla = data.get_data_yahoo('tsla', start, end)
tsla.head(8)

tsla_close = tsla['Close']
tsla_cummax = tsla_close.cummax()
tsla_cummax.head()

tsla_trailing_stop = tsla_cummax * .9
tsla_trailing_stop.head(8)

datetime.datetime.strptime('2017-6-1', "%Y-%m-%d")

def set_trailing_loss(symbol, start_date, end_date, perc):
    start_date = datetime.datetime.strptime(start_date, '%Y-%m-%d')
    end_date = datetime.datetime.strptime(end_date, '%Y-%m-%d')
    close = data.get_data_yahoo(symbol, start_date, end_date)['Close']
    
    return close.cummax() * perc


msft_trailing_stop = set_trailing_loss('msft', '2017-6-1', '2018-6-1', .85)
msft_trailing_stop.head()


college = pd.read_csv('./data/college.csv', index_col='INSTNM')
city = college['CITY']
city.head()

city.iloc[3]
city.iloc[[10, 20, 30]]
city.iloc[4:50:10]
city.iloc[14]
city.iloc[44]
city.loc['Heritage Christian University']
np.random.seed(1)
labels = list(np.random.choice(city.index, 4))
labels
city.loc[labels]

city.loc['Alabama State University':
    'Reid State Technical College' : 10]

college.iloc[99:102]
college.iloc[99]
college.iloc[101]
start = 'International Academy of Hair Design'
stop = 'Mesa Community College'
college.loc[start:stop] # include last label value

label = college.iloc[[60, 99, 3]].index.tolist()
college.loc[label]
college.iloc[[60, 99, 3]]

## select rows and colunms at once

college.iloc[:3, :4]
college.loc[:'Amridge University', : 'MENONLY']
college.iloc[:, [4, 6]].head()
college.loc[:, ['WOMENONLY', 'SATVRMID']].head()
college.iloc[[100, 200], [7, 5]]
rows = ['GateWay Community College', 'American Baptist Seminary of the West']
columns = ['SATMTMID', 'RELAFFIL']
college.loc[rows, columns]
college.head()
college.iloc[5, -4]
college.iloc[5] ## row = The University of Alabama
college.columns[-4] ## columns = PCTFLOAN
college.loc['The University of Alabama', 'PCTFLOAN']

college.iloc[90:80:-2, 5]
college.iloc[90:80:-2, :]
college.iloc[82, :]
start = 'Empire Beauty School-Flagstaff'
stop = 'Arizona State University-Tempe'
college.columns[5]
college.loc[start:stop:-2, 'RELAFFIL']

college.ix[:5, 'UGDS_WHITE':'UGDS_UNKN']
col_start = college.columns.get_loc('UGDS_WHITE')
col_start
col_stop = college.columns.get_loc('UGDS_UNKN') + 1
col_stop
col_start, col_stop 

# college.iloc[:5, col_start:col_stop] # error

# .iat, .at
college = pd.read_csv('./data/college.csv', index_col='INSTNM')
cn = 'Texas A & M University-College Station'
college.loc[cn, 'UGDS_WHITE']
college.at[cn, 'UGDS_WHITE']

# %timeit college.loc[cn, 'UGDS_WHITE']
# %timeit college.at[cn, 'UGDS_WHITE']

row_num = college.index.get_loc(cn)
col_num = college.columns.get_loc('UGDS_WHITE')
row_num, col_num
# %timeit college.iloc[row_num, col_num]
# %timeit college.iat[row_num, col_num]

state = college['STABBR']
state.iat[1000]
state.at['Stanford University']

# lazy row slicing

college[10:20:2]
city = college['CITY']
city[10:20:2]
start = 'Mesa Community College'
stop = 'Spokane Community College'
college[start:stop:1500]
city[start:stop:1500]
# college[:10, ['CITY', 'STABBR']] # unhashable type: 'slice'
first_ten_instnm = college.index[:10]
college.loc[first_ten_instnm, ['CITY', 'STABBR']]

# slicing with dictionary order

# college.loc['Sp': 'Su'] # error because not yet arranged
college = college.sort_index()
college.loc['Sp':'Su']
college.loc['Sp':'Su'].index
college.loc['S':'T']
college.loc['S':'T'].index
college = college.sort_index(ascending = False)
college.index.is_monotonic_decreasing
college.loc['E':'B']
college.loc['E':'B'].index

# Boolean indexing
## select rows with boolean values

movie = pd.read_csv('./data/movie.csv', index_col = 'movie_title')
movie.head()
movie_2_hours = movie['duration'] > 120
movie_2_hours.head(10)
movie_2_hours.sum()
movie_2_hours.mean()
movie['duration'].dropna().gt(120).mean()
movie_2_hours.describe()
movie_2_hours.value_counts(normalize = True)
actors = movie[['actor_1_facebook_likes', 'actor_2_facebook_likes']].dropna()
actors
(actors['actor_1_facebook_likes'] > actors['actor_2_facebook_likes']).mean()

# set multiful boolean condition 
# imdb_score > 8, content_rating == 'PG-13', title_year < 2000 or > 2009

movie = pd.read_csv('./data/movie.csv', index_col = 'movie_title')
criteria1 = movie.imdb_score > 8
criteria2 = movie.content_rating == 'PG-13'
criteria3 = ((movie.title_year < 2000) | (movie.title_year > 2009))
criteria3.head()

criteria_final = criteria1 & criteria2 & criteria3
criteria_final.head()

crit_a1 = movie.imdb_score > 8
crit_a2  = movie.content_rating == 'PG-13'
crit_a3 = (movie.title_year < 2000) | (movie.title_year > 2009)
final_crit_a = crit_a1 & crit_a2 & crit_a3

crit_b1 = movie.imdb_score < 5
crit_b2 = movie.content_rating == 'R'
crit_b3 = (movie.title_year >= 2000) & (movie.title_year <= 2010)
final_crit_b = crit_b1 & crit_b2 & crit_b3

final_crit_all = final_crit_a | final_crit_b # make final boolean series
final_crit_all.head()
movie[final_crit_all].head() # deliver final boolean series to dataframe

cols = ['imdb_score', 'content_rating', 'title_year']
movie_filtered = movie.loc[final_crit_all, cols] # deliver final boolean series to dataframe with .loc
movie_filtered.head()

final_crit_a2 = (movie.imdb_score > 8) &\
(movie.content_rating == 'PG-13') &\
((movie.title_year < 2000) | (movie.title_year > 2009))

final_crit_a2.equals(final_crit_a)

# using index like boolean index

college = pd.read_csv('./data/college.csv')
college[college['STABBR'] == 'TX'].head()

college2 = college.set_index('STABBR')
college2.loc['TX'].head()

# %timeit college[college['STABBR'] == 'TX']
# %timeit college2.loc['TX']
# %timeit college2 = college.set_index('STABBR')

states = ['TX', 'CA', 'NY']
college[college['STABBR'].isin(states)].head()
college2.loc[states].head()

# select with monotonic index

college = pd.read_csv('./data/college.csv')
college2 = college.set_index('STABBR')
college2.index.is_monotonic
college3 = college2.sort_index()
college3.index.is_monotonic

# %timeit college[college['STABBR']=='TX']
# %timeit college2.loc['TX']
# %timeit college3.loc['TX']

college_unique = college.set_index('INSTNM')
college_unique.index.is_unique
college[college['INSTNM'] == 'Stanford University']
college_unique.loc['Stanford University']

# multiple columns which is connected can be used as an index

college.index = college['CITY'] + ', ' + college['STABBR']
college = college.sort_index()
college.head()

college.loc['Miami, FL'].head()
## %%timeit
crit1 = college['CITY']=='Miami'
crit2 = college['STABBR'] == 'FL'
college[crit1 & crit2]
# %timeit college.loc['Miami, FL']

# predict Schlumberger stock value

slb = pd.read_csv('./data/slb_stock.csv', index_col = 'Date', parse_dates = ['Date'])
slb.head()
slb_close = slb['Close']
slb_summary = slb_close.describe(percentiles = [.1, .9])
slb_summary
upper_10 = slb_summary.loc['90%']
lower_10 = slb_summary.loc['10%']
criteria = (slb_close < lower_10) | (slb_close > upper_10)
slb_top_bottom_10 = slb_close[criteria]
## plotting
slb_close.plot(color = 'black', figsize = (12, 6))
slb_top_bottom_10.plot(marker = 'o', style = ' ', ## no line
                       ms = 4, color = 'lightgray')
xmin = criteria.index[0]
xmax = criteria.index[-1]
import matplotlib.pyplot as plt
plt.hlines(y = [lower_10, upper_10], xmin = xmin, xmax = xmax, color = 'black')

slb_close.plot(color = 'black', figsize=(12, 6))
plt.hlines(y = [lower_10, upper_10], xmin = xmin, xmax=xmax, color = 'lightgray')
plt.fill_between(x=criteria.index, y1=lower_10, y2=slb_close.values, color ='black')
plt.fill_between(x=criteria.index, y1=lower_10, y2=slb_close.values, 
                 where=slb_close < lower_10, color='lightgray')
plt.fill_between(x=criteria.index, y1=upper_10, y2=slb_close.values, 
                 where=slb_close > upper_10, color='lightgray')


# SQL where










