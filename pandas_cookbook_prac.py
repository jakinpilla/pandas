# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from os import getcwd, chdir
wd = getcwd()
import pandas as pd
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

# 리스트 할당을 통한 열의 레이블 변경
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























