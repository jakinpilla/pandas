# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from os import getcwd, chdir
getcwd()
import pandas as pd

chdir("E:/pandas")

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













