import pandas as pd
import numpy as np

from os import getcwd, chdir
#!/usr/bin/env python
chdir("C:\\Users\\Daniel\\Documents\\pandas")
getcwd()

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 100)

import os
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = 'C:/Anaconda3/Library/plugins/platforms'

movie = pd.read_csv("./data/movie.csv")
movie.head()

index = movie.index; index
columns = movie.columns; columns
data = movie.values; data
movie.info()
type(index)
type(columns)
type(data)

movie.dtypes
movie.get_dtype_counts()
movie['director_name']
movie.director_name

director = movie['director_name']
director.name
director.to_frame()

s_attr_methods = set(dir(pd.DataFrame))
len(s_attr_methods)

movie = pd.read_csv('./data/movie.csv')
director =movie['director_name']
actor_1_fb_likes = movie['actor_1_facebook_likes']

director.head()
actor_1_fb_likes.head()
actor_1_fb_likes.value_counts()

director.size
director.shape
len(director)

director.count()

actor_1_fb_likes.count()
actor_1_fb_likes.min(), actor_1_fb_likes.max(), actor_1_fb_likes.mean(), actor_1_fb_likes.median()

from sklearn import linear_model
reg = linear_model.LinearRegression()
reg.fit([[0,0], [1,1], [2,2]], [0, 1, 2])
reg.coef_




