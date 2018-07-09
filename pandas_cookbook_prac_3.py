# -*- coding: utf-8 -*-
"""
Created on Sat Jul  7 08:08:56 2018

@author: Daniel
"""

from os import getcwd, chdir
getcwd()
chdir('C:/Users/dsc/pandas')
import numpy as np
import pandas as pd

weight_loss = pd.read_csv('./data/weight_loss.csv')
weight_loss
weight_loss.query('Month == "Jan"')

def find_perc_loss(s):
    return (s - s.iloc[0]) / s.iloc[0]

bob_jan = weight_loss.query('Name == "Bob" and Month == "Jan"')
find_perc_loss(bob_jan['Weight'])
weight_loss.groupby(['Name', 'Month'])['Weight']
pcnt_loss = weight_loss.groupby(['Name', 'Month'])['Weight'].transform(find_perc_loss)
pcnt_loss.head(8)
pcnt_loss
weight_loss['Perc Weight Loss'] = pcnt_loss.round(3)
weight_loss.query('Name=="Bob" and Month in ["Jan", "Feb"]')
week4 = weight_loss.query('Week == "Week 4"')
week4
winner = week4.pivot(index='Month', columns='Name', values='Perc Weight Loss')
winner
winner["Winner"] = np.where(winner['Amy'] < winner['Bob'], 'Amy', 'Bob')
winner
winner.Winner.value_counts()
#   Name Month    Week  Weight  Perc Weight Loss
#0   Bob   Jan  Week 1     291             0.000
#2   Bob   Jan  Week 2     288            -0.010
#4   Bob   Jan  Week 3     283            -0.027
#6   Bob   Jan  Week 4     283            -0.027
#8   Bob   Feb  Week 1     283             0.000 # why this value is 0.0 ??
#10  Bob   Feb  Week 2     275            -0.028
#12  Bob   Feb  Week 3     268            -0.053
#14  Bob   Feb  Week 4     268            -0.053

(283-275)/283
(283-268)/283


week4a = week4.copy()
month_chron = week4a['Month'].unique() # or use drop_duplicates
month_chron
week4a['Month'] = pd.Categorical(week4a['Month'], categories=month_chron,
      ordered=True)
week4a.pivot(index='Month', columns='Name', values='Perc Weight Loss')

# SAT score weighted average per state with apply()
# groupby --> agg, filter, transform, apply
college = pd.read_csv('./data/college.csv')
subset = ['UGDS', 'SATMTMID', 'SATVRMID']
college2 = college.dropna(subset=subset)
college.shape
college2.shape

def weighted_math_average(df):
    weighted_math = df['UGDS'] * df['SATMTMID']
    return int(weighted_math.sum() / df['UGDS'].sum())

college2.groupby('STABBR').apply(weighted_math_average)

from collections import OrderedDict

def weighted_average(df):
    data = OrderedDict()
    weight_m = df['UGDS'] * df['SATMTMID']
    weight_v = df['UGDS'] * df['SATVRMID']
    wm_avg = weight_m.sum() / df['UGDS'].sum()
    wv_avg = weight_v.sum() / df['UGDS'].sum()
    
    data['weighted_math_avg'] = wm_avg
    data['weighted_verbal_avg'] = wv_avg
    data['math_avg'] = df['SATMTMID'].mean()
    data['verbal_avg'] = df['SATVRMID'].mean()
    data['count'] = len(df)
    return pd.Series(data, dtype='int')
    
college2.groupby('STABBR').apply(weighted_average).head(10)

from collections import OrderedDict
def weighted_average(df):
    data = OrderedDict()
    weight_m = df['UGDS'] * df['SATMTMID']
    weight_v = df['UGDS'] * df['SATVRMID']

    data['weighted_math_avg'] = weight_m.sum() / df['UGDS'].sum()
    data['weighted_verbal_avg'] = weight_v.sum() / df['UGDS'].sum()
    data['math_avg'] = df['SATMTMID'].mean()
    data['verbal_avg'] = df['SATVRMID'].mean()
    data['count'] = len(df)
    return pd.Series(data, dtype='int')

college2.groupby('STABBR').apply(weighted_average).head(10)

from scipy.stats import gmean, hmean
def calculate_means(df):
    df_means = pd.DataFrame(index=['Arithmeric', 'Weighted', 'Geometric', 'Harmonic'])
    cols = ['SATMTMID', 'SATVRMID']
    for col in cols:
        arithmetic = df[col].mean()
        weighted = np.average(df[col], weights=df['UGDS']) # weights is argument
        geometric = gmean(df[col])
        harmonic = hmean(df[col])
        df_means[col] = [arithmetic, weighted, geometric, harmonic]
    
    df_means['count'] = len(df)
    return df_means.astype(int)

college2.groupby('STABBR').apply(calculate_means).head(12)

# group by contnuous variables :: cut()

flights = pd.read_csv('./data/flights.csv')
flights.head()
flights.columns
flights.info()

bins = [-np.inf, 200, 500, 1000, 2000, np.inf]
cuts = pd.cut(flights['DIST'], bins=bins)
cuts.head()
cuts.value_counts()
flights.groupby(cuts)['AIRLINE'].value_counts(normalize=True).round(3).head(15)
flights.groupby(cuts)['AIR_TIME'].quantile(q=[.25, .5, .75]).div(60).round(2)
labels=['Under an Hour', '1 Hour', '1-2 Hours', '2-4 Hours', '4+ Hours']
cut2 = pd.cut(flights['DIST'], bins=bins, labels = labels)
flights.groupby(cuts)['AIRLINE'].value_counts(normalize=True).round(3)
flights.groupby(cuts)['AIRLINE'].value_counts(normalize=True).round(3).unstack()

flights = pd.read_csv('./data/flights.csv')
flights_ct = flights.groupby(['ORG_AIR', 'DEST_AIR']).size()
flights_ct.head()
flights_ct.loc[[('ATL', 'IAH'), ('IAH', 'ATL')]]
flights[['ORG_AIR', 'DEST_AIR']].apply(sorted, axis=1)
flights_sort = flights[['ORG_AIR', 'DEST_AIR']].apply(sorted, axis=1)
type(flights_sort) ## it does not return df, but series
flights_sort.head()
AIR1 = []
for i in range(len(flights_sort)):
    AIR1.append(flights_sort[i][0])

AIR2 = []
for i in range(len(flights_sort)):
    AIR2.append(flights_sort[i][1])

flights_sort = pd.DataFrame({'AIR1' : AIR1, 'AIR2' : AIR2})
flights_sort.head()

flights_ct2 = flights_sort.groupby(['AIR1', 'AIR2']).size()
flights_ct2

flights_ct2.loc[('ATL', 'IAH')]

data_sorted = np.sort(flights[['ORG_AIR', 'DEST_AIR']])
data_sorted
flights_sort2 = pd.DataFrame(data_sorted, columns=['AIR1', 'AIR2'])
flights_sort.head()
len(flights_sort)
flights_sort.columns
flights_sort2.head()
len(flights_sort2)
flights_sort2.columns

flights_sort2.equals(flights_sort)

# find the longeat streak of on-time flights

s = pd.Series([0, 1, 1, 0, 1, 1, 1, 0])
s
s1 = s.cumsum()
s1
s.mul(s1)
s.mul(s1).diff()
s.mul(s1).diff().where(lambda x : x<0)
s.mul(s1).diff().where(lambda x : x<0).ffill()
s.mul(s1).diff().where(lambda x : x<0).ffill().add(s1, fill_value=0)
s.mul(s1).diff().where(lambda x : x<0).ffill().add(s1, fill_value=0).max()

flights = pd.read_csv('./data/flights.csv')
flights['ARR_DELAY'][:5]
flights['ARR_DELAY'].lt(15)[:5]
flights['ARR_DELAY'].lt(15).astype(int)[:5]
flights['ON_TIME'] = flights['ARR_DELAY'].lt(15).astype(int)
flights[['AIRLINE', 'ORG_AIR', 'ON_TIME']].head(10)


def max_streak(s):
    s1 = s.cumsum()
    return s.mul(s1).diff().where(lambda x : x < 0).ffill().add(s1, fill_value=0).max()

flights.sort_values(['MONTH', 'DAY', 'SCHED_DEP'])\
.groupby(['AIRLINE', 'ORG_AIR'])['ON_TIME']\
.agg(['mean', 'size', max_streak]).round(2).head()

def max_delay_streak(df):
    df = df.reset_index(drop=True)
    s = 1 - df['ON_TIME']
    s1 = s.cumsum()
    streak = s.mul(s1).diff().where(lambda x : x<0).ffill().add(s1, fill_value=0)
    last_idx = streak.idxmax()
    first_idx = last_idx - streak.max() + 1
    df_return = df.loc[[first_idx, last_idx], ['MONTH', 'DAY']]
    df_return['streak'] = streak.max()
    df_return.index = ['first', 'last']
    df_return.index.name = 'type'
    return df_return

flights.sort_values(['MONTH', 'DAY', 'SCHED_DEP'])\
.groupby(['AIRLINE', 'ORG_AIR'])\
.apply(max_delay_streak)\
.sort_values('streak', ascending=False).head(10)

# restructuring data in tidy form
state_fruit = pd.read_csv('./data/state_fruit.csv', index_col=0)
state_fruit
state_fruit.stack()
state_fruit_tidy = state_fruit.stack().reset_index()
state_fruit_tidy
state_fruit_tidy.columns = ['state', 'fruit', 'weight']
state_fruit_tidy
state_fruit.stack().rename_axis(['state', 'fruit'])
state_fruit.stack().rename_axis(['state' ,'fruit']).reset_index(name = 'weight')


# melt() :: id_vars, value_vars
state_fruit2 = pd.read_csv('./data/state_fruit2.csv')
state_fruit2
state_fruit2.stack()
state_fruit2.set_index('State').stack()
state_fruit2.melt(id_vars = ['State'], value_vars = ['Apple', 'Orange', 'Banana'])
state_fruit2.melt(id_vars = ['State'], value_vars = ['Apple', 'Orange', 'Banana'], 
                  var_name= 'Fruit', value_name = 'Weight')

# id_vars : lists that are wanted to be a columns without being restuctured.
# value_vars : list that consist of columns' names which are wanted to be restructured as an single column
 # melt, stacking, unpivoting
state_fruit2
state_fruit2.melt()
state_fruit2.melt(id_vars='State')

movie = pd.read_csv('./data/movie.csv')
actor = movie[['movie_title', 'actor_1_name', 
               'actor_2_name', 'actor_3_name',
               'actor_1_facebook_likes', 'actor_2_facebook_likes', 
               'actor_3_facebook_likes']]
actor.head()

# wide_to_long()
list(actor.columns)

def change_col_name(col_name):
    col_name = col_name.replace('_name', '')
    if 'facebook' in col_name:
        fb_idx = col_name.find('facebook')
        col_name = col_name[:5] + col_name[fb_idx - 1 :] + col_name[5:fb_idx-1]
    return col_name

actor2 = actor.rename(columns=change_col_name)
actor2.head()
stubs = ['actor', 'actor_facebook_likes']
actor2_tidy = pd.wide_to_long(actor2, stubnames =stubs, 
                              i = ['movie_title'], 
                              j = 'actor_num',
                              sep = '_')

actor2_tidy.head()

df = pd.read_csv('./data/stackme.csv')
df2 = df.rename(columns={'a1' : 'group1_a1', 'b2' : 'group_b2', 
                         'd' : 'group2_a1', 'e' : 'group_b2'})

df2

pd.wide_to_long(df2, 
                stubnames=['group1', 'group2'],
                i=['State', 'Country', 'Test'],
                j = 'Label',
                suffix='.+',
                sep = '_')

# stack/unstack inverting
usecol_func = lambda x : 'UGDS_' in x or x == 'INSTNM'
usecol_func
college = pd.read_csv('./data/college.csv', index_col = 'INSTNM', usecols = usecol_func)
college.head()
college_stacked = college.stack()
college_stacked.head(18)
college_stacked.unstack()

# melt/pivot inverting
college2 = pd.read_csv('./data/college.csv', usecols = usecol_func)
college2.head()
college_melted = college2.melt(id_vars='INSTNM', 
                               var_name='Race', 
                               value_name='Percentage')

college_melted.head()
melted_inv = college_melted.pivot(index = 'INSTNM', 
                                  columns = 'Race', 
                                  values = 'Percentage')


college2_replication = melted_inv.loc[college2['INSTNM'], college2.columns[1:]].reset_index()
college2.equals(college2_replication)

college.head()
college.stack().unstack(0)
college.T
college.transpose()

# groupby() & unstacking
employee = pd.read_csv('./data/employee.csv')
employee.groupby('RACE')['BASE_SALARY'].mean().astype(int)
agg = employee.groupby(['RACE', 'GENDER'])['BASE_SALARY'].mean().astype(int)
agg
agg.unstack('GENDER')
agg.unstack('RACE')

agg2 = employee.groupby(['RACE', 'GENDER'])['BASE_SALARY'].agg(['mean', 'max', 'min']).astype(int)
agg2
agg2.unstack('GENDER')

# groupby() & pivot_table()
flights = pd.read_csv('./data/flights.csv')
fp = flights.pivot_table(index='AIRLINE',
                         columns='ORG_AIR',
                         values = 'CANCELLED', 
                         aggfunc = 'sum', fill_value=0).round(2)

fp.head()
fg = flights.groupby(['AIRLINE', 'ORG_AIR'])['CANCELLED'].sum()
fg.head()
fg_unstack = fg.unstack('ORG_AIR', fill_value=0)
fp.equals(fg_unstack)

flights.pivot_table(index=['AIRLINE', 'MONTH'], 
                    columns=['ORG_AIR', 'CANCELLED'],
                    values=['DEP_DELAY', 'DIST'],
                    aggfunc=[np.sum, np.mean], 
                    fill_value=0)

flights.groupby(['AIRLINE', 'MONTH', 'ORG_AIR', 'CANCELLED'])['DEP_DELAY', 'DIST']\
.agg(['mean', 'sum']).unstack(['ORG_AIR', 'CANCELLED'], fill_value=0).swaplevel(0,1,axis='columns')

college = pd.read_csv('./data/college.csv')
cg = college.groupby(['STABBR', 'RELAFFIL'])['UGDS', 'SATMTMID'].agg(['count', 'min', 'max']).head()
cg
cg = cg.rename_axis(['AGG_COLS', 'AGG_FUNCS'], axis='columns')
cg
cg.stack('AGG_FUNCS').head()
cg.stack('AGG_FUNCS').swaplevel('AGG_FUNCS', 'STABBR', axis='index').head()
cg.stack('AGG_FUNCS')\
.swaplevel('AGG_FUNCS', 'STABBR', axis='index')\
.sort_index(level='RELAFFIL', axis='index')\
.sort_index(level='AGG_COLS', axis='columns').head(6)
cg.stack('AGG_FUNCS').unstack(['RELAFFIL', 'STABBR'])
cg.stack(['AGG_FUNCS', 'AGG_COLS']).head(12)
cg.rename_axis([None, None], axis='index').rename_axis([None, None], axis='columns')

weightlifting = pd.read_csv('./data/weightlifting_men.csv')
weightlifting
wl_melt = weightlifting.melt(id_vars='Weight Category', 
                             var_name='sex_age',
                             value_name='Qual Total')
wl_melt.head()
sex_age = wl_melt['sex_age'].str.split(expand=True)
sex_age.head()
sex_age.columns = ['Sex', 'Age Group']
sex_age.head()
sex_age['Sex'] = sex_age['Sex'].str[0]
sex_age.head()
wl_cat_total = wl_melt[['Weight Category', 'Qual Total']]
wl_tidy = pd.concat([sex_age, wl_cat_total], axis='columns')
wl_tidy.head()

cols = ['Weight Category', 'Qual Total']
sex_age[cols] = wl_melt[cols]
sex_age[cols]
wl_melt[cols]


age_group = wl_melt.sex_age.str.extract('(\d{2}[-+](?:\d{2})?)', expand=False)
age_group
sex = wl_melt.sex_age.str[0]
new_cols = {'Sex' : sex, 'Age Group' : age_group}
wl_tidy2 = wl_melt.assign(**new_cols).drop('sex_age', axis='columns')
wl_tidy2.sort_index(axis=1).equals(wl_tidy.sort_index(axis=1))

inspections = pd.read_csv('./data/restaurant_inspections.csv', parse_dates = ['Date'])
inspections

inspections.set_index(['Name', 'Date', 'Info']).head(10)
inspections.set_index(['Name', 'Date', 'Info']).unstack('Info').head()
insp_tidy = inspections.set_index(['Name', 'Date', 'Info']).unstack('Info').reset_index(col_level=-1)
insp_tidy.head()
insp_tidy.columns = insp_tidy.columns.droplevel(0).rename(None)
insp_tidy.head()
inspections.set_index(['Name', 'Date', 'Info']).squeeze()\
.unstack('Info')\
.reset_index()\
.rename_axis(None, axis='columns')

inspections.pivot_table(index=['Name', 'Date'], 
                        columns='Info', 
                        values='Value',
                        aggfunc='first').reset_index().rename_axis(None, axis='columns')

cities = pd.read_csv('./data/texas_cities.csv')
cities
geolocations = cities.Geolocation.str.split(pat='. ', expand=True)
geolocations.columns = ['lat', 'lat_direc', 'log', 'log_direc']
geolocations
geoloc = geolocations.astype({'lat' : 'float', 'log' : 'float'})
geoloc.dtypes
geoloc
cities_tidy = pd.concat([cities['City'], geoloc], axis=1)
cities_tidy
geoloc.apply(pd.to_numeric, errors = 'ignore')

cities = pd.read_csv('./data/texas_cities.csv')
geolocations = cities.Geolocation.str.split(pat=' |, ', expand=True)
geolocations
cities.Geolocation.str.extract('([0-9.]+). (N|S), ([0-9.]+). (E|W)', expand=True)

sensors = pd.read_csv('./data/sensors.csv')
sensors
sensors.melt(id_vars=['Group', 'Property'], var_name='Year').head(10)
sensors.melt(id_vars=['Group', 'Property'], var_name='Year')\
.pivot_table(index=['Group', 'Year'], columns='Property', values='value')\
.reset_index().rename_axis(None, axis='columns')

sensors
sensors.set_index(['Group', 'Property'])
sensors.set_index(['Group', 'Property']).stack()
sensors.set_index(['Group', 'Property']).stack().unstack('Property')

sensors.set_index(['Group', 'Property']).stack().unstack('Property')\
.rename_axis(['Group', 'Year'], axis='index')

sensors.set_index(['Group', 'Property']).stack().unstack('Property')\
 
# Tidying when multiple observational units are stored in the same table

movie = pd.read_csv('./data/movie_altered.csv')
movie.head()
movie.insert(0, 'id', np.arange(len(movie)))
movie.head()
movie.columns
stubnames=['director', 'director_fb_likes', 'actor', 'actor_fb_likes']
movie_long = pd.wide_to_long(movie, 
                             stubnames=stubnames,
                             i='id',
                             j='num',
                             sep='_').reset_index()

# stubnames : The stub name(s). The wide format variables are assumed to start with the stub names.
# i : Column(s) to use as id variable(s)
# j : The name of the subobservation variable. What you wish to name your suffix in the long format.
movie_long.head()
movie_long.columns
movie_long.info()

movie_table = movie_long[['id', 'title', 'year', 'duration', 'rating']]
director_table = movie_long[['id', 'num', 'director', 'director_fb_likes']]
actor_table = movie_long[['id', 'num', 'actor', 'actor_fb_likes']]

movie_table = movie_table.drop_duplicates().reset_index(drop=True)
director_table = director_table.dropna().reset_index(drop=True)
actor_table = actor_table.dropna().reset_index(drop=True)

movie.memory_usage(deep=True).sum() # 2289466

movie_table.memory_usage(deep=True).sum() +\
director_table.memory_usage(deep=True).sum() +\
actor_table.memory_usage(deep=True).sum() # 2259676 ??

director_cate=pd.Categorical(director_table['director'])
director_cate.codes
director_table.insert(1, 'director_id', director_cate.codes)

actor_cate = pd.Categorical(actor_table['actor'])
actor_table.insert(1, 'actor_id', actor_cate.codes)

director_associative = director_table[['id', 'director_id', 'num']]
dcols = ['director_id', 'director', 'director_fb_likes']
director_unique = director_table[dcols].drop_duplicates().reset_index(drop=True)

actor_associative = actor_table[['id', 'actor_id', 'num']]
acols = ['actor_id', 'actor', 'actor_fb_likes']
actor_unique = actor_table[acols].drop_duplicates().reset_index(drop=True)

actors = actor_associative.merge(actor_unique, on='actor_id')\
.drop('actor_id', 1)\
.pivot_table(index='id', columns='num', aggfunc='first')

actors.columns = actors.columns.get_level_values(0) + '_' + \
actors.columns.get_level_values(1).astype(str)

directors = director_associative.merge(director_unique, on='director_id')\
.drop('director_id', 1)\
.pivot_table(index='id', columns='num', aggfunc='first')

directors.columns = directors.columns.get_level_values(0) + '_' +\
directors.columns.get_level_values(1).astype(str)

movie2 = movie_table.merge(directors.reset_index(), on='id', how='left')\
.merge(actors.reset_index(), on='id', how='left')

movie.equals(movie2[movie.columns]) ## False :: why??

# combining pandas object
# appending new rows to dataframe

names = pd.read_csv('./data/names.csv')
new_data_list = ['Aria', 1]
names.loc[4] = new_data_list
names
names.loc['five']=['Zach', 3]
names
names.loc[len(names)] = {'Name' : 'Zayd', 'Age' : 2}
names
names.loc[len(names)]=pd.Series({'Age' : 32, 'Name' : 'Dean'})
names
names.append({'Name':'Aria', 'Age':1}, ignore_index=True)
names = pd.read_csv('./data/names.csv')
names.index=['Canada', 'Canada', 'USA', 'USA']
names
s = pd.Series({'Name':'Zach', 'Age':3}, name=len(names))
s
names.append(s)

s1=pd.Series({'Name':'Zach', 'Age':3}, name=len(names))
s2=pd.Series({'Name':'Zayd', 'Age':2}, name='USA')
names.append([s1, s2])

bball_16=pd.read_csv('./data/baseball16.csv')
bball_16.head()
data_dict=bball_16.iloc[0].to_dict()
print(data_dict)
new_data_dict = {k: '' if isinstance(v, str) else np.nan for k, v in data_dict.items()}
data_dict.items()
print(new_data_dict)

random_data=[]
for i in range(1000):
    d=dict()
    for k, v in data_dict.items():
        if isinstance(v, str):
            d[k] = np.random.choice(list('abcde'))
        else:
            d[k] = np.random.randint(10)
    random_data.append(pd.Series(d, name=i + len(bball_16)))

random_data[0].head()

bball_16_copy = bball_16.copy()
for row in random_data:
    bball_16_copy = bball_16_copy.append(row)
    
bball_16_copy = bball_16.copy()
bball_16_copy = bball_16_copy.append(random_data)

stocks_2016=pd.read_csv('./data/stocks_2016.csv', index_col='Symbol')
stocks_2017=pd.read_csv('./data/stocks_2017.csv', index_col='Symbol')
s_list=[stocks_2016, stocks_2017]
pd.concat(s_list)
pd.concat(s_list, keys=['2016', '2017'], names=['Year', 'Symbol'])
pd.concat(s_list, keys=['2016', '2017'], axis='columns', names=['Year', None]) # outer join
pd.concat(s_list, join='inner', keys=['2016', '2017'], axis='columns', names=['Year', None]) # inner join

stocks_2016.append(stocks_2017)

base_url = 'http://www.presidency.ucsb.edu/data/popularity.php?pres={}'
trump_url = base_url.format(45)
df_list = pd.read_html(trump_url)
len(df_list) # 14 :: 14 df

df0 = df_list[0]
df0.shape
df0.head()
df_list=pd.read_html(trump_url, match='Start Date')
len(df_list)
df_list=pd.read_html(trump_url, match='Start Date', attrs={'align':'center'})
len(df_list)
trump=df_list[0]
trump.shape
trump.head()
df_list = pd.read_html(trump_url, match='Start Date', attrs={'align':'center'},
                       header=0, skiprows=[0,1,2,3,5], parse_dates=['Start Date', 'End Date'])
trump=df_list[0]
trump.head()
trump=trump.dropna(axis=1, how='all')
trump.head()
trump.isnull().sum()
trump=trump.ffill()
trump.head()
trump.dtypes

def get_pres_appr(pres_num):
    base_url = 'http://www.presidency.ucsb.edu/data/popularity.php?pres={}'
    pres_url = base_url.format(pres_num)
    df_list = pd.read_html(pres_url, match='Start Date',
                          attrs={'align':'center'},
                          header=0, skiprows=[0,1,2,3,5],
                          parse_dates=['Start Date', 'End Date'])
    pres=df_list[0].copy()
    pres=pres.dropna(axis=1, how='all')
    pres['President']=pres['President'].ffill()
    return pres.sort_values('End Date').reset_index(drop=True)

obama=get_pres_appr(44)
obama.head()

pres_41_45 = pd.concat([get_pres_appr(x) for x in range(41, 46)], ignore_index=True)

get_pres_appr(41)
get_pres_appr(42)
get_pres_appr(43)
get_pres_appr(44)
get_pres_appr(45)

pres_41_45.groupby('President').head(3)
pres_41_45['End Date'].value_counts().head(8)
pres_41_45 = pres_41_45.drop_duplicates(subset='End Date')
pres_41_45.shape
pres_41_45['President'].value_counts()
pres_41_45.groupby('President', sort=False).median().round(1)

import matplotlib.pyplot as plt
from matplotlib import cm
fig, ax = plt.subplots(figsize=(16, 6))
styles=['-.', '-', ':', '-', ':']
colors=[.9, .3, .7, .3, .9]
groups=pres_41_45.groupby('President', sort=False)
for style, color, (pres, df) in zip(styles, colors, groups):
    df.plot('End Date', 'Approving', ax=ax,
            label=pres, style=style, color=cm.Greys(color),
            title='Presidential Approval Rating')
    
days_func=lambda x: x - x.iloc[0]
pres_41_45['Days in Office']=pres_41_45.groupby('President')['End Date']\
.transform(days_func)
pres_41_45.head()
pres_41_45.groupby('President').head(3)
pres_41_45.dtypes
pres_41_45['Days in Office']=pres_41_45['Days in Office'].dt.days
pres_41_45['Days in Office'].head()
pres_pivot = pres_41_45.pivot(index='Days in Office', 
                              columns='President',
                              values='Approving')
pres_pivot.head()
plot_kwargs = dict(figsize=(16, 6), color=cm.gray([.3, .7]), 
                   style=['-', '--'], title='Approval Rating')
pres_pivot.loc[:250, ['Donald J. Trump', 'Barack Obama']].ffill().plot(**plot_kwargs)
pres_rm = pres_41_45.groupby('President', sort=False).rolling('90D', on='End Date')['Approving']\
.mean()
pres_rm.head()

styles=['-.', '-', ':', '-', ':']
colors = [.9, .3, .7, .3, .9]
color=cm.Greys(colors)
title='90 Day Approval Rating Rolling Average'
plot_kwargs=dict(figsize=(16, 6), style=styles, color=color, title=title)
correct_col_order = pres_41_45.President.unique()
pres_rm.unstack('President')[correct_col_order].plot(**plot_kwargs)

# difference between concat, join and merge
from IPython.display import display_html
years=2016, 2017, 2018
stock_tables = [pd.read_csv('./data/stocks_{}.csv'.format(year), 
                            index_col='Symbol') for year in years]

def display_frames(frames, num_spaces=0):
    t_style = '<table style="display: inline;"'
    tables_html = [df.to_html().replace('<table', t_style) for df in frames]
    space ='&nbsp;' * num_spaces
    display_html(space.join(tables_html), raw=True)

display_frames(stock_tables, 30)
stocks_2016, stocks_2017, stocks_2018 = stock_tables
stocks_2016
stocks_2017
stocks_2018
stock_tables

pd.concat(stock_tables, keys=[2016, 2017, 2018])
zip(years, stock_tables)
dict(zip(years, stock_tables))
pd.concat(dict(zip(years, stock_tables)), axis='columns')
stocks_2016.join(stocks_2017, lsuffix='_2016', rsuffix='_2017', how='outer')

other=[stocks_2017.add_suffix('_2017'), stocks_2018.add_suffix('_2018')]
stocks_2016.add_suffix('_2016').join(other, how='outer')

stock_join=stocks_2016.add_suffix('_2016').join(other, how='outer')
stock_concat = pd.concat(dict(zip(years, stock_tables)), axis='columns')
level_1 = stock_concat.columns.get_level_values(1)
level_0 = stock_concat.columns.get_level_values(0).astype(str)
stock_concat.columns = level_1 + '_' + level_0
stock_join.equals(stock_concat)

stocks_2016.merge(stocks_2017, left_index=True, right_index=True)

step1=stocks_2016.merge(stocks_2017, left_index=True, right_index=True, how='outer',
                       suffixes=('_2016', '_2017'))
stock_merge =step1.merge(stocks_2018.add_suffix('_2018'),left_index=True, right_index = True,
                         how='outer')
stock_concat.equals(stock_merge)

names=['prices', 'transactions']
food_tables = [pd.read_csv('./data/food_{}.csv'.format(name)) for name in names]
food_tables
food_prices, food_transactions = food_tables
display_frames(food_tables, 30)
food_transactions.merge(food_prices, on=['item', 'store'])
food_transactions.merge(food_prices.query('Date == 2017'),how='left')
food_prices_join=food_prices.query('Date == 2017').set_index(['item', 'store'])
food_prices_join
food_transactions.join(food_prices_join, on=['item', 'store'])

import glob

glob.glob('./data/gas prices/*.csv')

df_list=[]
for filename in glob.glob('./data/gas prices/*.csv'):
    df_list.append(pd.read_csv(filename, index_col = 'Week', parse_dates=['Week']))

gas=pd.concat(df_list, axis='columns')
gas.head()

from sqlalchemy import create_engine
engine = create_engine('sqlite:///data/chinook.db')

tracks = pd.read_sql_table('tracks', engine)
tracks.head()

genres = pd.read_sql_table('genres', engine)
genres

genre_track = genres.merge(tracks[['GenreId', 'Milliseconds']], on='GenreId', how='left')\
.drop('GenreId', axis='columns')

genre_track.head()

genre_time = genre_track.groupby('Name')['Milliseconds'].mean()
pd.to_timedelta(genre_time, unit='ms').dt.floor('s').sort_values()

cust = pd.read_sql_table('customers', engine, columns=['CustomerId', 'FirstName',
                                                       'LastName'])
invoice = pd.read_sql_table('invoices', engine, columns=['InvoiceId', 'CustomerId'])
ii = pd.read_sql_table('invoice_items', engine, 
                       columns=['InvoiceId', 'UnitPrice', 'Quantity'])

cust_inv = cust.merge(invoice, on='CustomerId').merge(ii, on='InvoiceId')
cust_inv.head()
total=cust_inv['Quantity']*cust_inv['UnitPrice']
cols=['CustomerId', 'FirstName', 'LastName']
cust_inv.assign(Total=total).groupby(cols)['Total'].sum()\
.sort_values(ascending=False).head()

sql_string1='''
select
    Name,
    time(avg(Milliseconds)/1000, 'unixepoch') as avg_time
from (
      select
          g.Name,
          t.Milliseconds
    from genres as g
    join
        tracks as t
        on
            g.genreid == t.genreid
    )
group by 
    Name
order by
    avg_time
'''
pd.read_sql_query(sql_string1, engine)

sql_string2='''
select
    c.customerid,
    c.FirstName,
    c.LastName,
    sum(ii.quantity*ii.unitprice) as Total
from
    customers as c
join
    invoices as i
    on c.customerid = i.customerid
join
    invoice_items as ii
        on i.invoiceid = ii.invoiceid
group by
    c.customerid, c.FirstName, c.LastName
order by 
    Total desc
'''
pd.read_sql_query(sql_string2, engine)













