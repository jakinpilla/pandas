# -*- coding: utf-8 -*-
"""
Created on Sat Jul  7 08:08:56 2018

@author: Daniel
"""

from os import getcwd, chdir
getcwd()
chdir('C:/Users/Daniel/pandas')
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

movie_table = movie_long[['id', 'year', 'duration', 'rating']]
director_table = movie_long[['id', 'num', 'director', 'director_fb_likes']]
actor_table = movie_long[['id', 'num', 'actor', 'actor_fb_likes']]

movie_table = movie_table.drop_duplicates().reset_index(drop=True)
director_table = director_table.dropna().reset_index(drop=True)
actor_table = actor_table.dropna().reset_index(drop=True)

movie.memory_usage(deep=True)









