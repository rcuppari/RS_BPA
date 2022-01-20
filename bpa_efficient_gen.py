# -*- coding: utf-8 -*-
"""
Created on Sat Nov 20 17:33:30 2021

@author: rcuppari
"""

import pandas as pd
import numpy as np
import scipy 
import matplotlib.pyplot as plt
###############################################################################
###############################################################################
## purpose: use pre-defined functions to identify correlations between 
## any number of variables and outcomes of interest 
## then undertake regression analysis with them 
###############################################################################
###############################################################################

## what is the outcome of interest that we want to correlate with? 
## clean it 
data = pd.read_csv("Drive_Data/hist_monthly_gen.csv")
data.columns = ['month','year','outcome']

## generation data is NOT normally distributed -- detrend monthly 
means = data[['outcome','month']].groupby('month').agg('mean')
means.columns = ['mean']
std = data.outcome.std()

obs_trended_ann = data.groupby('year').agg('mean')
data['outcome'] = (data['outcome'] - means.mean()['mean'])/std

#plt.hist(data['outcome'])
#print(scipy.stats.normaltest(data['outcome']))

## COOL! detrended data is normal -- now gonna run everything with the detrended data 
## and will have to retrend after the regression

outcome_mon = data
outcome_mon['day'] = 1
outcome_mon['Date'] = pd.to_datetime(outcome_mon[['year','month', 'day']])
outcome_mon = outcome_mon[['year','month','outcome']]

outcome_ann = data.groupby('year').agg('mean')
outcome_ann.reset_index(inplace=True)
outcome_ann = outcome_ann[['year','outcome']]
outcome_ann['day'] = 1
outcome_ann['month'] = 1
outcome_ann['Date'] = pd.to_datetime(outcome_ann[['year','month','day']])
outcome_ann = outcome_ann[['year','outcome']]

#outcome_ann['month'] = pd.to_numeric(outcome_ann['month'])
#outcome_mon['month'] = pd.to_numeric(outcome_mon['month'])

#outcome_ann['year'] = pd.to_numeric(outcome_ann['year'])
#outcome_mon['year'] = pd.to_numeric(outcome_mon['year'])

## provide all variables I want to read in 
read_in = ['EVI_CRB','GPM_CRB','GRACE_CRB','MODIS_LST_CRB','snow_CRB','TerraClim_CRB', 'runoff_ERA']

outcome = 'gen'
###############################################################################
###############################################################################
## script holding all of the long defined functions to do our work 
import corrs_func

## make a dictionary where each key is the name of the different dataset 
## make sure data is located in the "Drive_Data" folder or modify in script 
all_data_mon, all_data_ann = corrs_func.read_make_dictionary(read_in, outcome_mon, outcome_ann, detrend = True)

## use id_corrs to input the dictionary and the outcome of interest and 
## output the joined dataset and a dictionary with correlations 
## can change variable names based on whatever the outcome is 
## (or not if do separate scripts for each outcome)
corrs_ann = corrs_func.id_corrs(outcome_ann, all_data_ann, read_in)

## get all combinations of variables regressing with input dataframe 
reg_results_ann = corrs_func.combination_ann_regressions(all_data_ann, read_in)

## filter by the training minimum r2 (decimal value), test minimum r2, and 
## length of inputs
## the filtered function will also plot them 
reg_filtered_ann = corrs_func.filtered_ann_regressions(all_data_ann, read_in, 0.8, 0.8, 3, outcome)

######################################################################
## what happens when we run these regressions on a monthly basis?
######################################################################
#reg_results_mon = {}
#corrs_mon = {}
import mon_funcs
corrs_mon, reg_results_mon = mon_funcs.mon_corrs_reg(outcome_ann, all_data_mon, read_in)  

## the filtered function will also plot them 
reg_filtered_mon = mon_funcs.filtered_mon_regressions(all_data_mon, outcome_ann, read_in, 0.8, 0.8, 3, outcome)

######################################################################
## now let's look at the most efficient ones: 
## PET + runoff in june and precip in june
######################################################################
import statsmodels.api as sm
from sklearn.model_selection import train_test_split as tts

reg_data = all_data_ann['runoff_ERA'][['mean', 'outcome', 'year']]
#reg_data = reg_data.merge(all_data_ann['GRACE_CRB'][['lwe_thickness_jpl_mean','year']], on = 'year')
reg_data.dropna(axis=0, inplace=True)
obs_trended_ann.columns = ['month','og_outcome']

reg_data = reg_data.merge(obs_trended_ann['og_outcome'], left_on = 'year', right_index=True)

x = reg_data.drop(['year','outcome','og_outcome'],axis=1)
y = reg_data['outcome']

x = sm.add_constant(x)
X_train,X_test,y_train,y_test=tts(x,y,test_size=.2,random_state=1)
model = sm.OLS(y_train, X_train).fit()
print(model.summary())
             
predicted = model.predict(X_test)
training = model.predict(X_train)
pred_all = model.predict(x)

range = [y.min(), y.max()]
plt.scatter(training, y_train, label = 'test', marker = 's')
plt.scatter(predicted, y_test, label = 'training', marker = 'o')            
plt.plot(range, range, label = '1:1 line')
plt.legend(fontsize=16)
plt.xlabel("Predicted Residuals", fontsize=18)
plt.ylabel("Observed Residuals", fontsize=18)
plt.title("Predicting Generation with Annual Runoff (2007-2018)", fontsize=20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.show()
#### r2 = 0.67 ####

## now let's retrend and see what happens! :) 
pred_all_retrend = (pred_all * std) + means.mean()['mean']

range = [reg_data.og_outcome.min(), reg_data.og_outcome.max()]
plt.scatter(pred_all_retrend, reg_data['og_outcome'])
plt.xlabel("Predicted Generation (MW)", fontsize=18)
plt.ylabel("Observed Generation (MW)", fontsize=18)
plt.plot(range, range, label = '1:1 line')
plt.title("Predicting Generation with Annual Runoff (2007-2018)", fontsize=18)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.legend(fontsize=16)
plt.show()

comb = pd.concat([pred_all_retrend, reg_data['og_outcome'].reset_index()], axis=1)
print('r2 = ' + str(comb.corr()**2))



##############################################################
## nope ##
##############################################################

grace = all_data_mon['GRACE_CRB'][['lwe_thickness_csr_mean','year','month','outcome']]
grace_apr = grace[grace['month'] == 3]
grace_apr.columns = ['jpl_3','year','month','outcome']

grace_jun = grace[grace['month'] == 6]
grace_jun.columns = ['jpl_6','year','month', 'outcome']

runoff = all_data_mon['ERA_CRB'][['runoff_mean','year','month']]
runoff_jun = runoff[runoff['month'] == 6]
runoff_jun.columns = ['runoff_6','year','month']

runoff_apr = runoff[runoff['month'] == 4]
runoff_apr.columns = ['runoff_4','year','month']

reg_data = grace_apr.merge(grace_jun,on=['year']) ##grace jun, runoff apr
#reg_data = reg_data.merge(grace_apr,on=['year'])
#reg_data = reg_data.merge(runoff_jun,on=['year'])

#reg_data = reg_data.merge(outcome_ann, on = ['year'])   
reg_data.dropna(axis=0, inplace=True)


#reg_data = reg_data[reg_data['month'] == 4]
x = reg_data.drop(['outcome_x', 'jpl_3','outcome_y','year', 'month_x','month_y'],axis=1)
# = reg_data.drop(['outcome_x', 'year','month_x','month_y', 'outcome_y'],axis=1)
#x = grace_apr.drop(['year','month','outcome'], axis=1)
y = reg_data['outcome_x']
#y = grace_apr['outcome']

x = sm.add_constant(x)

X_train,X_test,y_train,y_test=tts(x,y,test_size=.2,random_state=3)
model = sm.OLS(y_train, X_train).fit()
print(model.summary())

predicted = model.predict(X_test)
training = model.predict(X_train)
pred_all = model.predict(x)

plt.scatter(training, y_train, label = 'test')
plt.scatter(predicted, y_test, label = 'training')            
z = np.polyfit(pred_all, y, 1)
p = np.poly1d(z)
plt.plot(pred_all,p(pred_all),"r--")
plt.legend(fontsize=16)
plt.xlabel("Predicted Values (cfs)", fontsize=18)
plt.ylabel("Observed Values (cfs)", fontsize=18)
plt.title("Predicting Generation (2004-2016)", fontsize=20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.show()


## relationship between generation and net revenue? 
df = pd.read_excel("Drive_Data/hist_rev.xlsx", header = 1)
df = df[['Year','Gross Power Sales']]
df.columns = ['year','adj NR']
df.dropna(inplace=True)

gen = data.groupby('year').agg('sum')

both = df.merge(gen, on='year')

plt.scatter(both['adj NR'], both['outcome'])
plt.xlabel("Gross Power Sales ($B)", fontsize = 16)
plt.ylabel("Annual Total Generation (MW)", fontsize = 16)
plt.xticks(fontsize = 14)
plt.yticks(fontsize = 14)

#reg_data = reg_data.merge(df, on = 'year')
x = both['outcome']
# = reg_data.drop(['outcome_x', 'year','month_x','month_y', 'outcome_y'],axis=1)
#x = grace_apr.drop(['year','month','outcome'], axis=1)
y = both['adj NR']
#y = grace_apr['outcome']

x = sm.add_constant(x)

X_train,X_test,y_train,y_test=tts(x,y,test_size=.2,random_state=3)
model = sm.OLS(y_train, X_train).fit()
print(model.summary())

predicted = model.predict(X_test)
training = model.predict(X_train)
pred_all = model.predict(x)

plt.scatter(training, y_train, label = 'test')
plt.scatter(predicted, y_test, label = 'training')            
z = np.polyfit(pred_all, y, 1)
p = np.poly1d(z)
plt.plot(pred_all,p(pred_all),"r--")
plt.legend(fontsize=16)
plt.xlabel("Predicted Values (cfs)", fontsize=18)
plt.ylabel("Observed Values (cfs)", fontsize=18)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.show()








