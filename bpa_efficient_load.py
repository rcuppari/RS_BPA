# -*- coding: utf-8 -*-
"""
Created on Sat Nov 20 16:32:45 2021

@author: rcuppari
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy 
from sklearn.linear_model import LinearRegression

import scipy.stats

###############################################################################
###############################################################################
## purpose: use pre-defined functions to identify correlations between 
## any number of variables and outcomes of interest 
## then undertake regression analysis with them 
###############################################################################
###############################################################################
outcome = 'load'
## what is the outcome of interest that we want to correlate with? 
## clean it 
data = pd.read_excel("Drive_Data/hist_demand_data.xlsx",sheet_name = 0)

## provide all variables I want to read in 
read_in = ['EVI_CRB','GPM_CRB','GRACE_CRB','MODIS_LST_CRB','snow_CRB','TerraClim_CRB']

###############################################################################
###############################################################################
## script holding all of the long defined functions to do our work 
import corrs_func

outcome_mon = data.groupby(['Month','Year']).agg('mean')
outcome_mon.reset_index(inplace=True)
outcome_mon = outcome_mon[['Year','Month','BPA']]
outcome_mon.columns = ['year','month','outcome']

outcome_ann = data.groupby('Year').agg('mean')
outcome_ann.reset_index(inplace=True)
outcome_ann = outcome_ann[['Year','BPA']]
outcome_ann.columns = ['year','outcome']

## make a dictionary where each key is the name of the different dataset 
## make sure data is located in the "Drive_Data" folder or modify in script 
all_data_mon, all_data_ann = corrs_func.read_make_dictionary(read_in, outcome_mon, outcome_ann)

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
reg_filtered_ann = corrs_func.filtered_ann_regressions(all_data_ann, read_in, 0.9, 0.9, 3, outcome)

######################################################################
## what happens when we run these regressions on a monthly basis?
######################################################################
#reg_results_mon = {}
#corrs_mon = {}
import mon_funcs
corrs_mon, reg_results_mon = mon_funcs.mon_corrs_reg(outcome_ann, all_data_mon, read_in)  

## the filtered function will also plot them 
reg_filtered_mon = mon_funcs.filtered_mon_regressions(all_data_mon, outcome_ann, read_in, 0.9, 0.9, 3, 'load')

######################################################################
## now let's look at the most efficient ones: 
## tmax/min and EVI annually, runoff/mean temp May, aet/tmin June 
######################################################################
import statsmodels.api as sm
import numpy as np
from sklearn.model_selection import train_test_split as tts

#reg_data = all_data_ann['EVI_CRB']
reg_data = all_data_ann['TerraClim_CRB'][['tmmn_mean','tmmx_mean','year','outcome']]
#reg_data = all_data_ann['ERA_CRB'][['temperature_2m_mean','outcome','year']]
reg_data.dropna(axis=0, inplace=True)

x = reg_data.drop(['year','outcome'],axis=1)
y = reg_data['outcome']

x = sm.add_constant(x)
X_train,X_test,y_train,y_test=tts(x,y,test_size=.2,random_state=3)
model = sm.OLS(y_train, X_train).fit()
print(model.summary())
             

predicted = model.predict(X_test)
training = model.predict(X_train)
pred_all = model.predict(x)
#pred_all2 = model.predict(x)

range = [y.min(), y.max()]
plt.scatter(training, y_train, label = 'test', marker = 's')
plt.scatter(predicted, y_test, label = 'training', marker = 'o')            
plt.plot(range, range, label = '1:1 line')
plt.legend(fontsize=16)
plt.xlabel("Predicted Values (MW)", fontsize=18)
plt.ylabel("Observed Values (MW)", fontsize=18)
plt.title("Predicting Load with Annual TMIN/TMAX (2010-2016)", fontsize=20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.legend(fontsize = 16)
plt.show()

##############################################################
## runoff/mean temp May, aet/tmin June 
## rubbish :( 
##############################################################

#reg_data = all_data_mon['ERA_CRB'][['year','month','aet_mean', 'outcome']]
#reg_data = reg_data[reg_data['month'] == 6]

reg_data = all_data_mon['TerraClim_CRB'][['outcome','aet_mean','tmmx_mean','year', 'month']]
reg_data = reg_data[reg_data['month'] == 6]
#reg_data2 = all_data_mon['EVI_CRB'][['year','month','mean']]
#reg_data2 = reg_data2[reg_data2['month']==7]
#reg_data = reg_data[reg_data['month'] == 5]
#reg_data = reg_data.merge(reg_data2, on = 'year')
#jun_temp = all_data_mon['TerraClim_CRB'][['tmmn_mean','year', 'month']]
#jun_temp = jun_temp[jun_temp['month'] == 1]

#reg_data = reg_data.merge(may_temp[['tmmx_mean','year']], on = 'year')
reg_data.dropna(axis=0, inplace=True)
#plt.plot(reg_data[['outcome']])

x = reg_data.drop(['year','outcome', 'month'],axis=1)
y = reg_data['outcome']

x = sm.add_constant(x)
X_train,X_test,y_train,y_test=tts(x,y,test_size=.2,random_state=2)
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
plt.title("Predicting Generation with May TMIN/TMAX (2010-2016)", fontsize=20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.show()


