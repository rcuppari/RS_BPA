# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 12:52:37 2021

@author: rcuppari
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy 
from sklearn.linear_model import LinearRegression
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import train_test_split as tts
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller

###############################################################################
###############################################################################
## purpose: use pre-defined functions to identify correlations between 
## any number of variables and outcomes of interest 
## then undertake regression analysis with them 
###############################################################################
###############################################################################
outcome = 'rev'
## what is the outcome of interest that we want to correlate with? 
## clean it 
data = pd.read_excel("Drive_Data/hist_rev.xlsx", header = 1)
#plt.scatter(data['Adjusted NR'], data['Reported Net Rev'])
#plt.plot(data['Gross Power Sales'])
#plt.plot(data['Adjusted NR'])

#outcome_ann = data[['Year','Net Power Sales']]
ann_outcome = data[['Year','Reported Net Rev']]
ann_outcome.columns = ['year','outcome']
ann_outcome.dropna(inplace=True)

outcome_mon = 0 

df = data[['Year','Reported Net Rev']]
df.columns = ['year','outcome']
df.dropna(inplace=True)
#data2 = data['Gross Power Sales'].dropna(axis = 0)
#print(scipy.stats.normaltest(data2))
#plt.hist(data2)
#plt.plot(data['outcome'])

## is the data non-stationary? 
## p > .05 => non-stationary 
#dftest = adfuller(df['outcome'], autolag = 'AIC')
## pretty close 

## provide all variables I want to read in 
read_in = ['EVI_CRB','GPM_CRB','GRACE_CRB','MODIS_LST_CRB','snow_CRB','TerraClim_CRB', \
           'NDWI_CRB', 'surf_ERA', 'temp_ERA', 'runoff_ERA', 'snow_ERA', 'precip_ERA', 'NDWI_CRB']
###############################################################################
###############################################################################
## script holding all of the long defined functions to do our work 
import corrs_func


## make a dictionary where each key is the name of the different dataset 
## make sure data is located in the "Drive_Data" folder or modify in script 
all_data_mon, all_data_ann = corrs_func.read_make_dictionary(read_in, outcome_mon, ann_outcome)

## use id_corrs to input the dictionary and the outcome of interest and 
## output the joined dataset and a dictionary with correlations 
## can change variable names based on whatever the outcome is 
## (or not if do separate scripts for each outcome)
corrs_ann = corrs_func.id_corrs(ann_outcome, all_data_ann, read_in)

import mon_funcs
corrs_mon, mon_reg = mon_funcs.mon_corrs_reg(ann_outcome, all_data_mon, read_in)

reg_filtered_mon = mon_funcs.filtered_mon_regressions(all_data_mon, ann_outcome, read_in, .6, .6, 3, outcome)

## get all combinations of variables regressing with input dataframe 
reg_results_ann = corrs_func.combination_ann_regressions(all_data_ann, read_in)

## filter by the training minimum r2 (decimal value), test minimum r2, and 
## length of inputs
## the filtered function will also plot them 
reg_filtered_ann = corrs_func.filtered_ann_regressions(all_data_ann, read_in, 0.6, 0.6, 3, outcome)


######################################################################
## now let's look at the most efficient one 
## mean 
######################################################################
all_data_mon = {}
for n, names in enumerate(read_in): 
    data = pd.read_csv('Drive_Data/' + names + '.csv')
    data.Date = pd.to_datetime(data.Date)
    data['mean_month'] = data.Date.dt.month
    data['mean_year'] = data.Date.dt.year
    data = data.loc[:,data.columns.str.contains('mean')]
    data = data.rename(columns = {'mean_year':'year','mean_month':'month'})
        
    data_mon = data.groupby(['year','month']).agg('mean')  
    data_mon.reset_index(inplace=True)  
    all_data_mon[names] = data_mon 

######################################################################
TerraClim = all_data_mon['TerraClim_CRB'][['tmmn_mean','tmmx_mean','pdsi_mean','year','month']]
ann_TC = TerraClim.groupby('year').agg('mean')
ann_TC.reset_index(inplace = True)

feb_TC = TerraClim[(TerraClim['month']== 3) | (TerraClim['month']== 4) | (TerraClim['month']== 5)]
feb_TC = feb_TC.groupby('year').agg('mean')
feb_TC.reset_index(inplace = True)
feb_TC.columns = ['year', 'tmmn_mean2', 'tmmx_mean2', 'pdsi_mean2', 'month']

snow = all_data_mon['snow_CRB']
ann_snow = snow.groupby('year').agg('max')
ann_snow.reset_index(inplace = True)

snow_feb = snow[(snow['month'] == 5) | (snow['month'] == 3) | (snow['month'] == 4)][['year', 'mean']]
snow_feb = snow_feb.groupby('year').agg('max')
snow_feb.reset_index(inplace = True)
snow_feb.columns = ['year','feb_snow']

snow_dec = snow[snow['month'] == 2][['year', 'mean']]
snow_dec = snow_dec.groupby('year').agg('max')
snow_dec.reset_index(inplace = True)
snow_dec.columns = ['year','dec_snow']

max_snow = snow.groupby('year').agg('max')
max_snow.reset_index(inplace=True)
max_snow.columns = ['year', 'month', 'max_snow']

grace = all_data_mon['GRACE_CRB']

grace_spr = grace[(grace['month']==6) | (grace['month']==5) | (grace['month']==7)]
grace_spr.columns = ['year', 'month', 'csr6', 'gfz6', 'jpl6']
grace_spr['grace_mean_spr'] = (grace_spr['csr6'] + grace_spr['gfz6'] + grace_spr['jpl6'])/3
grace_spr = grace_spr.groupby('year').agg('mean')
grace_spr.reset_index(inplace = True)

grace_aut = grace[(grace['month']==9) | (grace['month']==10) | (grace['month']==11)]
grace_aut.columns = ['year', 'month', 'csr10', 'gfz10', 'jpl10']
grace_aut['grace_mean_aut'] = (grace_aut['csr10'] + grace_aut['gfz10'] + grace_aut['jpl10'])/3
grace_aut = grace_aut.groupby('year').agg('mean')
grace_aut.reset_index(inplace = True)

ann_grace = all_data_ann['GRACE_CRB']
ann_grace.columns = ['year','csr','gfz','jpl','outcome']
#ann_grace['mean_grace'] = (ann_grace['csr'] + ann_grace['gfz'] + ann_grace['jpl'])/3

runoff = all_data_mon['surf_ERA']
runoff_spr = runoff[(runoff['month']==3) | (runoff['month']==5) | (runoff['month']==4)]
runoff_spr.columns = ['year', 'month', 'runoff']
runoff_spr = runoff_spr.groupby('year').agg('mean')
runoff_spr.reset_index(inplace = True)

#reg_data = feb_TC.merge(ann_outcome, on='year')#all_data_ann['MODIS_LST_CRB'][['year']].merge(feb_TC, on='year')
reg_data = feb_TC.merge(ann_snow, on='year')
#reg_data = reg_data.merge(ann_outcome, on = 'year')
reg_data = reg_data.merge(grace_spr, on='year')
reg_data = reg_data.merge(grace_aut, on='year')
reg_data = reg_data.merge(ann_grace, on='year')
reg_data = reg_data.merge(snow_feb, on='year')
reg_data = reg_data.merge(snow_dec, on='year')
reg_data = reg_data.merge(max_snow, on='year')
reg_data = reg_data.merge(ann_TC, on = 'year')
reg_data = reg_data.merge(runoff_spr, on = 'year')

## mean grace and pdsi mean are effective ish -- significant but r2 = /39
## grace aut + tmmx_mean + runoff state = 1 
y = reg_data['outcome']
x = reg_data[['tmmx_mean', 'grace_mean_aut']]

x = sm.add_constant(x)
X_train,X_test,y_train,y_test=tts(x,y,test_size=.2,random_state=4)
model = sm.OLS(y_train, X_train).fit()
print(model.summary())

predicted = model.predict(X_test)
training = model.predict(X_train)
                
range = [0, 850000]
plt.scatter(training, y_train, label = 'test')
plt.scatter(predicted, y_test, label = 'train')  
plt.plot(range, range, label = 'one to one line')          

######################################################################
outcome_ann2 = ann_outcome[ann_outcome['year']>2008]
reg_data = pd.concat([#all_data_ann['ERA_CRB']['total_precipitation_mean'], 
               all_data_ann['EVI_CRB']['mean'],
               all_data_ann['snow_CRB']['mean'],
               all_data_ann['MODIS_LST_CRB'][['mean']],
               all_data_ann['runoff_CRB'][['mean']],
               all_data_ann['GRACE_CRB']['gfz'],
               all_data_ann['TerraClim_CRB'][['pdsi_mean','tmmx_mean', 'tmmn_mean']]], axis=1)

evi = all_data_ann['EVI_CRB'][['mean','year']]

reg_data = evi.merge(all_data_ann['snow_CRB'][['mean','year']],on=['year']) ##grace jun, runoff apr
reg_data = reg_data.merge(all_data_ann['GRACE_CRB'][['lwe_thickness_csr_mean','year']],on=['year'])
reg_data = reg_data.merge(all_data_ann['MODIS_LST_CRB'][['mean','year']], on='year')
reg_data = reg_data.merge(all_data_ann['TerraClim_CRB'][['year','pdsi_mean','tmmx_mean', 'tmmn_mean','outcome']], on='year')
reg_data = reg_data.merge(all_data_ann['ERA_CRB'][['year','runoff_mean']], on = 'year')
reg_data.dropna(axis=0, inplace=True)
    
reg_data.columns = ['EVI', 'year', 'snow', 'grace', 'MODIS', 'pdsi', 'tmmx', 'tmmn', 'outcome', 'runoff']
reg_data.dropna(axis=0, inplace=True)

x = reg_data.drop(['outcome','snow','runoff','year','pdsi','MODIS','EVI'],axis=1)
y = reg_data['outcome']

#x = sm.add_constant(x)
X_train,X_test,y_train,y_test=tts(x,y,test_size=.2,random_state=3)
model = sm.OLS(y_train, X_train).fit()
print(model.summary())

predicted = model.predict(X_test)
training = model.predict(X_train)
                
#plt.scatter(training, y_train, label = 'test')
#plt.scatter(predicted, y_test, label = 'train')            

#plt.scatter(all_data_ann['MODIS_LST_CRB']['outcome'],all_data_ann['MODIS_LST_CRB']['mean'])
predicted = model.predict(X_test)
training = model.predict(X_train)
pred_all = model.predict(x)

plt.scatter(training, y_train, label = 'test')
plt.scatter(predicted, y_test, label = 'training')            
z = np.polyfit(pred_all, y, 1)
p = np.poly1d(z)
plt.plot(pred_all,p(pred_all),"r--")
plt.legend(fontsize=16)
plt.xlabel("Predicted Residuals ($M)", fontsize=18)
plt.ylabel("Observed Residuals ($M)", fontsize=18)
plt.title("Predicting Net Revenues ($ '000s)", fontsize=20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.show()

##############################################################
## monthly ##
##############################################################
grace = all_data_mon['GRACE_CRB'][['lwe_thickness_jpl_mean','year','month']]
grace10 = grace[(grace['month'] == 10) | (grace['month'] == 9 )]
grace10=grace10.groupby('year').agg('mean')
grace10.reset_index(inplace=True)
grace10.columns = ['year','grace10','month']

#grace_jun = grace[grace['month'] == 6]
#grace_jun.columns = ['grace_6','year','month']

runoff = all_data_ann['ERA_CRB'][['runoff_mean','year']]
#runoff_jun = runoff[runoff['month'] == 6]
#runoff_jun.columns = ['runoff_6','year','month']

## 3 might also just be the cleanest month... no weird outliers 
snow_all = all_data_mon['snow_CRB']
snow = snow_all[snow_all['month'] == 3]
snow = snow[['year','mean']]
snow.columns = ['year','snow']

evi = all_data_mon['EVI_CRB']
evi = evi[evi['month']==5]
evi = evi[['year','mean']]

temp = all_data_mon['TerraClim_CRB'][['year','tmmx_mean', 'tmmn_mean','month']]
temp = temp[temp['month'] == 4]
temp.columns = ['year','tmmx4','tmmn4','month']
temp.drop('month',axis=1,inplace=True)

precip = all_data_mon['ERA_CRB'][['year','month','total_precipitation_mean']]
precip = precip[precip['month'] == 1]
precip.drop('month',axis=1,inplace=True)

reg_data = grace10.merge(snow[['year','snow']],on=['year'])
reg_data = reg_data.merge(runoff,on=['year'])
#reg_data = reg_data.merge(all_data_ann['MODIS_LST_CRB'][['mean','year','outcome']], on='year')
reg_data = reg_data.merge(all_data_ann['GRACE_CRB'][['jpl','year','outcome']], on='year')
reg_data = reg_data.merge(all_data_ann['TerraClim_CRB'][['year','tmmx_mean', 'tmmn_mean']], on='year')
reg_data = reg_data.merge(evi, on='year')
reg_data = reg_data.merge(temp, on='year')
reg_data = reg_data.merge(precip, on='year')
reg_data.dropna(axis=0, inplace=True)


x = reg_data[['total_precipitation_mean']]
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
plt.xlabel("Predicted Values ($M)", fontsize=18)
plt.ylabel("Observed Values ($M)", fontsize=18)
plt.title("Predicting Net Revenues", fontsize=20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.show()

plt.scatter(reg_data['outcome'], reg_data['snow'])
plt.scatter(snow_all[snow_all['month'] == 3]['mean'], snow_all[snow_all['month'] == 4]['mean'])





