# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 08:48:21 2021

@author: rcuppari
"""
import numpy as np 
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns
import scipy 
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from sklearn.model_selection import train_test_split as tts
import scipy.stats

###############################################################################
##also check with streamflow data 
streamflow=pd.read_csv("Drive_Data/TDA6ARF_daily.csv")

streamflow['date']=pd.to_datetime(streamflow.date)
streamflow['year']=streamflow['date'].dt.year
streamflow['month']=streamflow['date'].dt.month
streamflow.columns=['date','flow','year','month']

tda_mon=streamflow.groupby(['year','month']).agg('mean')
tda_mon.reset_index(inplace=True)
tda_mon['date']=pd.to_datetime(tda_mon[['year', 'month']].assign(DAY=1))
tda_yr=streamflow.groupby('year').agg('mean')
tda_yr.reset_index(inplace=True)

##can use same codes as before slightly modified to get maximum corr with TDA flow

months=np.arange(1,13)
months2=['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec']

##read in data produced through google earth engine 
##already have taken some summary statistics to aggregate by 
##this is monthly data
GRACE=pd.read_csv("Drive_Data/GRACE_CSR_hybas4.csv")
GRACE['DT']=pd.to_datetime(GRACE['Date'])
GRACE['year']=GRACE.DT.dt.year
GRACE['month']=GRACE.DT.dt.month
grace_yr=GRACE.groupby(['HYBAS_ID','year']).agg('mean')
grace_yr.reset_index(inplace=True)

subbasins=grace_yr.DIST_SINK.unique()

#this actually came out daily oddly 
temp=pd.read_csv("Drive_Data/AVHRR_CSR_hybas4.csv")
temp['DT']=pd.to_datetime(temp['Date'])
temp['year']=temp.DT.dt.year
temp['month']=temp.DT.dt.month
temp_mon=temp.groupby(['HYBAS_ID','year','month']).agg('mean')
temp_mon.reset_index(inplace=True)
temp_yr=temp.groupby(['HYBAS_ID','year']).agg('mean')
temp_yr.reset_index(inplace=True)

#this actually came out daily oddly too
snow=pd.read_csv("Drive_Data/snow_CSR_hybas4.csv")
snow['DT']=pd.to_datetime(snow['Date'])
snow['year']=snow.DT.dt.year
snow['month']=snow.DT.dt.month
snow_mon=snow.groupby(['HYBAS_ID','year','month']).agg('mean')
snow_mon.reset_index(inplace=True)
snow_yr=snow.groupby(['HYBAS_ID','year']).agg('mean')
snow_yr.reset_index(inplace=True)

swe=pd.read_csv("Drive_Data/SWE_CSR_hybas4.csv")
swe['DT']=pd.to_datetime(swe['Date'])
swe['year']=swe.DT.dt.year
swe['month']=swe.DT.dt.month
swe_yr=swe.groupby(['HYBAS_ID','year']).agg('mean')
swe_yr.reset_index(inplace=True)


##now go through all subbasins and create dictionary with 
##correlations between generation and stats for each subbasin 

##create function to do this 
def id_corrs(variable):
    corr={}
    for s in subbasins:
        sub_var=variable[variable['HYBAS_ID']==s]
        join=pd.merge(sub_var[['year','mean','max','median','min']],tda_yr['flow','year'],on='year')
        corr[s]=join.corr()['Gen']
    return join, corr 

       
##then create function to retrieve maximum corr value 
##idxmax will return column for largest absolute value of each stat, 
##need to iterate over rows sadly to get the actual absolute between stats
def retrieve_index(v,obj):
    ##get maximum of each row 
    idx_stat = v.abs().idxmax(axis=1)
    #print(idx_stat)
    #print(idx_stat[0])
    #df.reset_index(inplace=True)
    ##compare between the rows to find absolute max 
    ##do this for each subbasin 
    for r in range(0,len(v)):
        max_col=0
        max_row=0
        max_val=0
        ##retrieve indices
        
        if obj=='Series': 
            compare = v[r]
            col=idx_stat        
        else: 
            col=idx_stat[0]
            compare = v[col].iloc[r]
            
        if abs(compare) > abs(max_val): 
            max_col=col
            max_row=r
    ##output will be the index of the largest correlation 
    return max_col,max_row
        
def retrieve_max_corr(corr): 
    #max_corr=pd.DataFrame(index=np.arange(0,len(subbasins)),columns=['Stat','Month','Value'])
    #max_corr.index=subbasins
    ##length of corrs you want to keep (everything except Gen:Gen and Gen:year)
    max_corr=pd.DataFrame(columns=['Stat','Month','Value'])
    for s in subbasins: 
        end=len(corr[s])-1
        v=corr[s]
        ##if it's annual, will only have one column -- check if this is true 
        ##or if it's monthly (>1 column)
        if isinstance(v,pd.Series)==True:
            v=v.iloc[1:end]
            obj='Series'
        else: 
            v=v.iloc[1:end,:]
            obj='no_ser'
        #print(v)
        ##the subbasin will be the key 
        for Key in corr.items(): 
            ##want to store for each subbasin a single value
            ##which is the largest correlation (whether positive or negative), 
            ##so consider absolute values
            max_col,max_row=retrieve_index(v,obj)
            #max_corr.loc[s,'Subbasin']=s
            ##want to use this for monthly and yearly data so add if statement 
            ##to track whether there are multiple columns or not (==monthly)
            if obj=='Series':
                maxi=v[max_col]
                max_corr.loc[s,'Month']='NA'
                #inner_df.loc[s,'Month']='NA'
                ##record maximum value and its index
                max_corr.loc[s,'Stat']=max_col
                #inner_df.loc[s,'Stat']=max_col
                
            else: 
                maxi=v[max_col].iloc[max_row]#.iloc[max_row]
                max_corr.loc[s,'Month']=max_col
                #inner_df.loc[s,'Month']=max_col
                ##record maximum value and its index
                max_corr.loc[s,'Stat']=v[v[:]==maxi].index[max_row]
                #inner_df.loc[s,'Stat']=v[v[:]==maxi].index[max_row]
            max_corr.loc[s,'Value']=maxi
            #max_corr.loc[s]=inner_df       
    return max_corr

##all variables I want to iterate through
variables=[grace_yr,swe_yr] #temp_yr,snow_yr
names=['grace_yr','swe_yr'] #'temp_yr','snow_yr',

master_dict_yr={}

for i in range(0,len(variables)):
    variable=variables[i]
    name=names[i]

    max_corr=[]
    df, corr =id_corrs(variable)
    
    max_corr=retrieve_max_corr(corr)

    #print('corrs for variable '+str(name))
    #print(max_corr)
    
    master_dict_yr[name]=max_corr


##REPEAT FOR MONTHLY LEVEL AGGREGATION
##consider also correlation between monthly variable and annual generation 
def id_corrs_mon(variable):
    corr_mon={}

    for s in subbasins:
        corr_df=pd.DataFrame(columns=months)
        for m in months: 
            ##retrieve all data for a single month of a single subbasin
            sub_var=variable[variable['HYBAS_ID']==s]
            sub_var=sub_var[sub_var['month']==m]
            ##make df with generation 
            join=pd.merge(sub_var[['year','mean','max','median','min']],tda_yr['flow','year'],on='year')
            ##make each column refer to corr between gen and a different month 
            ##for that same subbasin 
            corr_df.iloc[:,m-1]=join.corr()['Gen']

        corr_df.columns=months2            
        corr_mon[s]=corr_df
       
    return join,corr_mon

##all variables I want to iterate through
variables=[temp_mon,snow_mon]
##not sure how to automate this tbh... 
names=['temp_mon','snow_mon']

master_dict_mon={}

for i in range(0,len(variables)):
    variable=variables[i]
    name=names[i]

    max_corr=[]
#    variable,corr_mon=id_corrs_mon(variable)
    
#    max_corr=retrieve_max_corr(corr_mon)

    #print('corrs for variable '+str(name))
    #print(max_corr)
    
#    master_dict_mon[name]=max_corr

#############################################################
##now that we know which stats correlate most strongly with 
##generation, identify a regression and produce r2 
#############################################################
    
comb=pd.merge(tda_yr,grace_yr[grace_yr['HYBAS_ID']==489.9],on='year')
print(comb.corr()['flow'])
X=comb[['mean']]
X=sm.add_constant(X)

est=sm.OLS(comb['flow'],X)
est2=est.fit()
print(est2.summary())
r2=round(est2.rsquared_adj,3)
pred=est2.predict(X)

plt.scatter(pred,comb['flow'])
plt.xlabel("Predicted TDA Flow (cfs)",fontsize=14)
plt.ylabel("Observed TDA Flow (cfs)",fontsize=14)
plt.title("Use of annual GRACE sub-basin data to predict TDA flow",fontsize=16)
z = np.polyfit(pred, comb['flow'], 1)
p = np.poly1d(z)
plt.plot(pred,p(pred),"r--")
plt.annotate(('r2 = '+str(r2)),(160000,220000),fontsize=14)
