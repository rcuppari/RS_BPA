# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 08:37:25 2021

@author: rcuppari
"""
import numpy as np 
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns
import scipy 

generation=pd.read_csv("../BPA/Hist_data/hist_monthly_gen.csv")
GRACE=pd.read_csv("Drive_Data/GRACE_CSR_hybas4.csv")

GRACE['DT']=pd.to_datetime(GRACE['Date'])
GRACE['year']=GRACE.DT.dt.year
GRACE['month']=GRACE.DT.dt.month

grace_yr=GRACE.groupby(['DIST_SINK','year']).agg({'mean':'mean','stdDev':'mean'})
grace_yr.reset_index(inplace=True)
subbasins=grace_yr.DIST_SINK.unique()

gen_yr=generation.groupby('year').agg('mean')

##now go through all subbasins and create dictionary with 
##correlations between generation and stats for each subbasin 
corr={}
keys=subbasins
for s in range(0,len(subbasins)): 
    subbasin=subbasins[s]
    sub_grace=grace_yr[grace_yr['DIST_SINK']==subbasin]
    join=pd.merge(sub_grace[['year','mean','stdDev']],gen_yr['Gen'],on='year')
    corr[subbasin]=join.corr()['Gen']

max_corr=[]

##then create function to retrieve maximum corr value 
def retrieve_max_corr(corr): 
    max_corr=pd.DataFrame(index=[subbasins],columns=['Stat','Value'])
    for key, df in corr.items(): 
        ##length of corrs you want to keep (everything except Gen:Gen)
        end=len(df.values)-1
        v=df.values.ravel()[0:end]
        maxi=v.max()
        ##record maximum value and its index
        max_corr.loc[key,'Stat']=df[df[:]==maxi].index[0]
        max_corr.loc[key,'Value']=maxi
    return max_corr 

retrieve_max_corr(corr)    



























