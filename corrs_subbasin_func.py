# -*- coding: utf-8 -*-
"""
Created on Thu Nov 11 16:23:32 2021

@author: rcuppari
"""
import pandas as pd
import numpy as np 

##############################################################################
## code that takes a dataframe with the variable to be explored ("variable")
## and uses id_corr to identify all correlations
## func 2 is meant to retrieve the maximum correlation from all of those 
## correlations previously calculated, 
## idxmax will return column for largest absolute value of each stat, 
## need to iterate over rows sadly to get the actual absolute between stats
##now go through all subbasins and create dictionary with 
##correlations between generation and stats for each subbasin 

##create function to do this 
## variable 
def get_csvs(names, monthly = True): 
    ann_vars = {}
    mon_vars = {}
    for v, var in enumerate(names):
        variable = pd.read_csv('Drive_Data/' + str(var) + '_hybas4.csv')
#       print(var)
#        print(variable.head())
#        print()
        
        ## clean data to allow for aggregation 
        variable['DT']=pd.to_datetime(variable['Date'])
        variable['year']=variable.DT.dt.year
        variable['month']=variable.DT.dt.month
        variable.reset_index(inplace=True)
        subbasins = variable.HYBAS_ID.unique()
        
        variable_ann=variable.groupby(['HYBAS_ID','year']).agg({"mean": [np.min, np.max, np.mean]})
        variable_ann.reset_index(inplace=True)
        variable_ann.columns = ['HYBAS_ID','year','min','max','mean']

        if monthly == True: 
            variable_mon = variable.groupby(['HYBAS_ID','year','month']).agg({"mean": [np.mean, np.max, np.min]})
            variable_mon.reset_index(inplace=True)
            variable_mon.columns = ['HYBAS_ID','year','month','mean','max','min']
        ann_vars[var] = variable_ann 
        mon_vars[var] = variable_mon 
    
    return ann_vars, mon_vars
        
## can choose to do just annual, or also monthly (default is to have monthly)
def id_corrs(outcome, names, monthly = True):
    corr_ann={}
    corr_mon={}
        
    months=np.arange(1,13)
    months2=['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec']
    
    for v, var in enumerate(names):
        
        sub_dict_ann = {}
        sub_dict_mon = {}
#        print(var)
        variable = pd.read_csv('Drive_Data/' + str(var) + '_hybas4.csv')
#       print(var)
#        print(variable.head())
#        print()
        
        ## clean data to allow for aggregation 
        variable['DT']=pd.to_datetime(variable['Date'])
        variable['year']=variable.DT.dt.year
        variable['month']=variable.DT.dt.month
        variable.reset_index(inplace=True)
        subbasins = variable.HYBAS_ID.unique()
        
        variable_ann=variable.groupby(['HYBAS_ID','year']).agg({"mean": [np.min, np.max, np.mean]})
        variable_ann.reset_index(inplace=True)
        variable_ann.columns = ['HYBAS_ID','year','min','max','mean']
        
        if monthly == True: 
            try: 
                ## if have daily values with mean/min/max can just retrieve them 
                variable_mon = variable[['HYBAS_ID','year','month','min','max','mean']]
            except: 
                ## if not, need to do grouby but may get same value for all of them 
                variable_mon = variable.groupby(['HYBAS_ID','year','month']).agg({"mean": [np.mean, np.max, np.min]})
                variable_mon.reset_index(inplace=True)
                variable_mon.columns = ['HYBAS_ID','year','month','mean','max','min']
              
        for s in subbasins:
            ## make ann dictionary 
            sub_var_ann=variable_ann[variable_ann['HYBAS_ID']==s]
            # print(sub_var_ann['mean'].iloc[0])
            join_ann = pd.merge(sub_var_ann[['year','mean','max','min']],outcome[['year','outcome']],on='year')
            sub_dict_ann[s] = join_ann.corr()['outcome']
            #print(sub_dict_ann[s][1])
            
            ## repeat for monthly
            if monthly == True: 
                corr_df=pd.DataFrame(columns=months)
                for m in months: 
                    ## retrieve all data for a single month of a single subbasin
                    sub_var=variable_mon[variable_mon['HYBAS_ID']==s]
                    sub_var=sub_var[sub_var['month']==m]
                    sub_var.reset_index()
                    ## make df with generation 
                    join_mon=pd.merge(sub_var[['year','mean','max','min']], outcome, on='year')
                    ## make each column refer to corr between gen and a different month 
                    ## for that same subbasin 
                    corr_df.iloc[:,m-1]=join_mon.corr()['outcome']

                corr_df.columns=months2            
                sub_dict_mon[s] = corr_df
            
            corr_ann[names[v]] = sub_dict_ann
            corr_mon[names[v]] = sub_dict_mon
                
    return corr_ann, corr_mon, subbasins


def retrieve_index(v, obj):
    ## get index for the maximum corelation of each row 
    idx_stat = v.abs().idxmax(axis=1)

    ## compare between the rows to find absolute max 
    max_col = 0
    max_row = 0
    max_val = 0
    for r in range(0, len(idx_stat)):
        ## if idx_stat is a single value, it will go through the letters
        ## but that's okay for now 
        
        ## retrieve indices first for year 'series' and then for month     
        if obj=='Series': 
            col = idx_stat        
            compare = v[col]
            
        else: 
            col = idx_stat[r]
            compare = v[col].iloc[r]
        
        if abs(compare) > abs(max_val): 
            max_col = col
            max_row = r
            max_val = compare
            #print(max_row)
#            print(max_val)
            #print()
    ## output will be the index of the largest correlation 
    return max_col, max_row, idx_stat
        
def retrieve_max_corr(corr2, subbasins, names, ann = False): 

    max_corr = {}
    for n in names: 
        sub_max = pd.DataFrame(columns=['Stat','Month','Value'])

        corr = corr2[n]
#        print(n)
#        print(corr[s].head())
        for s in subbasins: 
            v = corr[s]
            end = len(v)-1
#            print(corr[s][2])
            #print(v)
            ## if it's annual, will only have one column -- check if this is true 
            ## or if it's monthly (>1 column)
            v = v.iloc[1:end]
            #print(v.head())
            if ann == True:
                obj = 'Series'
            else: 
                obj = 'no_ser'
            
            max_col, max_row, idx_stat = retrieve_index(v, obj)
            
            if obj == 'Series':
                maxi = v[max_col]#.iloc[max_row]
                sub_max.loc[s, 'Month'] = 'NA'
                sub_max.loc[s, 'Stat'] = max_col 
                sub_max.loc[s,'Value'] = maxi 
                #print(maxi)
    #                  max_corr.loc[s,'Month'] = 'NA'
                    ## record maximum value and its index
  #                  max_corr.loc[s,'Stat'] = max_col
                
            else: 
                maxi = v[max_col].iloc[max_row]
                sub_max.loc[s, 'Month'] = max_col 
                sub_max.loc[s,'Stat'] = idx_stat.index[max_row]
                sub_max.loc[s,'Value'] = maxi                    
#                    max_corr.loc[s,'Month'] = max_col
                    ## record maximum value and its index
                    ## in the monthly case, need to take index, not "max_col"
 #                   max_corr.loc[s,'Stat'] = idx_stat.index[max_row]
                            
           # print(sub_max.iloc[0,2])
            
 #max_corr.loc[s,'Value'] = maxi
        max_corr[n] = sub_max
    return max_corr


def id_corrs_mon(variable, subbasins, outcome):
    corr_mon={}
    months=np.arange(1,13)
    months2=['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec']

    for s in subbasins:
        corr_df=pd.DataFrame(columns=months)
        for m in months: 
            ##retrieve all data for a single month of a single subbasin
            sub_var=variable[variable['HYBAS_ID']==s]
            sub_var=sub_var[sub_var['month']==m]
            ##make df with generation 
            join=pd.merge(sub_var[['year','mean','max','median']], outcome, on='year')
            ##make each column refer to corr between gen and a different month 
            ##for that same subbasin 
            corr_df.iloc[:,m-1]=join.corr()['outcome']

        corr_df.columns=months2            
        corr_mon[s]=pd.DataFrame(corr_df)
       
    return join,corr_mon