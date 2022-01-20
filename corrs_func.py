# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 10:48:09 2021

@author: rcuppari
"""

import pandas as pd
import datetime as dt
from itertools import chain, combinations
import numpy as np 
import statsmodels.api as sm
from sklearn.model_selection import train_test_split as tts
import matplotlib.pyplot as plt

##############################################################################
def clean_data(data): 
    data['date']=pd.to_datetime(data.date)
    data['year'] = data['date'].dt.year
    data['month'] = data['date'].dt.month
    data.columns=['date','outcome','year','month']
    data.drop('date',axis=1,inplace=True)
    data_ann = data.groupby('year').agg('mean')
    data_ann.drop('month',axis=1,inplace=True)
    data_ann.reset_index(inplace=True)
    data_ann.columns = ['year','outcome']
    
    data_mon = data.groupby(['year','month']).agg('mean')
    data_mon.reset_index(inplace=True)
    data_mon.columns = ['year','month','outcome']
    
    return data_mon, data_ann


def read_make_dictionary(read_in, outcome_mon, outcome_ann, monthly = 'True', detrend = 'False'):
    all_data_year = {}
    all_data_mon = {}
    for n, names in enumerate(read_in): 
        data = pd.read_csv('Drive_Data/' + names + '.csv')
        data.Date = pd.to_datetime(data.Date)
        data['mean_month'] = data.Date.dt.month
        data['mean_year'] = data.Date.dt.year
        data = data.loc[:,data.columns.str.contains('mean')]
        data = data.rename(columns = {'mean_year':'year','mean_month':'month'})
        
        if detrend == 'True': 
            means = data.groupby('month').agg('mean')
            stds = data.std()
            means.reset_index(inplace=True)
            
            for c, col in enumerate(means.iloc[:,1:-1]):  
                print(col)
                merge = means[[col, 'month']].merge(data[[col,'month','year']], on = 'month')
                ## when merge, because the colnames from the grouped
                ## means are the same as the OG data names, need to 
                ## add "_x" for the means and "_y" for data
                detrend_data = (merge[col + '_y'] - merge[col + '_x'])/stds[col]
                
                data.loc[:, col] = detrend_data 
                
            if monthly == 'False':
                all_data_mon = 'na'
            else: 
                data_mon = data.groupby(['year','month']).agg('mean')  
                data_mon.reset_index(inplace=True)  
                data_mon = pd.merge(data_mon, outcome_ann, on = ['year'])
                all_data_mon[names] = data_mon 
        
            ann_data = data.groupby(['year']).agg('mean')
            ann_data.reset_index(inplace=True)
            ann_data.drop('month', axis=1, inplace=True)
            ann_data = pd.merge(ann_data, outcome_ann)
            
            
        else:     
            if monthly == 'False':
                all_data_mon = 'na'
            else: 
                data_mon = data.groupby(['year','month']).agg('mean')  
                data_mon.reset_index(inplace=True)  
                data_mon = pd.merge(data_mon, outcome_ann, on = ['year'])
                all_data_mon[names] = data_mon 
        
            ann_data = data.groupby(['year']).agg('mean')
            ann_data.reset_index(inplace=True)
            ann_data.drop('month', axis=1, inplace=True)
            ann_data = pd.merge(ann_data, outcome_ann)

        all_data_year[names] = ann_data
        
    return all_data_mon, all_data_year


def id_corrs(data_ann, all_data_ann, read_in):
    corrs_ann = {}
    
    for n, name in enumerate(read_in): 
        dict_subset_ann = all_data_ann[name]
        
        corr_ann = dict_subset_ann.corr()['outcome']
        corr_ann.drop('outcome', axis = 0, inplace=True)
        
        corrs_ann[name] = corr_ann
    return corrs_ann



def powerset(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

def combination_ann_regressions(df, read_in): 
    regression_output = pd.DataFrame()
    for n, names in enumerate(read_in): 
        data = df[names]
        x = data.drop(['outcome','year'], axis = 1)
        x = sm.add_constant(x)
        y = data['outcome']

        X_train,X_test,y_train,y_test=tts(x,y,test_size=.3,random_state=1)

        for subset in powerset(x.columns): 
#        print(subset)
            if len(subset) > 0: 
                model = sm.OLS(y_train, X_train[list(subset)]).fit()
    
    ## check the fit on test data 
                predicted = model.predict(X_test[list(subset)])
                pred_r2 = (predicted.corr(y_test)**2)
    ## store regression inputs, p-values, and adj rsquared 
            
                pval = list(model.pvalues)
                r2 = np.float(model.rsquared_adj)
                series = list(subset)
            
                new = {'inputs':[series], 'r2': r2, 'pval':[pval], 'test_r2': pred_r2} 
                new = pd.DataFrame(new)
                regression_output = pd.concat([regression_output, new], axis = 0)
   
    return regression_output


def filtered_ann_regressions(dictionary, read_in, train, test, length, outcome): 
    regression_output = pd.DataFrame()
    number = 0 
    for n, names in enumerate(read_in): 
        data = dictionary[names]
        x = data.drop(['outcome','year'], axis = 1)
        x = sm.add_constant(x)
        y = data['outcome']

        X_train,X_test,y_train,y_test=tts(x,y,test_size=.3,random_state=1)

        for subset in powerset(x.columns): 
#        print(subset)
            if len(subset) > 0: 
                model = sm.OLS(y_train, X_train[list(subset)]).fit()
    
    ## check the fit on test data 
                predicted = model.predict(X_test[list(subset)])
                pred_r2 = (predicted.corr(y_test)**2)
                
                training = model.predict(X_train[list(subset)])
                
    ## store regression inputs, p-values, and adj rsquared 
            
                pval = list(model.pvalues)
                r2 = np.float(model.rsquared_adj)
                series = list(subset)
                
                if ((len(series) < length) & (r2 >= train) & (pred_r2 >= test)): 
                    number += 1
                    new = {'inputs':[series], 'r2': r2, 'pval':[pval], 'test_r2': pred_r2} 
                    new = pd.DataFrame(new)
    
                    regression_output = pd.concat([regression_output, new], axis = 0)
                    
                    plt.figure()
                    plt.scatter(training, y_train, label = 'Training Set')
                    plt.scatter(predicted, y_test, label = 'Test Set')
                    plt.xlabel("Predicted " + outcome, fontsize = 16)
                    plt.ylabel("Observed " + outcome, fontsize = 16)
                    plt.legend()
                    plt.savefig('Regressions/Ann'+ str(outcome) + str(number))
                    plt.close()
                    
   
    return regression_output
