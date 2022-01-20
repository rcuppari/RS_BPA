# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 18:40:46 2021

@author: rcuppari
"""

import pandas as pd
import datetime as dt
import itertools
from itertools import chain, combinations
import numpy as np 
import statsmodels.api as sm
from sklearn.model_selection import train_test_split as tts
import matplotlib.pyplot as plt

###############################################################################
## here we take monthly data and regress the annual outcome against the months
## to see if any single month has high predictive value and when combined 
## with all others 
###############################################################################
def powerset(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))


def mon_corrs_reg(ann_outcome, all_data_mon, read_in, merged = True):
    corrs_mon = {}
    mon_dict = {}
    reg_dict = {}
    regression_output = pd.DataFrame()    
    
    months = ['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec']
    
#    data_mon['month_class'] = [str(q) for q in data_mon['month'].values]
    
    for n, name in enumerate(read_in): 
        
        ## in the input dataset, the outcome is matched by month and year
        if merged == True: 
            new_data = all_data_mon[name].drop('outcome', axis = 1)
        else: 
            new_data = all_data_mon[name]
        ## so now we want to take the annual value for the outcome (e.g., mean)
        ## and match it to the single month's data 
        new_data = new_data.merge(ann_outcome, on = ['year'])         

        
        for m, month in enumerate(months): 
            m = m+1
            subset_mon = new_data[new_data['month']==m]
        
            corr_mon = subset_mon.corr()['outcome']
            corr_mon.drop('outcome', axis = 0, inplace=True)
            
            mon_dict[m] = corr_mon 
            
            ## now want to get powerset with that month and see if 
            ## can find any predictive regressions with the outcome 
            ## using the single dataset (e.g., TerraClim // Modis)
            x = subset_mon.drop(['outcome','year'], axis = 1)
            x = sm.add_constant(x)
            y = subset_mon['outcome']

            X_train,X_test,y_train,y_test=tts(x,y,test_size=.3,random_state=1)

            for subset in powerset(x.columns): 

                if len(subset) > 0: 
                    model = sm.OLS(y_train, X_train[list(subset)]).fit()
    
            ## check the fit on test data 
                    predicted = model.predict(X_test[list(subset)])
                    pred_r2 = (predicted.corr(y_test)**2)
                    ## store regression inputs, p-values, and adj rsquared 
            
                    pval = list(model.pvalues)
                    r2 = np.float(model.rsquared_adj)
                    series = list(subset)
                    
                    new = {'inputs':[series], 'month':month, 'r2': r2, 'pval':[pval], 'test_r2': pred_r2} 
                    new = pd.DataFrame(new)
                    regression_output = pd.concat([regression_output, new], axis = 0)
            
 #           print(subset)
        corrs_mon[name] = mon_dict
            
    return corrs_mon, regression_output

def filtered_mon_regressions(dictionary, ann_outcome, read_in, train, test, length, outcome, merged = True):
    corrs_mon = {}
    mon_dict = {}
    regression_output = pd.DataFrame()    
    
    months = ['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec']
    
#    data_mon['month_class'] = [str(q) for q in data_mon['month'].values]
    
    for n, name in enumerate(read_in): 
        number = 0 
        
        if merged == True: 
        ## in the input dataset, the outcome is matched by month and year
            new_data = dictionary[name].drop('outcome', axis = 1)
        else: 
            new_data = dictionary[name]
        ## so now we want to take the annual value for the outcome (e.g., mean)
        ## and match it to the single month's data 
        new_data = new_data.merge(ann_outcome, on = 'year')         

        
        for m, month in enumerate(months): 
            m = m+1
            subset_mon = new_data[new_data['month']==m]
        
            corr_mon = subset_mon.corr()['outcome']
            corr_mon.drop('outcome', axis = 0, inplace=True)
            
            mon_dict[m] = corr_mon 
            
            ## now want to get powerset with that month and see if 
            ## can find any predictive regressions with the outcome 
            ## using the single dataset (e.g., TerraClim // Modis)
            x = subset_mon.drop(['outcome','year'], axis = 1)
            x = sm.add_constant(x)
            y = subset_mon['outcome']

            X_train,X_test,y_train,y_test=tts(x,y,test_size=.3,random_state=1)

            for subset in powerset(x.columns): 

                if len(subset) > 0: 
                    model = sm.OLS(y_train, X_train[list(subset)]).fit()
    
            ## check the fit on test data 
                    predicted = model.predict(X_test[list(subset)])
                    pred_r2 = (predicted.corr(y_test)**2)
                    ## store regression inputs, p-values, and adj rsquared 
                    training = model.predict(X_train[list(subset)])

                    pval = list(model.pvalues)
                    r2 = np.float(model.rsquared_adj)
                    series = list(subset)
                    if ((len(series) < length) & (r2 >= train) & (pred_r2 >= test)): 
                        new = {'inputs':[series], 'month':month, 'r2': r2, 'pval':[pval], 'test_r2': pred_r2} 
                        new = pd.DataFrame(new)
                        regression_output = pd.concat([regression_output, new], axis = 0)
            
                        number += 1
                        
                        fig = plt.figure()
                        plt.scatter(training, y_train, label = 'Training Set')
                        plt.scatter(predicted, y_test, label = 'Test Set')
                        plt.xlabel("Predicted " + outcome, fontsize = 16)
                        plt.ylabel("Observed " + outcome, fontsize = 16)
                        plt.legend()
                        plt.savefig('Regressions/Mon' + str(outcome) + str(number) )
                        plt.close()
                    
 #           print(subset)
        corrs_mon[name] = mon_dict
            
    return regression_output
