# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 13:23:32 2021

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

## list of all variables I want to iterate through
names=['snow_ERA', 'surf_ERA', 'runoff_ERA', 'precip_ERA', 'NDWI_CRB','AVHRR_CSR','EVI_CRB','GRACE_CSR','MODIS_LST_CRB', 'snow_CSR','SWE_CSR']

hist_rev=pd.read_excel("Drive_data/hist_rev.xlsx",header=1).loc[:,['Year', 'Reported Net Rev']]
hist_rev.columns=['year','outcome']
hist_rev04=hist_rev.iloc[:16,:] ##stop in 2004
hist_rev08=hist_rev.iloc[:13,:] ##stop in 2008

mean_rev=hist_rev.outcome.mean()
std_rev=hist_rev.outcome.std()
norm_hist=pd.DataFrame([(hist_rev.loc[i,'outcome']-mean_rev)/std_rev for i in range(0,len(hist_rev))])
norm_hist['year']=hist_rev['year']
norm_hist.columns = ['outcome', 'year']
##read in data produced through google earth engine 
##already have taken some summary statistics to aggregate by 
##want aggregated to the monthly timestep 

##then go through all subbasins and create dictionary with 
##correlations between generation and stats for each subbasin 
import corrs_subbasin_func
       
##use function to retrieve maximum corr value 
#for i in range(0,len(names)):
corr_ann, corr_mon, subbasins = corrs_subbasin_func.id_corrs(hist_rev, names)
    
max_corr_ann = corrs_subbasin_func.retrieve_max_corr(corr_ann, subbasins, names, ann = True)
max_corr_mon = corrs_subbasin_func.retrieve_max_corr(corr_mon, subbasins, names, ann = False)

##also have full basin variables to correlation with 
#full_bas=[grace_yr2,temp_yr2]
#names=['grace_crb','temp_crb']
#stats=['year','mean','max','min','rev']
#crb_corrs=pd.DataFrame(columns=names,index=stats)
#for i in range(0,len(full_bas)):
#    n=full_bas[i]
#    name=names[i]
#    join=pd.merge(n[['year','mean','max','min']],hist_rev,on='year')
#    crb_corrs.loc[:,name]=join.corr()['rev']
    
## takes the df with outcome, the test pred & actual
## and the train pred & actual 
## to show a plot comparing the two
## also insert label with units 
def plot_reg(df, test, train, label):
    range = [df['outcome'].min(), df['outcome'].max()]    
    plt.scatter(test['pred'], test['obs'], label='Test Data', marker='s')
    plt.scatter(train['pred'], train['obs'], label='Train Data', marker='o')
    plt.xlabel("Predicted " + label, fontsize=16)
    plt.ylabel("Observed " + label, fontsize=16)
    plt.plot(range,range, label = 'one to one line')
    plt.title("Predicting " + label, fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=16)
    plt.show()
    return 


#############################################################
##now that we know which stats correlate most strongly with 
##generation, identify a regression and produce r2 
#############################################################
## snow 430 in april or snow max apr 310; avhrr, min, mar 430 or mean, may 580; 
## grace, max, sep 930; grace min sep; 430 grace mean oct
## swe dec, min 690
ann_vars, mon_vars = corrs_subbasin_func.get_csvs(names)
snow = mon_vars['snow_CSR']
apr_snow = snow[(snow['HYBAS_ID']==7040379430) & (snow['month']==4)][['year','max']]
apr_snow.columns = ['year', 'apr_snow']

ann_snow = ann_vars['snow_CSR']
snow_190 = ann_snow[ann_snow['HYBAS_ID']==7040395190][['year','max']]

snow_era = ann_vars['snow_ERA']
snow_740 = snow_era[snow_era['HYBAS_ID']==7040388740][['year', 'mean']]
snow_740.columns = ['year', 'snow740']
snow_430 = snow_era[snow_era['HYBAS_ID']==7040379430][['year', 'mean']]
snow_430.columns = ['year', 'snow430']

#sub_var=sub_var.groupby('year').agg('mean')
avhrr = mon_vars['AVHRR_CSR']
mar_avhrr = avhrr[(avhrr['HYBAS_ID']==7040379430) & (avhrr['month'] == 3)][['year','mean']]
mar_avhrr.columns = ['year', 'mar_av']
may_avhrr = avhrr[(avhrr['HYBAS_ID']==7040395340) & (avhrr['month'] == 4)][['year','mean']]
may_avhrr.columns = ['year', 'may_av']
oct_avhrr = avhrr[(avhrr['HYBAS_ID']==7040388740) & (avhrr['month'] == 10)][['year','mean']]
oct_avhrr.columns = ['year', 'oct_av']
aug_avhrr = avhrr[(avhrr['HYBAS_ID']==7040388740) & (avhrr['month'] == 7)][['year','mean']]
aug_avhrr.columns = ['year', 'aug_av']

spr_avhrr = avhrr[((avhrr['month'] == 12) | (snow['month'] == 1) | \
                 (avhrr['month'] == 2)) & ((avhrr['HYBAS_ID'] == 7040388740) | \
                 (avhrr['HYBAS_ID'] ==7040379430) | \
                 (avhrr['HYBAS_ID'] ==7040379310) | \
                 (avhrr['HYBAS_ID'] ==7040388580))][['year', 'mean']]
spr_avhrr = spr_avhrr.groupby('year').agg('mean')
spr_avhrr.reset_index(inplace=True)
spr_avhrr.columns = ['year', 'avhrr']

grace = mon_vars['GRACE_CSR']
grace_740 = grace[((grace['HYBAS_ID']==7040388740) | (grace['HYBAS_ID']==7040379430)) \
                  & ((grace['month'] == 9) | (grace['month'] == 10))][['year','mean']]
grace_740 = grace_740.groupby('year').agg('mean')
grace_740.reset_index(inplace = True)
grace_740.columns = ['year', 'grace740']

grace_430 = grace[(grace['HYBAS_ID']==7040379430) & (grace['month'] == 10)][['year','mean']]
grace_430.columns = ['year', 'grace430']

grace_ann = mon_vars['GRACE_CSR']
grace_3169 = grace_ann[((grace_ann['HYBAS_ID']==7040379430) | (grace_ann['HYBAS_ID']==7040388740)) & \
                       ((grace_ann['month']==9) | (grace_ann['month']==10))][['year', 'mean']]
#grace_3169 = grace_ann[(grace_ann['HYBAS_ID']==7040379430) | (grace_ann['HYBAS_ID']==7040394690)][['year', 'mean']]
grace_3169 = grace_3169.groupby('year').agg('mean')
grace_3169.reset_index(inplace = True)
grace_3169.columns = ['year','grace3169']
#grace_310.columns = ['year', 'grace310']
#grace_690 = grace_ann[][['year','mean']]
#grace_690.columns = ['year', 'grace690']

precip = mon_vars['precip_ERA']
precip_feb = precip[(precip['HYBAS_ID']==7040379310) & (precip['month'] == 2)][['year','mean']]
precip_feb.columns = ['year', 'precip']

runoff2 = ann_vars['runoff_ERA']
runoff690 = runoff2[(runoff2['HYBAS_ID']==7040394690)][['year','mean']]
runoff690.columns = ['year', 'run690']
runoff680 = runoff2[(runoff2['HYBAS_ID']==7040394680)][['year','mean']]
runoff680.columns = ['year', 'run680']

runoff = mon_vars['surf_ERA']
surf680 = runoff[(runoff['HYBAS_ID']==7040394680) & (runoff['month'] == 6)][['year','mean']]
surf680.columns = ['year', 'surf680']
surf740 = runoff[(runoff['HYBAS_ID']==7040388740) & (runoff['month'] == 6)][['year','mean']]
surf740.columns = ['year', 'surf740']

evi = ann_vars['EVI_CRB']
evi_580 = evi[evi['HYBAS_ID']==7040388580][['year','mean']]
evi_580.columns = ['year', 'evi580']

evi_690 = evi[evi['HYBAS_ID']==7040388740][['year','mean']]
evi_690.columns = ['year', 'evi690']

modis = ann_vars['MODIS_LST_CRB']
#modis_430 = modis[modis['HYBAS_ID']==7040388740][['year','mean']]
#modis_430.columns = ['year', 'mod430']
#modis_430 = modis[modis['HYBAS_ID']==7040379430][['year','max']]
#modis_430.columns = ['year', 'mod430']

modis_430 = modis[(modis['HYBAS_ID']==7040379430)| (modis['HYBAS_ID']==7040388740)][['year','max']]
modis_430 = modis_430.groupby('year').agg('mean')
modis_430.reset_index(inplace = True)
modis_430.columns = ['year', 'mod430']

ndwi_ann = ann_vars['NDWI_CRB']
ndwi_740 = ndwi_ann[ndwi_ann['HYBAS_ID']==7040388740][['year','mean']]
ndwi_740.columns = ['year', 'ndwi740']

ndwi_310 = ndwi_ann[ndwi_ann['HYBAS_ID']==7040379310][['year','mean']]
ndwi_310.columns = ['year', 'ndwi430']

swe = mon_vars['SWE_CSR']
swe430 = swe[(swe['HYBAS_ID']==7040379430) & (swe['month'] == 12)][['year', 'min']]

##############################################################################
## bringing in the full CRB vars
read_in = ['TerraClim_CRB', 'snow_ERA', 'GRACE_CRB', 'EVI_CRB', 'MODIS_LST_CRB']
## script holding all of the long defined functions to do our work 
import corrs_func
outcome_mon = 0
ann_outcome = hist_rev
all_data_mon, all_data_ann = corrs_func.read_make_dictionary(read_in, outcome_mon, ann_outcome)

modis = all_data_ann['MODIS_LST_CRB']
modis.columns = ['year', 'modis', 'outcome']
#modis = modis[(modis['HYBAS_ID']==7040379430)| (modis['HYBAS_ID']==7040388740)]
#modis = modis.groupby('year').agg('mean')
#modis.reset_index(inplace = True)
#modis.columns = ['year', 'HYBAS_ID', 'mod_min', 'mod_max', 'mod_mean']

grace2 = all_data_mon['GRACE_CRB']
grace_aut = grace2[(grace2['month']==9) | (grace2['month']==10) | (grace2['month']==11)]
grace_aut.columns = ['year', 'month', 'csr10', 'gfz10', 'jpl10', 'outcome']
grace_aut['grace_mean_aut'] = (grace_aut['csr10'] + grace_aut['gfz10'] + grace_aut['jpl10'])/3
grace_aut = grace_aut.groupby('year').agg('mean')
grace_aut.reset_index(inplace = True)

ann_grace = all_data_ann['GRACE_CRB']
ann_grace.columns = ['year', 'csr', 'gfz', 'jpl', 'outcome']
ann_grace['avg_grace'] = (ann_grace['csr'] + ann_grace['gfz'] + ann_grace['jpl'])/3 

TerraClim = all_data_mon['TerraClim_CRB'][['tmmn_mean','tmmx_mean','pdsi_mean','year','month']]
wint = TerraClim[(TerraClim['month'] > 1) | (TerraClim['month'] < 6)]
sum = TerraClim[(TerraClim['month'] > 5) & (TerraClim['month'] < 9)]
wint = wint.groupby('year').agg('mean')
sum = sum.groupby('year').agg('mean')

wint.columns = ['wtmmn', 'wtmmx', 'wpdsi', 'wmonth']
sum.columns = ['stmmn', 'stmmx', 'spdsi', 'smonth']

ann_TC = TerraClim[['year', 'tmmx_mean']].groupby('year').agg('mean')
ann_TC.reset_index(inplace = True)

tmmx = TerraClim[['year', 'tmmx_mean']].groupby('year').agg('max')
tmmx.reset_index(inplace = True)
tmmx.columns = ['year', 'tmax']

tmmn = TerraClim[['year', 'tmmn_mean']].groupby('year').agg('min')
tmmn.reset_index(inplace = True)
tmmn.columns = ['year', 'tmin']

snow = all_data_mon['snow_ERA']
ann_snow = snow.groupby('year').agg('max')
ann_snow.reset_index(inplace = True)
ann_snow.columns = ['year', 'month', 'ann_snow', 'outcome']
ann_snow = ann_snow.iloc[:, :3]

#snow_feb = snow[((snow['month'] == 2)) & \
#                  (snow['HYBAS_ID'] ==7040379430)][['year', 'mean']]
snow = mon_vars['snow_CSR']
snow_feb = snow[(snow['HYBAS_ID'] ==7040379430) & (snow['month'] == 2)][['year', 'max']]                 
snow_feb = snow_feb.groupby('year').agg('max')
snow_feb.reset_index(inplace = True)
snow_feb.columns = ['year','feb_snow']

evi_ann = all_data_ann['EVI_CRB'][['year', 'mean']]
evi_ann = evi_ann.groupby('year').agg('mean')
evi_ann.reset_index(inplace = True)

comb = hist_rev.merge(precip_feb, on='year')
comb = pd.merge(comb, apr_snow, on='year')
comb = pd.merge(comb, snow_740, on='year')
comb = pd.merge(comb, snow_430, on='year')
comb = pd.merge(comb, evi_580, on='year')
comb = pd.merge(comb, surf680, on='year')
comb = pd.merge(comb, runoff680, on='year')
comb = pd.merge(comb, spr_avhrr, on='year')
comb = pd.merge(comb, mar_avhrr, on='year')
comb = pd.merge(comb, modis[['year', 'modis']], on='year')
comb = pd.merge(comb, tmmx, on='year')
comb = pd.merge(comb, tmmn, on='year')
comb = pd.merge(comb, modis_430, on='year')
#comb=pd.merge(comb, grace_3169, on='year')

#comb = pd.merge(comb, grace_aut[['grace_mean_aut', 'year']], on='year')
#comb = pd.merge(comb, grace_740, on='year')
comb = pd.merge(comb, snow_feb, on='year')
#comb = pd.merge(comb, ann_grace[['avg_grace', 'year']], on='year')
comb = pd.merge(comb, evi_ann, on = 'year')
#comb = pd.merge(comb, ann_snow, on = 'year')

## REGULAR, NOT DETRENDED
## works well: grace_mean_aut and annual tmax mean or max 
#X=comb[['grace_mean_aut', 'tmax', MEAN 'feb_snow']]
#X=sm.add_constant(X)

comb2 = comb[comb['year'] > 2002]
X=comb2[['evi580', 'feb_snow', 'modis']]
X=sm.add_constant(X)
X_train,X_test,y_train,y_test = tts(X, comb2['outcome'], test_size = .2, random_state = 3)
est=sm.OLS(y_train,X_train)
est2=est.fit()
print(est2.summary())

## depending on random_state evi 580 alone ranges from 0.62 to 0.76 (rs = 5) for Gross Power Sales
## with rs = 3, evi 580 + mod 740 mean = 0.64 for Reported Net Rev 
## with rs = 3, evi 580 + mod 430 max = 0.7 for RPR
## with rs = 3, evi 580 + mod 430 max + surf680 = 0.87 for RPR
## with rs = 3, evi 580 + surf 680 max = 0.78 for RPR
## with rs = 3, evi580 + feb 430 snow + modis avg = 0.83 for RPR

pred = est2.predict(X_test)
pred_train = est2.predict(X_train)
pred_all = est2.predict(X)

test_corr = pred.corr(y_test)
train_corr = pred_train.corr(y_train)

print('training r2 = ' + str(test_corr**2))
print('test r2 = ' + str(train_corr**2))


test = pd.concat([y_test, pred], axis = 1)
test.columns = ['obs', 'pred']

train = pd.concat([y_train, pred_train], axis = 1)
train.columns = ['obs', 'pred']
plot_reg(comb, test, train, "Reported Net Rev ($ '000)")

##############################################################################
## does including avg natural gas make a difference? 
## NOPE! Consistent with the historical analysis (fascinating )
##############################################################################
ng = pd.read_csv("../BPA/Hist_data/Henry_Hub_Historical.csv", header = 4)
ng.columns = ['Date', 'ng']
ng['Date'] = pd.to_datetime(ng['Date'], format = '%b-%y')
ng['year'] = ng.Date.dt.year
ann_ng = ng.groupby('year').agg('mean')
ann_ng.reset_index(inplace = True)

w_ng = ann_ng.merge(comb2, on = 'year')

X=w_ng[['evi580', 'feb_snow', 'modis', 'ng']]
X=sm.add_constant(X)
X_train,X_test,y_train,y_test = tts(X, comb2['outcome'], test_size = .2, random_state = 3)
est=sm.OLS(y_train,X_train)
est2=est.fit()
print(est2.summary())

pred = est2.predict(X_test)
pred_train = est2.predict(X_train)
pred_all = est2.predict(X)

test_corr = pred.corr(y_test)
train_corr = pred_train.corr(y_train)

print('training r2 = ' + str(test_corr**2))
print('test r2 = ' + str(train_corr**2))


test = pd.concat([y_test, pred], axis = 1)
test.columns = ['obs', 'pred']

train = pd.concat([y_train, pred_train], axis = 1)
train.columns = ['obs', 'pred']
plot_reg(comb, test, train, "Reported Net Rev ($ '000)")

##############################################################################
##so now we think this work. How does it compare to the TDA & gage based
##indices that we built? 
##############################################################################
key_hist=pd.read_csv('C:/Users/rcuppari/Desktop/CAPOW_PY36-master/Updated streamflow data/ann_hist.csv')

hist_rev2=pd.read_excel("Drive_data/hist_rev.xlsx",header=1).loc[:,['Year','Net Power Sales']]
hist_rev2.columns=['year','rev']

key2=key_hist[key_hist['year']>=2004][['TDA6ARF_daily','year']]
merge=key2[['year','TDA6ARF_daily']].merge(hist_rev2,on='year')

sns.regplot(merge['TDA6ARF_daily'],merge['rev'])

ORO=pd.read_csv("../BPA/Hist_data/ORO6H_daily.csv")
ORO['date']=pd.to_datetime(ORO.iloc[:,0])
ORO['year']=ORO.date.dt.year
ORO['month']=ORO.date.dt.month

mon_ORO=ORO.groupby(['year','month']).agg('mean')
mon_ORO.reset_index(inplace=True)

sep_ORO=mon_ORO[mon_ORO['month']==9]
sep_ORO.reset_index(inplace=True)

temps=pd.read_csv("../BPA/Hist_data/Wind_Temps_1970_2017.csv")
boise=temps[temps['STATION']=='USW00024131']
boise.loc[:,'TAVG']=pd.to_numeric(boise.loc[:,'TAVG'],errors='coerce')
boise=boise.dropna(subset=['TAVG'],how='all')

ann_boise=boise.groupby('Year').agg({'TAVG':'mean'})
ann_boise.loc[:,'year']=ann_boise.index

##join what we have
sep_ORO.index = sep_ORO.year
comb2 = sep_ORO.merge(ann_boise['TAVG'],left_index=True,right_index=True)
#comb2=comb2.merge(comb[['grace_430','temp_310','rev','year']],on='year')
comb2 = comb2.merge(key_hist[['BFE6L_daily','TDA6ARF_daily','year']],on = 'year')
comb2 = comb2.merge(hist_rev, on = 'year')
comb2.columns = ['index','year','month','ORO','TAVG','BFE','TDA', 'rev']

##############################################################################
##test out using residuals only though need to detrend with all available data
##to be consistent, should I use 2004-2009 as the baseline, which is the 
##GRACE benchmark for deviations?????? 
oro_m=(sep_ORO[(sep_ORO['year']<=2009) & (sep_ORO['year']>=2004)]).iloc[:,3].mean()
sep_ORO2=sep_ORO.iloc[:,3]/oro_m

key_hist_m=(key_hist[(key_hist['year']>=2004)&(key_hist['year']<=2009)]).mean()
tda_resids=key_hist['TDA6ARF_daily']/key_hist_m['TDA6ARF_daily']
bfe_resids=key_hist['BFE6L_daily']/key_hist_m['BFE6L_daily']
oro_resids=key_hist['ORO6H_daily']/key_hist_m['ORO6H_daily']

rev_m=(hist_rev[(hist_rev['year']>=2004)&(hist_rev['year']<=2009)])['outcome'].mean()
rev_resids=hist_rev['outcome']/rev_m
rev_resids.index=hist_rev.loc[:,'year']   
rev_resids.columns=['rev_resids']

#oro_resids=pd.DataFrame([(sep_ORO.iloc[i,3]-sep_ORO.iloc[:,3].mean())/sep_ORO.iloc[:,3].std() for i in range(0,len(sep_ORO))])
#bfe_resids=pd.DataFrame([(key_hist.iloc[i,2]-key_hist.iloc[:,2].mean())/key_hist.iloc[:,2].std() for i in range(0,len(key_hist))])
#tda_resids=pd.DataFrame([(key_hist.iloc[i,3]-key_hist.iloc[:,3].mean())/key_hist.iloc[:,3].std() for i in range(0,len(key_hist))])
#grace430_resids=pd.DataFrame([(sub_var.iloc[i,1]-sub_var.iloc[:,1].mean())/sub_var.iloc[:,1].std() for i in range(0,len(sub_var))])
#temp310_resids=pd.DataFrame([(sub_var7.iloc[i,1]-sub_var7.iloc[:,1].mean())/sub_var7.iloc[:,1].std() for i in range(0,len(sub_var7))])
#rev_resids=pd.DataFrame([(hist_rev.iloc[i,1]-hist_rev.iloc[:,1].mean())/hist_rev.iloc[:,1].std() for i in range(0,len(hist_rev))])
#rev_resids.loc[:,1]=hist_rev['year']
#rev_resids.columns=['rev_resid','year']
#sub_var=GRACE[(GRACE['HYBAS_ID']==7040379430)&(GRACE['month']==2)][['year','mean']]
#sub_var6=temp_mon[(temp_mon['HYBAS_ID']==7040379310)&(temp_mon['month']==9)][['year','mean']]

comb_resids=pd.concat([oro_resids,bfe_resids,tda_resids],axis=1)
comb_resids.columns=['ORO','BFE','TDA']
comb_resids.index=sep_ORO['year']

comb3=comb_resids.merge(comb[['grace_430','temp_310','year']],on='year')
comb3=comb3.merge(norm_hist,on='year')
comb3.columns=['year','ORO','BFE','TDA','grace_430','temp_310','rev']

print(comb3.corr()['rev'])
#print(comb3.corr()['rev_resid'])

X=comb3[['grace_430']]
X=sm.add_constant(X)
X_train,X_test,y_train,y_test=tts(X,comb3['rev'],test_size=.3,random_state=1)

est=sm.OLS(y_train,X_train)
est2=est.fit()
print(est2.summary())

r2=round(est2.rsquared_adj,3)
pred=est2.predict(X_test)
pred_train=est2.predict(X_train)
pred_all=est2.predict(X)

plt.scatter(pred,y_test,label='Test Data')
plt.scatter(pred_train,y_train,label='Training Data')
plt.xlabel("Predicted Revenues ($B)",fontsize=16)
plt.ylabel("Observed Revenues ($B)",fontsize=16)
plt.title("Use of Feb. GRACE 430 & Sep. Temp 310",fontsize=16)
plt.xticks([2050000,2100000,2200000,2300000,2400000,2500000],
           [2.05,2.1,2.2,2.3,2.4,2.5],fontsize=14)
plt.yticks([2050000,2100000,2200000,2300000,2400000,2500000],
           [2.05,2.1,2.2,2.3,2.4,2.5],fontsize=14)
z = np.polyfit(pred_all, comb['rev'], 1)
p = np.poly1d(z)
plt.plot(pred_all,p(pred_all),"r--")
plt.annotate(('r2 = '+str(r2)),(2100000,2350000),fontsize=16)
plt.show()

































