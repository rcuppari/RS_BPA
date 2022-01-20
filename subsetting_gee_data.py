# -*- coding: utf-8 -*-
"""
Created on Sat Jan  1 20:31:27 2022

@author: rcuppari
"""

import pandas as pd 

data = pd.read_csv("Drive_data/ERA_CRB_hybas4.csv")

surf = data[['Date','HYBAS_ID','surface_runoff']]
surf.columns = ['Date', 'HYBAS_ID', 'mean']

snow = data[['Date','HYBAS_ID','snow_cover']]
snow.columns = ['Date', 'HYBAS_ID', 'mean']

precip = data[['Date','HYBAS_ID','total_precipitation']]
precip.columns = ['Date', 'HYBAS_ID', 'mean']

runoff = data[['Date','HYBAS_ID','runoff']]
runoff.columns = ['Date', 'HYBAS_ID', 'mean']

temp = data[['Date','HYBAS_ID','temperature_2m']]
temp.columns = ['Date', 'HYBAS_ID', 'mean']

#surf.to_csv("Drive_Data/surf_ERA_hybas4.csv")
#snow.to_csv("Drive_Data/snow_ERA_hybas4.csv")
#precip.to_csv("Drive_Data/precip_ERA_hybas4.csv")
#runoff.to_csv("Drive_Data/runoff_ERA_hybas4.csv")
#temp.to_csv("Drive_Data/temp_ERA_hybas4.csv")

data = pd.read_csv("Drive_data/ERA_CRB.csv")

surf = data[['Date','HYBAS_ID','surface_runoff']]
surf.columns = ['Date', 'HYBAS_ID', 'mean']

snow = data[['Date','HYBAS_ID','snow_cover']]
snow.columns = ['Date', 'HYBAS_ID', 'mean']

precip = data[['Date','HYBAS_ID','total_precipitation']]
precip.columns = ['Date', 'HYBAS_ID', 'mean']

runoff = data[['Date','HYBAS_ID','runoff']]
runoff.columns = ['Date', 'HYBAS_ID', 'mean']

temp = data[['Date','HYBAS_ID','temperature_2m']]
temp.columns = ['Date', 'HYBAS_ID', 'mean']

surf.to_csv("Drive_Data/surf_ERA.csv")
snow.to_csv("Drive_Data/snow_ERA.csv")
precip.to_csv("Drive_Data/precip_ERA.csv")
runoff.to_csv("Drive_Data/runoff_ERA.csv")
temp.to_csv("Drive_Data/temp_ERA.csv")