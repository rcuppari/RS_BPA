# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 22:41:28 2021

@author: rcuppari
"""
import ee
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
#import geopandas as gpd
#import contextily as cx
#import shapely

#ee.Authenticate()
ee.Initialize()

##set consistent dates for all variables (better comparability)
start = ee.Date('2000-01-01')
end = ee.Date('2020-12-31')

##read in data
#GRIDMET, 1980-2021: standardized precip index, PDSI, evap drought demand, standardized precip ET index
CONUS_spi=ee.ImageCollection("GRIDMET/DROUGHT").select('spi14d').filterDate(start,end)
CONUS_et=ee.ImageCollection("GRIDMET/DROUGHT").select('eddi14d').filterDate(start,end)
CONUS_spei=ee.ImageCollection("GRIDMET/DROUGHT").select('spei14d').filterDate(start,end)
CONUS_pdsi=ee.ImageCollection("GRIDMET/DROUGHT").select('pdsi').filterDate(start,end)

#1980-2021
#water_area=ee.ImageCollection("")

#Terra Climate, 1958-2020
swe=ee.ImageCollection("IDAHO_EPSCOR/TERRACLIMATE").select('swe').filterDate(start,end)
terra_pdsi=ee.ImageCollection("IDAHO_EPSCOR/TERRACLIMATE").select('pdsi').filterDate(start,end)
aet=ee.ImageCollection("IDAHO_EPSCOR/TERRACLIMATE").select('aet').filterDate(start,end)
terra_tmax=ee.ImageCollection("IDAHO_EPSCOR/TERRACLIMATE").select('tmmx').filterDate(start,end)

#MODIS, 2000-2021
 ##NOTE: in Kelvins, need to convert
modis_temp=ee.ImageCollection("MODIS/006/MOD11A1").select('LST_Day_1km').filterDate(start,end)
snowcov=ee.ImageCollection("MODIS/006/MOD10A1").select(['NDSI_Snow_Cover','NDSI_Snow_Cover_Basic_QA']).filterDate(start,end)

##import basin boundaries in order to geographically clip data 
##and eventually loop over different spatial scales 
bounds = ee.FeatureCollection("WWF/HydroSHEDS/v1/Basins/hybas_3").filter(ee.Filter.eq('PFAF_ID',782))
#define as single feature
crb = ee.Feature(bounds.filterMetadata("PFAF_ID","equals",782).first())

##clip data with CRB boundaries 
CRB_spi=CONUS_spi.map(lambda image:image.clip(crb))
CRB_et=CONUS_et.map(lambda image:image.clip(crb))
CRB_spei=CONUS_spei.map(lambda image:image.clip(crb))
CRB_pdsiC=CONUS_pdsi.map(lambda image:image.clip(crb))

CRB_swe=swe.map(lambda image:image.clip(crb))
CRB_pdsiT=terra_pdsi.map(lambda image:image.clip(crb))
CRB_aet=aet.map(lambda image:image.clip(crb))
CRB_tmax=terra_tmax.map(lambda image:image.clip(crb))

CRB_temp=modis_temp.map(lambda image:image.clip(crb))
CRB_snowcov=snowcov.map(lambda image:image.clip(crb))












##function to transform image to dataframe to make it more manipulable in python 
def ee_array_to_df(arr, list_of_bands):
    """Transforms client-side ee.Image.getRegion array to pandas.DataFrame."""
    df = pd.DataFrame(arr)

    # Rearrange the header.
    headers = df.iloc[0]
    df = pd.DataFrame(df.values[1:], columns=headers)

    # Remove rows without data inside.
    df = df[['longitude', 'latitude', 'time', *list_of_bands]].dropna()

    # Convert the data to numeric values.
    for band in list_of_bands:
        df[band] = pd.to_numeric(df[band], errors='coerce')

    # Convert the time field into a datetime.
    df['datetime'] = pd.to_datetime(df['time'], unit='ms')

    # Keep the columns of interest.
    df = df[['time','datetime',  *list_of_bands]]

    return df





##CODE THAT NEEDS TO BE FIXED :( 
##use centroid of subbasin to check if it's inside of CRB
subbasin=ee.FeatureCollection("WWF/HydroSHEDS/v1/Basins/hybas_4")

##define function to do this checking for us 
def in_CRB(subbasin): 
    ##copy features
    #features = subbasin.getInfo()['features']
    ##find center (make the feature a geometry)
    center= ee.Geometry(subbasin).centroid()
    ##see if basin contains that center point
    inside_CRB=crb.contains(center,1)
    ##only keep those features with centroids within basins
    #subbasin2=subbasin.set('containedIn',inside_CRB)
    
    return subbasin.set('containedIn',inside_CRB) 
#ee.Feature(center).copyProperties(subbasin,features)
    
##need to use map
inCRB=in_CRB(subbasin)  
#inCRB=map(in_CRB,subbasin)
crb_subbasins=inCRB.filter(ee.Filter.eq("containedIn",True))
