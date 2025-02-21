# -*- coding: utf-8 -*-
"""
Created on Tue Dec 31 04:19:40 2024

@author: user
"""


        
def deg2km(lat1, lon1, lat2, lon2):
    
    import math
    
    # Radius of the Earth in kilometers
    R = 6371.0

    # Convert latitude and longitude from degrees to radians
    lat1 = math.radians(lat1)
    lon1 = math.radians(lon1)
    lat2 = math.radians(lat2)
    lon2 = math.radians(lon2)

    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    distance = R * c

    return distance    

def degree_to_meters(degree, latitude=35):
    import math
    # Earth's approximate radius in meters
    earth_radius = 6371000
    # Adjust the distance for the cosine of the latitude
    return degree * (math.pi / 180) * earth_radius * math.cos(math.radians(latitude))

# Ancillary functions from here
# Import only Geo-reference
def geotiffreadRef(SARtifname):
    import gdal
    import numpy as np

    gdal.AllRegister()

    ds = gdal.Open(SARtifname)
    tif_ref = ds.GetGeoTransform()
    rows, cols = ds.RasterXSize, ds.RasterYSize

    tif_ref = np.array(tif_ref)
    tif_ref[2] = tif_ref[0] + tif_ref[1] * (rows - 1)
    tif_ref[4] = tif_ref[3] + tif_ref[5] * (cols - 1)

    tif_ref.astype(np.double)

    return tif_ref, rows, cols


# Corresponding function to geotiffread of MATLAB
def geotiffread(SARtifname, num_band):
    import gdal
    import numpy as np

    gdal.AllRegister()

    ds = gdal.Open(SARtifname)

    if num_band == 3:
        band1 = ds.GetRasterBand(1)
        arr1 = band1.ReadAsArray()
        band2 = ds.GetRasterBand(2)
        arr2 = band2.ReadAsArray()
        band3 = ds.GetRasterBand(3)
        arr3 = band3.ReadAsArray()

        cols, rows = arr1.shape

        arr = np.zeros((cols, rows, 3))
        arr[:,:,0] = arr1
        arr[:,:,1] = arr2
        arr[:,:,2] = arr3

    elif num_band == 1:
        band1 = ds.GetRasterBand(1)
        arr = band1.ReadAsArray()

        cols, rows = arr.shape

    elif num_band == 2:
        band1 = ds.GetRasterBand(1)
        arr1 = band1.ReadAsArray()
        band2 = ds.GetRasterBand(2)
        arr2 = band2.ReadAsArray()
        
        cols, rows = arr1.shape

        arr = np.zeros((cols, rows, 2))
        arr[:,:,0] = arr1
        arr[:,:,1] = arr2

    else:
        print('cannot open except number of band is 1, 2 or 3')

    tif_ref = ds.GetGeoTransform()
    tif_ref = np.array(tif_ref)
    tif_ref[2] = tif_ref[0] + tif_ref[1] * (rows - 1)
    tif_ref[4] = tif_ref[3] + tif_ref[5] * (cols - 1)

    tif_ref.astype(np.double)

    return arr, tif_ref


# Read AIS data file
def readAIS(AISname):
    import pandas as pd
    import datetime
    import numpy as np

    #import cudf
    # import datetime.datetime.strptime
    
    try:
        df = pd.read_csv(AISname)
    except:
        df = AISname
    #df = pd.concat(df)
    
    #df = dd.read_csv(AISname, assume_missing=True)

    # Dynamic Information
    Ship_ID = df['MMSI']
    Ship_Date = df['Date']
    Ship_Time = df['Time']
    Ship_Lon = df['Lon']
    Ship_Lat = df['Lat']
    Ship_SOG = df['SOG']
    Ship_COG = df['COG']

    # Static Information
    Ship_Name = df['VesselName']
    Ship_Type = df['VesselType']
    Ship_DimA = df['DimA']
    Ship_DimB = df['DimB']
    Ship_DimC = df['DimC']
    Ship_DimD = df['DimD']
    Ship_Status = df['Status']
    

    # Transfer this into np.array
    Ship_Name = np.array(Ship_Name)
    Ship_Type = np.array(Ship_Type)
    Ship_Status = np.array(Ship_Status)
    
    Ship_ID = np.array(Ship_ID)
    
    #Ship_Lon = Ship_Lon.compute()
    #Ship_Lat = Ship_Lat.compute()
    #Ship_SOG = Ship_SOG.compute()
    #Ship_COG = Ship_COG.compute()

    Ship_Lon = np.array(Ship_Lon)
    Ship_Lat = np.array(Ship_Lat)
    Ship_SOG = np.array(Ship_SOG)
    Ship_COG = np.array(Ship_COG)
    Ship_DimA = np.array(Ship_DimA)
    Ship_DimB = np.array(Ship_DimB)
    Ship_DimC = np.array(Ship_DimC)
    Ship_DimD = np.array(Ship_DimD)

    # Transfer the time array into serial number
    Ship_Time = np.array(Ship_Time)
    Ship_Date = np.array(Ship_Date)
    
    #Ship_Time=Ship_Time.compute()
    #Ship_Date=Ship_Date.compute()
    
    Ship_Time_num = np.zeros(len(Ship_Time))
    for num in range(len(Ship_Time_num)):
        
        temp1 = Ship_Date[num]
        temp2 = Ship_Time[num]
        
        # print(temp)

        try:
            temp = datetime.datetime(int(temp1[0:4]), int(temp1[5:7]), int(temp1[8:10]), int(temp2[0:2]),
                                      int(temp2[3:5]), int(temp2[6:8]))

            # Ship_Time_num[num]=time.mktime(temp.timetuple())
            # temp = datetime.datetime.strptime(temp, '%Y-%m-%d %H:%M:%S.%f')
            # print(temp)
            Ship_Time_num[num] = time.mktime(temp.timetuple())
            # print(time.mktime(temp1.timetuple()))
        except:
            continue

    # Join StaticData: MMSI and Dimension    
    #Ship_ID = Ship_ID.compute()
    #Ship_DimA = Ship_DimA.compute()
    #Ship_DimB = Ship_DimB.compute()
    #Ship_DimC = Ship_DimC.compute()
    #Ship_DimD = Ship_DimD.compute()
    
    CSVexport = np.zeros((len(Ship_DimA), 5))
    
    for num in range(len(Ship_DimA)):

        try:
            # CSVtemp=CSVlines[num]
            #CSVexport[num, 0] = pd.to_numeric(Ship_ID[num])
            #CSVexport[num, 1] = pd.to_numeric(Ship_DimA[num])
            #CSVexport[num, 2] = pd.to_numeric(Ship_DimB[num])
            #CSVexport[num, 3] = pd.to_numeric(Ship_DimC[num])
            #CSVexport[num, 4] = pd.to_numeric(Ship_DimD[num])
            
            CSVexport[num, 0] = Ship_ID[num]
            CSVexport[num, 1] = Ship_DimA[num]
            CSVexport[num, 2] = Ship_DimB[num]
            CSVexport[num, 3] = Ship_DimC[num]
            CSVexport[num, 4] = Ship_DimD[num]
            
            
        except:
            continue

    return Ship_ID, Ship_Time_num, Ship_Lon, Ship_Lat, Ship_SOG, Ship_COG, Ship_Name, Ship_Type, Ship_Status, CSVexport

# Read AIS data file
def readAIS1(AISname):
    import pandas as pd
    import datetime
    import numpy as np
    import time
    import dask.dataframe as dd
    #import cudf
    # import datetime.datetime.strptime
    
    try:
        df = pd.read_csv(AISname)
    except:
        df = AISname
    #df = pd.concat(df)
    
    #df = dd.read_csv(AISname, assume_missing=True)

    # Dynamic Information
    Ship_ID = df['mmsi']
    Ship_Date = df['date']
    Ship_Time = df['time']
    Ship_Lon = df['lon']
    Ship_Lat = df['lat']
    Ship_SOG = df['sog']
    Ship_COG = df['cog']

    # Static Information
    Ship_Name = df['vesselname']
    Ship_Type = df['vesseltype']
    Ship_DimA = df['dima']
    Ship_DimB = df['dimb']
    Ship_DimC = df['dimc']
    Ship_DimD = df['dimd']
    Ship_Status = df['status']
    

    # Transfer this into np.array
    Ship_Name = np.array(Ship_Name)
    Ship_Type = np.array(Ship_Type)
    Ship_Status = np.array(Ship_Status)
    
    Ship_ID = np.array(Ship_ID)
    
    #Ship_Lon = Ship_Lon.compute()
    #Ship_Lat = Ship_Lat.compute()
    #Ship_SOG = Ship_SOG.compute()
    #Ship_COG = Ship_COG.compute()

    Ship_Lon = np.array(Ship_Lon)
    Ship_Lat = np.array(Ship_Lat)
    Ship_SOG = np.array(Ship_SOG)
    Ship_COG = np.array(Ship_COG)
    Ship_DimA = np.array(Ship_DimA)
    Ship_DimB = np.array(Ship_DimB)
    Ship_DimC = np.array(Ship_DimC)
    Ship_DimD = np.array(Ship_DimD)

    # Transfer the time array into serial number
    Ship_Time = np.array(Ship_Time)
    Ship_Date = np.array(Ship_Date)
    
    #Ship_Time=Ship_Time.compute()
    #Ship_Date=Ship_Date.compute()
    
    Ship_Time_num = np.zeros(len(Ship_Time))
    for num in range(len(Ship_Time_num)):
        
        temp1 = Ship_Date[num]
        temp2 = Ship_Time[num]
        
        # print(temp)

        try:
            temp = datetime.datetime(int(temp1[0:4]), int(temp1[5:7]), int(temp1[8:10]), int(temp2[0:2]),
                                      int(temp2[3:5]), int(temp2[6:8]))

            # Ship_Time_num[num]=time.mktime(temp.timetuple())
            # temp = datetime.datetime.strptime(temp, '%Y-%m-%d %H:%M:%S.%f')
            # print(temp)
            Ship_Time_num[num] = time.mktime(temp.timetuple())
            # print(time.mktime(temp1.timetuple()))
        except:
            continue

    # Join StaticData: MMSI and Dimension    
    #Ship_ID = Ship_ID.compute()
    #Ship_DimA = Ship_DimA.compute()
    #Ship_DimB = Ship_DimB.compute()
    #Ship_DimC = Ship_DimC.compute()
    #Ship_DimD = Ship_DimD.compute()
    
    CSVexport = np.zeros((len(Ship_DimA), 5))
    
    for num in range(len(Ship_DimA)):

        try:
            # CSVtemp=CSVlines[num]
            #CSVexport[num, 0] = pd.to_numeric(Ship_ID[num])
            #CSVexport[num, 1] = pd.to_numeric(Ship_DimA[num])
            #CSVexport[num, 2] = pd.to_numeric(Ship_DimB[num])
            #CSVexport[num, 3] = pd.to_numeric(Ship_DimC[num])
            #CSVexport[num, 4] = pd.to_numeric(Ship_DimD[num])
            
            CSVexport[num, 0] = Ship_ID[num]
            CSVexport[num, 1] = Ship_DimA[num]
            CSVexport[num, 2] = Ship_DimB[num]
            CSVexport[num, 3] = Ship_DimC[num]
            CSVexport[num, 4] = Ship_DimD[num]
            
            
        except:
            continue

    return Ship_ID, Ship_Time_num, Ship_Lon, Ship_Lat, Ship_SOG, Ship_COG, Ship_Name, Ship_Type, Ship_Status, CSVexport


# Read AIS data file
def readAISProc(AISname):
    import pandas as pd
    import datetime
    import numpy as np
    import time
    # import datetime.datetime.strptime

    df = pd.read_csv(AISname)

    # Dynamic Information
    Ship_ID = df['MMSI']
    Ship_Date = df['Date']
    Ship_Time = df['Time']
    Ship_Lon = df['Lon']
    Ship_Lat = df['Lat']
    Ship_SOG = df['SOG']
    Ship_COG = df['COG']

    # Static Information
    Ship_Name = df['VesselName']
    Ship_Type = df['VesselType']
    Ship_DimA = df['DimA']
    Ship_DimB = df['DimB']
    Ship_DimC = df['DimC']
    Ship_DimD = df['DimD']
   
    # Transfer this into np.array
    Ship_ID = np.array(Ship_ID)
    Ship_Lon = np.array(Ship_Lon)
    Ship_Lat = np.array(Ship_Lat)
    Ship_SOG = np.array(Ship_SOG)
    Ship_COG = np.array(Ship_COG)

    Ship_DimA = np.array(Ship_DimA)
    Ship_DimB = np.array(Ship_DimB)
    Ship_DimC = np.array(Ship_DimC)
    Ship_DimD = np.array(Ship_DimD)

    # Transfer the time array into serial number
    Ship_Time_num = np.zeros((Ship_Time.size))
    for num in range(len(Ship_Time_num)):
        
        temp1 = Ship_Date[num]
        temp2 = Ship_Time[num]
        
        # print(temp)

        try:
            temp = datetime.datetime(int(temp1[0:4]), int(temp1[5:7]), int(temp1[8:10]), int(temp2[0:2]),
                                      int(temp2[3:5]), int(temp2[6:8]))

            # Ship_Time_num[num]=time.mktime(temp.timetuple())
            # temp = datetime.datetime.strptime(temp, '%Y-%m-%d %H:%M:%S.%f')
            # print(temp)
            Ship_Time_num[num] = time.mktime(temp.timetuple())
            # print(time.mktime(temp1.timetuple()))
            
        except:
            continue

    # Join StaticData: MMSI and Dimension
    CSVexport = np.zeros((len(Ship_DimA), 5))
    for num in range(len(Ship_DimA)):

        try:
            # CSVtemp=CSVlines[num]
            CSVexport[num, 0] = pd.to_numeric(Ship_ID[num])
            CSVexport[num, 1] = pd.to_numeric(Ship_DimA[num])
            CSVexport[num, 2] = pd.to_numeric(Ship_DimB[num])
            CSVexport[num, 3] = pd.to_numeric(Ship_DimC[num])
            CSVexport[num, 4] = pd.to_numeric(Ship_DimD[num])
        except:
            continue

    return Ship_ID, Ship_Time_num, Ship_Lon, Ship_Lat, Ship_SOG, Ship_COG, Ship_Name, Ship_Type, CSVexport


# Sentinel-1 xml reader
# Required information: SAR start/endtime, 
def metadataReadL1D(SARmetaname):
    import xml.etree.ElementTree as ET
    import datetime
    import time

    tree = ET.parse(SARmetaname)
    root = tree.getroot()
    
    # Starttime/Endtime from metadata
    startdatetime=root[0][5].text
    startdatetime = datetime.datetime(int(startdatetime[0:4]), int(startdatetime[5:7]), int(startdatetime[8:10]),
                             int(startdatetime[11:13]),int(startdatetime[14:16]), int(startdatetime[17:19]))
    startdatetime=time.mktime(startdatetime.timetuple())
    
    enddatetime=root[0][6].text
    enddatetime = datetime.datetime(int(enddatetime[0:4]), int(enddatetime[5:7]), int(enddatetime[8:10]),
                             int(enddatetime[11:13]),int(enddatetime[14:16]), int(enddatetime[17:19]))
    enddatetime=time.mktime(enddatetime.timetuple())

    # Ascending/Descending
    AscDesc=root[2][0][0].text
    if AscDesc=='Ascending':
        AscDesc=1
    else:
        AscDesc=0
        
    # SlantRange, IncidencAngle
    SlantRng=[788000, 936000]  
    IncidenceAngle=[30, 45]     
    
    return startdatetime, enddatetime, AscDesc, SlantRng, IncidenceAngle
   

# Start and End time extraction of Sentinel-1
def SARStartEndTimeS1(SARtifname): 
    import datetime
    import time
    
    startdate=SARtifname[17:25]; starttime=SARtifname[26:32]
    enddate=SARtifname[33:41];   endtime=SARtifname[42:48]
    
    
    startdatetime = datetime.datetime(int(startdate[0:4]), int(startdate[4:6]), int(startdate[6:]), int(starttime[0:2]),
                              int(starttime[2:4]), int(starttime[4:]))
    startdatetime = time.mktime(startdatetime.timetuple())
    
    enddatetime = datetime.datetime(int(enddate[0:4]), int(enddate[4:6]), int(enddate[6:]), int(endtime[0:2]),
                              int(endtime[2:4]), int(endtime[4:]))
    enddatetime = time.mktime(enddatetime.timetuple())
    
    return startdatetime, enddatetime
    
    
# Transform geographic coordinate into intrinsic coordinate
def geographicToIntrinsic(tif_ref, lat, lon):
    import numpy as np
    from scipy.interpolate import interp1d

    max_lat = tif_ref[3]
    min_lat = tif_ref[4]
    max_lon = tif_ref[2]
    min_lon = tif_ref[0]
    space_lat = tif_ref[5]
    space_lon = tif_ref[1]

    num_lat = round(((max_lat - space_lat) - min_lat) / (-space_lat))
    num_lon = round(((max_lon + space_lon) - min_lon) / space_lon)

    lat_array = np.linspace(max_lat, min_lat, num_lat)
    lat_order = np.linspace(1, len(lat_array), len(lat_array))
    lon_array = np.linspace(min_lon, max_lon, num_lon)
    lon_order = np.linspace(1, len(lon_array), len(lon_array))

    lat_order = lat_order.astype(int)
    lon_order = lon_order.astype(int)

    try:
        lat_y = interp1d(lat_array, lat_order)
        y = lat_y(lat)
    except:
        lat_y = interp1d(lat_array, lat_order, fill_value='extrapolate')
        y = lat_y(lat)

    try:
        lon_x = interp1d(lon_array, lon_order)
        x = lon_x(lon)
    except:
        lon_x = interp1d(lon_array, lon_order, fill_value='extrapolate')
        x = lon_x(lon)

    return y, x    