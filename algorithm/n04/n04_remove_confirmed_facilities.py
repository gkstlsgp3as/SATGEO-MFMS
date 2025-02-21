import psycopg2
import pandas as pd
import datetime as dt
import time
import shapefile
from utils.cfg import Cfg
from typing import Tuple, List, Optional
import numpy as np
from shapely.geometry import shape, Point
from pyproj import Transformer
from utils.utils import *
import argparse
import logging

def remove_list_dist(
    input_facilities_whole_file: str,
    input_ais_file: str,
    input_vessel_detection_file: str,
    input_facilities_confirmed_file: str,
    input_zone_path: str,
    distance_threshold: int = 200    
) -> pd.DataFrame:
    """
    This function removes identified targets that are within a specified distance threshold
    from detected targets, returning a DataFrame of unidentified facilities.

    Parameters:
    - input_facilities_whole_file: Path to the CSV file containing data on all facilities.
    - input_ais_file: Path to the CSV file containing AIS data.
    - input_vessel_detection_file: Path to the CSV file containing detected vessel data.
    - input_facilities_confirmed_file: Path to the CSV file containing confirmed facilities data.
    - input_zone_path: Path to the shapefile defining marine zones.
    - distance_threshold: The distance threshold for identifying nearby facilities (in meters).

    Returns:
    - A DataFrame containing the unidentified facilities after removing those close to detected targets.
    """
    start = time.time()

    # Load datasets from specified files
    df_db = pd.read_csv(input_facilities_confirmed_file)
    df_ais = pd.read_csv(input_ais_file)
    df_ship = pd.read_csv(input_vessel_detection_file)

    # Prepare facility data by calculating midpoint coordinates
    lon_min_db = df_db['LonMin']
    lat_min_db = df_db['LatMin']
    lon_max_db = df_db['LonMax']
    lat_max_db = df_db['LatMax']

    lon_ais = df_ais['Lon']
    lat_ais = df_ais['Lat']
    date_acq = df_ais['Date']
    date_acq = date_acq.iloc[0]

    lon_db = 0.5 * (lon_min_db + lon_max_db)
    lat_db = 0.5 * (lat_min_db + lat_max_db)

    lon_ship = df_ship['Lon']
    lat_ship = df_ship['Lat']

    # Combine and reset index for facility, ship, and AIS coordinates
    lon_remove = pd.concat([lon_db, lon_ship, lon_ais], axis=0).reset_index(drop=True)
    lat_remove = pd.concat([lat_db, lat_ship, lat_ais], axis=0).reset_index(drop=True)

    df_facil = pd.read_csv(input_facilities_whole_file)
    lon_min_facil = df_facil['LonMin']
    lat_min_facil = df_facil['LatMin']
    lon_max_facil = df_facil['LonMax']
    lat_max_facil = df_facil['LatMax']

    lon_facil = 0.5 * (lon_min_facil + lon_max_facil)
    lat_facil = 0.5 * (lat_min_facil + lat_max_facil)

    # Calculate dimensions of facilities for area calculations
    length_facil = abs(lon_min_facil - lon_max_facil)
    width_facil = abs(lat_min_facil - lat_max_facil)

    # Initialize array to track identified numbers and distances
    sar_iden_num = (-1) * np.ones((len(lon_facil), 2))
    uniden_fac = pd.DataFrame(columns=['Lon', 'Lat', 'Length', 'Width', 'Date'])

    # Loop over each facility and calculate distances to remove points
    for num in range(len(lon_facil)):
        sar_facil_temp = np.array([lon_facil[num], lat_facil[num]])
        length_facil_temp = degree_to_meters(length_facil[num])
        width_facil_temp = degree_to_meters(width_facil[num])

        for num0 in range(len(lon_remove)):
            remove_temp = np.array([lon_remove[num0], lat_remove[num0]])
            sar_dist_temp = 1000 * deg2km(sar_facil_temp[1], sar_facil_temp[0], remove_temp[1], remove_temp[0])

            if sar_iden_num[num, 0] == -1 and sar_dist_temp < distance_threshold:
                sar_iden_num[num, 0] = num0
                sar_iden_num[num, 1] = sar_dist_temp
            elif sar_iden_num[num, 0] > -1 and sar_dist_temp < distance_threshold and sar_iden_num[num, 1] > sar_dist_temp:
                sar_iden_num[num, 0] = num0
                sar_iden_num[num, 1] = sar_dist_temp

        # Add facility to unidentified list if it is not within the distance threshold
        if sar_iden_num[num, 0] < 0:
            uniden_vessels_append = {
                'Lon': sar_facil_temp[0], 'Lat': sar_facil_temp[1],
                'Length': length_facil_temp, 'Width': width_facil_temp, 'Date': date_acq
            }
            uniden_fac = uniden_fac.append(uniden_vessels_append, ignore_index=True)

    # Read and process the shapefile to assign region numbers
    shapefile_path = input_zone_path
    sf = shapefile.Reader(shapefile_path)
    fields = sf.fields[1:]
    field_names = [field[0] for field in fields]
    records = sf.records()
    shapes = sf.shapes()

    facil_lon = np.array(uniden_fac['Lon'])
    facil_lat = np.array(uniden_fac['Lat'])
    unidentified_facilities_num = np.zeros((len(uniden_fac), 1))

    # Assign region numbers based on facility locations and shapefile zones
    for num in range(len(uniden_fac)):
        lon = facil_lon[num]
        lat = facil_lat[num]
        transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
        lon, lat = transformer.transform(lon, lat)
        point = Point(lon, lat)

        region_number = None
        for record, shp in zip(records, shapes):
            polygon = shape(shp.__geo_interface__)
            if polygon.contains(point):
                region_number = record[field_names.index('HAEGU_NO')]
                break

        unidentified_facilities_num[num] = region_number if region_number else -1

    uniden_fac_num_df = pd.DataFrame(unidentified_facilities_num, columns=['RegionNum'])
    unidentified_facilities_exp = pd.concat([uniden_fac, uniden_fac_num_df], axis=1)

    print('MFMS facility matching finished in:', time.time() - start, '[s]')
    unidentified_facilities_exp.to_csv('unidentifiedMFMS.csv', index=True)

    return unidentified_facilities_exp

def remove_confirmed_facilities(
    unidentified_facilities: pd.DataFrame,
    input_zone_file: str
) -> pd.DataFrame:
    """
    This function enriches unidentified facility data with region numbers based on their geographical
    coordinates and a provided shapefile, and identifies which of these facilities have been confirmed.

    Parameters:
    - unidentified_facilities: DataFrame containing facilities that need region assignment.
    - input_zone_file: Path to the shapefile used to determine region numbers.

    Returns:
    - DataFrame with added region numbers and an indication if they are confirmed.
    """
    start = time.time()

    shapefile_path = input_zone_file
    sf = shapefile.Reader(shapefile_path)
    fields = sf.fields[1:]
    field_names = [field[0] for field in fields]
    records = sf.records()
    shapes = sf.shapes()

    facil_lon = np.array(unidentified_facilities['Lon'])
    facil_lat = np.array(unidentified_facilities['Lat'])
    unidentified_facilities_num = np.zeros((len(unidentified_facilities), 1))

    # Assign a region number based on the geographic location of each facility
    for num, (lon, lat) in enumerate(zip(facil_lon, facil_lat)):
        transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
        lon_transformed, lat_transformed = transformer.transform(lon, lat)
        point = Point(lon_transformed, lat_transformed)

        region_number = None
        for record, shape in zip(records, shapes):
            if shape.contains(point):
                region_number = record[field_names.index('HAEGU_NO')]
                break

        unidentified_facilities_num[num] = region_number if region_number else -1

    unidentified_facilities_num_df = pd.DataFrame(unidentified_facilities_num, columns=['RegionNum'])
    unidentified_facilities_exp = pd.concat([unidentified_facilities, unidentified_facilities_num_df], axis=1)

    print('MFMS facility matching finished in:', time.time() - start, '[s]')
    unidentified_facilities_exp.to_csv('unidentifiedMFMS.csv', index=True)

    return unidentified_facilities_exp


def process(
    input_facilities_whole_file: str, 
    input_ais_file: str, 
    input_vessel_detection_file: str, 
    input_facilities_confirmed_file: str,
    distance_threshold: int, 
    input_zone_file: str
):  
    """
    This function orchestrates the processing of facility data by first identifying and removing 
    facilities close to known points and then appending regional data based on a shapefile.

    Parameters:
    - input_facilities_whole_file: Path to the file with data on all facilities.
    - input_ais_file: Path to the AIS data file.
    - input_vessel_detection_file: Path to the file with detected vessel data.
    - input_facilities_confirmed_file: Path to the file with confirmed facilities data.
    - distance_threshold: Threshold in meters for considering facilities as close.
    - input_zone_file: Path to the shapefile for regional data mapping.
    """
    # Identify and remove close facilities
    unidentified_facilities = remove_list_dist(
        input_facilities_whole_file,
        input_ais_file,
        input_vessel_detection_file,
        input_facilities_confirmed_file,
        input_zone_file,
        distance_threshold
    )
    
    # Process the results to remove confirmed facilities and append regional data
    remove_confirmed_facilities(unidentified_facilities, input_zone_file)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_grd_file",
        type=str,
        default="W:/ship_ais/MFMS/remove_known_facilities/input/S1A_IW_GRDH_1SDV_20231209T092333_20231209T092402_051576_0639F8_652A.tif",
        required=False,
        help="path to input grd image file",
    )
    
    parser.add_argument(
        "--input_meta_file",
        type=str,
        default="W:/ship_ais/MFMS/remove_known_facilities/input/s1a-iw-grd-vv-20231209t092333-20231209t092402-051576-0639f8-001.xml",
        required=False,
        help="path to satellite meta file",
    )
    
    parser.add_argument(
        "--input_facilities_whole_file",
        type=str,
        default="~",
        required=False,
        help="path to file of whole facilities",
    )
    
    parser.add_argument(
        "--input_facilities_confirmed_file",
        type=str,
        default="~",
        required=False,
        help="path to file of confiremd facilities",
    )
    
    parser.add_argument(
        "--input_ais_file",
        type=str,
        default="~",
        required=False,
        help="path to ais file"
    )
    
    parser.add_argument(
        "--input_vessel_detection_file",
        type=str,
        default="~",
        required=False,
        help="path to vessel detection file"
    )
    
    parser.add_argument(
        "--input_zone_file",
        type=str,
        default="~",
        required=False,
        help="path to marine zone file"
    )
    
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    start_time = time.time()

    args = get_args()
    
    process(args.input_facilities_whole_file, args.input_ais_file, args.input_vessel_detection_file, 
            args.input_facilities_confirmed_file, args.input_zone_file, Cfg.distance_threshold)

    processed_time = time.time() - start_time
    logging.info(f"{processed_time:.2f} seconds")


'''
if __name__ == '__main__':
    unidenFac = RemoveListDist(
        Cfg.SARFacilityname,
        Cfg.AISprocname,
        Cfg.VesslDetname,
        Cfg.DBlistname,
        idenDistance=200
    )
    unidenFacExp = addRegionNumber(unidenFac)
'''


