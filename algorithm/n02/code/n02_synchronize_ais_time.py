import psycopg2
import json
import pandas as pd
import sys
import datetime as dt
from typing import Any, Dict, List, Tuple, Optional
from utils.cfg import Cfg
import time
import logging

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

def select_data(config: Dict[str, Any], argv: List[str]) -> pd.DataFrame:
    """
    Fetch data from a PostgreSQL database within the specified time range.
    """
    conn = psycopg2.connect(
        database=config["database"],
        user=config["user"],
        password=config["password"],
        host=config["host"],
        port=config["port"]
    )
    conn.set_session(autocommit=True)
    cur = conn.cursor()

    if '.' in argv[2]:  # 마이크로초가 포함
        start_datetime: dt.datetime = dt.datetime.strptime(argv[1] + " " + argv[2], "%Y-%m-%d %H:%M:%S.%f")
    else:
        start_datetime = dt.datetime.strptime(argv[1] + " " + argv[2], "%Y-%m-%d %H:%M:%S")

    if '.' in argv[4]:
        end_datetime: dt.datetime = dt.datetime.strptime(argv[3] + " " + argv[4], "%Y-%m-%d %H:%M:%S.%f")
    else:
        end_datetime = dt.datetime.strptime(argv[3] + " " + argv[4], "%Y-%m-%d %H:%M:%S")

    database_name: str = argv[5]

    result: pd.DataFrame = pd.read_sql(f"""
        SELECT * FROM {database_name}
        WHERE (date::text || ' ' || time::text)::timestamp
        BETWEEN '{start_datetime}' AND '{end_datetime}'
    """, conn)

    cur.close()
    conn.close()
    return result


def metadata_read_l1d(input_meta_file: str) -> Tuple[float, float, int, List[int], List[int]]:
    """
    Read metadata from a Sentinel-1 XML file.
    """
    import xml.etree.ElementTree as ET
    import time
    import datetime

    tree = ET.parse(input_meta_file)
    root = tree.getroot()

    # Parse start time
    start_info = root[0][5].text
    start_dt = datetime.datetime(
        int(start_info[0:4]), int(start_info[5:7]), int(start_info[8:10]),
        int(start_info[11:13]), int(start_info[14:16]), int(start_info[17:19])
    )
    start_time = time.mktime(start_dt.timetuple())

    # Parse end time
    end_info = root[0][6].text
    end_dt = datetime.datetime(
        int(end_info[0:4]), int(end_info[5:7]), int(end_info[8:10]),
        int(end_info[11:13]), int(end_info[14:16]), int(end_info[17:19])
    )
    end_time = time.mktime(end_dt.timetuple())

    asc_desc_text = root[2][0][0].text
    asc_desc = 1 if asc_desc_text == "Ascending" else 0

    slant_range = [788000, 936000]
    incidence_angle = [30, 45]

    return start_time, end_time, asc_desc, slant_range, incidence_angle


def geotiff_read_ref(input_grd_file: str) -> Tuple[Any, int, int]:
    """
    Read geo-reference information from a TIFF file.
    """
    from osgeo import gdal
    import numpy as np

    gdal.AllRegister()
    ds = gdal.Open(input_grd_file)
    tif_ref = ds.GetGeoTransform()
    rows, cols = ds.RasterXSize, ds.RasterYSize

    tif_ref = np.array(tif_ref)
    tif_ref[2] = tif_ref[0] + tif_ref[1] * (rows - 1)
    tif_ref[4] = tif_ref[3] + tif_ref[5] * (cols - 1)
    tif_ref.astype(np.double)

    return tif_ref, rows, cols


def read_ais(input_ais_file: str):
    """
    Read AIS data from a CSV file.
    """
    import pandas as pd
    import datetime
    import numpy as np
    import time

    try:
        df = pd.read_csv(input_ais_file)
    except:
        df = input_ais_file

    ship_id = df['MMSI']
    ship_date = df['Date']
    ship_time = df['Time']
    ship_lon = df['Lon']
    ship_lat = df['Lat']
    ship_sog = df['SOG']
    ship_cog = df['COG']

    ship_name = df['VesselName']
    ship_type = df['VesselType']
    ship_dim_a = df['DimA']
    ship_dim_b = df['DimB']
    ship_dim_c = df['DimC']
    ship_dim_d = df['DimD']
    ship_status = df['Status']
    ship_destination = df['ToPortof']

    ship_name = np.array(ship_name)
    ship_type = np.array(ship_type)
    ship_status = np.array(ship_status)
    ship_destination = np.array(ship_destination)

    ship_id = np.array(ship_id)
    ship_lon = np.array(ship_lon)
    ship_lat = np.array(ship_lat)
    ship_sog = np.array(ship_sog)
    ship_cog = np.array(ship_cog)
    ship_dim_a = np.array(ship_dim_a)
    ship_dim_b = np.array(ship_dim_b)
    ship_dim_c = np.array(ship_dim_c)
    ship_dim_d = np.array(ship_dim_d)

    ship_time = np.array(df['Time'])
    ship_date = np.array(df['Date'])
    ship_time_num = np.zeros(len(ship_time))

    for num in range(len(ship_time_num)):
        temp1 = ship_date[num]
        temp2 = ship_time[num]
        try:
            temp_dt = datetime.datetime(
                int(temp1[0:4]), int(temp1[5:7]), int(temp1[8:10]),
                int(temp2[0:2]), int(temp2[3:5]), int(temp2[6:8])
            )
            ship_time_num[num] = time.mktime(temp_dt.timetuple())
        except:
            continue

    csv_export = np.zeros((len(ship_dim_a), 5))
    for num in range(len(ship_dim_a)):
        try:
            csv_export[num, 0] = ship_id[num]
            csv_export[num, 1] = ship_dim_a[num]
            csv_export[num, 2] = ship_dim_b[num]
            csv_export[num, 3] = ship_dim_c[num]
            csv_export[num, 4] = ship_dim_d[num]
        except:
            continue

    return (
        ship_id,
        ship_time_num,
        ship_lon,
        ship_lat,
        ship_sog,
        ship_cog,
        ship_name,
        ship_type,
        ship_status,
        ship_destination,
        csv_export
    )


def geographic_to_intrinsic(tif_ref: Any, lat: Any, lon: Any) -> Tuple[Any, Any]:
    """
    Convert geographic latitude and longitude to image intrinsic coordinates.
    """
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
    lat_order = np.linspace(1, len(lat_array), len(lat_array)).astype(int)
    lon_array = np.linspace(min_lon, max_lon, num_lon)
    lon_order = np.linspace(1, len(lon_array), len(lon_array)).astype(int)

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


def deg_to_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Compute distance in kilometers between two latitude/longitude points
    using the Haversine formula.
    """
    import math

    radius_earth = 6371.0
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)

    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad
    a = (math.sin(dlat / 2) ** 2 + math.cos(lat1_rad) * 
         math.cos(lat2_rad) * math.sin(dlon / 2) ** 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance = radius_earth * c
    return distance


def sar_ais_match_time_interp(input_ais_file: str, input_grd_file: str, input_meta_file: str, output_dir: str) -> pd.DataFrame:
    """
    Perform time interpolation for SAR-AIS matching.
    """
    import numpy as np
    import pandas as pd
    import time

    start_time = time.time()

    ship_id, ship_time_num, ship_lon, ship_lat, ship_sog, ship_cog, ship_name, ship_type, ship_status, ship_destination, csv_export = read_ais(input_ais_file)

    unique_array, unique_loc = np.unique(csv_export[:, 0], return_index=True)
    csv_export = csv_export[unique_loc, :]

    tif_ref, rows, cols = geotiff_read_ref(input_grd_file)
    start_datetime, end_datetime, asc_desc, slant_rng, incidence_angle = metadata_read_l1d(input_meta_file)

    # Remove lat/lon out of bounds
    for num in range(len(ship_lat)):
        if (ship_lat[num] > tif_ref[3] or ship_lat[num] < tif_ref[4]) or (ship_lon[num] > tif_ref[2] or ship_lon[num] < tif_ref[0]):
            ship_lat[num] = 0
            ship_lon[num] = 0

    # Time sorting within ±20 minutes
    for num in range(len(ship_time_num)):
        temp_time = ship_time_num[num]
        if temp_time > end_datetime + 600 or temp_time < start_datetime - 600:
            ship_lat[num] = 0
            ship_lon[num] = 0

    # Retain valid entries
    det = np.multiply(ship_lat, ship_lon)
    det_loc = np.argwhere(det != 0)
    det_loc = np.squeeze(det_loc)

    ship_id = ship_id[det_loc]
    ship_cog = ship_cog[det_loc]
    ship_sog = ship_sog[det_loc]
    ship_lat = ship_lat[det_loc]
    ship_lon = ship_lon[det_loc]
    ship_time_num = ship_time_num[det_loc]

    # Convert knots to m/s
    ship_sog = 0.5144 * ship_sog

    # Set up time array
    ship_id_unique = np.unique(ship_id)
    if asc_desc > 0:
        time_array = np.linspace(end_datetime, start_datetime, cols)
    else:
        time_array = np.linspace(start_datetime, end_datetime, cols)

    # [II] Time Array Interpolation
    interpolated_output = np.zeros((8, len(ship_id_unique)))

    for num1 in range(len(ship_id_unique)):
        ship_id_temp = ship_id_unique[num1]
        loc_temp = np.argwhere(ship_id == ship_id_temp)
        loc_temp = loc_temp[:, 0]

        ship_cog_temp = ship_cog[loc_temp]
        ship_sog_temp = ship_sog[loc_temp]
        ship_lat_temp = ship_lat[loc_temp]
        ship_lon_temp = ship_lon[loc_temp]
        ship_time_num_temp = ship_time_num[loc_temp]

        try:
            # Single AIS point
            if len(loc_temp) < 2:
                ship_y_temp, ship_x_temp = geographic_to_intrinsic(tif_ref, float(ship_lat_temp), float(ship_lon_temp))
                sar_time = time_array[int(ship_y_temp)]
                ship_time = float(ship_time_num_temp)
                diff_sar_ship_time = sar_time - ship_time

                vlat = float(ship_sog_temp) * np.cos(float(ship_cog_temp) * np.pi / 180) / 111000
                vlon = float(ship_sog_temp) * np.sin(float(ship_cog_temp) * np.pi / 180) / 111000

                interpolated_output[0, num1] = ship_id_temp
                interpolated_output[1, num1] = float(ship_lat_temp) + diff_sar_ship_time * vlat
                interpolated_output[2, num1] = float(ship_lon_temp) + diff_sar_ship_time * vlon
                interpolated_output[3, num1] = float(ship_cog_temp)
                interpolated_output[4, num1] = float(ship_sog_temp)
                interpolated_output[7, num1] = sar_time

            # Multiple AIS points
            else:
                loc1 = int(np.round(len(loc_temp) / 2))
                ship_cog_val = ship_cog_temp[loc1]
                ship_sog_val = ship_sog_temp[loc1]
                ship_lat_val = ship_lat_temp[loc1]
                ship_lon_val = ship_lon_temp[loc1]
                ship_time = float(ship_time_num_temp[loc1])

                ship_y_temp, ship_x_temp = geographic_to_intrinsic(tif_ref, float(ship_lat_val), float(ship_lon_val))
                sar_time = time_array[int(ship_y_temp)]
                diff_sar_ship_time = sar_time - ship_time

                vlat = float(ship_sog_val) * np.cos(float(ship_cog_val) * np.pi / 180) / 111000
                vlon = float(ship_sog_val) * np.sin(float(ship_cog_val) * np.pi / 180) / 111000

                interpolated_output[0, num1] = ship_id_temp
                interpolated_output[1, num1] = float(ship_lat_val) + diff_sar_ship_time * vlat
                interpolated_output[2, num1] = float(ship_lon_val) + diff_sar_ship_time * vlon
                interpolated_output[3, num1] = float(ship_cog_val)
                interpolated_output[4, num1] = float(ship_sog_val)
                interpolated_output[7, num1] = sar_time
        except:
            interpolated_output[0, num1] = 0
            interpolated_output[1, num1] = 0
            interpolated_output[2, num1] = 0
            interpolated_output[3, num1] = 0
            interpolated_output[4, num1] = 0
            interpolated_output[7, num1] = 0

    # Convert epoch times into formatted date/time
    date_output: List[str] = ["" for _ in range(len(ship_id_unique))]
    time_output: List[str] = ["" for _ in range(len(ship_id_unique))]
    import time

    for num1 in range(len(ship_id_unique)):
        time_utc = time.localtime(interpolated_output[7, num1])

        # Construct date string
        if time_utc[1] < 10:
            if time_utc[2] < 10:
                date_output[num1] = f"{time_utc[0]}-0{time_utc[1]}-0{time_utc[2]}"
            else:
                date_output[num1] = f"{time_utc[0]}-0{time_utc[1]}-{time_utc[2]}"
        else:
            if time_utc[2] < 10:
                date_output[num1] = f"{time_utc[0]}-{time_utc[1]}-0{time_utc[2]}"
            else:
                date_output[num1] = f"{time_utc[0]}-{time_utc[1]}-{time_utc[2]}"

        # Construct time string
        if time_utc[3] < 10:
            if time_utc[4] < 10:
                if time_utc[5] < 10:
                    time_output[num1] = f"0{time_utc[3]}:0{time_utc[4]}:0{time_utc[5]}"
                else:
                    time_output[num1] = f"0{time_utc[3]}:0{time_utc[4]}:{time_utc[5]}"
            else:
                if time_utc[5] < 10:
                    time_output[num1] = f"0{time_utc[3]}:{time_utc[4]}:0{time_utc[5]}"
                else:
                    time_output[num1] = f"0{time_utc[3]}:{time_utc[4]}:{time_utc[5]}"
        else:
            if time_utc[4] < 10:
                if time_utc[5] < 10:
                    time_output[num1] = f"{time_utc[3]}:0{time_utc[4]}:0{time_utc[5]}"
                else:
                    time_output[num1] = f"{time_utc[3]}:0{time_utc[4]}:{time_utc[5]}"
            else:
                if time_utc[5] < 10:
                    time_output[num1] = f"{time_utc[3]}:{time_utc[4]}:0{time_utc[5]}"
                else:
                    time_output[num1] = f"{time_utc[3]}:{time_utc[4]}:{time_utc[5]}"

    # Retrieve static info (dimensions)
    dim_output = np.zeros((4, len(ship_id_unique)))
    for num1 in range(len(ship_id_unique)):
        ship_id_temp = interpolated_output[0, num1]
        det_loc = int(np.argwhere(csv_export[:, 0] == ship_id_temp))
        dim_output[:, num1] = csv_export[det_loc, 1:]

    # Retrieve vessel name/type/destination/detloc/detlat
    ship_name_exp: List[Any] = []
    ship_type_exp: List[Any] = []
    ship_destination_exp: List[Any] = []
    
    for num1 in range(len(ship_id_unique)):
        ship_id_temp = interpolated_output[0, num1]
        loc_array = np.argwhere(ship_id == ship_id_temp)
        det_loc_int = int(loc_array[0])
        ship_name_exp.append(ship_name[det_loc_int])
        ship_type_exp.append(ship_type[det_loc_int])
        ship_destination_exp.append(ship_destination[det_loc_int])

    locexp = np.where(interpolated_output[0, :] != 0)
    interpolated_output = np.squeeze(interpolated_output[:, locexp])
    dim_output = np.squeeze(dim_output[:, locexp])

    locexp = locexp[0]
    ship_name_exp = [ship_name_exp[i] for i in locexp]
    ship_type_exp = [ship_type_exp[i] for i in locexp]
    ship_destination_exp = [ship_destination_exp[i] for i in locexp]
    date_output = [date_output[i] for i in locexp]
    time_output = [time_output[i] for i in locexp]

    df = pd.DataFrame(columns=[
        'Date', 'Time', 'MMSI', 'Lon', 'Lat',
        'COG', 'SOG', 'VesselName', 'VesselType',
        'DimA', 'DimB', 'DimC', 'DimD', 'Destination'
    ])

    df['Lat'] = interpolated_output[1, :].T
    df['Lon'] = interpolated_output[2, :].T
    df['COG'] = interpolated_output[3, :].T
    df['SOG'] = interpolated_output[4, :].T
    df['MMSI'] = interpolated_output[0, :].T
    df['DimA'] = dim_output[0, :].T
    df['DimB'] = dim_output[1, :].T
    df['DimC'] = dim_output[2, :].T
    df['DimD'] = dim_output[3, :].T
    df['VesselName'] = ship_name_exp
    df['VesselType'] = ship_type_exp
    df['Destination'] = ship_destination_exp
    df['Date'] = date_output
    df['Time'] = time_output

    df.to_csv(output_dir + 'AIStimeInterpolated.csv', index=True)
    print('SAR-AIS Time interpolation Finished in: ', time.time() - start_time, ' [s]')
    
    return df


def sar_azimuth_offset_corr(df: pd.DataFrame, input_grd_file: str, input_meta_file: str, output_dir: str, view_left_right: int = 1) -> pd.DataFrame:
    """
    Correct SAR-AIS azimuth offset based on input parameters.
    """
    import numpy as np
    import time

    start_time = time.time()

    tif_ref, rows, cols = geotiff_read_ref(input_grd_file)
    va = 7600  # Typical azimuth velocity for SAR in m/s

    start_time, end_time, asc_desc, slant_rng, incidence_angle = metadata_read_l1d(input_meta_file)
    azimuth_heading = 349 if asc_desc == 1 else 191

    slant_range = np.linspace(int(slant_rng[0]), int(slant_rng[1]), rows) if asc_desc > 0 else np.linspace(int(slant_rng[1]), int(slant_rng[0]), rows)
    inc_angle = np.linspace(int(incidence_angle[0]), int(incidence_angle[1]), rows) if asc_desc > 0 else np.linspace(int(incidence_angle[1]), int(incidence_angle[0]), rows)

    for index, row in df.iterrows():
        vy = row['SOG'] * np.sin((row['COG'] - azimuth_heading) * np.pi / 180)
        ship_y_temp, ship_x_temp = geographic_to_intrinsic(tif_ref, row['Lat'], row['Lon'])
        r0 = slant_range[int(ship_x_temp)] * np.cos(inc_angle[int(ship_x_temp)] * np.pi / 180)
        az_off = vy * r0 / va
        az_off_lat = az_off * np.cos(azimuth_heading * np.pi / 180) / 111000
        az_off_lon = az_off * np.sin(azimuth_heading * np.pi / 180) / 111000

        df.at[index, 'Lat'] += az_off_lat if view_left_right > 0 else -az_off_lat
        df.at[index, 'Lon'] += az_off_lon if view_left_right > 0 else -az_off_lon

    print('SAR-AIS Azimuth Offset Correction Finished in:', time.time() - start_time, 'seconds')
    df.to_csv(output_dir + 'AISmatchedwithSAR.csv', index=False)
    return df


def get_args():
    """
    Parse command-line arguments.
    """
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input_grd_file', 
        type=str, 
        default='../data/input/S1A_IW_GRDH_1SDV_20240829T092329_20240829T092358_055426_06C2AF_BF0D.tif',
        help='SAR image file'
    )
    parser.add_argument(
        '--input_meta_file', 
        type=str, 
        default='../data/input/s1a-iw-grd-vv-20240829t092329-20240829t092358-055426-06c2af-001.xml',
        help='SAR metadata file'
    )
    parser.add_argument(
        '--input_ais_file', 
        type=str, 
        default='../data/input/TAIS_20240829.csv',
        help='AIS data file'
    )
    parser.add_argument(
        '--input_vessel_detection_file', 
        type=str, 
        default='../data/input/S1A_IW_GRDH_1SDV_20240224T093202_20240224T093227_052699_066052_14E8.txt',
        help='Detected Vessel file'
    )
    parser.add_argument(
        '--output_dir', 
        type=str, 
        default='../data/output/',
        help='output directory'
    )
    return parser.parse_args()


if __name__ == '__main__':
    # Record start time for processing
    start_time = time.time()
    
    args = get_args()
    
    output_vessel_iden_file = args.output_dir + args.input_vessel_detection_file.split('/')[-1][:-4] + '_IdenVessel.csv'
    output_vessel_uniden_file = args.output_dir + args.input_vessel_detection_file.split('/')[-1][:-4] + '_UnIdenVessel.csv'
    
    ProcessedDATA1 = sar_ais_match_time_interp(args.input_ais_file, args.input_grd_file, args.input_meta_file, args.output_dir)
    ProcessedDATA2 = sar_azimuth_offset_corr(ProcessedDATA1, args.input_grd_file, args.input_meta_file, args.output_dir)
    
    # Calculate and log the processing time
    processed_time = time.time() - start_time
    logging.info(f"Processed SAR image classification in {processed_time:.2f} seconds")