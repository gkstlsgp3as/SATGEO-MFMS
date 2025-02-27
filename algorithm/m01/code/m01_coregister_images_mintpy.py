# -*- coding: utf-8 -*-
'''
@Time          : 2025/01/06 15:17
@Author        : Satgeo lab.
@File          : main.py
@Noice         : 
@Description   : Perform coregister_images_mintpy.

@Modificattion :
    @Author    :
    @Time      :
    @Detail    :
'''

import os,glob
import argparse
import numpy as np
from datetime import datetime, timedelta
import time
from subprocess import check_output      
from typing import List, Tuple, Optional
import re
import zipfile
import itertools
import requests
import logging
import shutil
import csv

def set_bbox(bbox: str) -> Tuple[str, float, float, float, float]:
    """
    Define the bounding box from the extracted table.

    Args:
        bbox (str): A string representation of the bounding box, 
                    with values separated by spaces (e.g., "ymin ymax xmin xmax").

    Returns:
        Tuple[str, float, float, float, float]: 
            A tuple containing the original bbox string and the parsed bounding box values 
            (ymin, ymax, xmin, xmax).

    Raises:
        ValueError: If the bounding box contains NaN values or is incorrectly specified.
    """
    # Split the bounding box string into individual values
    bbox_split = bbox.split(' ')
    try:
        ymin, ymax, xmin, xmax = map(float, bbox_split)
    except ValueError:
        raise ValueError("ERROR: Incorrect bbox specified, please ensure it contains four numeric values.")

    # Check for NaN values
    if np.isnan([ymin, ymax, xmin, xmax]).sum() > 0:
        raise ValueError("ERROR: Bounding box contains NaN values, please check the input.")

    print(f"Bounding box: {bbox}")
    return bbox, ymin, ymax, xmin, xmax

def select_reference_img(input_dir: str, output_dir: str) -> Optional[str]:
    """
    Selects the reference image from a directory of Sentinel-1 SLC files.

    The reference image is chosen as the middle file in the sorted list of SLC files.
    The date of the reference image is extracted from its filename.

    Args:
        input_dir (str): The directory containing Sentinel-1 SLC files.

    Returns:
        Optional[str]: The date of the selected reference image in 'YYYYMMDD' format,
                       or `None` if no SLC files are found in the directory.
    """
    # Get a sorted list of Sentinel-1 SLC files
    slc_list = sorted(glob.glob(f'{input_dir}/S1*'))
    
    for i in range(len(slc_list)):
        slc_file_list = os.path.basename(slc_list[i])
        # Open the CSV file in append mode
        with open(f"{output_dir}/SLC_files.list", "a") as file:
            file.write(f'{slc_file_list}\n')  # Append a single row
    
    if not slc_list:
        print("INFORM: No SLC files found in the specified directory.")
        return None

    # Select the middle SLC file
    reference = slc_list[round(len(slc_list) / 2) - 1]
    
    # Extract the date from the reference filename
    reference_date = os.path.basename(reference)[17:25]

    print(f"INFORM: Reference was selected as {reference}")
    return reference_date

def import_dem_file(input_dem_dir: str, input_slc_dir: str, work_dir: str) -> str:
    """
    Downloads and prepares the DEM files for the given Sentinel-1 input directory.

    Args:
        input_dem_dir (str): The directory containing DEM(.hgt) files.
        input_slc_dir (str): The directory containing Sentinel-1 SLC files.
        work_dir (str): The working directory where DEM files will be stored.

    Returns:
        str: The path to the prepared DEM file.
    """
    os.chdir(input_slc_dir)
    zipfiles: List[str] = sorted(glob.glob('S1*zip'))
    bboxes = []

    # Extract bounding boxes from Sentinel-1 SLC files
    for slc in zipfiles:
        kml_content = zipfile.ZipFile(slc, 'r').read(f'{slc[:-4]}.SAFE/preview/map-overlay.kml').decode('utf-8')
        coordinates = re.split(r',|\s', kml_content.split('</coordinates>')[0].split('<coordinates>')[1])
        bboxes.append(coordinates)

    bboxes = np.ndarray.flatten(np.array(bboxes))
    latitudes = np.array(bboxes[1::2], dtype='float')
    longitudes = np.array(bboxes[::2], dtype='float')

    # Define bounding box with rounded limits
    bbox1 = np.array([
        np.floor(latitudes.min()),
        np.ceil(latitudes.max()),
        np.floor(longitudes.min()),
        np.ceil(longitudes.max())
    ], dtype='int')

    # Generate DEM grid points
    lat_range = np.arange(bbox1[0], bbox1[1] + 1, 1)
    lon_range = np.arange(bbox1[2], bbox1[3] + 1, 1)
    dem_pairs = list(itertools.product(lat_range, lon_range))
        
    # Change directory to the DEM folder
    os.chdir(input_dem_dir)
    hgt_files = sorted(glob.glob('*.hgt'))
    # print(f"DEM files: {hgt_files}")

    # Prepare the DEM using dem.py
    dem_file = os.path.join(input_dem_dir, 'demfile.wgs84')

    os.system(f'dem.py -a stitch -b {bbox1[0]} {bbox1[1]} {bbox1[2]} {bbox1[3]} -l -c -f -o demfile')
    os.system(f'fixImageXml.py -i {dem_file} -f')

    # # Download SWBD files for watermask preparation
    # for lat, lon in dem_pairs:
    #     hemisphere = 'N' if lat > 0 else 'S'
    #     meridian = 'E' if lon > 0 else 'W'
    #     swbd_file = f"{hemisphere}{abs(lat):02d}{meridian}{abs(lon):03d}.raw"
    #     os.system(f'aws s3 cp s3://satgeo-db/DEM/SWBD_90/{swbd_file} {dem_dir}/')

    return dem_file

'''        
def import_orbit_file(input_slc_dir: str, input_orbit_dir: str, work_dir: str) -> List[str]:
    """
    Downloads orbit files for Sentinel-1 SLC data.

    Args:
        input_slc_dir (str): The directory containing Sentinel-1 SLC files.
        input_orbit_dir (str): The directory containing orbit files.
        work_dir (str): The working directory where orbit files will be stored.

    Returns:
        List[str]: A sorted list of paths to the downloaded orbit files.
    """
    server = 'https://step.esa.int/auxdata/orbits/Sentinel-1/POEORB/'
    server2 = 'https://step.esa.int/auxdata/orbits/Sentinel-1/RESORB/'

    # Get a sorted list of Sentinel-1 SLC files
    slc_list = sorted(glob.glob(f'{input_slc_dir}/S1*'))
    
    for slc in slc_list:
        fil = os.path.basename(slc)
        date = fil[17:25] + fil[26:32]
        year, month = date[:4], date[4:6]
        sensor = fil[:3]

        url = f"{server}/{sensor}/{year}/{month}/"

        # Fetch and parse the POEORB server response
        res = requests.get(url)
        src = res.text
        url_list = src.splitlines()[4:-2]
        
        date_time = datetime.strptime(date, '%Y%m%d%H%M%S')
        startdat = date_time - timedelta(days=1)
        enddat = date_time + timedelta(days=1)

        startdate = startdat.strftime('%Y%m%d')
        enddate = enddat.strftime('%Y%m%d')

        # Search for files matching the date range
        find_data = [s for s in url_list if startdate in s and enddate in s]

        if not find_data:
            # Fall back to RESORB if no POEORB file is found
            url = f"{server2}/{sensor}/{year}/{month}/"

            res = requests.get(url)
            src = res.text
            url_list = src.splitlines()[4:-2]

            startdate = date_time.strftime('%Y%m%d')
            find_data = [s for s in url_list if startdate in s]

            # Find the appropriate orbit file within the time range
            for entry in find_data:
                orbit_data = entry.split('"', 2)[1]
                start_time = datetime.strptime(orbit_data[42:50] + orbit_data[51:57], '%Y%m%d%H%M%S')
                end_time = datetime.strptime(orbit_data[58:66] + orbit_data[67:73], '%Y%m%d%H%M%S')

                if start_time < date_time < end_time:
                    break
            else:
                continue  # Skip to the next SLC if no match is found

        else:
            orbit_data = find_data[0].split('"', 2)[1]

        # Ensure the orbit directory exists
        os.makedirs(input_orbit_dir, exist_ok=True)
        os.chdir(input_orbit_dir)

        # Download and unzip the orbit file if it does not already exist
        orbit_path = os.path.join(input_orbit_dir, orbit_data[:-4])
        if not os.path.isfile(orbit_path):
            os.system(f'wget {url}/{orbit_data}')
            os.system(f'unzip {orbit_data}')
            os.system(f'rm {orbit_data}')

    # List and return the downloaded orbit files
    orb_list = sorted(glob.glob(os.path.join(input_orbit_dir, 'S1*EOF')))
    print("INFORM: Orbit files were downloaded")
    print(orb_list)

    return orb_list
'''

# workflow preparation for isce 
def prepare_isce_settings(output_dir: str, input_slc_dir: str, input_dem_dir: str, input_orbit_dir: str, bbox: str, rglooks: int, azlooks: int, pol: str, n_proc: int, reference: Optional[str]) -> str:
    """
    Prepares the ISCE workflow settings.

    Args:
        work_dir (str): The working directory where ISCE processing will take place.
        input_slc_dir (str): The directory containing input SLC files.
        input_dem_dir (str): The directory containing input DEM files.
        input_orbit_dir (str): The directory containing input orbit files.
        output_dir (str): The directory containing output coregistered files.
        bbox (str): The bounding box coordinates in the format "xmin xmax ymin ymax".
        rglooks (int): Number of range looks.
        azlooks (int): Number of azimuth looks.
        pol (str): Polarization type (e.g., "VV", "VH").
        reference (Optional[str]): The reference Sentinel-1 file for processing. If None, stack mode is used.

    Returns:
        str: The type of coregistration method used ("geometry" or "nesd").
    """
    dem_dirname = input_dem_dir
    orb_dirname = input_orbit_dir
    rg = rglooks
    az = azlooks

    os.chdir(output_dir)

    # Check if run_files directory exists
    run_files_dir = f"{output_dir}/run_files"
    if os.path.exists(run_files_dir):
        print("-----\n------ run_files directory exists -----\n--------")
    else:
        # Construct the stackSentinel.py command based on the presence of a reference
        stack_sentinel_cmd = (
            f"stackSentinel.py -s {input_slc_dir} -o {orb_dirname} -w {output_dir} "
            f"-d {dem_dirname}/demfile.wgs84 -a {output_dir} -b \"{bbox}\" "
            f"-r {rg} -z {az} -e 0.6 -W interferogram --num_proc {n_proc} -p {pol}"
        )
        if reference is not None:
            stack_sentinel_cmd += f" -m {reference}"
        
        print(stack_sentinel_cmd)
        os.system(stack_sentinel_cmd)

    # Execute the first run file
    os.chdir(run_files_dir)
    run_files = sorted(glob.glob("run*"))
    if run_files:
        os.chmod(run_files[0], 0o777)
        os.system(f'bash {run_files[0]}')
    else:
        raise FileNotFoundError(f"No run files found in {run_files_dir}")

    # Determine the coregistration method
    os.chdir(f"{output_dir}/reference")
    iw_list = sorted(glob.glob("IW*.xml"))
    
    if not iw_list:
        raise FileNotFoundError("No IW*.xml files found in the reference directory.")
    iw_file_base = iw_list[0][:-4]
    print(iw_file_base)

    burst_files = sorted(glob.glob(f"{iw_file_base}/burst_*.slc.vrt"))
    if len(burst_files) == 1:
        print("Geometry co-registration")
        coreg = "geometry"
    elif len(burst_files) > 1:
        print("NESD co-registration method")
        coreg = "nesd"
    else:
        raise RuntimeError("No burst SLC VRT files found.")

    return coreg
    
def run_isce(output_dir: str, input_slc_dir: str, input_dem_dir: str, input_orbit_dir: str, bbox: str, rglooks: int, azlooks: int, pol: str, n_proc: int, reference: Optional[str] = None) -> None:
    """
    Executes the ISCE workflow for Sentinel-1 data.

    Args:
        work_dir (str): Working directory for ISCE processing.
        input_slc_dir (str): The directory containing input SLC files.
        input_dem_dir (str): The directory containing input DEM files.
        input_orbit_dir (str): The directory containing input orbit files.
        output_dir (str): The directory containing output coregistered files.
        bbox (str): Bounding box coordinates as "xmin xmax ymin ymax".
        rglooks (int): Number of range looks.
        azlooks (int): Number of azimuth looks.
        pol (str): Polarization type (e.g., "VV", "VH").
        reference (Optional[str]): Reference Sentinel-1 file for processing, if any.

    Returns:
        None
    """
    dem_dirname = input_dem_dir
    orb_dirname = input_orbit_dir
    rg = rglooks
    az = azlooks

    os.chdir(output_dir)
    os.chdir(f"{output_dir}/reference")
    iwlist = sorted(glob.glob("IW*.xml"))[0][:-4]
    print(f"Selected IW file base: {iwlist}")

    burst_files = sorted(glob.glob(f"{iwlist}/burst_*.slc.vrt"))
    if len(burst_files) == 1:
        print("Using geometry co-registration method")
        set_geometry_coreg(input_slc_dir, output_dir, bbox, rg, az, pol, reference, dem_dirname, orb_dirname, iwlist, n_proc)
    elif len(burst_files) > 1:
        print("Using NESD co-registration method")
        set_nesd_coreg(output_dir)
    else:
        print("No burst extracted")

    print("INFORM: Coregistration finished")

def set_geometry_coreg(input_slc_dir: str, output_dir: str, bbox: str, rg: int, az: int, pol: str, reference: Optional[str], input_dem_dir: str, input_orbit_dir: str, iwlist: str, n_proc: int) -> None:
    """Handles geometry-based coregistration."""
    os.system(f"rm -rf {output_dir}/run_files")

    if reference is None:
        os.system(
            f"stackSentinel.py -s {input_slc_dir} -o {input_orbit_dir} -w {output_dir} "
            f"-d {input_dem_dir}/demfile.wgs84 -a {output_dir} -b \"{bbox}\" -r {rg} -z {az} "
            f"-e 0.6 -C geometry -W interferogram --num_proc {n_proc} -p {pol}"
        )
    else:
        os.system(
            f"stackSentinel.py -s {input_slc_dir} -o {input_orbit_dir} -w {output_dir} "
            f"-d {input_dem_dir}/demfile.wgs84 -a {output_dir}  -m {reference} -b \"{bbox}\" "
            f"-r {rg} -z {az} -e 0.6 -C geometry -W interferogram --num_proc {n_proc} -p {pol}"
        )

    run_isce_steps(output_dir)


def set_nesd_coreg(output_dir: str) -> None:
    """Handles NESD-based coregistration."""
    run_isce_steps(output_dir)


def run_isce_steps(output_dir: str) -> None:
    """Executes ISCE processing steps."""
    os.chdir(f"{output_dir}/run_files")
    cmd_list = sorted(glob.glob1(f"{output_dir}/run_files", "run*"))

    for cmd in cmd_list:
        os.chmod(f"{output_dir}/run_files/{cmd}", 0o777)
        os.system(f"./{cmd}")

    # if os.path.exists(f"{output_dir}/merged"):
    #     print("Merged directory exists")
    # else:
    #     print("Continuing ISCE workflow")
    #     for cmd in cmd_list[6:]:
    #         os.system(f"./{cmd}") 

def remove_remnant(output_dir: str) -> None:
    """Removes remnant output directories and files in a safe way."""
    
    # List of directories/files to remove
    paths_to_remove = [
        "ESD",
        "SAFE_files.txt",
        "coarse_interferograms",
        "configs",
        "coreg_secondarys",
        "geom_reference",
        "interferograms",
        "misreg",
        "secondarys",
        "stack"
    ]

    for path in paths_to_remove:
        full_path = os.path.join(output_dir, path)

        if os.path.exists(full_path):
            if os.path.isdir(full_path):
                shutil.rmtree(full_path)  # Remove directories
            else:
                os.remove(full_path)  # Remove files
            print(f"Removed: {full_path}")  # Optional logging
        else:
            print(f"Skipped (not found): {full_path}")  # Optional logging
         
            
def get_args():
    parser = argparse.ArgumentParser(description="m01_coregister_images_stamps")

    parser.add_argument(
        '--input_slc_dir', 
        type=str, 
        required=True,
        default='../data/input_slc/', 
        help='Directory path of input SLC data.'
    )
    
    parser.add_argument(
        '--input_dem_dir', 
        type=str, 
        required=True,
        default='../data/input_dem/', 
        help='Directory path of input DEM data.'
    )

    parser.add_argument(
        '--input_orbit_dir', 
        type=str, 
        required=True,
        default='../data/input_orbit/', 
        help='Directory path of input orbit data.'
    )

    parser.add_argument(
        '--output_dir', 
        type=str,
        required=True,
        default='../data/output/', 
        help='Path to output coregistration.'
    )

    parser.add_argument(
        '--bbox', 
        type=str, 
        required=True,
        help='Bounding box to coregister images "xmin xmax ymin ymax"'
    )

    parser.add_argument(
        '--pol', 
        type=str, 
        required=True,
        help='Polarization of the images to be used'
    )

    parser.add_argument(
        '--looks', 
        type=str, 
        required=True,
        help='Range and Azimuth looks e.g. "1 1"'
    )

    parser.add_argument(
        '--n_proc', 
        type=int, 
        required=True,
        default=1,  # Setting a default value
        help='Number of threads to process (Need to check)'
    )

    args = parser.parse_args()

    return args


def process(input_slc_dir, input_dem_dir, input_orbit_dir, output_dir, bbox, pol, looks, n_proc):
    
    if looks is None:
        rglooks = 1
        azlooks = 1
    else:
        looks_split = looks.split(' ')
        rglooks = looks_split[0]
        azlooks = looks_split[1]
           
    bbox, s, n, w, e = set_bbox(bbox)
    
    reference = select_reference_img(input_slc_dir)
    
    dem_file = import_dem_file(input_dem_dir, input_slc_dir, output_dir)
    
    # orb_lis = import_orbit_file(input_slc_dir, input_orbit_dir, output_dir)
    
    prepare_isce_settings(output_dir, input_slc_dir, input_dem_dir, input_orbit_dir, bbox, rglooks, azlooks, pol, n_proc, reference)
    
    run_isce(output_dir, input_slc_dir, input_dem_dir, input_orbit_dir, bbox, rglooks, azlooks, pol, n_proc, reference)
    
    remove_remnant(output_dir)
    
if __name__ == "__main__":

    start_time = time.time()
    
    args = get_args()
    
    process(args.input_slc_dir, args.input_dem_dir, args.input_orbit_dir, args.output_dir, args.bbox, args.pol, args.looks, args.n_proc)

    processed_time = time.time() - start_time
    logging.info(f"{processed_time:.2f} seconds")    


