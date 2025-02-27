#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@Time          : 2025/02/06 14:23
@Author        : Satgeo lab.
@File          : main.py
@Noice         : 
@Description   : Perform create_quickview_animation.

@Modificattion :
    @Author    :
    @Time      :
    @Detail    :
'''

import os
import glob
import cv2
import numpy as np
from osgeo import gdal, osr, gdalconst
import argparse
from typing import List, Tuple
# from skimage.feature import register_translation as phase_cross_correlation
# from skimage.feature.register_translation import _upsampled_dft
from skimage.registration import phase_cross_correlation
from skimage.registration._phase_cross_correlation import _upsampled_dft
from scipy.ndimage import shift
import time
import logging
from scipy.ndimage import map_coordinates

def get_geo_transform(extent: List[float], nlines: int, ncols: int) -> List[float]:
    """Compute the GeoTransform array."""
    resx = (extent[2] - extent[0]) / ncols
    resy = (extent[3] - extent[1]) / nlines
    return [extent[0] - resx, resx, 0, extent[3] + resy/2, 0, -resy]


def read_geotiff(filename: str) -> Tuple[np.ndarray, gdal.Dataset]:
    """Read a GeoTIFF file."""
    ds = gdal.Open(filename)
    if ds is None:
        raise FileNotFoundError(f"GeoTIFF file not found: {filename}")
    band = ds.GetRasterBand(1)
    
    return band.ReadAsArray(), ds


def read_geotiff_RGB(filename: str) -> Tuple[np.ndarray, gdal.Dataset]:
    """Read a 3-band RGB GeoTIFF file."""
    ds = gdal.Open(filename)
    if ds is None:
        raise FileNotFoundError(f"GeoTIFF file not found: {filename}")
    
    bands = [ds.GetRasterBand(i + 1).ReadAsArray() for i in range(3)]
    
    return np.array(bands), ds


def image_align_RGB(ref: np.ndarray, align: np.ndarray, patch=True, patch_size=128) -> Tuple[np.ndarray, np.ndarray]:
    """Align two RGB images using phase correlation."""
    img2_color = align.transpose(1, 2, 0)
    img1_color = ref.transpose(1, 2, 0)

    img1 = cv2.cvtColor(img1_color, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2_color, cv2.COLOR_BGR2GRAY)

    ref_nm = robust_normalization(img1)
    align_nm = robust_normalization(img2)
    
    if patch:
        # Prepare output images for coregistered results (patch-based)
        shift_x_map = np.zeros((ref_nm.shape[0] // patch_size, ref_nm.shape[1] // patch_size))
        shift_y_map = np.zeros((ref_nm.shape[0] // patch_size, ref_nm.shape[1] // patch_size))
        
        # Loop through patches
        rows, cols = ref_nm.shape
        for i in range(0, rows, patch_size):
            for j in range(0, cols, patch_size):
                # Extract patches
                reference_patch = ref_nm[i:i+patch_size, j:j+patch_size]
                align_patch = align_nm[i:i+patch_size, j:j+patch_size]
                
                # Handle edge patches
                if reference_patch.shape != (patch_size, patch_size):
                    continue  # Skip incomplete patches

                # Estimate shifts for the current patch
                shift_y, shift_x = match_patches_phase_with_nan(reference_patch, align_patch)
                shift_x_map[i // patch_size, j // patch_size] = shift_x
                shift_y_map[i // patch_size, j // patch_size] = shift_y
        
        # Fit a polynomial to shift maps to estimate global shifts
        grid_x, grid_y = np.meshgrid(np.arange(shift_x_map.shape[1]), np.arange(shift_x_map.shape[0]))
        grid_x = grid_x.ravel()
        grid_y = grid_y.ravel()
        shift_x_flat = shift_x_map.ravel()
        shift_y_flat = shift_y_map.ravel()
        
        # Polynomial fitting
        coeffs_x = np.polyfit(grid_x, shift_x_flat, 3)
        coeffs_y = np.polyfit(grid_y, shift_y_flat, 3)

        # Evaluate polynomial to get global shifts
        global_shift_x = np.polyval(coeffs_x, np.mean(grid_x))
        global_shift_y = np.polyval(coeffs_y, np.mean(grid_y))
        
        # Coregister the entire image using global shifts
        coregistered_image = np.zeros((ref.shape))
        coregistered_image[0,:,:] = coregistration(align[0,:,:], -global_shift_x, -global_shift_y, order=3)
        coregistered_image[1,:,:] = coregistration(align[1,:,:], -global_shift_x, -global_shift_y, order=3)
        coregistered_image[2,:,:] = coregistration(align[2,:,:], -global_shift_x, -global_shift_y, order=3)
        shift_arr = np.array([global_shift_x, global_shift_y])
        
        return shift_arr, coregistered_image
    
    else:
        # Coregister the entire image (no patches)
        shift_y, shift_x = match_patches_phase_with_nan(ref, align)
        coregistered_image = np.zeros((ref.shape))
        coregistered_image[0,:,:] = coregistration(align[0,:,:], -shift_x, -shift_y, order=3)
        coregistered_image[1,:,:] = coregistration(align[1,:,:], -shift_x, -shift_y, order=3)
        coregistered_image[2,:,:] = coregistration(align[2,:,:], -shift_x, -shift_y, order=3)
        shift_arr = np.array([shift_x, shift_y])
        
        return shift_arr, coregistered_image


def image_align_SAR(ref: np.ndarray, align: np.ndarray, patch=True, patch_size=128) -> Tuple[np.ndarray, np.ndarray]:
    """Align two SAR images using phase correlation."""
    
    ref_nm = robust_normalization(ref)
    align_nm = robust_normalization(align)
    
    if patch:
        # Prepare output images for coregistered results (patch-based)
        shift_x_map = np.zeros((ref_nm.shape[0] // patch_size, ref_nm.shape[1] // patch_size))
        shift_y_map = np.zeros((ref_nm.shape[0] // patch_size, ref_nm.shape[1] // patch_size))
        
        # Loop through patches
        rows, cols = ref_nm.shape
        for i in range(0, rows, patch_size):
            for j in range(0, cols, patch_size):
                # Extract patches
                reference_patch = ref_nm[i:i+patch_size, j:j+patch_size]
                align_patch = align_nm[i:i+patch_size, j:j+patch_size]
                
                # Handle edge patches
                if reference_patch.shape != (patch_size, patch_size):
                    continue  # Skip incomplete patches

                # Estimate shifts for the current patch
                shift_y, shift_x = match_patches_phase_with_nan(reference_patch, align_patch)
                shift_x_map[i // patch_size, j // patch_size] = shift_x
                shift_y_map[i // patch_size, j // patch_size] = shift_y
        
        # Fit a polynomial to shift maps to estimate global shifts
        grid_x, grid_y = np.meshgrid(np.arange(shift_x_map.shape[1]), np.arange(shift_x_map.shape[0]))
        grid_x = grid_x.ravel()
        grid_y = grid_y.ravel()
        shift_x_flat = shift_x_map.ravel()
        shift_y_flat = shift_y_map.ravel()
        
        # Polynomial fitting
        coeffs_x = np.polyfit(grid_x, shift_x_flat, 3)
        coeffs_y = np.polyfit(grid_y, shift_y_flat, 3)

        # Evaluate polynomial to get global shifts
        global_shift_x = np.polyval(coeffs_x, np.mean(grid_x))
        global_shift_y = np.polyval(coeffs_y, np.mean(grid_y))
        
        # Coregister the entire image using global shifts
        coregistered_image = coregistration(align, -global_shift_x, -global_shift_y, order=3)
        shift_arr = np.array([global_shift_x, global_shift_y])
        
        return shift_arr, coregistered_image
    
    else:
        # Coregister the entire image (no patches)
        shift_y, shift_x = match_patches_phase_with_nan(ref, align)
        coregistered_image = coregistration(align, -shift_x, -shift_y, order=3)      
        shift_arr = np.array([shift_x, shift_y])
        
        return shift_arr, coregistered_image


def robust_normalization(img, lower_percentile=1, upper_percentile=99):
    """Normalize image while handling outliers by clipping extreme values."""
    if img.dtype == np.uint8:  # If already uint8, return as is
        return img

    # Compute percentiles
    lower_bound = np.percentile(img, lower_percentile)
    upper_bound = np.percentile(img, upper_percentile)

    # Clip outliers
    img_clipped = np.clip(img, lower_bound, upper_bound)

    # Normalize to range [0, 255]
    img_normalized = (img_clipped - lower_bound) / (upper_bound - lower_bound) * 255
    return img_normalized.astype(np.uint8)


def coregistration(sec_image, shift_x, shift_y, order):
    """Apply sub-pixel shifts to coregister an image."""
    rows, cols = sec_image.shape
    # Create a meshgrid of coordinates
    x_coords, y_coords = np.meshgrid(np.arange(cols), np.arange(rows))
    
    # Compute new coordinates
    new_x_coords = x_coords + shift_x
    new_y_coords = y_coords + shift_y
    
    # Use map_coordinates for bilinear interpolation (handles complex data correctly)
    real_part = map_coordinates(sec_image, [new_y_coords, new_x_coords], order=order, mode='constant', cval=0.0)
    
    return real_part


def match_patches_phase_with_nan(patch1, patch2):
    """Match two patches using phase cross-correlation."""
    patch1 = np.abs(patch1)
    patch2 = np.abs(patch2)
    
    # Create masks for NaN values
    mask1 = ~np.isnan(patch1)
    mask2 = ~np.isnan(patch2)
    valid_mask = mask1 & mask2

    # Apply masks
    patch1_valid = np.where(valid_mask, patch1, 0)
    patch2_valid = np.where(valid_mask, patch2, 0)
    
    if np.all(patch1_valid == 0):
        return 0, 0  # No valid data
    else:
        # Phase cross-correlation with sub-pixel precision
        shift, _, _ = phase_cross_correlation(patch1_valid, patch2_valid, upsample_factor=100)
        return shift[0], shift[1]
    
    
def image_shift_save_RGB(output_dir: str, file: str, img2_color: np.ndarray, bbox: str) -> None:
    """Apply shift correction to RGB image and save as GeoTIFF."""

    img2_shift_B = img2_color[0, :, :]
    img2_shift_G = img2_color[1, :, :]
    img2_shift_R = img2_color[2, :, :]

    filepath = os.path.dirname(file)
    filename = os.path.basename(file)
    
    miny, maxy, minx, maxx = map(float, bbox.split())

    extent = [minx, miny, maxx, maxy]

    driver = gdal.GetDriverByName("GTiff")
    nlines, ncols = img2_shift_B.shape
    grid_data = driver.Create("grid_data", ncols, nlines, 3, gdal.GDT_Float32)

    grid_data.GetRasterBand(1).WriteArray(img2_shift_B)
    grid_data.GetRasterBand(2).WriteArray(img2_shift_G)
    grid_data.GetRasterBand(3).WriteArray(img2_shift_R)

    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)

    grid_data.SetProjection(srs.ExportToWkt())
    grid_data.SetGeoTransform(get_geo_transform(extent, nlines, ncols))

    output_file = os.path.join(output_dir, f"aligned_{filename}")
    driver.CreateCopy(output_file, grid_data, 0)

    print(f"Generated GeoTIFF: {output_file}")


def image_shift_save_SAR(output_dir: str, file: str, img2: np.ndarray, bbox: str) -> None:
    """Apply shift correction to SAR image and save as GeoTIFF."""

    filepath = os.path.dirname(file)
    filename = os.path.basename(file)
    
    miny, maxy, minx, maxx = map(float, bbox.split())

    extent = [minx, miny, maxx, maxy]

    driver = gdal.GetDriverByName("GTiff")
    nlines, ncols = img2.shape
    grid_data = driver.Create("grid_data", ncols, nlines, 1, gdal.GDT_Float32)
    grid_data.GetRasterBand(1).WriteArray(img2)

    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)

    grid_data.SetProjection(srs.ExportToWkt())
    grid_data.SetGeoTransform(get_geo_transform(extent, nlines, ncols))

    output_file = os.path.join(output_dir, f"aligned_{filename}")
    driver.CreateCopy(output_file, grid_data, 0)

    print(f"Generated GeoTIFF: {output_file}")

    os.remove("grid_data")
    
    
def image_crop(input_file: str, bbox: str) -> None:
    """
    Crops a GeoTIFF file using the provided bounding box.

    Args:
        output_dirname (str): Path to the output directory.
        bbox (Tuple[float, float, float, float]): Bounding box in (miny, maxy, minx, maxx) format.
        filename (str): Name of the file (without extension) to crop.
    """
    bbox_in = bbox.split(' ')
    if len(bbox_in) == 4:
        miny=float(bbox_in[0]);maxy=float(bbox_in[1]);minx=float(bbox_in[2]);maxx=float(bbox_in[3]);
        x = float(minx) - float(maxx)
        y = float(maxy) - float(miny)
        bbox = '{0:.4f} {1:.4f} {2:.4f} {3:.4f}'.format(miny,maxy,minx,maxx)
    elif len(bbox_in) == 2:
        x = bbox_in[0]
        y = bbox_in[1]
        miny=float(y)-0.02;maxy=float(y)+0.02;minx=float(x)-0.02;maxx=float(x)+0.02
        bbox = '{0:.4f} {1:.4f} {2:.4f} {3:.4f}'.format(miny,maxy,minx,maxx)
    else:
        print('Incorrect bbox specified, please check again')

    filepath = os.path.dirname(input_file)
    filename = os.path.basename(input_file)
    
    # Define input and output file paths
    inputpath = os.path.join(filepath, f"{filename}")
    outputpath = os.path.join(filepath, f"crop_{filename}")

    # Ensure input file exists
    if not os.path.exists(inputpath):
        raise FileNotFoundError(f"Input file not found: {inputpath}")

    # Perform cropping using GDAL Translate
    gdal.Translate(outputpath, inputpath, projWin=(minx, maxy, maxx, miny))
    print(f"Cropped image saved: {outputpath}")

    return outputpath


def process(output_dirname: str, reference: str, align: str, bbox: str, sensor: str) -> None:
    
    if args.sensor == 'SAR':
        
        crop_reference = image_crop(reference, bbox)
        crop_align = image_crop(align, bbox)
        
        reference_tiff, ds = read_geotiff(crop_reference)
        align_tiff, ds = read_geotiff(crop_align)
        
        align_tiff = cv2.resize(align_tiff,dsize=(reference_tiff.shape[1],reference_tiff.shape[0])) # matching grid size of align_tiff with reference
        
        shift_arr, img2 = image_align_SAR(reference_tiff, align_tiff, True, 128)
        image_shift_save_SAR(output_dirname, align, img2, bbox)
        os.system(f'mv {crop_reference} {output_dirname}')
        
    elif sensor == 'Optical':

        crop_reference = image_crop(reference, bbox)
        crop_align = image_crop(align, bbox)
        
        reference_tiff, ds = read_geotiff_RGB(crop_reference)
        align_tiff, ds = read_geotiff_RGB(crop_align)
        
        align_tiff2 = np.zeros((3,reference_tiff.shape[1],reference_tiff.shape[2]))
        
        for i in range(len(align_tiff2)):
            img = align_tiff[i,:,:]
            img_resize = cv2.resize(img,dsize=(reference_tiff.shape[2],reference_tiff.shape[1]))
            align_tiff2[i,:,:] = img_resize
        
        align_tiff2 = np.array(align_tiff2, dtype=int).astype('uint16')
        shift_arr, img2_color = image_align_RGB(reference_tiff, align_tiff2, True, 128)
        image_shift_save_RGB(output_dirname, align, img2_color, bbox)
        os.system(f'mv {crop_reference} {output_dirname}')
       

def get_args():
    parser = argparse.ArgumentParser(description="m02_create_quickview_animation")
    
    parser.add_argument(
        "--output_dir", 
        type=str, 
        required=True, 
        help="Output directory path"
    )
    
    parser.add_argument(
        "--input_ref_file", 
        type=str, 
        required=True, 
        help="Reference image"
    )
    
    parser.add_argument(
        "--input_aln_file", 
        type=str, 
        required=True, 
        help="Image for alignment"
    )
    
    parser.add_argument(
        "--bbox", 
        type=str, 
        required=True, 
        help="Bounding box (s n w e)"
    )
    
    parser.add_argument(
        "--sensor", 
        type=str, 
        required=True, 
        choices=["SAR", "Optical"], 
        help="Sensor type (SAR/Optical)"
    )

    args = parser.parse_args()

    return args

        
if __name__ == "__main__":  
    
    start_time = time.time()
    
    args = get_args()
    
    process(args.output_dir, args.input_ref_file, args.input_aln_file, args.bbox, args.sensor)

    processed_time = time.time() - start_time
    logging.info(f"{processed_time:.2f} seconds")   
       