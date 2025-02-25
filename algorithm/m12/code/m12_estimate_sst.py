# -*- coding: utf-8 -*-
'''
@Time          : 
@Author        : Hee Young Yoo
@File          : m12_estimate_sst.py
@Description   : A script to calculate the mean Sea Surface Temperature (SST) within the ROI from Sentinel-3 SLSTR Level 2 WST data.
@How to use    : python m12_estimate_sst.py --input_dir {image path} --output_dir {output path} --input_meta_file {metafile path}
'''

import argparse
import os
import time
import json
import glob
import logging
import numpy as np
from osgeo import gdal
from typing import Optional, Dict, List

# Configure logging (info, warnings, errors)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def calculate_mean_from_tif(sst_file: str) -> Optional[float]:
	"""
	Calculate the mean SST value from a GeoTIFF file, excluding NaN and NoData values.

	:param sst_file: Path to the SST GeoTIFF file.
	:return: Mean SST value as a float, or None if an error occurs.
	"""
	try:
		sst_dataset = gdal.Open(sst_file, gdal.GA_ReadOnly)
		if not sst_dataset:
			logging.warning(f"Failed to open file: {sst_file}. Skipping.")
			return None

		sst_band = sst_dataset.GetRasterBand(1)
		if sst_band is None:
			logging.warning(f"No raster band found in: {sst_file}. Skipping.")
			return None

		# NoData 값 확인
		nodata_value = sst_band.GetNoDataValue()

		# 데이터 읽기
		sst_array = sst_band.ReadAsArray().astype(np.float32)
		if sst_array is None:
			logging.warning(f"Failed to read raster data: {sst_file}. Skipping.")
			return None

		# NoData 값이 존재하는 경우, 이를 NaN으로 변환하여 평균 계산에서 제외
		if nodata_value is not None:
			sst_array[sst_array == nodata_value] = np.nan

		mean_value = np.nanmean(sst_array)  # NaN 값 제외하고 mean 계산

		sst_dataset = None  # 메모리 해제

		return float(mean_value) if not np.isnan(mean_value) else None
	except Exception as e:
		logging.error(f"Exception occurred while processing {sst_file}: {e}")
		return None


def read_metadata(meta_file: str) -> Dict[str, str]:
	"""
	Read metadata JSON file and extract ROI_ID and Timestamp.

	:param meta_file: Path to the metadata JSON file.
	:return: A dictionary containing {"ROI_ID": str, "Timestamp": str}. 
			 Returns default values if the file is missing or invalid.
	"""
	if not os.path.exists(meta_file):
		logging.warning(f"Metadata file not found: {meta_file}. Using default values.")
		return {"ROI_ID": "Unknown", "Timestamp": "Unknown"}

	try:
		with open(meta_file, "r", encoding="utf-8") as json_file:
			data = json.load(json_file)
			return {
				"ROI_ID": data.get("ROI_ID", "Unknown"),
				"Timestamp": data.get("Timestamp", "Unknown")
			}
	except (json.JSONDecodeError, IOError) as e:
		logging.error(f"Failed to read metadata file {meta_file}: {e}")
		return {"ROI_ID": "Unknown", "Timestamp": "Unknown"}


def save_json(output_path: str, roi_id: str, timestamp: str, filename: str, mean_value: float) -> None:
	"""
	Save the calculated mean SST value as a JSON file.

	:param output_path: Directory to save the JSON file.
	:param roi_id: ROI identifier.
	:param timestamp: Timestamp of the dataset.
	:param filename: Original TIFF filename (without extension).
	:param mean_value: Calculated mean SST value.
	"""
	json_data = {
		"ROI_ID": roi_id,
		"Timestamp": timestamp,
		"SST_Mean": float(mean_value)
	}

	output_json_path = os.path.join(output_path, f"{filename}_sst.json")
	try:
		with open(output_json_path, "w", encoding="utf-8") as json_file:
			json.dump(json_data, json_file, indent=4, ensure_ascii=False)
		logging.info(f"Saved JSON: {output_json_path}")
	except Exception as e:
		logging.error(f"Failed to save JSON file {output_json_path}: {e}")


def process(input_folder: str, output_folder: str, input_meta_file: str) -> None:
	"""
	Process all SST TIFF files in the input folder and save mean SST values as JSON files.

	:param input_folder: Directory containing input TIFF images.
	:param output_folder: Directory to save output JSON files.
	:param input_meta_file: Path to the metadata JSON file.
	"""
	sst_files = glob.glob(os.path.join(input_folder, "*.tif"))
	metadata = read_metadata(input_meta_file)

	roi_id = metadata.get("ROI_ID", "Unknown")
	timestamp = metadata.get("Timestamp", "Unknown")

	if not sst_files:
		logging.warning(f"No TIFF files found in: {input_folder}")
		return

	# 순차적으로 파일 처리 
	for sst_file in sst_files:
		filename = os.path.splitext(os.path.basename(sst_file))[0]
		mean_value = calculate_mean_from_tif(sst_file)

		if mean_value is not None:
			save_json(output_folder, roi_id, timestamp, filename, mean_value)


def get_args():
	
	parser = argparse.ArgumentParser(description="Calculate Mean SST from Sentinel-3 SLSTR Data")
	parser.add_argument(
		"-I", "--input_dir",
		type=str,
		required=True,
		default='../data/input/',
		help="path of input data directory",
	)
	parser.add_argument(
		"-O", "--output_dir",
		type=str,
		required=True,
		default='../data/output/',
		help="path of output data directory"
	)
	parser.add_argument(
		"-M", "--input_meta_file",
		type=str,
		required=True,
		default='../data/input/metainfo.json',
		help="path of output data directory"
	)
	
	args = parser.parse_args()

	return args


def main() -> None:
	"""
	Main execution function. Parses command-line arguments and starts processing.
	"""
	start_time = time.time()
	
	args = get_args()
	
	process(args.input_dir, args.output_dir, args.input_meta_file)
	
	processed_time = time.time() - start_time
	logging.info(f"{processed_time:.2f} seconds")


if __name__ == "__main__":
	main()
