import os
import json
import rasterio
import numpy as np
import argparse
import time
import logging


# 구름 마스크 생성 함수
def create_cloud_mask(qa_array):
    cloud_mask = (qa_array & (1 << 3)) != 0  # 비트 3: 구름
    cloud_shadow_mask = (qa_array & (1 << 4)) != 0  # 비트 4: 구름 그림자
    combined_mask = cloud_mask | cloud_shadow_mask  # 둘 중 하나라도 True이면 마스킹
    return combined_mask


# Landsat 데이터 경로 로드 함수
def load_landsat_path(data_dir):
    base_name = [f for f in os.listdir(data_dir) if f.endswith('_B4.TIF')][0].replace('_B4.TIF', '')
    band4_path = os.path.join(data_dir, base_name + '_B4.TIF')
    band5_path = os.path.join(data_dir, base_name + '_B5.TIF')
    band10_path = os.path.join(data_dir, base_name + '_B10.TIF')
    qa_path = os.path.join(data_dir, base_name + '_QA_PIXEL.TIF')
    return band4_path, band5_path, band10_path, qa_path


# 메타데이터 읽기 함수 (JSON 기반)
def read_metadata_json(meta_file):
    with open(meta_file, 'r') as f:
        metadata = json.load(f)
    rad_mult = metadata["RADIANCE_MULT_BAND_10"]
    rad_add = metadata["RADIANCE_ADD_BAND_10"]
    k1_constant = metadata["K1_CONSTANT_BAND_10"]
    k2_constant = metadata["K2_CONSTANT_BAND_10"]
    return rad_mult, rad_add, k1_constant, k2_constant


# 밝기 온도 계산 함수
def calculate_brightness_temperature(band10_path, meta_file):
    rad_mult, rad_add, k1_constant, k2_constant = read_metadata_json(meta_file)
    with rasterio.open(band10_path) as band10_src:
        band10 = band10_src.read(1).astype(np.float64)
        TOA = rad_mult * band10 + rad_add
        
        # TOA 값이 0 이하인 경우 NaN으로 변환하여 계산 오류 방지
        TOA = np.where(TOA <= 0, np.nan, TOA)
        brightness_temperature = np.where(
            TOA > 0, k2_constant / np.log((k1_constant / TOA) + 1), np.nan
        )
    
    return brightness_temperature


# 방출율 계산 함수
def calculate_emissivity(band4_path, band5_path, ndvi_thresholds=(0.2, 0.5), emissivity_values=(0.996, 0.973)):
    with rasterio.open(band4_path) as band4_src, rasterio.open(band5_path) as band5_src:
        band4 = band4_src.read(1).astype(np.float64)
        band5 = band5_src.read(1).astype(np.float64)
    with np.errstate(divide='ignore', invalid='ignore'):
        ndvi = np.where((band5 + band4) == 0, 0, (band5 - band4) / (band5 + band4))
    ndvi_min, ndvi_max = ndvi_thresholds
    pv = ((ndvi - ndvi_min) / (ndvi_max - ndvi_min)) ** 2
    emissivity_soil, emissivity_vegetation = emissivity_values
    emissivity = np.where(
        ndvi < ndvi_min, emissivity_soil,
        np.where(ndvi >= ndvi_max, emissivity_vegetation, emissivity_soil * pv + emissivity_vegetation * (1 - pv))
    )
    emissivity = np.where(ndvi < 0, 0.991, emissivity)
    emissivity = np.where((ndvi >= 0) & (ndvi < 0.2), 0.996, emissivity)
    return emissivity


def calculate_st(brightness_temperature, emissivity):
    st = brightness_temperature / (1 + (0.00115 * brightness_temperature / 1.4388) * np.log(emissivity)) - 273.15
    #이상값 필터링
    st = np.where((st < -50) | (st > 80), np.nan, st)

    return st


# 최종 결과 저장 함수
def save_st_as_geotiff(st, output_path, band10_src):
    profile = band10_src.profile
    profile.update(count=1, dtype=rasterio.float64)
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(st, 1)
    print("Surface Temperature (ST) calculated and saved to", output_path)


# 데이터 처리 함수
def process(input_dir: str, output_dir: str, meta_file: str):
    band4_path, band5_path, band10_path, qa_path = load_landsat_path(input_dir)
    with rasterio.open(band10_path) as band10_src:
        brightness_temperature = calculate_brightness_temperature(band10_path, meta_file)
        emissivity = calculate_emissivity(band4_path, band5_path)
        st = calculate_st(brightness_temperature, emissivity)
        with rasterio.open(qa_path) as qa_src:
            qa_pixel = qa_src.read(1)
            cloud_mask = create_cloud_mask(qa_pixel)
            masked_st = np.where(cloud_mask, np.nan, st)
        save_st_as_geotiff(masked_st, os.path.join(output_dir, 'output_st.tif'), band10_src)


def get_args():
    parser = argparse.ArgumentParser(description="Calculate ST from Landsat8/9 Data") 
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
        help="path of metadata file"
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