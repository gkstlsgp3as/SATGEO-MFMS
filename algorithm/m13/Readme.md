Calculate ST within the ROI from Landsat8/9 data
python m13_estimate_ss.py --input_dir {image path} --output_dir {output path} --meta_file {metafile path} 


pip install rasterio numpy argparse json logging


# Input 폴더내 필요한 입력 파일 (Landsat 8/9 데이터)
# 1. <basename>_B4.TIF  (Red 밴드)
# 2. <basename>_B5.TIF  (Near-Infrared 밴드)
# 3. <basename>_B10.TIF (Thermal Infrared 밴드)
# 4. <basename>_QA_PIXEL.TIF (품질 보증 마스크 - 구름 정보 포함)
# 5. metadata.json (메타데이터 파일 - 방사 보정 계수 포함)

# 출력 파일
# - output_st.tif (지표면 온도 (ST) 결과 GeoTIFF 파일)

