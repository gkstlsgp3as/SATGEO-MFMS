Calculate the mean SST within the ROI from Sentinel-3 SLSTR Level 2 WST data
python m12_estimate_sst.py --input_dir {image path} --output_dir {output path} --meta_file {metafile path} 

현재폴더 기준

python m12_estimate_sst.py --input_dir Input --output_dir output --meta_file Input/metainfo.json

pip install numpy gdal

ROI별 SST를 추출한 파일들이 Input으로 들어감 

Input path 안에는 ROI_ID 폴더 안에 cropped SST가 tiff 파일로 들어있고 영상촬영시간 관련 metainfo가 json으로 저장되어 있음
Output에는 ROI_ID로 되어 있는 Json 파일이 생성되고 그안에는 ROI_ID, Timestamp, SST_mean이 들어가 있음

여기에서의 sample data는 ROI_ID 한개에 대해서만 들어있지만 실제로는 Input내 모든 파일을 처리하도록 되어 있음

