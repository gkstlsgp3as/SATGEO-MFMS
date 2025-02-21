# 유류 탐지 딥 러닝 알고리즘 실행 방법

## 시스템 환경
- os version: Ubuntu 20.04.6 LTS
- Python version: 3.8(`3.7에서 테스트환경 세팅의 어려움이 있어서 3.8로 변경해서 진행`)

## Prerequisite
```shell
pip install -r requirements.txt
```

## 입력 커맨드 파라미터 설명
- `--weight`: weight 파일(`.h5` 포맷)
- `--input-tif-path`: 전처리 끝난 입력 SAR 영상 경로 (Sentinel-1)
- `--output-tif-path`: 결과 .tif 파일을 저장할 경로 (이진영상)
- `--output-json-path`: 결과 .json 파일을 저장할 경로
- `--patchsize`: weightfile에 적용하는 입력 patch size
- `-c`: 입력 영상 channel 선택(`-c 1 -c 2`의 경우 입력영상의 첫번째와 두번째 channel을 사용하게 됨)

## 실행 방법
```shell
python ./src/run.py \
--weight=./weights/oilspill.h5 \
--input-tif-path=./sample/input/input.tif \
--output-tif-path=./sample/output/output.tif \
--output-json-path=./sample/output/output.json \
--patchsize=224 \
-c 1 -c 2
```


## input, output 폴더 구조
```
# oilspill과 같이 input 이미지가 1개만 필요한 경우
.
├── input
|   ├── 0
|   |   ├── input.tif
|   |   └── metadata.xml
|
├── output
|   ├── 0
|   |   ├── output.tif
|   |   └── output.json

# 변화탐지와 같이 input 이미지가 쌍으로 필요한 경우(1, 2 폴더는 한번에 여러쌍의 결과를 얻고 싶을 때 사용)
.
├── input
|   ├── 0
|		|   ├── metadata_pre.xml
|		|   ├── input_pre.tif
|		|   ├── metadata_post.xml
|		|   ├── input_post.tif
|   ├── 1
|		|   ├── metadata_pre.xml
|		|   ├── input_pre.tif
|		|   ├── metadata_post.xml
|		|   ├── input_post.tif
|   ├── 2
|		|   ├── ...
|   |   └── 
|
├── output
|   ├── 0
|		|   ├── output.tif
|   ├── 1
|		|   ├── output.tif
|   ├── 2
|		|   ├── output.tif

# 변위탐지와 같이 여러 날짜의 영상이 한번에 필요한 경우
.
├── input
|   ├── 0
|   |   ├── input_{date}.tif
|   |   ├── metadata_{date}.xml
|   |   ├── input_{date}.tif
|   |   ├── metadata_{date}.xml
|   |   |   ...
|   |   └
|
├── output
|   ├── 0
|   |   └── output.tif
```

## 도커 실행 관련
### docker build
```shell
docker build -t snu_sample_oilspill .
# dns 문제 등으로 외부 인터넷에서 파일을 못 받아오는 경우에 --network=host 옵션 추가
```
### docker run
```shell
docker run -v $(pwd)/sample/input:/platform/data/input/0 -v $(pwd)/sample/output:/platform/data/output/0 --gpus all --rm snu_test --patchsize 224 -c 1 -c 2
```