FROM nvidia/cuda:11.2.2-cudnn8-runtime-ubuntu20.04

ENV TZ=Asia/Seoul

# 기본 패키지 업데이트 및 필수 패키지 설치(GDAL 포함)
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    python3.8 python3-pip python3.8-dev libgdal-dev gdal-bin python3-gdal 
RUN apt-get install build-essential
RUN export CPLUS_INCLUDE_PATH=/usr/include/gdal
RUN export C_INCLUDE_PATH=/usr/include/gdal
    
# 기본 Python 버전을 Python 3.8로 설정
RUN ln -sf /usr/bin/python3.8 /usr/bin/python3 && ln -sf /usr/bin/python3 /usr/bin/python

# 최신 pip 설치
RUN python3 -m pip install --upgrade pip

# 필요한 패키지 추가 설치
COPY requirements.txt /app/requirements.txt
RUN python3 -m pip install -r /app/requirements.txt

# weight 폴더 복사
COPY weights /weights

# 소스코드 복사
COPY src /app
WORKDIR /app

# entrypoint 설정
ENTRYPOINT ["python3", "m14_detect_oil_spills.py"]

# CMD는 실행 시 override 가능
# CMD ["--patchsize", "224", "-c", "1", "-c", "2"]
