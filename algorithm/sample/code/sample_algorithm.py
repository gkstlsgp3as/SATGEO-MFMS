# -*- coding: utf-8 -*-
'''
@Time          : 
@Author        : 
@File          : 
@Noice         : 
@Description   : 
@How to use    : 

@Modificattion :
    @Author    :
    @Time      :
    @Detail    :
'''

import argparse
from utils.cfg import Cfg

from app.config.settings import settings
#from app.models.{모델_py_파일} import {모델_클래스}
from app.service.{서비스_py_파일} import {서비스_함수}

def sub_algorithm():
    ## 코드

def algorithm(db: Session, args: type):
    input_dir = settings.SAMPLE_INPUT_PATH
    output_dir = settings.SAMPLE_OUTPUT_PATH
    meta_file = settings.SAMPLE_META_FILE

from tensorflow.keras.models import load_model
import numpy as np
from osgeo import gdal
import time
import argparse
import json
import os
import logging


logging.basicConfig(level=logging.INFO)


def sub_algorithm(model, input_img, width):
    
    ## 코드

    return results


def process(args: type, output_json_file: type):
    
    results = algorithm()
    
    # TODO: Create output json file
    # output.json 파일안에 threshold 검사에 필요한 값이 아래와 같은 형태로 정의되어 있어야 합니다.
    # thresholdEstimationValue에 정의된 값이 유의, 위험 등으로 설정한 값보다 큰 경우에 alert을 주도록 시스템 내 구현되어 있습니다.
    # Ex. {"thresholdEstimationValue": 10.4}
    # 아래는 임시로 작성한 코드입니다.
    ### start - create output.json ###
    THRESHOLD_ESTIMATION_VALUE_KEY = "thresholdEstimationValue"
    data = {
        THRESHOLD_ESTIMATION_VALUE_KEY: 10.4
    }
    with open(output_json_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    ### end - create output.json ###


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--argument_1",
        type=str,
        default="~",
        required=False,
        help="~",
    )
    parser.add_argument(
        "--argument_2",
        type=str,
        default="~",
        required=False,
        help=""
    )
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    startTime = time.time()

    args = get_args()

    process(args.input_grd_file, args.output_grd_file, args.output_json_file, args.model_weight_file, args.patch_size, args.channel_list)

    processedTime = time.time() - startTime
    logging.info(f"{processedTime:.2f} seconds")

    
    
