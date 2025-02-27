# -*- coding: utf-8 -*-
'''
@Time          : 20/04/25 15:49
@Author        : huguanghao
@File          : demo.py
@Noice         :
@Modificattion :
    @Author    :
    @Time      :
    @Detail    :
'''

from utils.tool.utils import *
from utils.tool.torch_utils import *
from utils.tool.darknet2pytorch import Darknet
import argparse
from utils.gdal_preprocess import *
import cv2
from utils.cfg import Cfg
from models import Yolov4
import os

"""hyper parameters"""
use_cuda = True

# geotiff 테스트 이미지를 rgb 데이터로 변환하여 list에 저장
def get_gdal_testset(imagefolder):
    from utils.cfg import Cfg

    test_images = []
    img_list = [x for x in os.listdir(imagefolder) if x.endswith('tif')]
    for i in img_list:
        image_path = imagefolder + i

        # RGB로 변환
        if Cfg.NewTest == 0:
            rgb_band = band_to_rgb(image_path, Cfg.Satelliteband)
        else:
            rgb_band = band_to_rgb(image_path, Cfg.Satelliteband, True)

        #rgb_band = band_to_rgb(image_path, Cfg.Satelliteband)

        test_images.append(rgb_band)

    return test_images


def detect(model, imglist, div, input_size, score_thresh):
    total_preds, shore_bboxes, detect_times = [], [], []
    for i in imglist:
        final_bboxes = []

        lines = line_detection(i)

        rgb_band = cv2.cvtColor(i, cv2.COLOR_BGR2RGB)
        div_img_list, div_coord = division_testset(input_band=rgb_band, div_num=div)
        line_div_list, _ = division_testset(input_band=lines, div_num=div)

        for d_id, d_img in enumerate(div_img_list):
            if np.average(line_div_list[d_id]) == 1.:
                continue
            div_x, div_y = div_coord[d_id][0], div_coord[d_id][1]

            sized = cv2.resize(d_img, (input_size,input_size))

            # Gaussian Blur
            kernel = 3
            sized = cv2.GaussianBlur(sized, (kernel,kernel), 0)
            
            start = time.time()
            boxes = do_detect(model=model, img=sized, conf_thresh=score_thresh, nms_thresh=0.4, use_cuda=1)
            detect_time = time.time() - start
            detect_times.append(detect_time)


            d_height, d_width = d_img.shape[:2]
            boxes = boxes[0]
            for n in range(len(boxes)):
                box = boxes[n]
                # origin image
                x1 = int(box[0] * d_width) + div_x
                y1 = int(box[1] * d_height) + div_y
                x2 = int(box[2] * d_width) + div_x
                y2 = int(box[3] * d_height) + div_y

                # Line filter
                w, h = x2-x1, y2-y1
                cx = x1 + int(w/2)
                cy = y1 + int(h/2)
                
                if lines[cy, cx] == 0:
                    final_bboxes.append([x1,y1,x2,y2])
                else:
                    shore_bboxes.append([x1,y1,x2,y2])
        total_preds.append(len(final_bboxes))
    
    return final_bboxes


def detect_listInference(model, input_dir, output_dir, div, input_size, score_thresh, model_name=None, line_det=False, kernel=3, csv_path='./milestone/rgb_divInfer.csv',bandnumber=3):
    from utils.cfg import Cfg
    import pandas as pd

    img_list = [x for x in os.listdir(input_dir) if x.endswith('tif')]
    total_preds = []

    for i in img_list:
        final_bboxes, shore_bboxes = [], []
        detect_times = []

        img_path = input_dir + i

        # Read the Geotiff file and save reference information
        imgfile_temp = gdal.Open(img_path)
        xoff, ca, cb, yoff, cd, ce = imgfile_temp.GetGeoTransform()

        if Cfg.NewTest == 0:
            rgb_band = band_to_rgb(img_path,bandnumber)
        else:
            rgb_band = band_to_rgb(img_path, bandnumber, True)

        height, width = rgb_band.shape[:2]

        # 라인 검출결과 육지로 판단된 좌표의 픽셀값을 1로, 이 외 값은 전부 0
        lines = line_detection(rgb_band)

        # 모델 추론을 위한 전처리
        # BGR --> RGB
        rgb_band = cv2.cvtColor(rgb_band, cv2.COLOR_BGR2RGB)

        # 테스트 이미지를 1/div_num 만큼 width, height를 분할하고, 크롭된 이미지와 위치좌표를 반환
        div_img_list, div_coord = division_testset(input_band=rgb_band, div_num=div)
        line_div_list, _ = division_testset(input_band=lines, div_num=div)

        for d_id, d_img in enumerate(div_img_list):
            # 크롭 이미지에서 라인 검출결과가 전부 1일 경우 해당 이미지는 전부 육지에 해당하기때문에 추론하지 않음
            if np.average(line_div_list[d_id]) == 1.:
                continue

            # 원본 이미지 좌표로 변환하기 위해 분활 좌표를 저장
            div_x, div_y = div_coord[d_id][0], div_coord[d_id][1]
            # 모델 입력 사이즈로 이미지 크기 변환
            sized = cv2.resize(d_img, (input_size, input_size))

            # Gaussian Blur
            if kernel > 0:
                sized = cv2.GaussianBlur(sized, (kernel, kernel), 0)

            start = time.time()

            # 검출(do not modify the threshold parameters!)
            boxes = do_detect(model=model, img=sized, conf_thresh=score_thresh, nms_thresh=0.4, use_cuda=1)

            detect_time = time.time() - start
            detect_times.append(detect_time)
            # 분할 이미지의 height, width
            d_height, d_width = d_img.shape[:2]
            boxes = boxes[0]
            for n in range(len(boxes)):
                box = boxes[n]
                # 분할 이미지에서 검출한 bbox좌표를 원본 이미지 좌표로 변환
                x1 = int(box[0] * d_width) + div_x
                y1 = int(box[1] * d_height) + div_y
                x2 = int(box[2] * d_width) + div_x
                y2 = int(box[3] * d_height) + div_y

                if x1 < 0: x1 = 0
                if x2 > width: x2 = width
                if y1 < 0: y1 = 0
                if y2 > width: y2 = height

                # bbox 중심값 계산
                w, h = x2 - x1, y2 - y1
                cx = x1 + int(w / 2)
                cy = y1 + int(h / 2)

                if line_det:
                    # 검출 bbox의 중심 좌표가 육지로 판단된 좌표에 해당할 경우 false positive로 판단하고 검출결과에서 제외
                    if lines[cy, cx] == 0:
                        final_bboxes.append([x1, y1, x2, y2])
                    else:
                        shore_bboxes.append([x1, y1, x2, y2])
                else:
                    final_bboxes.append([x1, y1, x2, y2])

        total_preds.append(len(final_bboxes))

        try:
            # Detected Vessel export to csv: Image-dependent output
            final_bboxes = np.array(final_bboxes)
            image_df = pd.DataFrame(columns=['image', 'X', 'Y', 'W', 'H','Lon','Lat'])
            image_df['image'] = [i for x in range(len(final_bboxes))]
            image_df['X'] = final_bboxes[:, 0]
            image_df['Y'] = final_bboxes[:, 1]
            image_df['W'] = final_bboxes[:, 2] - final_bboxes[:, 0]
            image_df['H'] = final_bboxes[:, 3] - final_bboxes[:, 1]
            image_df['Lon'] = ca * ((final_bboxes[:, 2] + final_bboxes[:, 0]) / 2) + cb * (
                    (final_bboxes[:, 3] + final_bboxes[:, 1]) / 2) + xoff
            image_df['Lat'] = cd * ((final_bboxes[:, 2] + final_bboxes[:, 0]) / 2) + ce * (
                    (final_bboxes[:, 3] + final_bboxes[:, 1]) / 2) + yoff

            print('Vessels Detected in This Scene! Saved at '+output_dir+i[:-4]+'.csv')
            image_df.to_csv(output_dir+i[:-4]+'.csv', index=False, encoding='utf-8')

        except:
            print('No Vessel Detected in This Scene!')
            image_df = pd.DataFrame(columns=['image', 'X', 'Y', 'W', 'H', 'Lon', 'Lat'])
            image_df.to_csv(output_dir+i, index=False, encoding='utf-8')
            
        return image_df


# IoU calculation between 2 boxes
def getIoU(bb1, bb2):
    bb1_x1 = bb1[0]
    bb1_y1 = bb1[1]
    bb1_x2 = bb1[0] + bb1[2]
    bb1_y2 = bb1[1] + bb1[3]

    bb2_x1 = bb2[0]
    bb2_y1 = bb2[1]
    bb2_x2 = bb2[0] + bb2[2]
    bb2_y2 = bb2[1] + bb2[3]

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1_x1, bb2_x1)
    y_top = max(bb1_y1, bb2_y1)
    x_right = min(bb1_x2, bb2_x2)
    y_bottom = min(bb1_y2, bb2_y2)

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1_x2 - bb1_x1) * (bb1_y2 - bb1_y1)
    bb2_area = (bb2_x2 - bb2_x1) * (bb2_y2 - bb2_y1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    # assert iou >= 0.0
    # assert iou <= 1.0

    return iou


# Import only Geo-reference
def geotiffreadRef(tif_name):
    import gdal
    import numpy as np

    gdal.AllRegister()

    ds = gdal.Open(tif_name)
    gt = ds.GetGeoTransform()
    rows, cols = ds.RasterXSize, ds.RasterYSize

    gt = np.array(gt)
    gt[2] = gt[0] + gt[1] * (rows - 1)
    gt[4] = gt[3] + gt[5] * (cols - 1)

    gt.astype(np.double)

    return gt, rows, cols


def geographicToIntrinsic(tif_ref, lat, lon):
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
    lat_order = np.linspace(1, len(lat_array), len(lat_array))
    lon_array = np.linspace(min_lon, max_lon, num_lon)
    lon_order = np.linspace(1, len(lon_array), len(lon_array))

    lat_order = lat_order.astype(int)
    lon_order = lon_order.astype(int)

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


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-I', '--input_dir', 
        type=str,
        default='../data/input/',
        help='path of your image folder.'
    )
    parser.add_argument(
        '-O', '--output_dir', 
        type=str,
        default='../data/output/',
        help='path of your output folder.'
    )
    parser.add_argument(
        '--model_weight_file', 
        type=str,
        default='../weights/S1_epoch2_div15_Maxrng0.15_0.5_50_pre87.696_rec91.651_f1s89.630.pth',
        help='path of trained model.'
    )
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    from utils.cfg import Cfg
    start_time = time.time()
    
    args = get_args()
    
    weight_path = args.model_weight_file
    weight_name = weight_path.split('/')[-1]
    
    # Pytorch(.pth) model load
    model = Yolov4(yolov4conv137weight=None, n_classes=1, inference=True)
    pretrained_dict = torch.load(weight_path, map_location=torch.device('cuda'))
    model.load_state_dict(pretrained_dict)

    if use_cuda:
        model.cuda()

    detect_listInference(model, args.input_dir,
                args.output_dir,
                div=Cfg.division,
                input_size=Cfg.inputsize,
                score_thresh=Cfg.scorethresh,
                model_name=weight_name,
                line_det=False,
                kernel=3,
                bandnumber=Cfg.Satelliteband)
    
    processed_time = time.time() - start_time
    logging.info(f"{processed_time:.2f} seconds")