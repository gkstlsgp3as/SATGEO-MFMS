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

from tool.utils import *
from tool.torch_utils import *
from tool.darknet2pytorch import Darknet
import argparse
from gdal_preprocess import *
import cv2
from cfg import Cfg
from models import Yolov4

"""hyper parameters"""
use_cuda = True

# geotiff 테스트 이미지를 rgb 데이터로 변환하여 list에 저장
def get_gdal_testset(imagefolder):
    from cfg import Cfg

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
            # sized = cv2.medianBlur(sized,kernel)
            # sized = cv2.morphologyEx(sized, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))

            start = time.time()
            # model, img, conf_thresh, nms_thresh, use_cuda=1
            boxes = do_detect(model=model, img=sized, conf_thresh=score_thresh, nms_thresh=0.4, use_cuda=1)
            detect_time = time.time() - start
            detect_times.append(detect_time)

            # # Draw division line
            # rgb_band = cv2.line(rgb_band, (div_x, 0), (div_x, rgb_band.shape[0]), (255,0,0), 1)
            # rgb_band = cv2.line(rgb_band, (0, div_y), (rgb_band.shape[1], div_y), (255,0,0), 1)

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
    #print('Prediction: {}\tShore: {}\tTime: {:.4f}sec'.format(sum(total_preds), len(shore_bboxes), sum(detect_times)))
    #return sum(total_preds)
    return final_bboxes


def detect_list(model, imgfolder, div, input_size, score_thresh, model_name=None, draw=False, line_det=False, kernel=3, bandnumber=3):

    from cfg import Cfg

    img_list = [x for x in os.listdir(imgfolder) if x.endswith('tif')]
    total_preds = []
    result_df = pd.DataFrame(columns=['image', 'X', 'Y', 'W', 'H'])
    
    # 여기서의 i는 전체 영상 1장 의미
    for i in img_list:
        final_bboxes, shore_bboxes = [], []
        detect_times = []

        img_path = imgfolder + i

        # RGB로 변환
        if Cfg.NewTest == 0:
            rgb_band = band_to_rgb(image_path, Cfg.Satelliteband)
        else:
            rgb_band = band_to_rgb(image_path, Cfg.Satelliteband, True)

        #rgb_band = band_to_rgb(img_path,bandnumber)

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
            sized = cv2.resize(d_img, (input_size,input_size))

            # Gaussian Blur
            if kernel > 0:
                sized = cv2.GaussianBlur(sized, (kernel,kernel), 0)

            start = time.time()
            # 검출
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
                w, h = x2-x1, y2-y1
                cx = x1 + int(w/2)
                cy = y1 + int(h/2)
                
                if line_det:
                    # 검출 bbox의 중심 좌표가 육지로 판단된 좌표에 해당할 경우 false positive로 판단하고 검출결과에서 제외
                    if lines[cy, cx] == 0:
                        final_bboxes.append([x1,y1,x2,y2])
                    else:
                        shore_bboxes.append([x1,y1,x2,y2])
                else:
                    final_bboxes.append([x1,y1,x2,y2])
        
        total_preds.append(len(final_bboxes))
        print('Image: {}\tPrediction: {}\tFP: {}\tTime: {:.4f}sec'.format(i, len(final_bboxes), len(shore_bboxes), sum(detect_times)))


        if draw:
            # band1 이미지에 검출 bbox 그리기
            b1_image = rgb_band[:,:,0]
            b1_image = np.dstack((b1_image, b1_image, b1_image))

            # 예측 bbox
            for b in final_bboxes:
                b1_image = cv2.rectangle(b1_image, (b[0], b[1]), (b[2], b[3]), (0,0,255), 1)

            # # FP로 판단된 bbox
            # for s in shore_bboxes:
            #     b1_image = cv2.rectangle(b1_image, (s[0], s[1]), (s[2], s[3]), (0,255,0), 1)

            # 결과 이미지 저장
            cv2.imwrite('./output/'+i, b1_image)

            # csv 저장
            final_bboxes = np.array(final_bboxes)
            image_df = pd.DataFrame(columns=['image', 'X', 'Y', 'W', 'H'])
            image_df['image'] = [i for x in range(len(final_bboxes))]
            image_df['X'] = final_bboxes[:, 0]
            image_df['Y'] = final_bboxes[:, 1]
            image_df['W'] = final_bboxes[:, 2] - final_bboxes[:, 0]
            image_df['H'] = final_bboxes[:, 3] - final_bboxes[:, 1]

            result_df = result_df.append(image_df, ignore_index=True)

    print('==> Total Prediction: ', sum(total_preds))
    if draw:
        csv_name = model_name.split('.')[0]
        csv_name = csv_name.split('_')[1] + '_score{}_div{}_size{}_kernel{}.csv'.format(str(score_thresh).split('.')[1], str(div), str(input_size), str(kernel))
        result_df.to_csv(csv_name, index=False, encoding='utf-8')
    
    return sum(total_preds)

# Training 중 중간평가를 위한 list detection
def detect_listTrain(model, imgfolder, div, input_size, score_thresh, line_det=False, kernel=3, csv_path='./milestone/rgb_divTest.csv'):
    from cfg import Cfg
    from train import bboxprecisionrecall, calc_precision_recall

    bandnumber = Cfg.Satelliteband
    img_list = [x for x in os.listdir(imgfolder) if x.endswith('tif')]
    total_preds = []

    df = pd.read_csv(csv_path)
    ExpACC=[]
    ExpDET=[]
    ExpGT=[]

    for i in img_list:
        final_bboxes, shore_bboxes = [], []
        detect_times = []

        img_path = imgfolder + i
        if Cfg.NewTest == 0:
            rgb_band = band_to_rgb(img_path, bandnumber)
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

        #print('Image: {}\tPrediction: {}\tFP: {}\tTime: {:.4f}sec'.format(i, len(final_bboxes), len(shore_bboxes),
        #                                                                  sum(detect_times)))

        # Import GT data corresponding to SAR images
        dfTempIdx=df.index[df['image'] == i].tolist()
        dfTemp=df.loc[dfTempIdx,:]
        dfTemp=dfTemp[['X','Y','W','H']]

        # convert Dataframe into List
        dfTemp['W'] = dfTemp['W'] + dfTemp['X']
        dfTemp['H'] = dfTemp['H'] + dfTemp['Y']

        dfTemp=dfTemp.values.tolist()
        #print(dfTemp)
        #print(final_bboxes)

        # This part should be modified
        #try:
        #    pr, rec, f1, Acc, Det, Grt = bboxprecisionrecall(final_bboxes, dfTemp, 0.2)
        #    print('Image: {}\tAccurate: {}\tTotal Detected: {}\tGround Truth: {}'.format(i, Acc, Det, Grt))
        #    print('Image: {}\tPrecision: {}\tRecall: {}\tF1: {}'.format(i, 100 * pr, 100 * rec, 100 * f1))
        #except:
        #    pr=0; rec=0; f1=0
            #print('Image: {}\tPrecision: {}\tRecall: {}\tF1: {}'.format(i, 100 * pr, 100 * rec, 100 * f1))

        # Evaluation of Dataset
        resultTemp = bboxprecisionrecall(dfTemp, final_bboxes, 0.2)
        #print(resultTemp)
        pr, rec, f1 = calc_precision_recall(resultTemp)

        print('Image: {}\tAccurate: {}\tTotal Detected: {}\tGround Truth: {}'.format(i, resultTemp['true_positive'], resultTemp['true_positive']+resultTemp['false_positive'], resultTemp['true_positive']+resultTemp['false_negative']))
        print('Image: {}\tPrecision: {}\tRecall: {}\tF1: {}'.format(i, 100 * pr, 100 * rec, 100 * f1))

       
        #ExpPR.append(100 * pr)
        #ExpRE.append(100 * rec)
        #ExpF1.append(100 * f1)
        ExpACC.append(resultTemp['true_positive'])
        ExpDET.append(resultTemp['true_positive'] + resultTemp['false_positive'])
        ExpGT.append(resultTemp['true_positive'] + resultTemp['false_negative'])

    # calculate average PR, RE, F1 for entire dataset
    ExpACCsum = sum(ExpACC)
    ExpDETsum = sum(ExpDET)
    ExpGTsum = sum(ExpGT)

    try:
        PRmean = 100 * ExpACCsum / ExpDETsum
        REmean = 100 * ExpACCsum / ExpGTsum

    except:
        PRmean=0; REmean=0

    try:
        F1mean = 2 * PRmean * REmean / (PRmean + REmean)
    except:
        F1mean = 0

    #ExpPRSum = sum(ExpPR)
    #ExpPRMean = ExpPRSum / len(ExpPR)

    #ExpRESum = sum(ExpRE)
    #ExpREMean = ExpRESum / len(ExpRE)

    #ExpF1Sum = sum(ExpF1)
    #ExpF1Mean = ExpF1Sum / len(ExpF1)

    #print('Average Precision: {}\tAverage Recall: {}\tAverage F1: {}'.format(ExpPRMean, ExpREMean, ExpF1Mean))
    print('Average Precision: {}\tAverage Recall: {}\tAverage F1: {}'.format(PRmean, REmean, F1mean))

    #return ExpPRMean, ExpREMean, ExpF1Mean
    return PRmean, REmean, F1mean

# Inference 평가를 위한 list detection
def detect_listInference(model, imgfolder, div, input_size, score_thresh, model_name=None, line_det=False, kernel=3, csv_path='./milestone/rgb_divInfer.csv',bandnumber=3):
    from cfg import Cfg
    from train import bboxprecisionrecall, calc_precision_recall
    import pandas as pd

    #bandnumber = Cfg.Satelliteband
    img_list = [x for x in os.listdir(imgfolder) if x.endswith('tif')]
    total_preds = []

    result_df = pd.DataFrame(columns=['image', 'X', 'Y', 'W', 'H'])

    df = pd.read_csv(csv_path)
    ExpACC=[]
    ExpDET=[]
    ExpGT=[]

    for i in img_list:
        final_bboxes, shore_bboxes = [], []
        detect_times = []

        img_path = imgfolder + i

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

        #print('Image: {}\tPrediction: {}\tFP: {}\tTime: {:.4f}sec'.format(i, len(final_bboxes), len(shore_bboxes),
        #                                                                  sum(detect_times)))

        # Import GT data corresponding to SAR images
        dfTempIdx=df.index[df['image'] == i].tolist()
        dfTemp=df.loc[dfTempIdx,:]
        dfTemp=dfTemp[['X','Y','W','H']]

        # convert Dataframe into List
        dfTemp['W'] = dfTemp['W'] + dfTemp['X']
        dfTemp['H'] = dfTemp['H'] + dfTemp['Y']

        dfTemp=dfTemp.values.tolist()
        #print(dfTemp)
        #print(final_bboxes)

        # This part should be modified
        #try:
        #    pr, rec, f1, Acc, Det, Grt = bboxprecisionrecall(final_bboxes, dfTemp, 0.2)
        #    print('Image: {}\tAccurate: {}\tTotal Detected: {}\tGround Truth: {}'.format(i, Acc, Det, Grt))
        #    print('Image: {}\tPrecision: {}\tRecall: {}\tF1: {}'.format(i, 100 * pr, 100 * rec, 100 * f1))
        #except:
        #    pr=0; rec=0; f1=0
            #print('Image: {}\tPrecision: {}\tRecall: {}\tF1: {}'.format(i, 100 * pr, 100 * rec, 100 * f1))

        # Evaluation of Dataset
        resultTemp = bboxprecisionrecall(dfTemp, final_bboxes, 0.2)
        #print(resultTemp)
        pr, rec, f1 = calc_precision_recall(resultTemp)



        ExpACC.append(resultTemp['true_positive'])
        ExpDET.append(resultTemp['true_positive']+resultTemp['false_positive'])
        ExpGT.append(resultTemp['true_positive']+resultTemp['false_negative'])

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

            print('Image: {}\tAccurate: {}\tTotal Detected: {}\tGround Truth: {}'.format(i, resultTemp['true_positive'],
                                                                                         resultTemp['true_positive'] +
                                                                                         resultTemp['false_positive'],
                                                                                         resultTemp['true_positive'] +
                                                                                         resultTemp['false_negative']))
            print('Image: {}\tPrecision: {}\tRecall: {}\tF1: {}'.format(i, 100 * pr, 100 * rec, 100 * f1))
            result_df = result_df.append(image_df, ignore_index=True)

        except:
            print('No Vessel Detected in This Scene!')
            image_df = pd.DataFrame(columns=['image', 'X', 'Y', 'W', 'H', 'Lon', 'Lat'])
            result_df = result_df.append(image_df, ignore_index=True)



    # calculate average PR, RE, F1 for entire dataset
    #ExpPRSum = sum(ExpPR)
    #ExpPRMean = ExpPRSum / len(ExpPR)

    #ExpRESum = sum(ExpRE)
    #ExpREMean = ExpRESum / len(ExpRE)

    #ExpF1Sum = sum(ExpF1)
    #ExpF1Mean = ExpF1Sum / len(ExpF1)

    ExpACCsum=sum(ExpACC)
    ExpDETsum=sum(ExpDET)
    ExpGTsum=sum(ExpGT)

    try:
        PRmean=100*ExpACCsum/ExpDETsum
        REmean = 100 * ExpACCsum / ExpGTsum
        F1mean = 2 * PRmean * REmean / (PRmean + REmean)
    except:
        PRmean=0
        REmean = 0
        F1mean = 0



    #print('Average Precision: {}\tAverage Recall: {}\tAverage F1: {}'.format(ExpPRMean, ExpREMean, ExpF1Mean))
    print('Average Precision: {}\tAverage Recall: {}\tAverage F1: {}'.format(PRmean, REmean, F1mean))

    # Final export of csv detection files
    csv_name = model_name.split('.')[0]
    csv_name = csv_name + '_div' + str(Cfg.division) + '_Maxrng' + str(
        Cfg.max1) + '_' + str(Cfg.max2) + '_' + str(Cfg.max3) + '_pre' + str(round(PRmean, 3)) + '_rec' + str(
        round(REmean, 3)) + '_f1s' + str(round(F1mean, 3)) + '.csv'

    #csv_name = csv_name.split('_')[1] + '_score{}_div{}_size{}_kernel{}.csv'.format(str(score_thresh).split('.')[1],
    #                                                                                str(div), str(input_size),
    #                                                                                str(kernel))

    result_df.to_csv(csv_name, index=False, encoding='utf-8')

    # Selective Final Post-processing: Remove wind mill
    OverlapEliminate(referenceinput='./milestone/DB_LL.csv',detectioninput=csv_name)

# Text file 형식으로 되어 있는 학습자료를 csv파일 형식으로 옮김
# Attention: file with 1 trainingData is unapprehendible
def concat_txt2csv_inference(importPath, exportPath='./milestone/rgb_divInfer.csv', typeNum=1):
    import glob
    import pandas as pd
    import numpy as np
    from cfg import Cfg

    txt_path = importPath
    txt_name = glob.glob(txt_path)

    # Massive array to concatenate all training data
    txt_data = np.empty((0, 4), int)
    txt_name_export = []

    for temp in txt_name:
        # Open assigned *.txt file
        filedata = np.loadtxt(temp, 'str', delimiter=',')

        # Should exclude images with 1 vessel
        # print(filedata)

        txt_name_temp = filedata[:, -1]
        filedata = filedata[:, :-1]

        # Replace the last row into file name: *.tif
        find_word = Cfg.Satellite
        pos = temp.find(find_word)

        temp_name = temp[pos:-4]
        temp_name = temp_name + '.tif'

        # Assign array full of filename
        txt_name_temp = [temp_name] * len(filedata)

        # Insert data into txt_data folder
        txt_data = np.concatenate((txt_data, filedata), axis=0)
        txt_name_export = txt_name_export + txt_name_temp

    # array export to rgb_train.csv
    df = pd.DataFrame(columns=['image', 'X', 'Y', 'W', 'H'])
    df['image'] = txt_name_export
    df['X'] = txt_data[:, 0]
    df['Y'] = txt_data[:, 1]
    df['W'] = txt_data[:, 2]
    df['H'] = txt_data[:, 3]

    df.to_csv(exportPath, index=False)


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

    # if len(lat)==1:
    #    y=float(y)

    # if len(lon)==1:
    #    x=float(x)

    return y, x


def OverlapEliminate(referenceinput='./DB_LL.csv', detectioninput='./Detection.csv'):
    import numpy as np
    import pandas as pd
    from cfg import Cfg
    import os
    import gdal_preprocess

    # Read vessel detection output (after inference)
    df_det = pd.read_csv(detectioninput)

    df_detX = df_det['X']
    df_detY = df_det['Y']
    df_detW = df_det['W']
    df_detH = df_det['H']
    df_detLon = df_det['Lon']
    df_detLat = df_det['Lat']
    df_Imagename = df_det['image']

    df_Imagename_unique = np.unique(df_Imagename)

    # Read WindTurbine Data
    df_ref = pd.read_csv(referenceinput)

    LatMin = df_ref['LatMin']
    LonMin = df_ref['LonMin']
    LatMax = df_ref['LatMax']
    LonMax = df_ref['LonMax']

    # Generate the np array who has identical size with df
    # DecisionArray=np.ones(len(df_detX))

    # Construct the array for DB
    DetectionScore = np.zeros(len(df_detX))
    print(len(DetectionScore))

    # Read each image data if possible
    for num in range(len(df_Imagename_unique)):

        inputName = os.path.join(Cfg.test_img_path, df_Imagename_unique[num])
        print(inputName)
        try:
            gt, rows, cols = geotiffreadRef(inputName)

        except:
            continue

        # Find the location which corresponds to the image name
        tifName = df_Imagename_unique[num]
        loc = df_Imagename.str.find(tifName)

        loc = loc.to_numpy()
        loc_exp = np.where(loc == 0)
        loc_exp = loc_exp[0]

        # Generate the np array who has identical size with df
        DecisionArray = np.zeros((len(LatMin), 4))

        # If importing Geotiff file succeeded ...
        for num1 in range(len(LatMin)):
            Yexp, Xexp = geographicToIntrinsic(gt, LatMax[num1], LonMin[num1])
            Yconf, Xconf = geographicToIntrinsic(gt, LatMin[num1], LonMax[num1])

            Yexp = int(Yexp)
            Xexp = int(Xexp)
            Yconf = int(Yconf)
            Xconf = int(Xconf)
            Wexp = abs(Xconf - Xexp)
            Hexp = abs(Yconf - Yexp)

            DecisionArray[num1, 0] = Xexp
            DecisionArray[num1, 1] = Yexp
            DecisionArray[num1, 2] = Wexp
            DecisionArray[num1, 3] = Hexp

        # Construct the array for DB
        # DetectionScore=np.zeros(len(df_detX))

        # Data comparison with DB
        for num1 in loc_exp:
            bboxDET = [int(df_detX[num1]), int(df_detY[num1]), int(df_detW[num1]), int(df_detH[num1])]
            bboxDET = np.array(bboxDET, dtype='float')

            for num2 in range(len(DecisionArray)):
                bboxAIS = DecisionArray[num2, :]

                iouscore = getIoU(bboxAIS, bboxDET)
                DetectionScore[num1] = DetectionScore[num1] + iouscore*100

    # Find where Detection Score Data is zero ONLY!
    loc_expFIN = np.where(DetectionScore == 0)
    loc_expFIN = loc_expFIN[0]

    print(len(loc_expFIN))

    # Export where Detection Score Data is zero ONLY!
    df_detX = df_detX[loc_expFIN]
    df_detY = df_detY[loc_expFIN]
    df_detW = df_detW[loc_expFIN]
    df_detH = df_detH[loc_expFIN]
    df_detLon = df_detLon[loc_expFIN]
    df_detLat = df_detLat[loc_expFIN]
    df_Imagename = df_Imagename[loc_expFIN]

    dfEx = pd.DataFrame(columns=['image', 'X', 'Y', 'W', 'H', 'Lon', 'Lat'])
    dfEx['X'] = df_detX
    dfEx['Y'] = df_detY
    dfEx['W'] = df_detW
    dfEx['H'] = df_detH
    dfEx['image'] = df_Imagename
    dfEx['Lon'] = df_detLon
    dfEx['Lat'] = df_detLat

    dfEx.to_csv(detectioninput, index=False)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-weightfile', type=str,
                        default='./checkpoints/K5_epoch53_div21_Maxrng150_200_250_pre62.921_rec66.986_f1s64.89.pth',
                        help='path of trained model.', dest='weightfile')
    parser.add_argument('-imgfolder', type=str,
                        default='/disk3/objdt/ship_Komp5/test_dataset_SLC/',
                        help='path of your image folder.', dest='imgfolder')
    parser.add_argument('-gtfolder', type=str,
                        default='/disk3/objdt/ship_Komp5/test_dataset_SLC/*.txt',
                        help='path of your ground truth folder.', dest='gtfolder')
    parser.add_argument('-division', type=int,
                        default=5,
                        help='division image', dest='division')
    parser.add_argument('-inputsize', type=int,
                        default=1024, #1024 1088
                        help='input image size', dest='inputsize')
    parser.add_argument('-scorethresh', type=int,
                        default=0.0382, #0.02 #0.2
                        help='confidence score threshold', dest='scorethresh')
    parser.add_argument('-bandnum', type=int,
                        default=1,
                        help='number of bands', dest='bandnum')
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    from cfg import Cfg
    args = get_args()
    
    weight_path = args.weightfile
    weight_name = weight_path.split('/')[-1]
    
    # Pytorch(.pth) model load
    model = Yolov4(yolov4conv137weight=None, n_classes=1, inference=True)
    pretrained_dict = torch.load(weight_path, map_location=torch.device('cuda'))
    model.load_state_dict(pretrained_dict)

    if use_cuda:
        model.cuda()

    concat_txt2csv_inference(importPath=args.gtfolder)

    detect_listInference(model, args.imgfolder,
                div=Cfg.division,
                input_size=Cfg.inputsize,
                score_thresh=Cfg.scorethresh,
                model_name=weight_name,
                line_det=False,
                kernel=3,
                bandnumber=Cfg.Satelliteband)

    #detect_list(model, args.imgfolder,
    #            div=args.division,
    #            input_size=args.inputsize,
    #            score_thresh=args.scorethresh,
    #            model_name=weight_name,
    #            draw=True,
    #            line_det=False,
    #            kernel=3,
    #            bandnumber=args.bandnum)
