import numpy as np
from osgeo import gdal
import os
import cv2
import pandas as pd
import json

# labelme로 레이블링한 어노테이션 정보를 ai factory 양식에 맞게 변환
def get_labelme_annotation():
    final_df = pd.DataFrame(columns=['image', 'X', 'Y', 'W', 'H'])
    json_list = [x for x in os.listdir('./dataset/train_dataset/') if x.endswith('json')]

    for json_path in json_list:
        image_name = json_path.split('.')[0] + '.tif'
        
        with open('./train_rgb_dataset/'+json_path) as j:
            json_data = json.load(j)
        bboxes = []
        for i in json_data['shapes']:
            box_info = i['points']
            if len(box_info) != 2:
                continue
            
            x1, y1 = box_info[0]
            x2, y2 = box_info[1]

            if x1 >= x2:
                tmp_x = x1
                x1 = x2
                x2 = tmp_x

            box = [int(x) for x in [x1, y1, x2, y2]]

            bboxes.append(box)

        # to csv
        arr_bboxes = np.array(bboxes)
        df = pd.DataFrame(columns=['image', 'X', 'Y', 'W', 'H'])
        df['image'] = [image_name for x in range(len(arr_bboxes))]
        df['X'] = arr_bboxes[:, 0]
        df['Y'] = arr_bboxes[:, 1]
        df['W'] = arr_bboxes[:, 2] - arr_bboxes[:, 0]
        df['H'] = arr_bboxes[:, 3] - arr_bboxes[:, 1]

        final_df = final_df.append(df, ignore_index=True)
        # print(final_df.shape)

    final_df.to_csv('./rgb_train.csv', index=False)

# Text file 형식으로 되어 있는 학습자료를 csv파일 형식으로 옮김
# Attention: file with 1 trainingData is unapprehendible
def concat_txt2csv(exportPath='./milestone/rgb_train_pre.csv', typeNum=1):
    import glob
    import pandas as pd
    import numpy as np
    from cfg import Cfg

    if typeNum==1:
        txt_path = Cfg.train_txt_path
    else:
        txt_path = Cfg.test_txt_path

    txt_name = glob.glob(txt_path)

    # Massive array to concatenate all training data
    txt_data = np.empty((0, 4), int)
    txt_name_export = []

    for temp in txt_name:
        try:
            # Open assigned *.txt file
            #filedata = np.loadtxt(temp, 'str', delimiter=',')
            filedata = np.loadtxt(temp,dtype='str', delimiter=',')

            # Should exclude images with 1 vessel
            #print(filedata)

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
        except:
            aaaaa=1


    # array export to rgb_train.csv
    df = pd.DataFrame(columns=['image', 'X', 'Y', 'W', 'H'])
    df['image'] = txt_name_export
    df['X'] = txt_data[:, 0]
    df['Y'] = txt_data[:, 1]
    df['W'] = txt_data[:, 2]
    df['H'] = txt_data[:, 3]

    df.to_csv(exportPath, index=False)
    
# 학습자료 중 잘못 취득되고 있는것 제거(ocean)
def confirm_trainset(inputpath='./milestone/rgb_train_pre.csv', outputpath='./milestone/rgb_train.csv', bandnumber=3, threshold=0.5):

    import numpy as np
    import pandas as pd
    from cfg import Cfg
    import os

    # import images for training
    imagePath = Cfg.train_img_path
    bandnumber = Cfg.Satelliteband

    # import generated csv file in concat_txt2csv
    df = pd.read_csv(inputpath)
    #imagelist = pd.Series.tolist(df['image'])
    imagelist = df['image']
    imagelist_Unique = imagelist.unique()

    X = np.array(df['X'])
    Y = np.array(df['Y'])
    W = np.array(df['W'])
    H = np.array(df['H'])

    # iteration for each training images
    for num in range(len(imagelist_Unique)):

        # Image list order
        imageTempName = imagelist_Unique[num]
        imagelistAns = (imagelist == imageTempName)


        imagelistOrder = imagelist[imagelistAns]
        imagelistOrder = np.array(imagelistOrder.index)
        #print(imagelistOrder)

        TempX = X[imagelistOrder]
        TempY = Y[imagelistOrder]
        TempW = W[imagelistOrder]
        TempH = H[imagelistOrder]


        # open Geotiff file
        imageTempFullName = os.path.join(imagePath, imageTempName)

        if bandnumber==3:
            tif_data, tif_ref = geotiffread(imageTempFullName, bandnumber)
            mean_tif = np.quantile(tif_data[:, :, 0], threshold)
        elif bandnumber==1:
            tif_data, tif_ref = geotiffread(imageTempFullName, bandnumber)
            mean_tif = np.quantile(tif_data, threshold)
            #mean_tif = 0

        # sort out trainingData bboxes in ocean region
        for num1 in range(len(TempX)):

            # Sort out the value demonstrating low value than entire average
            if bandnumber == 3:
                tif_data_crop = tif_data[TempY[num1]:TempY[num1] + TempH[num1], TempX[num1]:TempX[num1] + TempW[num1], 0]

            elif bandnumber==1:
                tif_data_crop = tif_data[TempY[num1]:TempY[num1] + TempH[num1], TempX[num1]:TempX[num1] + TempW[num1]]


            if np.mean(tif_data_crop) < mean_tif:
                X[imagelistOrder[num1]] = -1

    LocSave = np.squeeze(np.array(np.where(X > 0)).T)

    # Conclusive trainingData export as csv file
    X = X[LocSave]
    Y = Y[LocSave]
    W = W[LocSave]
    H = H[LocSave]

    imagelist_exp = imagelist[LocSave]
    imagelist_exp = pd.Series.tolist(imagelist_exp)

    dfEx = pd.DataFrame(columns=['image', 'X', 'Y', 'W', 'H'])
    dfEx['X'] = X
    dfEx['Y'] = Y
    dfEx['W'] = W
    dfEx['H'] = H
    dfEx['image'] = imagelist_exp

    dfEx.to_csv(outputpath, index=False)
    
# 이미지를 분할하고 각 이미지의 bbox 정보를 분활된 이미지에 맞게 변환
def division_trainset(csv_path='./milestone/rgb_train.csv', div_num=40, typeNum=1):
    from cfg import Cfg
    import pandas as pd

    #origin_image_folder = './dataset/train_dataset/'

    df = pd.read_csv(csv_path)
    image_list = list(df['image'].unique())

    # typeNum=1(training), other(test)
    if typeNum==1:
        origin_image_folder = Cfg.train_img_path
        f = open('./data/div_{}.txt'.format(str(div_num)), 'w')
    else:
        origin_image_folder = Cfg.test_img_path
        f = open('./dataTest/div_test_{}.txt'.format(str(div_num)), 'w')

    for i in image_list:
        last_name = i.split('_')[-1]
        image_path = os.path.join(origin_image_folder, i)

        bandnumber = Cfg.Satelliteband

        # RGB로 변환
        if Cfg.NewTest == 0:
            rgb_image = band_to_rgb(image_path, bandnumber)
        else:
            rgb_image = band_to_rgb(image_path, bandnumber,True)
        
        # x, y, w, h를 x1, y1, x2, y2로 변환
        image_df = df[df['image'] == i].reset_index(drop=True)
        image_df['X2'] = image_df['X'] + image_df['W']
        image_df['Y2'] = image_df['Y'] + image_df['H']
        bboxes = image_df[['X', 'Y', 'X2', 'Y2']].to_numpy()

        # 분할 구간 설정
        h, w = rgb_image.shape[:2]

        hd = [x for x in range(0, h, int(h / div_num))]
        wd = [x for x in range(0, w, int(w / div_num))]
        hd[-1] += h - hd[-1]
        wd[-1] += w - wd[-1]

        for h_id, div_h in enumerate(hd[:-1]):
            for w_id, div_w in enumerate(wd[:-1]):
                # 분할된 이미지의 좌표
                x1, y1 = div_w, div_h
                x2, y2 = wd[w_id+1], hd[h_id+1]

                # 이미지 크롭
                crop = rgb_image[y1:y2, x1:x2]

                div_boxes = []
                save_name = str(x1) + '_' + str(y1) + '_' + str(x2) + '_' + str(y2) + '_' + last_name
                line = save_name
                for b in bboxes:
                    # 현재 분할된 이미지의 x, y 구간
                    if (x1 <= b[0] <= x2) and (y1 <= b[1] <= y2):
                        # 원본 bbox 좌표를 분할된 이미지 좌표로 변환
                        dx1 = b[0] - x1
                        dy1 = b[1] - y1
                        dx2 = b[2] - x1
                        dy2 = b[3] - y1
                        div_boxes.append([dx1, dy1, dx2, dy2])

                        # crop = cv2.rectangle(crop, (dx1, dy1), (dx2, dy2), (0,0,255), 1)
                        if typeNum == 1:
                            save_path = './data/div_{}_train'.format(str(div_num))
                        else:
                            save_path = './dataTest/div_test_{}_train'.format(str(div_num))

                        os.makedirs(save_path, exist_ok=True)
                        # cv2.imwrite(save_path + '/' + save_name, crop)
                        cv2.imwrite(os.path.join(save_path, save_name), crop)
                        
                        
                if len(div_boxes) > 0:
                    for d in div_boxes:
                        strd = [str(x) for x in d]
                        one_box = strd[0] + ',' + strd[1] + ',' + strd[2] + ',' + strd[3] + ',' + str(0)
                        line += ' ' + one_box
                    f.write(line+'\n')



# 학습자료를 분할(SAR image와 TrainingDataset 전부)
def division_testset(input_band=None, div_num=20):
    '''
    # 미완성 Sliding Window Crop
    hw = input_band.shape[:2]
    overlab = 5

    xy_coords = []
    for i in hw:
        coords = []
        d = int(i / div_num)
        cnt = 0
        while True:
            if cnt == 0:
                coord = (0, d)
                coords.append(coord)
            else:
                # last coord x
                c1 = coords[-1][1] - overlab
                c2 = c1 + d
                coord = (c1, c2)
                coords.append(coord)
            
            if coords[-1][1] > i - d:
                coords[-1] = (coords[-1][0], i)
                break
            cnt += 1

        xy_coords.append(coords)
    
    hd = xy_coords[0]
    wd = xy_coords[1]
    img_list, div_coord = [], []
    for y in hd:
        for x in wd:
            # Div position
            x1, y1 = x[0], y[0]
            x2, y2 = x[1], y[1]
            # Crop
            crop = input_band[y1:y2, x1:x2]
            img_list.append(crop)
            div_coord.append([x1,y1])

    return img_list, div_coord
    '''

    img_list, div_coord = [], []

    h, w = input_band.shape[:2]
    div_width = int(w / div_num)
    div_height = int(h / div_num)
    hd = [x for x in range(0, h, div_height)]
    wd = [x for x in range(0, w, div_width)]
    
    hd[-1] += h - hd[-1]
    wd[-1] += w - wd[-1]
    
    for h_id, div_h in enumerate(hd[:-1]):
        for w_id, div_w in enumerate(wd[:-1]):
            # Div position
            x1, y1 = div_w, div_h
            x2, y2 = wd[w_id+1], hd[h_id+1]
            # Crop
            crop = input_band[y1:y2, x1:x2]
            img_list.append(crop)
            div_coord.append([div_w, div_h])

    return img_list, div_coord
    
# band 1 ~ 3을 0 ~ 255 값을 갖는 rgb로 변환
def band_to_rgb(tif_path,bandnumber,partest=False):
    from cfg import Cfg
    import numpy as np
    from sklearn import preprocessing

    raster = gdal.Open(tif_path)

    # transformation of 3-banded SAR image
    if bandnumber==3:
        bands = []
        for i in range(raster.RasterCount):
            band = raster.GetRasterBand(i+1)
            meta = band.GetMetadata()
            if band.GetMinimum() is None or band.GetMaximum()is None:
                band.ComputeStatistics(0)

            band_data = np.array(raster.GetRasterBand(i+1).ReadAsArray())

            if i == 0:
                # band 1의 최대값을 0.1로
                max_num = Cfg.max1
                min_num = Cfg.min1

            elif i == 1:
                # band 2의 최대값을 0.5로
                max_num = Cfg.max2
                min_num = Cfg.min2

            else:
                # band 3의 최대값을 1.0으로
                max_num = Cfg.max3
                min_num = Cfg.min3

                # GRDH DPIVD measurement
                if partest == True:

                    DoP=np.divide(Svh,Svv)
                    p1=np.divide(Svv,Svh+Svv)
                    SEiN=10*np.multiply(Svh,Svv)
                    #SEiN2=preprocessing.normalize(np.power(SEiN,2))

                    #band_data=np.multiply(DoP,np.multiply(p1,SEiN2))
                    band_data = np.multiply(DoP, np.multiply(p1, SEiN))
                    #band_data = band_data + 30

                    # Remove the data as 0 where infinite, NaN or negative
                    band_data[np.isinf(band_data)]=0
                    band_data[np.where(band_data<0)]=0
                    band_data[np.isnan(band_data)] = 0


                    #band_data=10*np.log10(band_data)

                    max_num = np.quantile(band_data,0.98)
                    min_num = 0
                    #min_num = np.quantile(band_data,0.05)

                    #print(np.max(band_data))


            # 0 ~ 255 이미지로 변환
            if partest == True:
                if i == 0:
                    Svh = band_data
                elif i == 1:
                    Svv = band_data

            band_data[band_data > max_num] = max_num
            band_data[band_data < min_num] = min_num
            band_data = band_data * (255./max_num)
            #band_data = band_data * ((255 - min_num)/ (max_num - min_num))

            bands.append(band_data)

         # band 1, 2, 3을 RGB로 변환
        rgb = np.dstack((bands[2], bands[1], bands[0]))
        #rgb = np.array(rgb)
        rgb = np.array(rgb, np.uint8)

    # transformation of single-banded SAR image
    elif bandnumber==1:
        max_num = Cfg.max1
        min_num = Cfg.min1

        band_data1 = np.array(raster.GetRasterBand(1).ReadAsArray())
        # max_num = np.quantile(band_data1, 0.9, axis=None)
        # band_data1=band_data1/0.8191
        band_data1[band_data1 > max_num] = max_num
        band_data1[band_data1 < min_num] = min_num
        band_data1 = band_data1 * (255. / max_num)

        rgb = np.zeros((band_data1.shape[0], band_data1.shape[1], 3))
        rgb[:, :, 0] = band_data1

        # For Band2(Min/Max)
        max_num = Cfg.max2
        min_num = Cfg.min2

        band_data2 = np.array(raster.GetRasterBand(1).ReadAsArray())
        # max_num = np.quantile(band_data2, 0.9, axis=None)
        # band_data2 = band_data2 / 0.8191
        band_data2[band_data2 > max_num] = max_num
        band_data2[band_data2 < min_num] = min_num
        band_data2 = band_data2 * (255. / max_num)

        rgb[:, :, 1] = band_data2

        # For Band3(Min/Max)
        max_num = Cfg.max3
        min_num = Cfg.min3

        band_data3 = np.array(raster.GetRasterBand(1).ReadAsArray())
        # max_num = np.quantile(band_data3, 0.9, axis=None)
        # band_data3 = band_data3 / 0.8191
        band_data3[band_data3 > max_num] = max_num
        band_data3[band_data3 < min_num] = min_num
        band_data3 = band_data3 * (255. / max_num)

        rgb[:, :, 2] = band_data3

        #rgb = np.dstack((rgb[:,:,2], rgb[:,:,1], rgb[:,:,0]))
        #rgb = np.array(rgb)
        rgb = np.array(rgb, np.uint8)


    # transformation of double-banded SAR image(B1,B2,B2)
    elif bandnumber==2:
        max_num = Cfg.max1
        min_num = Cfg.min1

        band_data1 = np.array(raster.GetRasterBand(1).ReadAsArray())
        # max_num = np.quantile(band_data1, 0.9, axis=None)
        # band_data1=band_data1/0.8191
        band_data1[band_data1 > max_num] = max_num
        band_data1[band_data1 < min_num] = min_num
        band_data1 = band_data1 * ((255 - min_num) / (max_num - min_num))

        rgb = np.zeros((band_data1.shape[0], band_data1.shape[1], 3))
        rgb[:, :, 0] = band_data1

        # For Band2(Min/Max)
        max_num = Cfg.max2
        min_num = Cfg.min2

        band_data2 = np.array(raster.GetRasterBand(2).ReadAsArray())
        # max_num = np.quantile(band_data2, 0.9, axis=None)
        # band_data2 = band_data2 / 0.8191
        band_data2[band_data2 > max_num] = max_num
        band_data2[band_data2 < min_num] = min_num
        band_data2 = band_data2 * ((255 - min_num) / (max_num - min_num))

        rgb[:, :, 1] = band_data2

        # For Band3(Min/Max)
        max_num = Cfg.max3
        min_num = Cfg.min3

        band_data3 = np.array(raster.GetRasterBand(2).ReadAsArray())
        # max_num = np.quantile(band_data3, 0.9, axis=None)
        # band_data3 = band_data3 / 0.8191
        band_data3[band_data3 > max_num] = max_num
        band_data3[band_data3 < min_num] = min_num
        band_data3 = band_data3 * ((255 - min_num)/(max_num - min_num))


        rgb[:, :, 2] = band_data3

        rgb = np.array(rgb, np.uint8)

    return rgb


# 외각 라인 검출(육지를 제거하기 위해)
def line_detection(input_array):
    input_image = np.array(input_array, np.uint8)
    
    # 비교적 잡음이 적은 band 1 영상에 대해 수행
    gray_image = input_image[:,:,2]
    blur_image = cv2.medianBlur(gray_image, 5) 

    # band 1 침식과정을 통해 흰색 노이즈 제거
    erode_image = cv2.erode(blur_image, (3,3), iterations=5)

    # threshhold
    thr = 35
    ret, thresh = cv2.threshold(erode_image, thr, 255, 0)

    # 육지 정보를 저장할 이미지
    line_filter = np.zeros(input_array.shape[:2], np.uint8)
    
    # 외각 라인 검출
    try:
        ext_contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    except:
        _, ext_contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for c in ext_contours:
        # 각 라인들의 면적
        area = cv2.contourArea(c)
        # 면적이 600 이상일 경우 육지로 판단하고 해당 위치의 픽셀값을 1로
        # 600 미만일 경우 0
        if area >= 600:
            line_filter = cv2.drawContours(line_filter, [c], -1, 1, -1)
    
    return line_filter

# Corresponding function to geotiffread of MATLAB
def geotiffread(tif_name, num_band):
    import gdal
    import numpy as np

    gdal.AllRegister()

    ds = gdal.Open(tif_name)

    if num_band == 3:
        band1 = ds.GetRasterBand(1)
        arr1 = band1.ReadAsArray()
        band2 = ds.GetRasterBand(2)
        arr2 = band2.ReadAsArray()
        band3 = ds.GetRasterBand(3)
        arr3 = band3.ReadAsArray()

        cols, rows = arr1.shape

        arr = np.zeros((cols, rows, 3))
        arr[:, :, 0] = arr1
        arr[:, :, 1] = arr2
        arr[:, :, 2] = arr3

    elif num_band == 1:
        band1 = ds.GetRasterBand(1)
        arr = band1.ReadAsArray()

        cols, rows = arr.shape


    else:
        print('cannot open except number of band is 1 or 3')

    gt = ds.GetGeoTransform()
    gt = np.array(gt)
    gt[2] = gt[0] + gt[1] * (rows - 1)
    gt[4] = gt[3] + gt[5] * (cols - 1)

    gt.astype(np.double)

    return arr, gt

# Median filtering on Oversampled image(Especially K5 0.3m)
def median_filter(img, filter_size=(5,5), stride=1):
    import numpy as np
    from PIL import Image
    import matplotlib.pyplot as plt

    img_shape = np.shape
    result_shape=tuple(np.int64(np.array(image_shape)-np.array(filter_size))/stride+1)

    result=np.zeros(result_shape)
    for h in range(0,result_shape[0],stride):
        for w in range(0,result_shape[1],stride):
            tmp=img[h:h+filter_size[0],w:w+filter_size[1]]
            tmp=np.sort(tmp.ravel())
            result[h,w]=tmp[int(filter_size[0]*filter_size[1]/2)]

    return result

if __name__ == '__main__':
    # get_labelme_annotation()
    division_trainset(div_num=40)
   


    

