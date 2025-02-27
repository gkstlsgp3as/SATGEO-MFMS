import numpy as np
from osgeo import gdal
import os
import cv2
import pandas as pd
import json
from tifffile import imwrite


def landmask(tif_name, vecPath):
    from osgeo import gdal
    from osgeo import ogr
    from PIL import Image
    
    ras_ds = gdal.Open(tif_name, gdal.GA_ReadOnly)
    gt = ras_ds.GetGeoTransform()

    vec_ds = ogr.Open(vecPath)  # landmask
    lyr = vec_ds.GetLayer()

    filename='../data/output/masked.tif'
    drv_tiff = gdal.GetDriverByName("GTiff") 
    chn_ras_ds = drv_tiff.Create(filename, ras_ds.RasterXSize, ras_ds.RasterYSize, 1, gdal.GDT_Float32)
    chn_ras_ds.SetGeoTransform(gt)

    gdal.RasterizeLayer(chn_ras_ds, [1], lyr) 
    chn_ras_ds.GetRasterBand(1).SetNoDataValue(0.0) 
    chn_ras_ds = None

    raster = gdal.Open(filename)
    band_data = np.array(raster.GetRasterBand(1).ReadAsArray())
    return np.array(band_data, np.float32)

'''
def landmask_tif(tif_path, dem_path):
    import rasterio
    import glob
    from rasterio.mask import mask

    # Open the smaller TIFF to use as a reference for cropping
    with rasterio.open(tif_path) as smaller:
        smaller_bounds = smaller.bounds

    # Open the larger TIFF which needs to be cropped
    with rasterio.open(dem_path) as larger:
        # Calculate the overlap of the larger TIFF on the smaller one
        # This can be done by creating a mask in the shape of the smaller TIFF's bounds
        # Convert the bounds into a GeoJSON feature (bbox polygon)
        geojson = [{
            'type': 'Polygon',
            'coordinates': [[
                [smaller_bounds.left, smaller_bounds.bottom],
                [smaller_bounds.left, smaller_bounds.top],
                [smaller_bounds.right, smaller_bounds.top],
                [smaller_bounds.right, smaller_bounds.bottom],
                [smaller_bounds.left, smaller_bounds.bottom]
            ]]
        }]

        # Mask the larger TIFF to get only the area overlapping with the smaller TIFF
        out_image, out_transform = mask(larger, geojson, crop=True)
        out_meta = larger.meta.copy()

        # Update the metadata to match the new dimensions
        out_meta.update({
            "driver": "GTiff",
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform
        })

        # Write the cropped area to a new TIFF
        with rasterio.open('../data/output/masked.tif', 'w', **out_meta) as dest:
            dest.write(out_image)

    # Optionally, display the merged image
    merged_raster = cv2.imread('../data/output/masked.tif', cv2.IMREAD_UNCHANGED)
    return np.array(merged_raster, np.float32)
'''

def landmask_tif(tif_path, dem_path):
    import rasterio
    from rasterio.windows import from_bounds
    from rasterio.enums import Resampling
    
    # Open the small reference TIFF to get its bounds, transform, and dimensions
    with rasterio.open(tif_path) as small_tif:
        small_bounds = small_tif.bounds
        small_transform = small_tif.transform
        small_width = small_tif.width
        small_height = small_tif.height

    # Open the big TIFF to be cropped and/or padded
    with rasterio.open(dem_path) as big_tif:
        # Calculate the overlap window
        window = from_bounds(
            left=small_bounds.left,
            bottom=small_bounds.bottom,
            right=small_bounds.right,
            top=small_bounds.top,
            transform=big_tif.transform
        )

        # Read the data from the overlap window with padding if necessary
        big_data = big_tif.read(window=window, boundless=True, fill_value=0,
                                out_shape=(big_tif.count, small_height, small_width),
                                resampling=Resampling.nearest)

        # Define new metadata for the output TIFF
        out_meta = big_tif.meta.copy()
        out_meta.update({
            "driver": "GTiff",
            "height": small_height,
            "width": small_width,
            "transform": small_transform,
            "dtype": 'float32'
        })

        # Write the cropped (and padded if necessary) data to a new TIFF
        with rasterio.open('../data/output/masked.tif', 'w', **out_meta) as out_tif:
            out_tif.write(big_data)
            
    out_tif = cv2.imread('../data/output/masked.tif', cv2.IMREAD_UNCHANGED)
    return np.array(out_tif, np.float32)


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


def division_testset(input_band=None, img_size=640):#, landmask=None):
    from utils.cfg import Cfg
    img_list, div_coord = [], []
    
    # 분할 구간 설정
    h, w = input_band.shape[:2]

    hd = [x for x in range(0, h, img_size-200)]
    wd = [x for x in range(0, w, img_size-200)]
    hd[-1] = h - img_size; wd[-1] = w - img_size
    hd.sort(); wd.sort()
    for h_id, div_h in enumerate(hd[:-1]):
        for w_id, div_w in enumerate(wd[:-1]):
            # 분할된 이미지의 좌표
            x1, y1 = div_w, div_h
            x2, y2 = div_w+img_size, div_h+img_size

            dw = x2-x1; dh = y2-y1
            # Crop
            img = input_band[y1:y2, x1:x2]
            #mask = landmask[y1:y2, x1:x2]>0
            if img.mean() == 0: continue
            #if img.mean()*(1-mask).mean() == 0: continue
            #if mask.shape != img.shape:
            #    temp = np.zeros((640, 640))
            #    temp[:mask.shape[0], :mask.shape[1]] = mask
            #    mask = temp
            #mask = np.repeat(np.expand_dims(mask, 2), 3, 2)
            #img = img*(1-mask)

            if Cfg.img_mode != 'org':
                if Cfg.img_mode == 'vh+vv':
                    newimg = img[...,1]+img[...,2]
                elif Cfg.img_mode == 'grayscale':
                    newimg = 0.229*img[...,0]+0.587*img[...,1]+0.114*img[...,2]
                elif Cfg.img_mode == 'vv^2+vh^2':
                    newimg = img[...,1]**2 +img[...,2]**2
                elif Cfg.img_mode == 'vv*vh':
                    newimg = img[...,0]*img[...,1]
                img = np.dstack((newimg, newimg, newimg))
                if img.max() != 0:
                    img = (img - img.min()) / (img.max() - img.min())
            
            
            img_list.append(img)
            div_coord.append([dw, dh, div_w, div_h])

            #save_name = str(x1) + '_' + str(y1) + '_' + str(x2) + '_' + str(y2) + '_' + 'test.tif'
        
            #imwrite(os.path.join('./data/test', save_name),crop)

    return img_list, div_coord    
       
       
# band 1 ~ 3을 0 ~ 255 값을 갖는 rgb로 변환
def band_to_input(tif_path,bandnumber,partest=False):
    from utils.cfg import Cfg
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

            max_num = Cfg.max[i]
            min_num = Cfg.min[i]

            # fill nan with neighbors
            mask = np.isnan(band_data)
            band_data[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), band_data[~mask])
            band_data[band_data > max_num] = max_num
            band_data[band_data < min_num] = min_num

            band_data = band_data * ((1 - min_num) / (max_num - min_num))
            #band_data = band_data * ((255 - min_num)/ (max_num - min_num))

            bands.append(band_data)

         # band 1, 2, 3을 RGB로 변환
        rgb = np.dstack((bands[2], bands[1], bands[0]))

    # transformation of single-banded SAR image
    elif bandnumber==1:
        max_num = Cfg.max[0]
        min_num = Cfg.min[0]

        band_data1 = np.array(raster.GetRasterBand(1).ReadAsArray())
        # max_num = np.quantile(band_data1, 0.9, axis=None)
        # band_data1=band_data1/0.8191
        band_data1[band_data1 > max_num] = max_num
        band_data1[band_data1 < min_num] = min_num
        band_data1 = band_data1 * ((1 - min_num) / (max_num - min_num))

        rgb = np.zeros((band_data1.shape[0], band_data1.shape[1], 3))
        rgb = np.dstack((band_data1, band_data1, band_data1))
        #rgb = np.array(rgb, np.uint8)

    # transformation of double-banded SAR image(B1,B2,B2)
    elif bandnumber==2:
        max_num = Cfg.max[0]
        min_num = Cfg.min[0]

        band_data1 = np.array(raster.GetRasterBand(1).ReadAsArray())
        # max_num = np.quantile(band_data1, 0.9, axis=None)
        # band_data1=band_data1/0.8191
        band_data1[band_data1 > max_num] = max_num
        band_data1[band_data1 < min_num] = min_num
        band_data1 = band_data1 * ((1 - min_num) / (max_num - min_num))

        rgb = np.zeros((band_data1.shape[0], band_data1.shape[1], 3))
        rgb[:, :, 0] = band_data1

        # For Band2(Min/Max)
        max_num = Cfg.max[1]
        min_num = Cfg.min[1]

        band_data2 = np.array(raster.GetRasterBand(2).ReadAsArray())
        # max_num = np.quantile(band_data2, 0.9, axis=None)
        # band_data2 = band_data2 / 0.8191
        band_data2[band_data2 > max_num] = max_num
        band_data2[band_data2 < min_num] = min_num
        band_data2 = band_data2 * ((1 - min_num) / (max_num - min_num))

        rgb[:, :, 1] = band_data2
        rgb[:, :, 2] = band_data2

    elif bandnumber > 3:
        for bn in range(bandnumber):
            max_num = Cfg.max[bn]
            min_num = Cfg.min[bn]

            band_data = np.array(raster.GetRasterBand(bn+1).ReadAsArray())
            # max_num = np.quantile(band_data1, 0.9, axis=None)
            # band_data1=band_data1/0.8191
            band_data.clip(min_num, max_num)
            band_data = band_data * ((1 - min_num) / (max_num - min_num))

            rgb = np.zeros((band_data.shape[0], band_data.shape[1], bandnumber))
            rgb[:, :, bn] = band_data

    return rgb


# 외각 라인 검출(육지를 제거하기 위해)
def line_detection(input_array):
    input_image = np.array(input_array*255, np.uint8)
    # Image.fromarray(input_image*255).save('./milestone/line/grey_10560_0_11200_640_9D5A.png')

    # 비교적 잡음이 적은 band 1 영상에 대해 수행
    gray_image = input_image[:,:,2]
    # Image.fromarray(gray_image).save('./milestone/line/gray_'+save_name.replace('tif','png'))

    blur_image = cv2.medianBlur(gray_image, 5) 
    # Image.fromarray(blur_image).save('./milestone/line/blur_'+save_name.replace('tif','png'))


    # band 1 침식과정을 통해 흰색 노이즈 제거
    erode_image = cv2.erode(blur_image, (3,3), iterations=1)
    # Image.fromarray(erode_image).save('./milestone/line/erode_'+save_name.replace('tif','png'))

    # threshhold
    thr = 15
    ret, thresh = cv2.threshold(erode_image, thr, 255, 0)
    # Image.fromarray(thresh).save('./milestone/line/thres_'+save_name.replace('tif','png'))

    # 육지 정보를 저장할 이미지
    #line_filter = np.zeros(input_array.shape[:2], np.uint8)
    
    # 외각 라인 검출
    #try:
    #    ext_contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #except:
    #    _, ext_contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    #for c in ext_contours:
        # 각 라인들의 면적
    #    area = cv2.contourArea(c)
        # 면적이 600 이상일 경우 육지로 판단하고 해당 위치의 픽셀값을 1로
        # 600 미만일 경우 0
    #    if area >= 1:
    #        line_filter = cv2.drawContours(line_filter, [c], -1, 1, -1)
    
    return thresh

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
    result_shape=tuple(np.int64(np.array(img_shape)-np.array(filter_size))/stride+1)

    result=np.zeros(result_shape)
    for h in range(0,result_shape[0],stride):
        for w in range(0,result_shape[1],stride):
            tmp=img[h:h+filter_size[0],w:w+filter_size[1]]
            tmp=np.sort(tmp.ravel())
            result[h,w]=tmp[int(filter_size[0]*filter_size[1]/2)]

    return result

