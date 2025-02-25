import json
import time
import math

import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import shape
from typing import Tuple
import argparse
from scipy.stats import f
from numpy import ma
from osgeo import gdal, osr
from utils.sarpy.acd_sglr import sglr
from scipy.ndimage import gaussian_filter
from mpl_toolkits.axes_grid1 import make_axes_locatable
import rasterio
from utils.sarpy import s1_load
from utils.sarpy.acd_sglr import vincentyDistance, applyFilterBulbSize


def AmplitudeCDchip(input_grd_before_file, input_grd_after_file, output_file):
    import time
    from utils.sarpy.acd_sglr import sglr 
    from rasterio.transform import from_origin
    
    startT = time.time()
    
    beforeSAR,beforeSAR_gt = geotiffread_rasterio(input_grd_before_file)
    afterSAR,afterSAR_gt = geotiffread_rasterio(input_grd_after_file)
    
    # Assign initial parameters
    ratio = beforeSAR[:,:,1] / afterSAR[:,:,1]
    m = 4.4
    #m = 1
    dfn = np.nanmean(ratio) * m * 20
    dfd = np.nanmean(ratio) * m * 4
    
    dt = f.ppf(0.001, dfn * m, dfd * m) / 10
    print(f'p-value threshold : {dt}')
    
    # Perform p-value estimation
    m2logQ = sglr.calM2logQchip(beforeSAR, afterSAR, m=4.4)      
    #m2logQ = m2logQ - np.min(m2logQ)
    p_value = 1 - sglr.chi2Cdf(m2logQ, 50)
    p_value = sglr.chi2Cdf(m2logQ, 50)
    #p_value = sglr.chi2Cdf(m2logQ, 2)
    
    print(ratio)
    print(m2logQ)
    print(p_value)

    c_map = np.zeros_like(p_value)
    c_map[p_value < dt] = 1
    cmap_filteres = applyFilterBulbSize(c_map, 1, 30)
    
    # Extract changed region
    mask_overlay = ma.masked_where(cmap_filteres == 0, cmap_filteres)
    nrows, ncols, nband = beforeSAR.shape
    
    p_value_num = p_value
    p_value_num[np.isnan(p_value_num)] = 0
    p_value_num = gaussian_filter(p_value_num, sigma=5)
    
    # Export as geotiff    
    try:
        # geotiffwrite in rasterio
        from rasterio.transform import from_origin
        transform = from_origin(beforeSAR_gt[0], beforeSAR_gt[1], beforeSAR_gt[1], -beforeSAR_gt[5])
        crs = 'EPSG:4326'
        
        with rasterio.open(
            output_file, 'w',
            driver='GTiff',
            height=beforeSAR.shape[0],
            width=beforeSAR.shape[1],
            count=3,  # number of layers/bands
            dtype=beforeSAR.dtype,
            crs=crs,
            transform=transform
        ) as dst:                           
            dst.write(p_value_num, 1)  
            dst.write(m2logQ, 2)  
            dst.write(mask_overlay, 3)  
    
        print('============================================================')
        print(time.time()-startT,'sec')
        
    
    except:
        # GDAL을 사용하여 GeoTIFF 파일 생성
        driver = gdal.GetDriverByName('GTiff')
        dataset = driver.Create('output.tif', ncols, nrows, 1, gdal.GDT_Float32)

        # 변환 매트릭스 설정
        geotransform = (beforeSAR_gt[0], beforeSAR_gt[1], 0, beforeSAR_gt[4], 0, beforeSAR_gt[5])
        dataset.SetGeoTransform(geotransform)

        # CRS 설정 (WGS 84 - EPSG:4326)
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(4326)
        dataset.SetProjection(srs.ExportToWkt())
        
        # 데이터 작성
        dataset.GetRasterBand(1).WriteArray(p_value_num)
        
        # 파일 닫기 및 저장
        dataset.FlushCache()
        dataset = None
        print('============================================================')
        print(time.time()-startT,'sec')
      
def geotiffread_rasterio(tif_name):
    import rasterio
    import numpy as np
    
    with rasterio.open(tif_name) as dataset:
        #print(dataset.transform)
                
        # Read geotiff file
        band1=dataset.read(1)
        band2=dataset.read(2)

        cols, rows = band1.shape        
        arr = np.zeros((cols, rows, 2))
        arr[:, :, 0] = band1
        arr[:, :, 1] = band2
        
        # Read geotiff metadata
        gt_transform = dataset.transform 
        gt=np.zeros((6,1))
        
        gt[0]=gt_transform.c; gt[3]=gt_transform.f
        gt[1]=gt_transform.a; gt[5]=gt_transform.e
        gt[2] = gt[0] + gt[1] * (rows - 1)
        gt[4] = gt[3] + gt[5] * (cols - 1)
        
        gt.astype(np.double)
        
        return arr, gt
    


def geotiffread_S1(tif_name):
    import gdal
    import numpy as np

    gdal.AllRegister()

    ds = gdal.Open(tif_name)    

    band1 = ds.GetRasterBand(1)
    arr1 = band1.ReadAsArray()
    band2 = ds.GetRasterBand(2)
    arr2 = band2.ReadAsArray()
    
    cols, rows = arr1.shape    
    arr = np.zeros((cols, rows, 2))
    arr[:, :, 0] = arr1
    arr[:, :, 1] = arr2
    
    gt = ds.GetGeoTransform()
    gt = np.array(gt)
    gt[2] = gt[0] + gt[1] * (rows - 1)
    gt[4] = gt[3] + gt[5] * (cols - 1)

    gt.astype(np.double)

    return arr, gt


def process(input_grd_before_file, input_grd_after_file, output_file):
    
    AmplitudeCDchip(input_grd_before_file, input_grd_after_file, output_file)
    

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input_grd_before_file', 
        type=str, 
        required=True,
        help='before SAR image file'
        )
    
    parser.add_argument(
        '--input_grd_after_file', 
        type=str, 
        required=True,
        help='after SAR image file'
        )
    
    parser.add_argument(
        '--output_file', 
        type=str, 
        help='output change detection image file',
        default='../data/output/output.tif'
        )

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    
    process(args.input_grd_before_file, args.input_grd_after_file, args.output_file)
    
   
    print('============================================================')
