# -*- coding: utf-8 -*-
'''
@Time          : 2024/12/18 00:00
@Author        : Satgeo
@File          : n01_detect_all_facilities.py
@Noice         : 
@Description   : Detect marine facilities from SAR image.
@How to use    : python n01_detect_all_facilities.py --input_dir {input_path} --output_dir {output_path} --model_weight_file {model_file}

@Modificattion :
    @Author    :
    @Time      :
    @Detail    :
'''

import argparse
import time
from pathlib import Path

import cv2
import torch
import numpy as np 
import os

from models.experimental import attempt_load
from utils.datasets import LoadSAR
from utils.general import (check_img_size, set_logging, check_file,
                           non_max_suppression)
from utils.torch_utils import (select_device, load_classifier,
                               time_synchronized, TracedModel)
from utils.plots import set_target
import logging
from utils.cfg import Cfg

def detect(input_dir='data/images', output_dir='runs/detect',
           model_weight_file='yolov8.pt', input_dem_file='../data/dem/dem.tif',
           gpu_device='', output_proj='4326'):
    """
    Perform detection on images and save results to specified directory.

    Args:
        input_dir (str): Directory containing input images.
        output_dir (str): Directory to save detection results.
        model_weight_file (str): Path to the model weight file.
        input_dem_file (str): Path to DEM file for land masking.
        gpu_device (str): Specify GPU device; 'cpu' for CPU.
        output_proj (str): Output projection coordinates system.
    """
    
    trace = not Cfg.no_trace
    # Directories
    save_dir = Path(output_dir, exist_ok=True)  # define the project
    (save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(gpu_device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(model_weight_file, map_location=device)  # load FP32 mode
    stride = int(model.stride.max())  # model stride
    img_size = check_img_size(Cfg.img_size, s=stride)  # check img_size
    if trace:
        model = TracedModel(model, device, img_size)

    if half:
        model.half()  # to FP16

    # Get datasets
    img_formats = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo']  # acceptable image suffixes
    
    if os.path.isdir(input_dir):
        files = os.listdir(input_dir)
    else:
        files = [input_dir]
    image_ext = np.unique(np.array([x.split('.')[-1].lower() for x in files if x.split('.')[-1].lower() in img_formats]))
    files = [f for f in files if f.split('.')[-1] in image_ext]
    t = time.time()
    dataset = LoadSAR(input_dir, img_size=img_size, stride=stride, save_dir=save_dir, landmask=input_dem_file)
     # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    
    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, img_size, img_size).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = img_size
    old_img_b = 3

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()
    
    target = set_target(Cfg.target_class)
    if target == -1: 
        print("The assigned name ", Cfg.target_class, " is invalid. \n The name should be either [bridge/ship/wind/windturbin/water/flood/oil/oilspill/fire/forestfire/landslide/eqrthquake/subsidence]\n")
        quit()
    if Cfg.output_band == 1:
        from utils.plots import plot_one_box_1band, set_colors_1band, set_colors
        Plot = plot_one_box_1band
        colors_1band = set_colors_1band(names, target=target)
        colors = set_colors(names, target=target)
    else:
        from utils.plots import plot_one_box, set_colors
        Plot = plot_one_box
        colors = set_colors(names, target=target)

    t0 = time.time()
    for path, input_band, div_img_list, div_coord, shapes, projection, geotransform, landmask in dataset:  
        # 테스트 이미지를 1/div_num 만큼 width, height를 분할하고, 크롭된 이미지와 위치좌표를 반환
        p = Path(path)
        
        b1_image = np.repeat(np.expand_dims(np.uint8(input_band[:,:,1]*255),2),Cfg.output_band,axis=2)
        SLC = True if p.stem.split('_')[2] == 'SLC' else False
        if not SLC:
            xoff, ca, cb, yoff, cd, ce = geotransform
        tot_det = torch.Tensor().to(device)
        
        for d_id, img0 in enumerate(div_img_list):
            div_x, div_y = div_coord[d_id][0], div_coord[d_id][1]

            img = img0[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416; already applied
            img = np.ascontiguousarray(img)
                            
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            #img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)
    
            # Warmup
            if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
                old_img_b = img.shape[0]
                old_img_h = img.shape[2]
                old_img_w = img.shape[3]
                for i in range(3):
                    model(img, augment=False)[0]
    
            # Inference
            t1 = time_synchronized()
            with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
                pred = model(img, augment=False)[0]
                
            t2 = time_synchronized()
            pred[:,:,0] = pred[:,:,0] + div_coord[d_id][2]
            pred[:,:,1] = pred[:,:,1] + div_coord[d_id][3]
            tot_det = torch.cat([tot_det, pred])
            # Process detections
        # Apply NMS
        max_det = 1000
        tot_det = non_max_suppression(tot_det, Cfg.conf_thres, Cfg.iou_thres, agnostic=False, max_det=max_det)
        t3 = time_synchronized()  
        latmaxs = []; latmins = []; lonmaxs = []; lonmins = []
        classes = []
        
        for d_id, dets in enumerate(tot_det):  # detections per image
            s, im0 = '', img0

            save_path = str(save_dir / p.name)#.replace('.tif','_{}.tif'.format(d_id)))  # img.jpg
            if len(dets)==0:
                continue
    
            # Print results
            for c in dets[:, -1].unique():
                n = (dets[:, -1] == c).sum()  # detections per class
                s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
            
            
            for *xyxy, conf, cls in reversed(dets):
                if not SLC:
                    x1 = min(xyxy[0::2]); x2 = max(xyxy[0::2]) # top-left
                    y1 = min(xyxy[1::2]); y2 = max(xyxy[1::2]) # bottom-right
                    
                    coord = [int(el.detach().cpu().numpy()) for el in xyxy]
                    
                    if landmask[coord[1],coord[0]]*landmask[coord[3],coord[0]]*\
                        landmask[coord[1],coord[2]]*landmask[coord[3],coord[2]] == 0: 
                        continue
                    
                    lonmin = ca * x1 + cb * y1 + xoff
                    lonmax = ca * x2 + cb * y1 + xoff
                    latmin = cd * x1 + ce * y1 + yoff
                    latmax = cd * x1 + ce * y2 + yoff
                
                    if (dataset.epsg != output_proj): 
                        # when the input coordinate system is different to dst coordinate system
                        dataset.point.AddPoint(float(lonmin), float(latmin))
                        if output_proj != Cfg.outepsg: # if proj argument is different from cfg 
                            dataset.coordTransform = dataset.change_projection(int(output_proj))
                        dataset.point.Transform(dataset.coordTransform) # 3857 to 4326
                        
                        latmin = dataset.point.GetX() # pop x
                        lonmin = dataset.point.GetY() # pop y
                        
                        dataset.point.AddPoint(float(lonmax), float(latmax))
                        if output_proj != Cfg.outepsg: # if proj argument is different from cfg 
                            dataset.coordTransform = dataset.change_projection(int(output_proj))
                        dataset.point.Transform(dataset.coordTransform) # 3857 to 4326
                        
                        latmax = dataset.point.GetX() # pop x
                        lonmax = dataset.point.GetY() # pop y
                        
                    latmaxs.append(latmax.detach().cpu().numpy()); lonmaxs.append(lonmax.detach().cpu().numpy()); 
                    latmins.append(latmin.detach().cpu().numpy()); lonmins.append(lonmin.detach().cpu().numpy())
                    classes.append(target)
                    
                label = None if Cfg.hide_labels else (names[int(cls)] if Cfg.hide_conf else f'{names[int(cls)]} {conf:.2f}')
                    
                xyxy = [x1, y1, x2, y2]
                xyxy = [int(el) for el in xyxy]
                if Cfg.output_band == 1:
                    Plot(xyxy, b1_image, label=label, color=colors_1band[int(cls)], line_thickness=1)
                else:
                    Plot(xyxy, b1_image, label=label, color=colors[int(cls)], line_thickness=1)    

        if Cfg.output_format == 'csv':
            import pandas as pd
            df = pd.DataFrame({'LatMin': latmins, 'LonMin': latmins, 'LatMax': latmaxs, 'LonMax': lonmaxs, 'Class': classes})
            df_fl = df.to_csv(save_dir / (p.stem + '_' + target + '.csv'), index=False)
    
        # Save results (image with detections)
        from osgeo import gdal 
        import zipfile
        save_path = str(save_dir / Path(path).name.replace('.tif', '_' + target + '.tif'))
        print(f" The image with the result is saved in: {save_path}")
    
        b1_image = np.array(b1_image, dtype=np.uint8)
        cv2.imwrite(save_path, b1_image)   # find what to plot   
    
        if not SLC: # slc는 geotransform 안함. 
            outfile = gdal.Open(save_path, gdal.GA_Update) 
            outfile.SetGeoTransform(geotransform) # 
            outfile.SetProjection(projection) # gdal에서 원하는 좌표계로 변형
            outfile.FlushCache() # 저장

        print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')
        
    print(f'Done. ({time.time() - t0:.3f}s)')


def process(input_dir, output_dir, model_weight_file, input_dem_file, gpu_device, output_proj):
    
    model_weight_file = check_file(model_weight_file[0])
    
    with torch.no_grad():
        detect(input_dir, output_dir, model_weight_file, input_dem_file, gpu_device, output_proj)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-I', '--input_dir', 
        type=str, 
        required=True,
        default='../data/input', 
        help='input_dir'
    )  
    parser.add_argument(
        '-O', '--output_dir', 
        type=str,
        required=True,
        default='../data/output/', 
        help='save results to project/name'
    )
    parser.add_argument(
        '--model_weight_file', 
        nargs='+', 
        type=str, 
        default='best.pt', 
        help='model.pt path(s)'
    )
    parser.add_argument(
        '--input_dem_file', 
        type=str, 
        default='../data/dem/dem.tif',
        help='path to dem for landmasking'
    )
    parser.add_argument(
        '--gpu_device', 
        default='', 
        help='cuda device, i.e. 0 or 0,1,2,3 or cpu'
    )
    parser.add_argument(
        '--output_proj', 
        default='4326', 
        help='define the projection coordinates to transform'
    )
    
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    start_time = time.time()

    args = get_args()
    
    process(args.input_dir, args.output_dir, args.model_weight_file, args.input_dem_file, args.gpu_device, args.output_proj)

    processed_time = time.time() - start_time
    logging.info(f"{processed_time:.2f} seconds")


    
