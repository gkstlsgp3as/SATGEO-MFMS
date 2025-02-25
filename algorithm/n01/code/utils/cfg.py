# -*- coding: utf-8 -*-
'''
@Time          : 2020/05/06 21:05
@Author        : Tianxiaomo
@File          : Cfg.py
@Noice         :
@Modificattion :
    @Author    :
    @Time      :
    @Detail    :

'''
import os
from easydict import EasyDict


_BASE_DIR = os.path.dirname(os.path.abspath(__file__))

Cfg = EasyDict()

Cfg.outepsg = 4326

# RGB
Cfg.color_3band = [[255,0,0], [255,255,255], [0,255,0], [255,0,255]] 
Cfg.color_1band = [[255], [200], [150], [100]] 
# 우선 255 -> 255, 0, 0 / 200 -> 0, 0, 255 / 150 -> 0, 255, 0 / 100 -> 255, 0, 255 순으로 매핑해두었습니다. 

# ship, 'trn', 'veh', 'ped' => red, blue, green, purple
Cfg.classes = 3

Cfg.iou_type = 'ciou'  # 'iou', 'giou', 'diou', 'ciou', 'gaussian'
Cfg.img_mode = 'vv*vh' #  'grayscale', 'vv^2+vh^2', 'vv*vh' or 'org'

# Additional Input information and variables
# Name of each image: S1, CSK, K5, ICEYE
# Number of satellite image band: 1 or 3
Cfg.Satellite = 'S1'
Cfg.Satelliteband = 3
Cfg.division = 40 # K5: 10, S1: 15, 20 15

# New Band Test(True=1, False=0)
Cfg.NewTest = 0

Cfg.minTh = 0
# Bands
Cfg.min = [0, 0, 0]
Cfg.max = [0.15, 0.5, 50]
 # 0.15 150 / 0.5 200 / 50 250

Cfg.size=2


Cfg.output_format = 'csv'
Cfg.output_band = 3

Cfg.hide_labels = True
Cfg.hide_conf = True

Cfg.no_trace = True

Cfg.img_size = 640

Cfg.target_class = 'infra'
Cfg.conf_thres = 0.05
Cfg.iou_thres = 0.3
Cfg.max_det = 1000
