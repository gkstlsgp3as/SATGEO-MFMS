# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 21:41:06 2025

@author: user
"""

import os
from easydict import EasyDict

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))

Cfg = EasyDict()

Cfg.weightfile1 = NN
Cfg.weightfile2 = NN
Cfg.patch_size1 = 224
Cfg.patch_size2 = 224
Cfg.channel = 'VV', 'Incidence_angle' # VV, Incidence angle, 

