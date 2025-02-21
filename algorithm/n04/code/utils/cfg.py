# -*- coding: utf-8 -*-
"""
Created on Tue Dec 31 04:19:40 2024

@author: user
"""

import os
from easydict import EasyDict


_BASE_DIR = os.path.dirname(os.path.abspath(__file__))

Cfg = EasyDict()

# input data path
Cfg.distance_threshold = 200
