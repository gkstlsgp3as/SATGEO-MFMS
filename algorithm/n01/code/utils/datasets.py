# Dataset utils and dataloaders

import glob
import math
import os
import random
import logging
from itertools import repeat
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ExifTags


from utils.general import xywhn2xyxy, xyn2xy, segment2box, resample_segments, colorstr
from utils.torch_utils import torch_distributed_zero_first

from osgeo import gdal, ogr, osr
from utils.cfg import Cfg

# Parameters
help_url = 'https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data'
img_formats = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo']  # acceptable image suffixes
vid_formats = ['mov', 'avi', 'mp4', 'mpg', 'mpeg', 'm4v', 'wmv', 'mkv']  # acceptable video suffixes
logger = logging.getLogger(__name__)

num_threads = min(8, os.cpu_count())  # number of multiprocessing threads

# Get orientation exif tag
for orientation in ExifTags.TAGS.keys():
    if ExifTags.TAGS[orientation] == 'Orientation':
        break


def get_hash(files):
    # Returns a single hash value of a list of files
    return sum(os.path.getsize(f) for f in files if os.path.isfile(f))


def exif_size(img):
    # Returns exif-corrected PIL size
    s = img.size  # (width, height)
    try:
        rotation = dict(img._getexif().items())[orientation]
        if rotation == 6:  # rotation 270
            s = (s[1], s[0])
        elif rotation == 8:  # rotation 90
            s = (s[1], s[0])
    except:
        pass

    return s

def create_dataloader(path, imgsz, batch_size, stride, opt, hyp=None, augment=False, cache=False, pad=0.0, rect=False,
                      rank=-1, world_size=1, workers=8, image_weights=False, quad=False, prefix='', polygon=False):
    # Make sure only the first process in DDP process the dataset first, and the following others can use the cache
    with torch_distributed_zero_first(rank):
        LoadModels = LoadImagesAndLabels if not polygon else Polygon_LoadImagesAndLabels if polygon else Rotated_LoadImagesAndLabels
        dataset = LoadModels(path, imgsz, batch_size,
                                      augment=augment,  # augment images
                                      hyp=hyp,  # augmentation hyperparameters
                                      rect=rect,  # rectangular training
                                      cache_images=cache,
                                      single_cls=opt.single_cls,
                                      stride=int(stride),
                                      pad=pad,
                                      image_weights=image_weights,
                                      prefix=prefix,
                                      yaml_file=path.split('/')[-3]+'.yaml')

    batch_size = min(batch_size, len(dataset))
    nw = min([os.cpu_count() // world_size, batch_size if batch_size > 1 else 0, workers])  # number of workers
    sampler = torch.utils.data.distributed.DistributedSampler(dataset) if rank != -1 else None
    loader = torch.utils.data.DataLoader if image_weights else InfiniteDataLoader
    # Use torch.utils.data.DataLoader() if dataset.properties will update during training else InfiniteDataLoader()
    dataloader = loader(dataset,
                        batch_size=batch_size,
                        num_workers=nw,
                        sampler=sampler,
                        pin_memory=True,
                        collate_fn=LoadImagesAndLabels.collate_fn4 if quad else LoadImagesAndLabels.collate_fn)
    return dataloader, dataset


class InfiniteDataLoader(torch.utils.data.dataloader.DataLoader):
    """ Dataloader that reuses workers

    Uses same syntax as vanilla DataLoader
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        object.__setattr__(self, 'batch_sampler', _RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)



class LoadImages:  # for inference
    def __init__(self, path, img_size=640, stride=32):
        p = str(Path(path).absolute())  # os-agnostic absolute path
        if '*' in p:
            files = sorted(glob.glob(p, recursive=True))  # glob
        elif os.path.isdir(p):
            files = sorted(glob.glob(os.path.join(p, '*.*')))  # dir
        elif os.path.isfile(p):
            files = [p]  # files
        else:
            raise Exception(f'ERROR: {p} does not exist')

        images = [x for x in files if x.split('.')[-1].lower() in img_formats]
        ni = len(images)

        self.img_size = img_size
        self.stride = stride
        self.files = images 
        self.nf = ni   # number of files
        self.mode = 'image'
        
        assert self.nf > 0, f'No images found in {p}. ' \
                            f'Supported formats are:\nimages: {img_formats}\n'

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count == self.nf:
            raise StopIteration
        path = self.files[self.count]

        # Read image
        self.count += 1
        img0 = cv2.imread(path, cv2.IMREAD_UNCHANGED)  # BGR
        assert img0 is not None, 'Image Not Found ' + path
        #print(f'image {self.count}/{self.nf} {path}: ', end='')

        # Padded resize
        img = letterbox(img0, self.img_size, stride=self.stride)[0]

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        return path, img, img0

    def __len__(self):
        return self.nf  # number of files


class LoadSAR:
    def __init__(self, path, img_size=640, stride=32, save_dir='', landmask=''):
        
        
        p = str(Path(path).absolute())  # os-agnostic absolute path
        if '*' in p:
            files = sorted(glob.glob(p, recursive=True))  # glob
        elif os.path.isdir(p):
            files = sorted(glob.glob(os.path.join(p, '*.*')))  # dir
        elif os.path.isfile(p):
            files = [p]  # files
        else:
            raise Exception(f'ERROR: {p} does not exist')

        images = [x for x in files if x.split('.')[-1].lower() in img_formats]
        ni = len(images)
        
        self.img_size = img_size
        self.stride = stride
        self.files = images
        self.nf = ni # number of files
        self.mode = 'image'
        self.save_dir = save_dir

        self.point = ogr.Geometry(ogr.wkbPoint) 
        self.inSpatialRef = osr.SpatialReference() 
        self.coordTransform = ''
        self.epsg = ''
        self.landmask = landmask

        assert self.nf > 0, f'No images found in {p}. ' \
                            f'Supported formats are:\nimages: {img_formats}'
    def __iter__(self):
        self.count = 0
        return self
                            
    def __next__(self):
        from utils.cfg import Cfg
        from utils.gdal_preprocess import band_to_input, division_testset, landmask_tif
        
        if self.count == self.nf:
            raise StopIteration
        path = self.files[self.count]

        # Read image
        self.count += 1
        
        geotiffile = gdal.Open(path) 
        
        projection = geotiffile.GetProjection() 
        geotransform = geotiffile.GetGeoTransform() 

        proj = osr.SpatialReference(wkt=projection)
        print("original epsg: ", proj.GetAttrValue('AUTHORITY',1))
        self.epsg = proj.GetAttrValue('AUTHORITY',1)
        self.inSpatialRef.ImportFromEPSG(int(self.epsg)) 
        self.outSpatialRef = osr.SpatialReference() 
        self.outSpatialRef.ImportFromEPSG(Cfg.outepsg) 
        self.coordTransform = osr.CoordinateTransformation(self.inSpatialRef, self.outSpatialRef) #transform: 3857 -> 4326
        
        # RGB로 변환
        if Cfg.NewTest == 0:
            input_band = band_to_input(path, Cfg.Satelliteband)
        else:
            input_band = band_to_input(path, Cfg.Satelliteband, True)
        #rgb_band = band_to_rgb(img_path,bandnumber)

        #height, width = rgb_band.shape[:2]
        
        # 모델 추론을 위한 전처리
        # BGR --> RGB
        #input_band = input_band.transpose(2, 0, 1) 
        #rgb_band = cv2.cvtColor(rgb_band, cv2.COLOR_BGR2RGB)
        self.landmask = landmask_tif(path, self.landmask)>0
        div_img_list, div_coord = division_testset(input_band=input_band, img_size=self.img_size)#, landmask=self.landmask)
        return path, input_band, div_img_list, div_coord, input_band.shape, projection, geotransform, 1-self.landmask
        
    def __len__(self):
        return self.nf  # number of files
    
    def change_projection(self, proj):
        self.outSpatialRef.ImportFromEPSG(proj) 
        coordTransform = osr.CoordinateTransformation(self.inSpatialRef, self.outSpatialRef) #transform: 3857 -> 4326

        return coordTransform

# gwang add
def load_image_label(self, i):
    # loads 1 image from dataset index 'i', returns im, original hw, resized hw
    im = self.imgs[i]
    label = self.labels[i].copy() # labels (array): (num_gt_perimg, [cls_id, poly])
    # print('i', i)
    # print('self.img_npy', self.img_npy)
    if im is None:  # not cached in ram
        npy = self.img_npy[i]
        # npy = None
        if npy and npy.exists():  # load npy
            im = np.load(npy)
        else:  # read image
            path = self.img_files[i]
            im = cv2.imread(path, cv2.IMREAD_UNCHANGED)  # BGR
            assert im is not None, f'Image Not Found {path}'
        h0, w0 = im.shape[:2]  # orig hw
        r = self.img_size / max(h0, w0)  # ratio
        if r != 1:  # if sizes are not equal
            im = cv2.resize(im, (int(w0 * r), int(h0 * r)),
                            interpolation=cv2.INTER_AREA if r < 1 and not self.augment else cv2.INTER_LINEAR)
            label[:, 1:] *= r
        return im, (h0, w0), im.shape[:2], label  # im, hw_original, hw_resized, resized_label
    else:
        return self.imgs[i], self.img_hw0[i], self.img_hw[i], self.labels[i]  # im, hw_original, hw_resized, resized_label    
    
def load_mosaic(self, index):
    # loads images in a 4-mosaic

    labels4, segments4 = [], []
    s = self.img_size
    yc, xc = [int(random.uniform(-x, 2 * s + x)) for x in self.mosaic_border]  # mosaic center x, y
    indices = [index] + random.choices(self.indices, k=3)  # 3 additional image indices
    for i, index in enumerate(indices):
        # Load image
        img, _, (h, w) = load_image(self, index)

        # place img in img4
        if i == 0:  # top left
            img4 = np.full((s * 2, s * 2, img.shape[2]), 0.45, dtype=img.dtype)  # base image with 4 tiles; original: np.uint8
            x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
        elif i == 1:  # top right
            x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
            x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
        elif i == 2:  # bottom left
            x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
        elif i == 3:  # bottom right
            x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
            x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

        img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
        padw = x1a - x1b
        padh = y1a - y1b

        # Labels
        labels, segments = self.labels[index].copy(), self.segments[index].copy()
        if labels.size:
            labels[:, 1:] = xywhn2xyxy(labels[:, 1:], w, h, padw, padh)  # normalized xywh to pixel xyxy format
            segments = [xyn2xy(x, w, h, padw, padh) for x in segments]
        labels4.append(labels)
        segments4.extend(segments)

    # Concat/clip labels
    labels4 = np.concatenate(labels4, 0)
    for x in (labels4[:, 1:], *segments4):
        np.clip(x, 0, 2 * s, out=x)  # clip when using random_perspective()
    # img4, labels4 = replicate(img4, labels4)  # replicate

    # Augment
    img4, labels4 = random_perspective(img4, labels4, segments4,
                                       degrees=self.hyp['degrees'],
                                       translate=self.hyp['translate'],
                                       scale=self.hyp['scale'],
                                       shear=self.hyp['shear'],
                                       perspective=self.hyp['perspective'],
                                       border=self.mosaic_border)  # border to remove

    return img4, labels4


def load_mosaic9(self, index):
    # loads images in a 9-mosaic

    labels9, segments9 = [], []
    s = self.img_size
    indices = [index] + random.choices(self.indices, k=8)  # 8 additional image indices
    for i, index in enumerate(indices):
        # Load image
        img, _, (h, w) = load_image(self, index)

        # place img in img9
        if i == 0:  # center
            img9 = np.full((s * 3, s * 3, img.shape[2]), 0.45, dtype=img.dtype)  # base image with 4 tiles; original: np.uint8
            h0, w0 = h, w
            c = s, s, s + w, s + h  # xmin, ymin, xmax, ymax (base) coordinates
        elif i == 1:  # top
            c = s, s - h, s + w, s
        elif i == 2:  # top right
            c = s + wp, s - h, s + wp + w, s
        elif i == 3:  # right
            c = s + w0, s, s + w0 + w, s + h
        elif i == 4:  # bottom right
            c = s + w0, s + hp, s + w0 + w, s + hp + h
        elif i == 5:  # bottom
            c = s + w0 - w, s + h0, s + w0, s + h0 + h
        elif i == 6:  # bottom left
            c = s + w0 - wp - w, s + h0, s + w0 - wp, s + h0 + h
        elif i == 7:  # left
            c = s - w, s + h0 - h, s, s + h0
        elif i == 8:  # top left
            c = s - w, s + h0 - hp - h, s, s + h0 - hp

        padx, pady = c[:2]
        x1, y1, x2, y2 = [max(x, 0) for x in c]  # allocate coords

        # Labels
        labels, segments = self.labels[index].copy(), self.segments[index].copy()
        if labels.size:
            labels[:, 1:] = xywhn2xyxy(labels[:, 1:], w, h, padx, pady)  # normalized xywh to pixel xyxy format
            segments = [xyn2xy(x, w, h, padx, pady) for x in segments]
        labels9.append(labels)
        segments9.extend(segments)

        # Image
        img9[y1:y2, x1:x2] = img[y1 - pady:, x1 - padx:]  # img9[ymin:ymax, xmin:xmax]
        hp, wp = h, w  # height, width previous

    # Offset
    yc, xc = [int(random.uniform(0, s)) for _ in self.mosaic_border]  # mosaic center x, y
    img9 = img9[yc:yc + 2 * s, xc:xc + 2 * s]

    # Concat/clip labels
    labels9 = np.concatenate(labels9, 0)
    labels9[:, [1, 3]] -= xc
    labels9[:, [2, 4]] -= yc
    c = np.array([xc, yc])  # centers
    segments9 = [x - c for x in segments9]

    for x in (labels9[:, 1:], *segments9):
        np.clip(x, 0, 2 * s, out=x)  # clip when using random_perspective()
    # img9, labels9 = replicate(img9, labels9)  # replicate

    # Augment
    img9, labels9 = random_perspective(img9, labels9, segments9,
                                       degrees=self.hyp['degrees'],
                                       translate=self.hyp['translate'],
                                       scale=self.hyp['scale'],
                                       shear=self.hyp['shear'],
                                       perspective=self.hyp['perspective'],
                                       border=self.mosaic_border)  # border to remove

    return img9, labels9


## anxiliary functions
def load_image(self, index):
    # loads 1 image from dataset, returns img, original hw, resized hw
    
    img = self.imgs[index]
    if img is None:  # not cached
        path = self.img_files[index]
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)  # BGR
        if img.max() > 1:
            img = img/255
        assert img is not None, 'Image Not Found ' + path
        h0, w0 = img.shape[:2]  # orig hw
        r = self.img_size / max(h0, w0)  # ratio
        if r != 1:  # if sizes are not equal
            img = cv2.resize(img, (int(w0 * r), int(h0 * r)),
                             interpolation=cv2.INTER_AREA if r < 1 and not self.augment else cv2.INTER_LINEAR)
        
        if Cfg.img_mode != 'org':
            if Cfg.img_mode == 'vh+vv':
                newimg = img[...,1]+img[...,2]
            elif Cfg.img_mode == 'grayscale':
                newimg = 0.229*img[...,0]+0.587*img[...,1]+0.114*img[...,2]
                newimg = img[...,1]**2 +img[...,2]**2
            elif Cfg.img_mode == 'vv*vh':
                newimg = img[...,1]*img[...,2]
            img = np.dstack((newimg, newimg, newimg))
            if img.max() != 0:
                img = (img - img.min()) / (img.max() - img.min())
            
        return img, (h0, w0), img.shape[:2]  # img, hw_original, hw_resized
    else:
        return self.imgs[index], self.img_hw0[index], self.img_hw[index]  # img, hw_original, hw_resized


class Albumentations:
    # Polygon YOLOv5 Albumentations class (optional, only used if package is installed)
    def __init__(self):
        self.transform = None
        try:
            import albumentations as A

            self.transform = A.Compose([
                A.MedianBlur(p=0.05),
                A.ToGray(p=0.1),
                A.RandomBrightnessContrast(p=0.35),
                A.CLAHE(p=0.2),
                A.InvertImg(p=0.3)],
                A.augmentations.transforms.GaussNoise (var_limit=(10.0, 50.0), mean=0, per_channel=True, 
                                                       always_apply=False, p=0.5),
                A.transforms.RandomGamma (gamma_limit=(80, 120), eps=None, always_apply=False, p=0.5),
                )
                #transforms.RandomApply([AddGaussianNoise(0., 1.)], p=0.5))
                # Not support for any position change to image

            logging.info(colorstr('albumentations: ') + ', '.join(f'{x}' for x in self.transform.transforms if x.p))
        except ImportError:  # package not installed, skip
            pass
        except Exception as e:
            logging.info(colorstr('albumentations: ') + f'{e}')

    def __call__(self, im, p=1.0):
        if self.transform and random.random() < p:
            new = self.transform(image=im)  # transformed
            im = new['image']
        return im

# Ancillary functions --------------------------------------------------------------------------------------------------------------------
def letterbox(img, new_shape=(640, 640), color=(0.45, 0.45, 0.45), auto=True, scaleFill=False, scaleup=True, stride=32):
    # for rectangular inference, pad with gray(RGB: 114,114,114) pixels
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)
    
def random_perspective(img, targets=(), segments=(), degrees=10, translate=.1, scale=.1, shear=10, perspective=0.0,
                       border=(0, 0)):
    # torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-10, 10))
    # targets = [cls, xyxy]

    height = img.shape[0] + border[0] * 2  # shape(h,w,c)
    width = img.shape[1] + border[1] * 2

    # Center
    C = np.eye(3)
    C[0, 2] = -img.shape[1] / 2  # x translation (pixels)
    C[1, 2] = -img.shape[0] / 2  # y translation (pixels)

    # Perspective
    P = np.eye(3)
    P[2, 0] = random.uniform(-perspective, perspective)  # x perspective (about y)
    P[2, 1] = random.uniform(-perspective, perspective)  # y perspective (about x)

    # Rotation and Scale
    R = np.eye(3)
    a = random.uniform(-degrees, degrees)
    # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
    s = random.uniform(1 - scale, 1 + scale)
    # s = 2 ** random.uniform(-scale, scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)

    # Translation
    T = np.eye(3)
    T[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * width  # x translation (pixels)
    T[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * height  # y translation (pixels)

    # Combined rotation matrix
    M = T @ S @ R @ P @ C  # order of operations (right to left) is IMPORTANT
    if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():  # image changed
        if perspective:
            img = cv2.warpPerspective(img, M, dsize=(width, height), borderValue=(0.45, 0.45, 0.45))
        else:  # affine
            img = cv2.warpAffine(img, M[:2], dsize=(width, height), borderValue=(0.45, 0.45, 0.45))

    # Visualize
    # import matplotlib.pyplot as plt
    # ax = plt.subplots(1, 2, figsize=(12, 6))[1].ravel()
    # ax[0].imshow(img[:, :, ::-1])  # base
    # ax[1].imshow(img2[:, :, ::-1])  # warped

    # Transform label coordinates
    n = len(targets)
    if n:
        use_segments = any(x.any() for x in segments)
        new = np.zeros((n, 4))
        use_segments=False
        if use_segments:  # warp segments
            segments = resample_segments(segments)  # upsample
            for i, segment in enumerate(segments):
                xy = np.ones((len(segment), 3))
                xy[:, :2] = segment
                xy = xy @ M.T  # transform
                xy = xy[:, :2] / xy[:, 2:3] if perspective else xy[:, :2]  # perspective rescale or affine

                # clip
                new[i] = segment2box(xy, width, height)

        else:  # warp boxes
            xy = np.ones((n * 4, 3))
            xy[:, :2] = targets[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
            xy = xy @ M.T  # transform
            xy = (xy[:, :2] / xy[:, 2:3] if perspective else xy[:, :2]).reshape(n, 8)  # perspective rescale or affine

            # create new boxes
            x = xy[:, [0, 2, 4, 6]]
            y = xy[:, [1, 3, 5, 7]]
                        
            new = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

            # clip
            new[:, [0, 2]] = new[:, [0, 2]].clip(0, width)
            new[:, [1, 3]] = new[:, [1, 3]].clip(0, height)

        # filter candidates
        i = box_candidates(box1=targets[:, 1:5].T * s, box2=new.T, area_thr=0.01 if use_segments else 0.10)
        targets = targets[i]
        targets[:, 1:5] = new[i]

    return img, targets

def box_candidates(box1, box2, wh_thr=2, ar_thr=20, area_thr=0.1, eps=1e-16):  # box1(4,n), box2(4,n)
    # Compute candidate boxes: box1 before augment, box2 after augment, wh_thr (pixels), aspect_ratio_thr, area_ratio
    w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
    w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
    ar = np.maximum(w2 / (h2 + eps), h2 / (w2 + eps))  # aspect ratio
    return (w2 > wh_thr) & (h2 > wh_thr) & (w2 * h2 / (w1 * h1 + eps) > area_thr) & (ar < ar_thr)  # candidates


## Augmentation ----------------------------------------------------------------------------------------------------------

def augment_hsv(img, hgain=0.5, sgain=0.5, vgain=0.5):
    rnd = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
    if img.dtype == np.uint8:
        hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
        dtype = img.dtype  # org: uint8; 

        x = np.arange(0, 256, dtype=rnd.dtype) # original: int16
        # In OpenCV, H: 0-180, S: 0-255, V: 0-255
        lut_hue = ((x * rnd[0]) % 180).astype(dtype) # color type; originally, 0-360º but only be represented to 180 wrt 8 bit
        lut_sat = np.clip(x * rnd[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * rnd[2], 0, 255).astype(dtype)
        
        # LUT only supports 8-bit LUTs
        img_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val))) # LUT = look up table -> ref: https://mrsnake.tistory.com/142
        img = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR) # no need to return
    else: # float
        hue, sat, val = cv2.split(cv2.cvtColor(np.array(img, dtype=np.uint8), cv2.COLOR_BGR2HSV))
        dtype = img.dtype  # org: uint8; now: float32

        x = np.arange(0, 256, dtype=rnd.dtype) # original: int16
        # In OpenCV, H: 0-180, S: 0-255, V: 0-255
        lut_hue = ((x * rnd[0]) % 180).astype(dtype) # color type; originally, 0-360º but only be represented to 180 wrt 8 bit
        lut_sat = np.clip(x * rnd[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * rnd[2], 0, 255).astype(dtype)
        
        # LUT only supports 8-bit LUTs
        img_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val))) # LUT = look up table -> ref: https://mrsnake.tistory.com/142
        img = cv2.cvtColor(img_hsv/255, cv2.COLOR_HSV2BGR)*255
    return img  
