
from shapely.geometry import shape
from utils.sarpy import s1_load
import numpy as np
from scipy.stats import chi2
from matplotlib import pyplot as plt
import json
from typing import Tuple
from scipy.special import gammainc, gamma
from typing import Tuple
from math import radians, atan, tan, sin, cos, sqrt, atan2
from pyproj import CRS
from numpy.typing import NDArray
import cv2
from numba import jit


@staticmethod
def getEllipsoidParameters(epsg: int) -> Tuple[float, float, float]:
    crs = CRS.from_epsg(epsg)
    ellipsoid = crs.ellipsoid
    a = ellipsoid.semi_major_metre
    b = ellipsoid.semi_minor_metre
    f = ellipsoid.inverse_flattening
    return a, b, 1 / f


@staticmethod
def vincentyDistance(lat1: float, lon1: float, lat2: float, lon2: float, epsg: int = 4326) -> float:
    a, b, f = getEllipsoidParameters(epsg)

    L = radians(lon2 - lon1)
    U1 = atan((1 - f) * tan(radians(lat1)))
    U2 = atan((1 - f) * tan(radians(lat2)))
    sinU1 = sin(U1)
    cosU1 = cos(U1)
    sinU2 = sin(U2)
    cosU2 = cos(U2)

    lamb = L
    iterations = 1000
    converged = False

    for _ in range(iterations):
        sin_lambda = sin(lamb)
        cos_lambda = cos(lamb)
        sin_sigma = sqrt((cosU2 * sin_lambda) ** 2 + (cosU1 * sinU2 - sinU1 * cosU2 * cos_lambda) ** 2)
        if sin_sigma == 0:
            # co-incident points
            return 0
        cos_sigma = sinU1 * sinU2 + cosU1 * cosU2 * cos_lambda
        sigma = atan2(sin_sigma, cos_sigma)
        sin_alpha = cosU1 * cosU2 * sin_lambda / sin_sigma
        cos_sq_alpha = 1 - sin_alpha ** 2
        cos2sigma_m = cos_sigma - 2 * sinU1 * sinU2 / cos_sq_alpha if cos_sq_alpha != 0 else 0
        C = f / 16 * cos_sq_alpha * (4 + f * (4 - 3 * cos_sq_alpha))
        lambda_prev = lamb
        lamb = L + (1 - C) * f * sin_alpha * (sigma + C * sin_sigma * (cos2sigma_m + C * cos_sigma * (-1 + 2 * cos2sigma_m ** 2)))
        if abs(lamb - lambda_prev) < 1e-12:
            converged = True
            break

    if not converged:
        raise ValueError("Vincenty formula failed to converge")

    u_sq = cos_sq_alpha * (a ** 2 - b ** 2) / (b ** 2)
    A = 1 + u_sq / 16384 * (4096 + u_sq * (-768 + u_sq * (320 - 175 * u_sq)))
    B = u_sq / 1024 * (256 + u_sq * (-128 + u_sq * (74 - 47 * u_sq)))
    delta_sigma = B * sin_sigma * (cos2sigma_m + B / 4 * (cos_sigma * (-1 + 2 * cos2sigma_m ** 2) - B / 6 * cos2sigma_m * (-3 + 4 * sin_sigma ** 2) * (-3 + 4 * cos2sigma_m ** 2)))

    return b * A * (sigma - delta_sigma)


def applyFilterBulbSize(labels_image: NDArray[np.float64], interpolation_coef: float = 1.0, bulb_size: int = 20) -> NDArray[np.float64]:
    labels_image = labels_image.astype(np.float32)
    nb_blobs, im_with_separated_blobs, stats, _ = cv2.connectedComponentsWithStats(labels_image.astype(np.uint8))
    sizes = stats[1:, -1]
    nb_blobs -= 1
    min_size: int = int(bulb_size * interpolation_coef)
    img: NDArray[np.float64] = np.zeros_like(im_with_separated_blobs, dtype=np.float64)
    for blob in range(nb_blobs):
        if sizes[blob] >= min_size:
            img[im_with_separated_blobs == blob + 1] = 1

    return img


class sglr:
    def __init__(self, config) -> None:
        with open(config, 'r') as file:
            self.config = json.load(file)
        with open(self.config["roi_path"], 'r') as file:
            self.roi = json.load(file)

        target_feature = None
        for feature in self.roi['features']:
            if feature['properties']['ID'] == self.config["roi_id"]:
                target_feature = feature
                break
        geometry = shape(target_feature['geometry'])
        bounds = geometry.bounds
        centroid = geometry.centroid
        self.centroid_coords: Tuple = (centroid.y, centroid.x)
        self.bbxdist = {
            'left': vincentyDistance(centroid.y, centroid.x, centroid.y, bounds[0]),
            'right': vincentyDistance(centroid.y, centroid.x, centroid.y, bounds[2]),
            'bottom': vincentyDistance(centroid.y, centroid.x, bounds[1], centroid.x),
            'top': vincentyDistance(centroid.y, centroid.x, bounds[3], centroid.x)}

    def det(im):
        return im.bands[0] * im.bands[1]

    def safeLog(x):
        return np.log(np.where(x <= 0, 1e-10, x))

    @classmethod
    def gammaDist(L, u, y):
        return (L / (u * gamma(L))) * ((L * y / u) ** (L - 1)) * np.exp(-L * y / u)

    @classmethod
    def SGLR(u_t, u_tp, L):
        term1 = 2 * L * np.log(np.sqrt(u_t / u_tp) + np.sqrt(u_tp / u_t))
        term2 = 2 * L * np.log(2)
        return term1 - term2

    def applySGLR(im1, im2, L):
        u_t = np.mean(im1, axis=(1, 2))
        u_tp = np.mean(im2, axis=(1, 2))
        return sglr.SGLR(u_t, u_tp, L)

    def chi2Cdf(chi2_values, df):
        return gammainc(df / 2, chi2_values / 2)

    @jit(nopython=True)
    def constructCovarianceMatrices(im1_vv, im1_vh, im2_vv, im2_vh):
        rows, cols = im1_vv.shape

        X = np.zeros((rows, cols))
        Y = np.zeros((rows, cols))

        for i in range(rows):
            for j in range(cols):
                x1 = im1_vv[i, j]
                x2 = im1_vh[i, j]
                y1 = im2_vv[i, j]
                y2 = im2_vh[i, j]
                cov_matrix_im1 = np.array([[x1, x2], [x2, x1]])
                cov_matrix_im2 = np.array([[y1, y2], [y2, y1]])
                X[i, j] = cov_matrix_im1[0, 1]
                Y[i, j] = cov_matrix_im2[0, 1]

        return X, Y

    def calM2logQ(im1, im2, m, mt=1):
        if mt == 1: # VV*VH
            det_im1 = sglr.det(im1)
            det_im2 = sglr.det(im2)
            
        elif mt == 2:
            det_im1, det_im2 = sglr.constructCovarianceMatrices(im1.bands[0], im1.bands[1], im2.bands[0], im2.bands[1])

        log_det_im1 = sglr.safeLog(det_im1)
        log_det_im2 = sglr.safeLog(det_im2) # Log(VV*VH)
        sum_im1_im2 = [im1.bands[0] + im2.bands[0], im1.bands[1] + im2.bands[1]] # before + after
        det_im1_plus_im2 = sum_im1_im2[0] * sum_im1_im2[1]  # VV * VH (b+a)
        log_det_im1_plus_im2 = sglr.safeLog(det_im1_plus_im2) # Log(VV*VH) (b+a)
        result = log_det_im1 + log_det_im2 - 2 * log_det_im1_plus_im2 + 4 * np.log(2)
        result = result * (-2 * m)
        return result

    def calM2logQvar(im1, im2, m, mt=1):
        if mt == 1: # VV*VH
            det_im1 = sglr.det(im1)
            det_im2 = sglr.det(im2)
            
        elif mt == 2:
            det_im1, det_im2 = sglr.constructCovarianceMatrices(im1.bands[0], im1.bands[1], im2.bands[0], im2.bands[1])

        log_det_im1 = sglr.safeLog(det_im1)
        log_det_im2 = sglr.safeLog(det_im2) # Log(VV*VH)
        sum_im1_im2 = [im1.bands[0] - im2.bands[0], im1.bands[1] - im2.bands[1]] # before + after
        det_im1_plus_im2 = sum_im1_im2[0] * sum_im1_im2[1]  # VV * VH (b-a)
        log_det_im1_plus_im2 = sglr.safeLog(det_im1_plus_im2) # Log(VV*VH) (b-a)
        result = log_det_im1 - log_det_im2 - 2 * log_det_im1_plus_im2 + 4 * np.log(2)
        result = result * (-2 * m)
        return result
    
    def calM2logQchip(im1, im2, m):
        # VV*VH
        det_im1 = im1[:,:,0]*im1[:,:,1]
        det_im2 = im2[:,:,0]*im2[:,:,1]
            
        log_det_im1 = sglr.safeLog(det_im1)
        log_det_im2 = sglr.safeLog(det_im2) # Log(VV*VH)
        sum_im1_im2 = [im1[:,:,0] - im2[:,:,0], im1[:,:,1] - im2[:,:,1]] # before + after
        det_im1_plus_im2 = sum_im1_im2[0] * sum_im1_im2[1]  # VV * VH (b-a)
        log_det_im1_plus_im2 = sglr.safeLog(det_im1_plus_im2) # Log(VV*VH) (b-a)
        result = log_det_im1 - log_det_im2 - 2 * log_det_im1_plus_im2 + 4 * np.log(2)
        #result = result * (-2 * m)
        return result


