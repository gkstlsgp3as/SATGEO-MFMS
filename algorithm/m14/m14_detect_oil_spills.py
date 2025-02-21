from tensorflow.keras.models import load_model
import numpy as np
from osgeo import gdal
import time
import argparse
import json
import os
import logging
from gdal import Open, GetDriverByName, GDT_Byte

logging.basicConfig(level=logging.INFO)


def detect(model, input_img, width):
    """
    Performs image segmentation by processing patches of the input image using a provided model.
    
    Parameters:
    - model: The deep learning model to use for prediction.
    - input_img: Numpy array of the input image (channels, height, width).
    - width: The width of the square patch to process at a time.
    
    Returns:
    - A segmented image array where detected areas are marked.
    """
    img_width = input_img.shape[2]
    img_height = input_img.shape[1]
    channel = input_img.shape[0]
    height = width  # Assumes square patches for simplicity.

    dm_width = divmod(img_width, width)
    dm_height = divmod(img_height, height)
    n_width = dm_width[0]
    n_height = dm_height[0]
    zero_padded_width = img_width
    zero_padded_height = img_height

    if dm_width[1] != 0:
        n_width += 1
        zero_padded_width = n_width * width
    if dm_height[1] != 0:
        n_height += 1
        zero_padded_height = n_height * height

    zero_padded_img = np.zeros((channel, zero_padded_height, zero_padded_width))
    zero_padded_img[:, 0:img_height, 0:img_width] = input_img

    temp_img = np.zeros((zero_padded_height, zero_padded_width))

    # Processing each patch in the image
    for y in range(n_height):
        for x in range(n_width):
            cropped = zero_padded_img[:, y * height: (y + 1) * height, x * width: (x + 1) * width]
            normalized_crop = normalize_patch(cropped)
            output = model.predict(normalized_crop)
            temp_img[y * height: (y + 1) * height, x * width: (x + 1) * width] = output
            
        percent = (y + 1) * 100 / n_height
        logging.info(f"..... {percent:.2f}% completed")

    out_img = temp_img[:img_height, :img_width]
    return out_img


def normalize_patch(patch):
    """
    Normalize the patch for model prediction.
    
    Parameters:
    - patch: The image patch to normalize.
    
    Returns:
    - The normalized image patch.
    """
    patch[0, ...] = np.where(patch[0, ...] > 0.1, 0.1, patch[0, ...]) / 0.1
    patch[1, ...] = (patch[1, ...] - 0) / (60 - 0)
    return np.expand_dims(np.moveaxis(patch, 0, -1), axis=0)


def process(input_grd_file, output_grd_file, output_json_file, weight, patch_size, channel_list):
    """
    Main processing function to handle image data loading, processing, and saving the results.
    
    Parameters:
    - input_grd_file: Path to the input GDAL-supported image file.
    - output_grd_file: Path for saving the processed image.
    - output_json_file: Path for saving processing metadata.
    - weight: Path to the model weights.
    - patch_size: Size of the patch to process.
    - channel_list: List of channels to use from the input image.
    """
    os.makedirs(os.path.dirname(output_grd_file), exist_ok=True)
    os.makedirs(os.path.dirname(output_json_file), exist_ok=True)

    img_open = Open(input_grd_file, gdal.GA_ReadOnly)
    gdal_img = img_open.ReadAsArray()
    transform = img_open.GetGeoTransform()
    projection = img_open.GetProjection()

    # Select specific channels if provided, else use all channels
    if not channel_list:
        process_img = gdal_img
    else:
        process_img = np.zeros((len(channel_list), gdal_img.shape[1], gdal_img.shape[2]))
        for i, channel in enumerate(channel_list):
            process_img[i, :, :] = gdal_img[channel - 1, :, :]

    # Load model
    logging.info("Loading Deep Learning Model...")
    model = load_model(weight, compile=False)
    output = detect(model, process_img, patch_size)
    output_masked = np.where(process_img[0] == 0, 0, output)

    # Save the processed image
    driver = GetDriverByName("GTiff")
    out_ds = driver.Create(output_grd_file, xsize=gdal_img.shape[2], ysize=gdal_img.shape[1], bands=1, eType=GDT_Byte)
    out_ds.GetRasterBand(1).WriteArray(output_masked)
    out_ds.SetProjection(projection)
    out_ds.SetGeoTransform(transform)
    out_ds = None

    
    # TODO: Create output json file
    # output.json 파일안에 threshold 검사에 필요한 값이 아래와 같은 형태로 정의되어 있어야 합니다.
    # thresholdEstimationValue에 정의된 값이 유의, 위험 등으로 설정한 값보다 큰 경우에 alert을 주도록 시스템 내 구현되어 있습니다.
    # Ex. {"thresholdEstimationValue": 10.4}
    # 아래는 임시로 작성한 코드입니다.
    ### start - create output.json ###
    THRESHOLD_ESTIMATION_VALUE_KEY = "thresholdEstimationValue"
    data = {
        THRESHOLD_ESTIMATION_VALUE_KEY: 10.4
    }
    with open(output_json_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    ### end - create output.json ###


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_weight_file",
        type=str,
        default="/weights/oilspill.h5",
        required=False,
        help="path of trained model.",
    )
    parser.add_argument(
        "--input_grd_file",
        type=str,
        default="/platform/data/input/0/input.tif",
        required=False,
        help="path of your input tif file."
    )
    parser.add_argument(
        "--output_grd_file",
        type=str,
        default="/platform/data/output/0/output.tif",
        required=False,
        help="path of your output tif file."
    )
    parser.add_argument(
        "--output_json_file",
        type=str,
        default="/platform/data/output/0/output.json",
        required=False,
        help="path of your output json file."
    )
    parser.add_argument(
        "--patch_size",
        type=int,
        default=1024,
        required=False,
        help="input image patch size"
    )
    parser.add_argument(
        "--channel", "-c", 
        type=int, 
        action="append", 
        help="-c 1 -c 3"
    )
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    start_time = time.time()

    args = get_args()

    process(args.input_grd_file, args.output_grd_file, args.output_json_file, args.model_weight_file, args.patch_size, args.channel_list)

    processed_time = time.time() - start_time
    logging.info(f"{processed_time:.2f} seconds")
