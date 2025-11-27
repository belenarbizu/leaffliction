
import cv2
import os
import rembg
from plantcv import plantcv as pcv
from pathlib import Path


def gaussian_blur(image, mask=None, plot=True, destination=None, file_name=None):
    """
    Apply Gaussian blur to the image.
    """
    # convert to grayscale using HSV channel 's'
    gray = pcv.rgb2gray_hsv(rgb_img=image, channel="s")
    # convert image to binary (white or black) using a threshold value of 60
    binary = pcv.threshold.binary(gray_img=gray, threshold=60, object_type="light")
    blurred_image = pcv.gaussian_blur(img=binary, ksize=(5, 5), sigma_x=0, sigma_y=None)
    if plot:
        pcv.plot_image(blurred_image, title="Gaussian Blurred Image")
    if destination and file_name:
        try:
            save_path = Path(destination) / f"{file_name}_gaussian_blur.JPG"
            os.makedirs(destination, exist_ok=True)
            cv2.imwrite(str(save_path), blurred_image)
        except Exception as e:
            print(f"Error saving Gaussian blurred image: {e}")
    return blurred_image


def mask(image, mask=None, plot=True, destination=None, file_name=None):
    """
    Apply masking to the image.
    """
    gray = pcv.rgb2gray_hsv(image, channel='s')
    mask_binary = pcv.threshold.binary(gray, threshold=85, object_type='light')
    masked_image = pcv.apply_mask(image, mask=mask_binary, mask_color="white")
    if plot:
        pcv.plot_image(masked_image, title="Masked Image")
    if destination and file_name:
        try:
            save_path = Path(destination) / f"{file_name}_masked.JPG"
            os.makedirs(destination, exist_ok=True)
            cv2.imwrite(str(save_path), masked_image)
        except Exception as e:
            print(f"Error saving masked image: {e}")
    return masked_image


def roi_object(image, mask=None, plot=True, destination=None, file_name=None):
    """
    Define region of interest on the image.
    """
    # remove background
    image_without_bg = rembg.remove(image)
    # convert to grayscale using hsv channel 's'
    l_grayscale = pcv.rgb2gray_hsv(image_without_bg, channel='s')
    # create binary image using threshold. the pixels above the threshold are set to white
    l_thresh = pcv.threshold.binary(gray_img=l_grayscale, threshold=85, object_type='light')
    # fill small objects. eliminate objects smaller than 200 pixels to reduce noise
    filled = pcv.fill(bin_img=l_thresh, size=200)
    # define ROI
    roi = pcv.roi.rectangle(img=image, x=0, y=0, h=image.shape[0], w=image.shape[1])
    # filter the filled image using the ROI
    kept_mask = pcv.roi.filter(mask=filled, roi=roi, roi_type='partial')
    roi_image = image.copy()
    # highlight the ROI in green
    roi_image[kept_mask != 0] = (0, 255, 0)
    if plot:
        pcv.plot_image(roi_image, title="ROI Image")
    if destination and file_name:
        try:
            save_path = Path(destination) / f"{file_name}_roi.JPG"
            os.makedirs(destination, exist_ok=True)
            cv2.imwrite(str(save_path), roi_image)
        except Exception as e:
            print(f"Error saving ROI image: {e}")
    return kept_mask


def analyze_image(image, mask=None, plot=True, destination=None, file_name=None):
    """
    Analyze the image using the given mask.
    """
    mask = roi_object(image, plot=False)
    analyze = pcv.analyze.size(img=image, labeled_mask=mask)
    if plot:
        pcv.plot_image(analyze, title="Analyzed Image")
    if destination and file_name:
        try:
            save_path = Path(destination) / f"{file_name}_analyzed.JPG"
            os.makedirs(destination, exist_ok=True)
            cv2.imwrite(str(save_path), analyze)
        except Exception as e:
            print(f"Error saving analyzed image: {e}")
    return analyze


def edges_image(image, mask=None, plot=True, destination=None, file_name=None):
    """
    Find edges in the image.
    """
    if mask is None:
        mask = image
    edges = cv2.Canny(mask, 150, 200)
    if plot:
        pcv.plot_image(edges)
    if destination and file_name:
        try:
            save_path = Path(destination) / f"{file_name}_edges.JPG"
            os.makedirs(destination, exist_ok=True)
            cv2.imwrite(str(save_path), edges)
        except Exception as e:
            print(f"Error saving edges image: {e}")
    return edges


def negative_image(image, mask=None, plot=True, destination=None, file_name=None):
    """
    Invert the image colors (negative).
    """
    negative = cv2.bitwise_not(image)
    if plot:
        pcv.plot_image(negative)
    if destination and file_name:
        try:
            save_path = Path(destination) / f"{file_name}_negative.JPG"
            os.makedirs(destination, exist_ok=True)
            cv2.imwrite(str(save_path), negative)
        except Exception as e:
            print(f"Error saving negative image: {e}")
    return negative

