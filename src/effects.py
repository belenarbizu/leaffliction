
import cv2
import os
import rembg
import numpy as np
from plantcv import plantcv as pcv
from pathlib import Path


def gaussian_blur(image, mask=None, plot=True):
    """
    Apply Gaussian blur to the image.
    """
    try:
        # convert to grayscale using HSV channel 's'
        gray = pcv.rgb2gray_hsv(rgb_img=image, channel="s")
        # convert image to binary (white or black) using a threshold value of 60
        binary = pcv.threshold.binary(gray_img=gray, threshold=60, object_type="light")
        blurred_image = pcv.gaussian_blur(img=binary, ksize=(5, 5), sigma_x=0, sigma_y=None)
        if plot:
            pcv.plot_image(blurred_image, title="Gaussian Blurred Image")
    except Exception as e:
        print(f"Error creating Gaussian blurred image: {e}")
        return None
    return blurred_image


def mask(image, mask=None, plot=True):
    """
    Apply masking to the image.
    """
    try:
        gray = pcv.rgb2gray_hsv(image, channel='s')
        mask_binary = pcv.threshold.binary(gray, threshold=85, object_type='light')
        masked_image = pcv.apply_mask(image, mask=mask_binary, mask_color="white")
        if plot:
            pcv.plot_image(masked_image, title="Masked Image")
    except Exception as e:
        print(f"Error creating masked image: {e}")
        return None
    return masked_image


def roi_object(image, mask=None, plot=True):
    """
    Define region of interest on the image.
    """
    try:
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
    except Exception as e:
        print(f"Error creating ROI image: {e}")
        return None
    return roi_image, kept_mask


def analyze_image(image, mask=None, plot=True):
    """
    Analyze the image using the given mask.
    """
    try:
        analyze = pcv.analyze.size(img=image, labeled_mask=mask)
        if plot:
            pcv.plot_image(analyze, title="Analyzed Image")
    except Exception as e:
        print(f"Error creating analyzed image: {e}")
        return None
    return analyze


def edges_image(image, mask=None, plot=True):
    """
    Find edges in the image.
    """
    try:
        if mask is None:
            mask = image
        edges = cv2.Canny(mask, 150, 200)
        if plot:
            pcv.plot_image(edges)
    except Exception as e:
        print(f"Error creating edges image: {e}")
        return None
    return edges


def negative_image(image, mask=None, plot=True):
    """
    Invert the image colors (negative).
    """
    try:
        negative = cv2.bitwise_not(image)
        if plot:
            pcv.plot_image(negative)
    except Exception as e:
        print(f"Error creating negative image: {e}")
        return None
    return negative


def posterize(image, mask=None, plot=True):
    """
    It takes all the millions of original colors and groups them into just a few levels (4).
    """
    try:
        levels = 4
        shift = 256 // levels
        posterized_image = (image // shift) * shift
        if plot:
            pcv.plot_image(posterized_image, title="Posterized Image")
    except Exception as e:
        print(f"Error creating posterized image: {e}")
        return None
    return posterized_image


def sharpen(image, mask=None, plot=True):
    """
    Simple sharpening filter.
    """
    try:
        kernel = np.array([[0,-1,0],
                        [-1,5,-1],
                        [0,-1,0]])
        sharp = cv2.filter2D(image, -1, kernel)
        if plot:
            pcv.plot_image(sharp, title="Sharpened")
    except Exception as e:
        print(f"Error creating sharpened image: {e}")
        return None
    return sharp
