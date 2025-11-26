import argparse
import cv2
from plantcv import plantcv as pcv
from pathlib import Path
import rembg
import os
from utils import get_directory
import matplotlib.pyplot as plt
import numpy as np


def validate_source_directory(src):
    """
    Validate source directory: must exist and contain subdirectories with photos.
    """
    src_path = Path(src)
    if not src_path.exists():
        print(f"Error: Source directory {src} does not exist.")
        exit(1)
    if not src_path.is_dir():
        print(f"Error: {src} is not a directory.")
        exit(1)

    # Check if subdirectories contain images
    has_images = False
    for subdir in src_path.iterdir():
        if subdir.is_dir():
            images = list(subdir.glob('*.jpg')) + list(subdir.glob('*.jpeg')) + list(subdir.glob('*.png'))
            if images:
                has_images = True
                break

    if not has_images:
        print(f"Error: No images found in subdirectories of {src}")
        exit(1)

    return src_path


def validate_source_file(src):
    """
    Validate source file: must exist and be a valid image file.
    """
    src_path = Path(src)
    if not src_path.exists():
        print(f"Error: Source file {src} does not exist.")
        exit(1)
    if not src_path.is_file():
        print(f"Error: {src} is not a file.")
        exit(1)

    image = cv2.imread(str(src_path))
    if image is None:
        print(f"Error: Unable to load image at {src}")
        exit(1)

    return src_path, image


def validate_destination_directory(dst, create=True):
    """
    Validate destination directory: must be a directory (or create if create=True).
    """
    dst_path = Path(dst)
    if dst_path.exists() and not dst_path.is_dir():
        print(f"Error: Destination {dst} exists but is not a directory.")
        exit(1)

    if not dst_path.exists() and create:
        os.makedirs(dst_path, exist_ok=True)
        print(f"Created destination directory: {dst}")

    return dst_path


def validate_inputs(src, dst):
    """
    Main validation logic:
    - If src is directory: check it exists and has subdirectories with photos, create dst if needed
    - If src is file: check it exists as valid image, dst must be valid directory
    """
    src_path = Path(src)
    
    # Case 1: src is a directory
    if src_path.is_dir():
        src_path = validate_source_directory(src)
        if dst is None:
            print("Error: Destination directory required for batch processing.")
            exit(1)
        dst_path = validate_destination_directory(dst, create=True)
        return 'directory', src_path, dst_path
    
    # Case 2: src is a file
    else:
        src_path, image = validate_source_file(src)
        if dst is None:
            print("Error: Destination directory required for single image processing.")
            exit(1)
        dst_path = validate_destination_directory(dst, create=True)
        return 'file', src_path, dst_path, image


# TRANSFORMATION FUNCTIONS


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
    # before had mask here i deleted it analyze = pcv.analyze.size(img=image, labeled_mask=mask)
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


def get_all_transformations():
    """
    Return all 6 transformation functions.
    """
    return {
        'gaussian': gaussian_blur,
        'mask': mask,
        'roi': roi_object,
        'analyze': analyze_image,
        'edges': edges_image,
        'negative': negative_image,
    }


def process_directory_with_filter(src_path, dst_path, filter_name):
    """
    Apply ONE specific filter to all images in all subdirectories.
    """
    transformations = get_all_transformations()
    
    if filter_name not in transformations:
        print(f"Error: Unknown filter '{filter_name}'")
        return
    
    filter_func = transformations[filter_name]
    
    for subdir in src_path.iterdir():
        if not subdir.is_dir():
            continue
        
        subdir_name = subdir.name
        image_files = list(subdir.glob('*.jpg')) + list(subdir.glob('*.jpeg')) + list(subdir.glob('*.png'))
        
        for img_file in image_files:
            image = cv2.imread(str(img_file))
            if image is None:
                continue
            
            base_name = img_file.stem
            ext = img_file.suffix
            
            # Apply filter
            transformed = filter_func(image, plot=False)
            
            # Save transformed image
            out_path = dst_path / subdir_name / f"{base_name}_{filter_name}{ext}"
            os.makedirs(out_path.parent, exist_ok=True)
            cv2.imwrite(str(out_path), transformed)
            print(f"Saved: {out_path}")


def process_directory_all_filters(src_path, dst_path):
    """
    Apply ALL 6 filters to all images in all subdirectories.
    """
    transformations = get_all_transformations()
    
    for subdir in src_path.iterdir():
        if not subdir.is_dir():
            continue
        
        subdir_name = subdir.name
        image_files = list(subdir.glob('*.jpg')) + list(subdir.glob('*.jpeg')) + list(subdir.glob('*.png'))
        
        for img_file in image_files:
            image = cv2.imread(str(img_file))
            if image is None:
                continue
            
            base_name = img_file.stem
            ext = img_file.suffix
            
            for filter_name, filter_func in transformations.items():
                try:
                    transformed = filter_func(image, plot=False)
                    out_path = dst_path / subdir_name / f"{base_name}_{filter_name}{ext}"
                    os.makedirs(out_path.parent, exist_ok=True)
                    cv2.imwrite(str(out_path), transformed)
                    print(f"Saved: {out_path}")
                except Exception as e:
                    print(f"Error processing {img_file} with {filter_name}: {e}")


def process_single_image_with_filter(image, filter_name, dst_path, file_name):
    """
    Apply ONE specific filter to a single image and display original + transformed.
    """
    transformations = get_all_transformations()

    if filter_name not in transformations:
        print(f"Error: Unknown filter '{filter_name}'")
        return

    filter_func = transformations[filter_name]
    transformed = filter_func(image, plot=False)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    if len(transformed.shape) == 2:
        plt.imshow(transformed, cmap='gray')
    else:
        plt.imshow(cv2.cvtColor(transformed, cv2.COLOR_BGR2RGB))
    plt.title(f"Transformed - {filter_name}")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

    ext = '.JPG'
    out_path = dst_path / f"{file_name}_comparison_{filter_name}{ext}"
    os.makedirs(out_path.parent, exist_ok=True)

    if len(transformed.shape) == 2:
        transformed_bgr = cv2.cvtColor(transformed, cv2.COLOR_GRAY2BGR)
    else:
        transformed_bgr = transformed

    comparison = cv2.hconcat([image, transformed_bgr])
    cv2.imwrite(str(out_path), comparison)
    print(f"Saved comparison: {out_path}")


def process_single_image_all_filters(image, dst_path, file_name):
    """
    Apply ALL 6 filters to a single image and display original + all 6 transformations in grid.
    """
    transformations = get_all_transformations()

    results = {'original': image}
    masked_image = mask(image)
    for filter_name, filter_func in transformations.items():
        try:
            results[filter_name] = filter_func(image, masked_image, plot=False)
        except Exception as e:
            print(f"Error applying {filter_name}: {e}")

    # Display in 2x4 grid (1 original + 6 transformations)
    plt.figure(figsize=(16, 8))

    plot_idx = 1
    for name in ['original'] + list(transformations.keys()):
        plt.subplot(2, 4, plot_idx)
        img_to_show = results[name]
        if len(img_to_show.shape) == 2:
            plt.imshow(img_to_show, cmap='gray')
        else:
            plt.imshow(cv2.cvtColor(img_to_show, cv2.COLOR_BGR2RGB))
        plt.title(name.capitalize())
        plt.axis('off')
        plot_idx += 1

    plt.tight_layout()
    plt.show()

    ext = '.JPG'
    out_path = dst_path / f"{file_name}_all_transformations{ext}"
    os.makedirs(out_path.parent, exist_ok=True)

    grid_images = [results['original']]
    for filter_name in transformations.keys():
        img = results[filter_name]
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        grid_images.append(img)

    row1 = cv2.hconcat(grid_images[:4])
    row2 = cv2.hconcat(grid_images[4:])
    grid = cv2.vconcat([row1, row2])
    cv2.imwrite(str(out_path), grid)
    print(f"Saved all transformations grid: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Apply transformations to images.")
    parser.add_argument("-src", "--source", type=str, help="Source: directory (with subdirectories) or image file", required=True)
    parser.add_argument("-dst", "--destination", type=str, help="Destination directory (required)", required=True)
    parser.add_argument("-f", "--filter", type=str, default=None, 
                        help="Apply specific filter (gaussian, mask, roi, analyze, edges, negative). If not provided, apply all 6.")
    args = parser.parse_args()

    src = args.source
    dst = args.destination
    filter_name = args.filter

    src_path = Path(src)

    # src is a DIRECTORY
    if src_path.is_dir():
        print(f"Mode: Batch processing directory")
        src_path = validate_source_directory(src)
        dst_path = validate_destination_directory(dst, create=True)

        if filter_name:
            print(f"Applying filter: {filter_name} to all images in subdirectories...")
            process_directory_with_filter(src_path, dst_path, filter_name)
        else:
            print(f"Applying all 6 filters to all images in subdirectories...")
            process_directory_all_filters(src_path, dst_path)

    # src is a FILE
    else:
        print(f"Mode: Single image processing")
        src_path, image = validate_source_file(src)
        dst_path = validate_destination_directory(dst, create=True)
        file_name = src_path.stem

        if filter_name:
            print(f"Applying filter: {filter_name} to single image...")
            process_single_image_with_filter(image, filter_name, dst_path, file_name)
        else:
            print(f"Applying all 6 filters to single image and creating grid...")
            process_single_image_all_filters(image, dst_path, file_name)


if __name__ == "__main__":
    main()
