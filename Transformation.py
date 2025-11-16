import argparse
import cv2
from plantcv import plantcv as pcv
from pathlib import Path
import rembg
import os


def check_path(file_path):
    """
    Check if the given file path exists and is a file. Load and return the image if valid.
    """
    path = Path(file_path)
    if not path.exists():
        print(f"Error: The path {path} does not exist.")
        exit(1)
    if path.is_dir():
        print(f"Error: The path {path} is a directory, expected a file.")
        exit(1)
    if path.is_file():
        image = cv2.imread(str(path))
        if image is None:
            print(f"Error: Unable to load image at {path}")
            exit(1)
        return image


def check_directory(dir_path):
    """
    Check if the given directory path exists. Load and return all images in the directory.
    """
    path = Path(dir_path)
    if not path.exists():
        print(f"Error: The directory {dir_path} does not exist.")
        exit(1)
    images = []
    files_names = []
    for img_file in path.glob('*'):
        image = cv2.imread(str(img_file))
        if image is not None:
            images.append(image)
            files_names.append(img_file.name.removesuffix(".JPG"))
    if not images:
        print(f"Error: No valid images found in directory {dir_path}")
        return
    return images, files_names


def gaussian_blur(image, plot=True, destination=None, file_name=None):
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
    try:
        if destination and file_name:
            save_path = Path(destination) / f"{file_name}_gaussian_blur.JPG"
            os.makedirs(destination, exist_ok=True)
            cv2.imwrite(str(save_path), blurred_image)
    except Exception as e:
        print(f"Error saving Gaussian blurred image: {e}")


def mask(image, plot=True, destination=None, file_name=None):
    """
    Apply masking to the image.
    """
    gray = pcv.rgb2gray_hsv(image, channel='s')
    mask_binary = pcv.threshold.binary(gray, threshold=85, object_type='light')
    masked_image = pcv.apply_mask(image, mask=mask_binary, mask_color="white")
    if plot:
        pcv.plot_image(masked_image, title="Masked Image")
    try:
        if destination and file_name:
            save_path = Path(destination) / f"{file_name}_masked.JPG"
            os.makedirs(destination, exist_ok=True)
            cv2.imwrite(str(save_path), masked_image)
    except Exception as e:
        print(f"Error saving masked image: {e}")
    return masked_image


def roi_object(image, masked, plot=True, destination=None, file_name=None):
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
    try:
        if destination and file_name:
            save_path = Path(destination) / f"{file_name}_roi.JPG"
            os.makedirs(destination, exist_ok=True)
            cv2.imwrite(str(save_path), roi_image)
    except Exception as e:
        print(f"Error saving ROI image: {e}")
    return kept_mask


def analyze_image(image, mask, plot=True, destination=None, file_name=None):
    """
    Analyze the image using the given mask.
    """
    analyze = pcv.analyze.size(img=image, labeled_mask=mask)
    if plot:
        pcv.plot_image(analyze, title="Analyzed Image")
    try:
        if destination and file_name:
            save_path = Path(destination) / f"{file_name}_analyzed.JPG"
            os.makedirs(destination, exist_ok=True)
            cv2.imwrite(str(save_path), analyze)
    except Exception as e:
        print(f"Error saving analyzed image: {e}")


def main():
    parser = argparse.ArgumentParser(description="Display image transformations.")
    parser.add_argument("path", type=str, help="File path image to be transformed.", default=None, nargs='?')
    parser.add_argument("-src", "--source", type=str, help="Directory source of the image.", default=None, nargs='?')
    parser.add_argument("-dst", "--destination", type=str, help="Directory destination to save transformed images.", default=None, nargs='?')
    parser.add_argument("-gaussian", "--gaussian", action="store_true", help="Apply Gaussian blur to the image.")
    parser.add_argument("-mask", "--mask", action="store_true", help="Apply masking to the image.")
    parser.add_argument("-roi", "--roi", action="store_true", help="Define region of interest on the image.")
    parser.add_argument("-analyze", "--analyze", action="store_true", help="Analyze the image.")
    args = parser.parse_args()

    if (args.path is None and args.source is None) or (args.path is not None and args.source is not None) or (args.path is not None and args.destination is not None):
        print("Error: Please provide either a file path or a source directory.")
        exit(1)
    if (args.source is not None and args.destination is None) or (args.source is None and args.destination is not None):
        print("Error: Please provide both source and destination directories for batch processing.")
        exit(1)

    if args.path:
        image = check_path(args.path)
        gaussian_blur(image)
        masked_image = mask(image)
        roi_mask = roi_object(image, masked_image)
        analyze_image(image, roi_mask)
    if args.source and args.destination:
        images, files_names = check_directory(args.source)
        for img, file_name in zip(images, files_names):
            if args.gaussian:
                gaussian_blur(img, plot=False, destination=args.destination, file_name=file_name)
            if args.mask:
                masked_image = mask(img, plot=False, destination=args.destination, file_name=file_name)
            if args.roi:
                masked_image = mask(img, plot=False,)
                roi_mask = roi_object(img, masked_image, plot=False, destination=args.destination, file_name=file_name)
            if args.analyze:
                masked_image = mask(img, plot=False,)
                roi_mask = roi_object(img, masked_image, plot=False)
                analyze_image(img, roi_mask, plot=False, destination=args.destination, file_name=file_name)
            

if __name__ == "__main__":
    main()