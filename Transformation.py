import argparse
import cv2
from plantcv import plantcv as pcv
from pathlib import Path


def check_path(file_path):
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
    path = Path(dir_path)
    if not path.exists():
        print(f"Error: The directory {dir_path} does not exist.")
        exit(1)
    images = []
    for img_file in path.glob('*'):
        image = cv2.imread(str(img_file))
        if image is not None:
            images.append(image)
    if not images:
        print(f"Error: No valid images found in directory {dir_path}")
        return
    return images


def gaussian_blur(image, plot=True):
    # convert to grayscale using HSV channel 's'
    gray = pcv.rgb2gray_hsv(rgb_img=image, channel="s")
    # convert image to binary (white or black) using a threshold value of 60
    binary = pcv.threshold.binary(gray_img=gray, threshold=60, object_type="light")
    blurred_image = pcv.gaussian_blur(img=binary, ksize=(5, 5), sigma_x=0, sigma_y=None)
    if plot:
        pcv.plot_image(blurred_image, title="Gaussian Blurred Image")


def mask(image, plot=True):
    gray = pcv.rgb2gray_hsv(image, channel='s')
    mask_binary = pcv.threshold.binary(gray, threshold=85, object_type='light')
    masked_image = pcv.apply_mask(image, mask=mask_binary, mask_color="white")
    if plot:
        pcv.plot_image(masked_image, title="Masked Image")


def roi_object(image, plot=True):
    pcv.params.debug = "plot" if plot else None
    pcv.roi.rectangle(img=image, x=0, y=0, h=image.shape[0], w=image.shape[1])


def main():
    parser = argparse.ArgumentParser(description="Display image transformations.")
    parser.add_argument("path", type=str, help="File path image to be transformed.", default=None, nargs='?')
    parser.add_argument("-src", "--source", type=str, help="Directory source of the image.", default=None, nargs='?')
    parser.add_argument("-dst", "--destination", type=str, help="Directory destination to save transformed images.", default=None, nargs='?')
    parser.add_argument("-gaussian", "--gaussian", action="store_true", help="Apply Gaussian blur to the image.")
    parser.add_argument("-mask", "--mask", action="store_true", help="Apply masking to the image.")
    parser.add_argument("-roi", "--roi", action="store_true", help="Define region of interest on the image.")
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
        mask(image)
        roi_object(image)
    if args.source and args.destination:
        images = check_directory(args.source)
        for img in images:
            gaussian_blur(img, plot=False)
            mask(img, plot=False)
            roi_object(img, plot=False)

if __name__ == "__main__":
    main()