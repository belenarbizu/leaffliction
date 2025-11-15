import argparse
import cv2
from plantcv import plantcv as pcv
from pathlib import Path


def check_path(file_path):
    path = Path(file_path)
    if path.is_file():
        image = cv2.imread(str(path))
        if image is None:
            print(f"Error: Unable to load image at {path}")
            return
        return image, "file"
    elif path.is_dir():
        images = []
        for img_file in path.glob('*'):
            image = cv2.imread(str(img_file))
            if image is not None:
                images.append(image)
        if not images:
            print(f"Error: No valid images found in directory {path}")
            return
        return images, "directory"


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
    parser.add_argument("path", type=str, help="File path image or directory with images to be transformed.")
    args = parser.parse_args()

    image, image_type = check_path(args.path)
    if image_type == "directory":
        for img in image:
            gaussian_blur(img, plot=False)
            mask(img, plot=False)
            roi_object(img, plot=False)
    else:
        gaussian_blur(image)
        mask(image)
        roi_object(image)

if __name__ == "__main__":
    main()