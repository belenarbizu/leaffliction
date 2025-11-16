from html import parser
import os
import sys
import cv2
import argparse
import numpy as np
import matplotlib.pyplot as plt
from utils import get_directory, count_images


def rotate_image(image, angle=45):
	"""
	Rotate the input image by the specified angle (left to impliment).
	"""
	rotated_image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
	return rotated_image


def flip_image(image):
	"""
	Flip the input image horizontally.
	"""
	flipped_image = cv2.flip(image, 1)
	return flipped_image


def blur_image(image, ksize=(10, 10)):
	""""
	Blur the input image.
	"""
	blurred_image = cv2.blur(image, ksize)
	return blurred_image


def contrast_image(image, alpha=2.0, beta=-10):
	"""
	Adjust the contrast of the input image.
	"""
	contrasted_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
	return contrasted_image


def scale_image(image, zoom_factor = 1.5):
	"""
	Scale the input image by croping and then resizing back to original.
	"""
	h, w = image.shape[:2]
	ch, cw = int(h/zoom_factor), int(w/zoom_factor)
	start_h, start_w = (h - ch)//2, (w - cw)//2
	crop = image[start_h:start_h+ch, start_w:start_w+cw]
	zoomed_image = cv2.resize(crop, (w, h))
	return zoomed_image


def shear_image(image, factor=0.2):
	"""
	Apply a projective transformation to the input image using 2x3 matrix.
	"""
	h, w = image.shape[:2]
	tx, ty = 0, 20  # Shift 100 pixels right and 50 pixels down
	translation_matrix = np.float32([
		[1, factor, tx],
		[factor, 1, ty]]
	)
	sheared_image = cv2.warpAffine(image, translation_matrix, (w, h))
	return sheared_image


def display_images(org_image, titles, images):
	titles = list(titles)
	images = list(images)
	titles.insert(0, "Original")
	images.insert(0, org_image)
	plt.figure(figsize=(18, 10))
	for i in range(len(images)):
		plt.subplot(2, 4, 1 + i)
		plt.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))
		plt.title(titles[i])
		plt.axis('off')

	plt.tight_layout()
	plt.show()


def data_augmentation(dir):
	augmented_directory = "augmented_directory/"
	os.makedirs(augmented_directory, exist_ok=True)
	images = get_directory(dir)
	print("Found", len(images), "images for augmentation.")
	# save in directory
	# out_path = os.path.join(augmented_directory, f"{os.path.basename(img_path)[:-4]}_{aug_name}" + '.JPG')
	# cv2.imwrite(out_path, aug_image)


def main():
	parser = argparse.ArgumentParser(description="Data Augmentation in directory and transformation on a single image")
	parser.add_argument("path", type=str, help="Path to directory to grow or the image to transform")
	args = parser.parse_args()
	if os.path.isdir(args.path):
		data_augmentation(args.path)
	else:
		image = cv2.imread(args.path)
		if image is None:
			print("Image not found:", image)
			return

		augmentations = {
			"Rotate": rotate_image(image),
			"Filp":	flip_image(image),
			"Blur": blur_image(image),
			"Contrast": contrast_image(image),
			"Scaling": scale_image(image),
			"Shear": shear_image(image),
		}
		display_images(image, augmentations.keys(), augmentations.values())


if __name__ == "__main__":
	main()
