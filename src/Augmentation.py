import os
import cv2
import argparse
import numpy as np
from pathlib import Path
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


def get_augmentations(image):
	return {
		"Rotate": rotate_image(image),
		"Filp":	flip_image(image),
		"Blur": blur_image(image),
		"Contrast": contrast_image(image),
		"Scaling": scale_image(image),
		"Shear": shear_image(image),
	}


def data_augmentation(dir):
	"""
	Augment images in subdirectories to match the maximum image count.
	Each subdirectory with fewer images will be augmented with transformed versions.
	"""
	augmented_directory = "augmented_directory/"
	class_name = os.path.basename(os.path.normpath(dir))
	os.makedirs(augmented_directory, exist_ok=True)
	images = get_directory(dir)
	num_images = count_images(images)
	target_count = max(num_images.values())
	print("Target image count per subdirectory:", target_count)
	images_by_subdir = {}

	for img in images:
		dir_name = img.parent.name
		if dir_name not in images_by_subdir:
			images_by_subdir[dir_name] = []
		images_by_subdir[dir_name].append(img)

	for subdir_name, count in num_images.items(): # loop through subdirs
		needed_images = target_count - count

		for img in images_by_subdir[subdir_name]:
			base_name = os.path.basename(str(img))
			image = cv2.imread(str(img))
			out_path = os.path.join(
				augmented_directory,
				class_name,
				subdir_name,
				base_name
			)
			os.makedirs(os.path.dirname(out_path), exist_ok=True)
			cv2.imwrite(out_path, image)

		print(f"\nCopied {count} files in {subdir_name}, needing {needed_images} more to reach {target_count}")

		if needed_images <= 0:
			print(f"'{subdir_name}': Already has {target_count} images (max count)")
			continue
		
		print(f"Augmenting '{subdir_name}': {count} -> {target_count} images (adding {needed_images})")

		for img in images_by_subdir[subdir_name]: # loop thorugh all photos
			if needed_images <= 0:
				break

			image = cv2.imread(str(img))
			base_name = os.path.splitext(os.path.basename(str(img)))[0]
			img_path = Path(img)
			ext = img_path.suffix

			augmentations = get_augmentations(image)
			
			for i in range(len(augmentations.values())): # save in directory
				if needed_images <= 0:
					break
				aug_name = list(augmentations.keys())[i]
				aug_image = list(augmentations.values())[i]
				out_path = os.path.join(
					augmented_directory,
					class_name,
					subdir_name,
					f"{base_name}_{aug_name}{ext}"
				)
				cv2.imwrite(out_path, aug_image)			
				needed_images -= 1

		if needed_images > 0:
			print(f"Impossible to rach {target_count}: (lacking {needed_images})")
		else:
			print(f"Copied {target_count - count} files in {subdir_name}")


def main():
	parser = argparse.ArgumentParser(description="Data Augmentation in directory and transformation on a single image")
	parser.add_argument("path", type=str, help="Path to directory to grow or the image to transform")
	args = parser.parse_args()
	if os.path.isdir(args.path):
		for root, dirs, files in os.walk(args.path):
			image_paths = [os.path.join(root, f) for f in files if f.lower().endswith((".jpg", ".jpeg", ".png"))]
			if len(image_paths) == 0 :
				continue
			print("\nProcessing directory:", root)
			print("IMAGE PATHS:", image_paths)
			# target_count = max(count_images(image_paths).values())
			data_augmentation(root, )
	else:
		image = cv2.imread(args.path)
		if image is None:
			print("Image not found:", image)
			return

		augmentations = get_augmentations(image)
		display_images(image, augmentations.keys(), augmentations.values())


if __name__ == "__main__":
	main()
