import os
import sys
import cv2


# def augment_data(input_data):	"""
# 	Perform data augmentation on the input data.

# 	Parameters:
# 	input_data (list): A list of data samples to be augmented.

# 	Returns:
# 	list: A list of augmented data samples.
# 	"""
# 	augmented_data = []


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


def main():
	augmented_directory = "augmented_directory/"
	os.makedirs(augmented_directory, exist_ok=True)
	img_path = sys.argv[1]
	image = cv2.imread(img_path)

	augmentations = {
		"Filp":	flip_image,
		"Rotate": rotate_image,

	}
	for aug_name, aug_func in augmentations.items():
		aug_image = aug_func(image)
		out_path = os.path.join(augmented_directory, f"{os.path.basename(img_path)[:-4]}_{aug_name}" + '.JPG'	)
		cv2.imwrite(out_path, aug_image)


if __name__ == "__main__":
	main()