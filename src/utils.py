import os
from pathlib import Path
from collections import Counter, defaultdict


def get_subdirectory(directory):
    """
    Returns a dict:
    {
        "Apple_healthy": [list of files],
        "Apple_rust": [list of files],
        ...
    }
    """
    grouped = defaultdict(list)
    path = Path(directory)

    for file in path.rglob("*"):
        if file.is_file():
            grouped[file.parent.name].append(file)

    return grouped


def get_directory(directory):
    """
    Get all image files in the given directory and its subdirectories.
    Returns a list of file paths.
    """
    try:
        images = []
        path = Path(directory)
        if not os.path.exists(directory):
            print(f"Error: The directory '{directory}' does not exist.")
            return []
        if path.is_file():
            print("Error: Please provide a directory path, not a file path.")
            return []
        for file in path.rglob('*'):
            if file.is_file():
                images.append(file)
        return images
    except FileNotFoundError:
        print(f"Error: '{directory}' doesn't exist.")
        return 0


def count_images(images):
    """
    Count the number of images in each subdirectory.
    Returns a Counter object with directory names as keys and image counts as values.
    """
    num_images = Counter()
    for img in images:
        dir_name = img.parent.name
        num_images[dir_name] += 1
    return num_images


def data_augmentation(parent_dir):
	"""
	Augment images in subdirectories to match the maximum image count.
	Each subdirectory with fewer images will be augmented with transformed versions.
	"""
	parent_path = Path(parent_dir)
	
	if not parent_path.is_dir():
		print(f"Error: '{parent_dir}' is not a directory.")
		return
	
	# Get all subdirectories and count images in each
	subdirs_info = defaultdict(lambda: {"images": [], "count": 0})
	
	for subdir in parent_path.iterdir():
		if subdir.is_dir():
			images = list(subdir.glob('*'))
			image_files = [img for img in images if img.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']]
			subdirs_info[subdir.name]["images"] = image_files
			subdirs_info[subdir.name]["count"] = len(image_files)
			subdirs_info[subdir.name]["path"] = subdir
	
	if not subdirs_info:
		print(f"No subdirectories found in '{parent_dir}'.")
		return
	
	# Find the maximum image count
	max_count = max(info["count"] for info in subdirs_info.values())
	print(f"Maximum image count: {max_count}")
	
	# List of augmentation functions to use
	augmentation_functions = [
		("rotate", rotate_image),
		("flip", flip_image),
		("blur", blur_image),
		("contrast", contrast_image),
		("scale", scale_image),
		("shear", shear_image),
	]
	
	# Augment each subdirectory to reach max_count
	for subdir_name, info in subdirs_info.items():
		current_count = info["count"]
		target_count = max_count
		needed_images = target_count - current_count
		
		if needed_images <= 0:
			print(f"✓ '{subdir_name}': Already has {current_count} images (max count)")
			continue
		
		print(f"\nAugmenting '{subdir_name}': {current_count} → {target_count} images (adding {needed_images})")
		
		original_images = info["images"]
		augmented_count = 0
		aug_func_index = 0
		
		# Loop through and create augmented images
		for i in range(needed_images):
			# Cycle through original images if we need more augmentations than originals
			original_image_path = original_images[i % len(original_images)]
			image = cv2.imread(str(original_image_path))
			
			if image is None:
				print(f"  Warning: Could not read {original_image_path}")
				continue
			
			# Cycle through augmentation functions
			aug_func_name, aug_func = augmentation_functions[aug_func_index % len(augmentation_functions)]
			aug_image = aug_func(image)
			
			# Save augmented image
			base_name = original_image_path.stem
			ext = original_image_path.suffix
			aug_filename = f"{base_name}_aug_{augmented_count}{ext}"
			out_path = info["path"] / aug_filename
			
			cv2.imwrite(str(out_path), aug_image)
			augmented_count += 1
			aug_func_index += 1
			
			print(f"  Saved: {aug_filename}")
		
		print(f"✓ '{subdir_name}': Completed ({augmented_count} augmented images added)")

