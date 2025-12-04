import os
import cv2
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
    Returns a Counter object with directory names
    as keys and image counts as values.
    """
    num_images = Counter()
    for img in images:
        dir_name = img.parent.name
        num_images[dir_name] += 1
    return num_images


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

    # Recursively search for any images in this directory or subdirectories
    has_images = False
    for img_file in src_path.rglob('*'):
        if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
            has_images = True
            break

    if not has_images:
        print(f"Error: No images found in {src} or its subdirectories")
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
