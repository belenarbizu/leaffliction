import os
from pathlib import Path
from collections import Counter


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
