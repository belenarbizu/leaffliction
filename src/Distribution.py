import os
import argparse
import pathlib 
from pathlib import Path
from collections import Counter
import matplotlib.pyplot as plt


def get_directory(directory):
    """
    Get all image files in the given directory and its subdirectories.
    Returns a list of file paths.
    """
    try:
        images = []
        path = pathlib.Path(directory)
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


def plot_pie(num_images, dir_name):
    """
    Plot a pie chart of the image distribution.
    """
    plt.figure(figsize=(10, 8))

    plt.pie(num_images.values(), labels=num_images.keys(), autopct='%1.1f%%', startangle=140, colors=['tan', 'peru', 'darkgoldenrod', 'saddlebrown', 'olivedrab', 'darkolivegreen', 'rosybrown', 'grey'])
    plt.title(f'Image Distribution: {dir_name}')
    plt.axis('equal')

    plt.savefig(f'{dir_name}_pie_chart.png')


def plot_bar(num_images, dir_name):
    """
    Plot a bar chart of the image distribution.
    """
    plt.figure(figsize=(10, 8))

    bars = plt.bar(num_images.keys(), num_images.values(), color=['tan', 'peru', 'darkgoldenrod', 'saddlebrown', 'olivedrab', 'darkolivegreen', 'rosybrown', 'grey'])
    plt.xlabel('Subdirectories')
    plt.ylabel('Number of Images')
    plt.title(f'Image Distribution: {dir_name}')
    plt.xticks(rotation=45)

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.5, yval, ha='center', va='bottom')

    plt.tight_layout()

    plt.savefig(f'{dir_name}_bar_chart.png')


def main():
    parser = argparse.ArgumentParser(description="Prompt pie chart and bar chart of the directory")
    parser.add_argument("directory", type=str, help="Directory path to analyze")
    args = parser.parse_args()

    if len(vars(args)) != 1:
        print("Usage: python3 ./Distribution directory")
        exit(1)

    directory = args.directory
    images = get_directory(directory)
    if not images:
        exit(1)
    num_images = count_images(images)
    plot_pie(num_images, Path(directory).name)
    plot_bar(num_images, Path(directory).name)


if __name__ == "__main__":
    main()