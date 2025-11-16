import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from utils import get_directory, count_images


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
    num_images = count_images(images)
    plot_pie(num_images, Path(directory).name)
    plot_bar(num_images, Path(directory).name)


if __name__ == "__main__":
    main()