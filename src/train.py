import argparse
import os
import random


def split_dataset(data_dir, split_ratio):
    """
    Split the dataset into training and validation sets based on the given ratio.
    """
    train_df = os.apth.join(data_dir, 'train')
    test_df = os.path.join(data_dir, 'val')
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']

def main():
    parser = argparse.ArgumentParser(description="Split dataset and train model.")
    parser.add_argument("data_dir", type=str, help="Directory containing the dataset.", default=None)
    parser.add_argument("-split", "--split_ratio", type=float, help="Ratio to split the dataset", default=0.8)

if __name__ == "__main__":
    main()