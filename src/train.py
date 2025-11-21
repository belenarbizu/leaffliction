import argparse
import os
import tensorflow as tf
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings("ignore")


def split_dataset(data_dir, split_ratio):
    """
    Split the dataset into training and validation sets based on the given ratio.
    """
    train_df = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=1 - split_ratio,
        subset="training",
        seed=42,
        image_size=(256, 256),
        # batch_size=32
    )
    shape = tf.TensorShape([None,140,140,3])

    val_df = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=1 - split_ratio,
        subset="validation",
        seed=42,
        image_size=(256, 256),
        # batch_size=32
    )


def main():
    parser = argparse.ArgumentParser(description="Split dataset and train model.")
    parser.add_argument("data_dir", type=str, help="Directory containing the dataset.", default=None)
    parser.add_argument("-split", "--split", type=float, help="Ratio to split the dataset", default=0.8)
    args = parser.parse_args()
    split_dataset(args.data_dir, args.split)


if __name__ == "__main__":
    main()