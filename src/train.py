import argparse
import os
import tensorflow as tf
import warnings
from utils import get_directory
from tensorflow.keras import layers, models
from pathlib import Path
import pandas as pd
import zipfile

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings("ignore")


def check_dir(directory):
    subdirs = [os.path.join(directory, d) for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
    if len(subdirs) < 2:
        print("The directory must contain at least two subdirectories for classification.")
        exit(1)


def split_dataset(data_dir, split_ratio):
    """
    Split the dataset into training and validation sets based on the given ratio.
    """
    train_df, val_df = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=1 - split_ratio,
        subset="both",
        seed=42,
        image_size=(128, 128)
    )
    return train_df, val_df


def train(train_data, val_data):
    """
    Train a simple CNN model on the training data and validate on the validation data.
    """
    model = models.Sequential([
        layers.Rescaling(1.0 / 255),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(32, (1, 1), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dense(256, activation='relu'),
        layers.Dense(len(train_data.class_names), activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(train_data, validation_data=val_data, epochs=3)
    return model


def create_zip(model, class_names):
    model.save("model/model.h5")
    pd.DataFrame({'class_names': class_names}).to_csv("model/class_names.csv", index=False)

    with zipfile.ZipFile("model.zip", "w") as zipf:
        zipf.write("model/model.h5")
        zipf.write("model/class_names.csv")

    os.system("sha1sum model.zip > model/sha1sum.txt")


def main():
    parser = argparse.ArgumentParser(description="Split dataset and train model.")
    parser.add_argument("data_dir", type=str, help="Directory containing the dataset.", default=None)
    parser.add_argument("-split", "--split", type=float, help="Ratio to split the dataset", default=0.8)
    args = parser.parse_args()

    check_dir(args.data_dir)
    images = get_directory(args.data_dir)
    if not images:
        exit(1)
    train_data, val_data = split_dataset(args.data_dir, args.split)

    model = train(train_data, val_data)
    results = model.evaluate(val_data)
    print(f"Validation Loss: {results[0]}, Validation Accuracy: {results[1]}")

    create_zip(model, train_data.class_names)


if __name__ == "__main__":
    main()