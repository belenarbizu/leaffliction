import argparse
from pathlib import Path
import zipfile
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np


def check_path(image_path):
    try:
        path = Path(image_path)
        if not path.is_file():
            print(f"Error: The specified image file does not exist: {image_path}")
    except Exception as e:
        print(f"Error: Invalid path provided: {image_path}.")
        exit(1)


def open_model():
    try:
        with zipfile.ZipFile("model.zip", "r") as zip_ref:
            zip_ref.extractall(".")
    except Exception as e:
        print(f"Error extracting model: {e}")
        exit(1)
    return "model/model.h5", "model/class_names.csv"


def load_class_names(class_names_path):
    try:
        df = pd.read_csv(class_names_path)
        class_names = df.iloc[:, 0].tolist()
        return class_names
    except Exception as e:
        print(f"Error loading class names: {e}")
        return []


def predict_image(image_path, model_path, class_names):
    model = load_model(model_path)
    img = image.load_img(image_path, target_size=(128, 128))
    img_array = image.img_to_array(img)
    img_normalized = img_array / 255.0  # Normalize the image
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    predictions = model.predict(img_array, verbose=0)
    predicted_class_index = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class_index]
    predicted_class_name = class_names[predicted_class_index]
    print(f"Predicted class: {predicted_class_name} with confidence {confidence:.2f}")


def main():
    argparser = argparse.ArgumentParser(description="Predict using a trained model.")
    argparser.add_argument(type=str, dest="image", help="Path to the image file to classify.")
    args = argparser.parse_args()

    check_path(args.image)
    model_path, class_names_path = open_model()
    class_names = load_class_names(class_names_path)
    if not class_names:
        print("No class names found.")
        return
    predict_image(args.image, model_path, class_names)


if __name__ == "__main__":
    main()