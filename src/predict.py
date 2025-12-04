import argparse
from pathlib import Path
import zipfile
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from utils import display_image


def check_path(image_path):
    """
    Check if the provided image path is valid.
    """
    try:
        path = Path(image_path)
        if not path.exists() or not path.is_file():
            print("Error: The specified path is not a file or does not exist.")
            exit(1)
    except Exception as e:
        print(f"Error: {e}.")
        exit(1)


def open_model(model_zip_path):
    """
    Extract the model and class names from the zip file.
    Returns the paths to the model file and class names file.
    """
    try:
        with zipfile.ZipFile(model_zip_path, "r") as zip_ref:
            zip_ref.extractall(".")
    except Exception as e:
        print(f"Error extracting model: {e}")
        exit(1)
    return f"model/{model_zip_path.split('_model.zip')[0]}_model.h5", \
        f"model/{model_zip_path.split('_model.zip')[0]}_class_names.csv"


def load_class_names(class_names_path):
    """
    Load class names from the CSV file.
    Returns a list of class names.
    """
    try:
        df = pd.read_csv(class_names_path)
        class_names = df.iloc[:, 0].tolist()
        return class_names
    except Exception as e:
        print(f"Error loading class names: {e}")
        return []


def predict_image(image_path, model_path, class_names):
    """
    Predict the class of the given image using the loaded model.
    """
    model = load_model(model_path)

    img = image.load_img(image_path, target_size=(128, 128))
    img_array = image.img_to_array(img)
    _ = img_array / 255.0  # Normalize the image
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    predictions = model.predict(img_array, verbose=0)
    predicted_class_index = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class_index]
    predicted_class = class_names[predicted_class_index]

    print(f"\nPredicted class: {predicted_class}. Confidence {confidence:.2f}")
    return predictions, predicted_class


def main():
    argparser = argparse.ArgumentParser(
        description="Predict using a trained model."
    )
    argparser.add_argument(
        type=str,
        dest="image",
        help="Path to the image file to classify."
    )
    argparser.add_argument(
        "-m",
        "--model",
        required=True,
        type=str,
        dest="model",
        help="Path to the model zip file."
    )
    args = argparser.parse_args()

    check_path(args.image)
    model_path, class_names_path = open_model(args.model)
    class_names = load_class_names(class_names_path)
    if not class_names:
        print("No class names found.")
        return
    predictions, predicted_class = predict_image(
        args.image, model_path, class_names
    )
    display_image(args.image, predictions, predicted_class, class_names)


if __name__ == "__main__":
    main()
