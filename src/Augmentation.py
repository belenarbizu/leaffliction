import os
import cv2
import argparse
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from utils import get_directory, count_images, validate_source_directory, validate_source_file


def rotate_image(image, angle=45):
    """
    Rotate the input image by the specified angle (left to impliment).
    """
    rotated_image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    return rotated_image


def flip_image(image):
    """
    Flip the input image horizontally.
    """
    flipped_image = cv2.flip(image, 1)
    return flipped_image


def blur_image(image, ksize=(10, 10)):
    """"
    Blur the input image.
    """
    blurred_image = cv2.blur(image, ksize)
    return blurred_image


def contrast_image(image, alpha=2.0, beta=-10):
    """
    Adjust the contrast of the input image.
    """
    contrasted_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return contrasted_image


def scale_image(image, zoom_factor=1.5):
    """
    Scale the input image by croping and then resizing back to original.
    """
    h, w = image.shape[:2]
    ch, cw = int(h/zoom_factor), int(w/zoom_factor)
    start_h, start_w = (h - ch)//2, (w - cw)//2
    crop = image[start_h:start_h+ch, start_w:start_w+cw]
    zoomed_image = cv2.resize(crop, (w, h))
    return zoomed_image


def shear_image(image, factor=0.2):
    """
    Apply a projective transformation to the input image using 2x3 matrix.
    """
    h, w = image.shape[:2]
    tx, ty = 0, 20  # Shift 100 pixels right and 50 pixels down
    translation_matrix = np.float32([
        [1, factor, tx],
        [factor, 1, ty]]
    )
    sheared_image = cv2.warpAffine(image, translation_matrix, (w, h))
    return sheared_image


def display_images(org_image, titles, images):
    titles = list(titles)
    images = list(images)
    titles.insert(0, "Original")
    images.insert(0, org_image)
    plt.figure(figsize=(16, 8))

    for i in range(len(images)):
        plt.subplot(2, 4, 1 + i)
        plt.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))
        plt.title(titles[i])
        plt.axis('off')

    plt.tight_layout()
    plt.show()


def get_augmentations(image):
    return {
        "Rotate": rotate_image(image),
        "Filp":	flip_image(image),
        "Blur": blur_image(image),
        "Contrast": contrast_image(image),
        "Scaling": scale_image(image),
        "Shear": shear_image(image),
    }


def data_augmentation(dir, target_count, images, diseases):
    """
    Augment images in subdirectories to match the maximum image count.
    Each subdirectory with fewer images
    will be augmented with transformed versions.
    """
    class_name = Path(dir).name
    pure_name = class_name.split("_")[0]
    target_count = target_count[pure_name]
    count = diseases[class_name]

    augmented_directory = "augmented_directory/"
    os.makedirs(augmented_directory, exist_ok=True)

    needed_images = target_count - count

    # Copy originals
    for img in images:
        base_name = Path(img).name
        image = cv2.imread(str(img))
        out_path = os.path.join(
            augmented_directory,
            pure_name,
            class_name,
            base_name
        )
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        cv2.imwrite(out_path, image)

    print(f"\nCopied {count} files in {class_name}, need {needed_images} more")

    if needed_images <= 0:
        return

    # Generate augmented images
    for img in images:
        if needed_images <= 0:
            break

        image = cv2.imread(str(img))
        base_name = Path(img).stem
        ext = Path(img).suffix

        augmentations = get_augmentations(image)

        for aug_name, aug_image in augmentations.items():
            if needed_images <= 0:
                break

            out_path = os.path.join(
                augmented_directory,
                pure_name,
                class_name,
                f"{base_name}_{aug_name}{ext}"
            )
            cv2.imwrite(out_path, aug_image)
            needed_images -= 1

    print(f"Final count: {target_count} images in {class_name}")


def get_target_count(root_path):
    """
    Identify class direcotries inside parent folder and photo count
    in each of them.
    """
    root = Path(root_path)
    level1 = [d for d in root.iterdir() if d.is_dir()]
    prefixes = {d.name.split("_")[0] for d in level1}

    # MULTI-CLASS SCENARIO (Apple + Grape)
    if len(prefixes) > 1:
        target_count = {}
        class_counts = {}

        for cls in level1:
            images = get_directory(cls)
            counts = count_images(images)

            class_counts[cls.name] = counts
            target_count[cls.name] = max(counts.values())

        return target_count, class_counts

    # SINGLE-CLASS SCENARIO (Apple only)
    class_name = prefixes.pop()
    images = get_directory(root)
    counts = count_images(images)

    target_count = {class_name: max(counts.values())}
    return target_count, {class_name: counts}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str)
    args = parser.parse_args()
    src_path = Path(args.path)

    # src is a DIRECTORY
    if src_path.is_dir():
        src_path = validate_source_directory(src_path)
        target_count, class_counts = get_target_count(args.path)
        print("TARGET COUNT:", target_count)

        # Loop over each class (Apple, Grape)
        for class_name, diseases in class_counts.items():
            if (Path(args.path).name == class_name):
                class_dir = args.path
            else:
                class_dir = os.path.join(args.path, class_name)

            # Loop over each disease folder inside this class
            for disease_name in diseases.keys():
                disease_dir = os.path.join(class_dir, disease_name)
                images = get_directory(disease_dir)

                data_augmentation(
                    disease_dir,
                    target_count,
                    images,
                    diseases
                )

    # src is a FILE
    else:
        _, image = validate_source_file(src_path)
        augmentations = get_augmentations(image)
        display_images(image, augmentations.keys(), augmentations.values())


if __name__ == "__main__":
    main()
