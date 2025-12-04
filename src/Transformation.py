import argparse
import cv2
from pathlib import Path
import os
from utils import (
    validate_source_directory,
    validate_source_file,
    validate_destination_directory
)
import matplotlib.pyplot as plt
from effects import (
    gaussian_blur,
    mask,
    roi_object,
    negative_image,
    analyze_image,
    edges_image,
    posterize,
    sharpen
)


def get_all_transformations(advanced=False):
    """
    Return 6 transformation functions.
    """
    base = {
        'gaussian': gaussian_blur,
        'mask': mask,
        'negative': negative_image,
        'edges': edges_image,
    }
    if advanced:
        base.update({
            'analyze': analyze_image,
            'roi': roi_object,
        })
    else:
        base.update({
            'posterize': posterize,
            'sharpen': sharpen,
        })
    return base


def collect_image_files(src_path):
    """
    Return list of image files under `src_path` (case-insensitive extensions).
    """
    exts = {'.jpg', '.jpeg', '.png'}
    files = [p for p in src_path.rglob('*') if p.suffix.lower() in exts]
    return sorted(files)


def process_directory_with_filter(src_path, dst_path, filter, advanced=False):
    """
    Apply ONE specific filter to all images found in the directory tree.
    Copies original images to destination, then adds transformed versions.
    """
    transformations = get_all_transformations(advanced)

    if filter not in transformations:
        print(f"Error: Unknown filter '{filter}'")
        return

    filter_func = transformations[filter]

    image_files = collect_image_files(src_path)
    print(f"Processing: {src_path} with filter: {filter}")
    print(f"Found {len(image_files)} images")

    print("Copying original images...")
    for img_file in image_files:
        rel_path = img_file.relative_to(src_path)
        out_path = dst_path / rel_path
        os.makedirs(out_path.parent, exist_ok=True)
        cv2.imwrite(str(out_path), cv2.imread(str(img_file)))
    print(f"Copied {len(image_files)} original images")

    print(f"Applying {filter} filter...")
    for img_file in image_files:
        image = cv2.imread(str(img_file))
        if image is None:
            continue

        rel_path = img_file.relative_to(src_path)
        base_name = img_file.stem
        ext = img_file.suffix

        if filter_func == analyze_image:
            roi_result = roi_object(image, mask=None, plot=False)[1]
            transformed = analyze_image(image, roi_result, plot=False)
        elif filter_func == roi_object:
            transformed = roi_object(image, mask=None, plot=False)[0]
        else:
            transformed = filter_func(image, mask=None, plot=False)

        out_path = dst_path / rel_path.parent / f"{base_name}_{filter}{ext}"
        os.makedirs(out_path.parent, exist_ok=True)
        cv2.imwrite(str(out_path), transformed)
    print(f"Saved originals + {filter} transformations in {dst_path}")


def process_directory_all_filters(src_path, dst_path, advanced=False):
    """
    Apply ALL 6 filters to all images found recursively in the directory tree.
    Copies original images to destination, then adds 6 transformed versions.
    """
    image_files = collect_image_files(src_path)
    print(f"Found {len(image_files)} images")

    print("Copying original images...")
    for img_file in image_files:
        rel_path = img_file.relative_to(src_path)
        out_path = dst_path / rel_path
        os.makedirs(out_path.parent, exist_ok=True)
        cv2.imwrite(str(out_path), cv2.imread(str(img_file)))
    print(f"Copied {len(image_files)} original images")

    print("Applying all 6 filters...")
    for img_file in image_files:
        image = cv2.imread(str(img_file))
        if image is None:
            continue

        rel_path = img_file.relative_to(src_path)
        base_name = img_file.stem
        ext = img_file.suffix

        # Apply each filter
        results = {}
        results['mask'] = mask(image, mask=None, plot=False)
        results['gaussian'] = gaussian_blur(image, mask=None, plot=False)
        results['negative'] = negative_image(image, mask=None, plot=False)
        results['edges'] = edges_image(image, mask=None, plot=False)

        if advanced:
            rot_res = roi_object(image, mask=None, plot=False)
            if isinstance(rot_res, tuple):
                results['roi'] = rot_res[0]
            else:
                results['roi'] = rot_res
            results['analyze'] = analyze_image(image, rot_res[1], plot=False)

        else:
            results['posterize'] = posterize(image, mask=None, plot=False)
            results['sharpen'] = sharpen(image, mask=None, plot=False)

        for name, img_to_save in results.items():
            if img_to_save is None:
                continue
            out_path = dst_path / rel_path.parent / f"{base_name}_{name}{ext}"
            os.makedirs(out_path.parent, exist_ok=True)
            cv2.imwrite(str(out_path), img_to_save)

    print(f"Saved originals + all 6 transformations in {dst_path}")


def process_image_with_filter(image, filter, dst_path, file, advanced=False):
    """
    Apply ONE filter to a image and display original + transformed.
    """
    transformations = get_all_transformations(advanced)

    if filter not in transformations:
        print(f"Error: Unknown filter '{filter}'")
        return

    filter_func = transformations[filter]
    if filter == 'analyze':
        roi_result = roi_object(image, mask=None, plot=False)[1]
        transformed = analyze_image(image, roi_result, plot=False)
    elif filter == 'roi':
        transformed = roi_object(image, mask=None, plot=False)[0]
    else:
        transformed = filter_func(image, mask=None, plot=False)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    if len(transformed.shape) == 2:
        plt.imshow(transformed, cmap='gray')
    else:
        plt.imshow(cv2.cvtColor(transformed, cv2.COLOR_BGR2RGB))
    plt.title(f"Transformed - {filter}")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

    if dst_path is not None:
        ext = '.JPG'
        out_path = dst_path / f"{file}_comparison_{filter}{ext}"
        os.makedirs(out_path.parent, exist_ok=True)

        if len(transformed.shape) == 2:
            transformed_bgr = cv2.cvtColor(transformed, cv2.COLOR_GRAY2BGR)
        else:
            transformed_bgr = transformed

        comparison = cv2.hconcat([image, transformed_bgr])
        cv2.imwrite(str(out_path), comparison)
        print(f"Saved comparison: {out_path}")


def process_image_all_filters(image, dst_path, file_name, advanced=False):
    """
    Apply 6 filters to a image and display original + transformations in grid.
    """
    results = {'original': image}
    results['mask'] = mask(image, mask=None, plot=False)
    results['gaussian'] = gaussian_blur(image, mask=None, plot=False)
    results['negative'] = negative_image(image, mask=None, plot=False)
    results['edges'] = edges_image(image, mask=None, plot=False)

    if advanced:
        roi_result = roi_object(image, mask=None, plot=False)
        if isinstance(roi_result, tuple):
            results['roi'] = roi_result[0]
        else:
            results['roi'] = roi_result
        results['analyze'] = analyze_image(image, roi_result[1], plot=False)

    else:
        results['posterize'] = posterize(image, mask=None, plot=False)
        results['sharpen'] = sharpen(image, mask=None, plot=False)

    plt.figure(figsize=(16, 8))
    plot_idx = 1
    for name, img_to_show in results.items():
        plt.subplot(2, 4, plot_idx)
        if img_to_show is None:
            plot_idx += 1
            continue
        if len(img_to_show.shape) == 2:
            plt.imshow(img_to_show, cmap='gray')
        else:
            plt.imshow(cv2.cvtColor(img_to_show, cv2.COLOR_BGR2RGB))
        plt.title(name.capitalize())
        plt.axis('off')
        plot_idx += 1

    plt.tight_layout()
    plt.show()

    if dst_path is not None:
        ext = '.JPG'
        out_path = dst_path / f"{file_name}_all_transformations{ext}"
        os.makedirs(out_path.parent, exist_ok=True)

        transformed_images = []
        for _, img_to_save in results.items():
            if img_to_save is None:
                continue
            if len(img_to_save.shape) == 2:
                img_to_save = cv2.cvtColor(img_to_save, cv2.COLOR_GRAY2BGR)
            transformed_images.append(img_to_save)

        comparison = cv2.hconcat(transformed_images)
        cv2.imwrite(str(out_path), comparison)
        print(f"Saved all transformations: {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Apply transformations to images."
    )
    parser.add_argument(
        "-src",
        "--source",
        type=str,
        help="Source: directory (with subdirectories) or image file",
        required=True
    )
    parser.add_argument(
        "-dst",
        "--destination",
        type=str,
        help="Destination directory (is source is directory then required)",
        required=False
    )
    parser.add_argument(
        "-f",
        "--filter",
        type=str,
        default=None,
        help="Apply specific filter (gaussian, mask, edges, negative,"
        "(BASIC: sharpen, posterized), (ADVANCED: roi, analyze))."
        "If not provided, apply 6 basic.")
    parser.add_argument(
        "--advanced",
        action="store_true",
        help="Use advanced filters (roi, analyze)"
        "instead of basic (posterize, sharpen).")
    args = parser.parse_args()

    src = args.source
    dst = args.destination
    filter_name = args.filter
    adv = args.advanced

    src_path = Path(src)

    # src is a DIRECTORY
    if src_path.is_dir():
        print("Mode: Batch processing directory")
        src_path = validate_source_directory(src)
        dst_path = validate_destination_directory(dst)

        if filter_name:
            print(f"Applying filter: {filter_name} to all images in dirs...")
            process_directory_with_filter(
                src_path, dst_path, filter_name, adv
            )
        else:
            print("Applying all 6 filters to all images in dirs...")
            process_directory_all_filters(src_path, dst_path, adv)

    # src is a FILE
    else:
        print("Mode: Single image processing")
        src_path, image = validate_source_file(src)
        dst_path = None
        if dst is not None:
            dst_path = validate_destination_directory(dst)
        file = src_path.stem

        if filter_name:
            print(f"Applying filter: {filter_name} to single image...")
            process_image_with_filter(image, filter_name, dst_path, file, adv)
        else:
            print("Applying 6 filters to single image and creating grid...")
            process_image_all_filters(image, dst_path, file, adv)


if __name__ == "__main__":
    main()
