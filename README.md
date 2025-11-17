# leaffliction
### 1. Analysis of the Data Set

--- 
### 2. Image Transformation and Data Augmentation

1. Transform a single image
Displays the original and transformed versions on screen:
```bash
python3 ./src/Augmentation.py "images/leaves/Apple/Apple_healthy/image (1).JPG"
```
**The program will:** 
Generate and display transformed images such as:
- Rotation
- Horizontal flip
- Blur
- Contrast adjustment
- Scaling (zoom)
- Shear transformation
![transformations](computed_images/transformations.jpg)

---

2. Process a directory
If the path points to a directory, the tool detects subdirectories (if any) and performs data augmentation for each:
```bash
python3 ./src/Augmentation.py "images/leaves/"
```

**The program will:** 
- Copy original images into augmented_directory/
- Generate additional images if a subdirectory has fewer images than the largest one
- Preserve subdirectory structure

---

## 3. Image Transformation