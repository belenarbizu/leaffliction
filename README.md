# leaffliction
### 1. Analysis of the Data Set

This program will analyze a plant directory, counts all images in its subfolders, and generates bar and pie charts showing the distribution of classes.
```bash
python3 ./src/Distribution.py "./Apple"
```
<img width="700" height="600" alt="Apple_bar_chart" src="https://github.com/user-attachments/assets/2ece7a09-e61c-4af7-9de5-edd99b656616" />
<img width="700" height="600" alt="Apple_pie_chart" src="https://github.com/user-attachments/assets/0b6bdb82-b23c-40c3-aa4f-ee045abe9d25" />

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
