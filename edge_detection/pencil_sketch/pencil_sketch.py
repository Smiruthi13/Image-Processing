
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Load the original image
image = cv2.imread("girl.jpeg")

if image is None:
    raise FileNotFoundError("Image not found! Check input.jpg")

print("Image loaded successfully:", image.shape)

# Convert BGR to RGB for Matplotlib
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 2. Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

# 3. Invert the grayscale image
inverted = 255 - gray

# 4. Apply Gaussian blur
blurred = cv2.GaussianBlur(inverted, (21, 21), 0)

# 5. Invert the blurred image
inverted_blur = 255 - blurred

# 6. Apply color dodge blending
sketch = cv2.divide(gray, inverted_blur, scale=256)

# 7. Display all processing stages
titles = [
    "1. Original Image",
    "2. Grayscale",
    "3. Inverted Image",
    "4. Gaussian Blur",
    "5. Inverted Blur",
    "6. Final Pencil Sketch"
]

images = [
    image,
    gray,
    inverted,
    blurred,
    inverted_blur,
    sketch
]

plt.figure(figsize=(15, 10))

for i in range(6):
    plt.subplot(2, 3, i + 1)

    if i == 0:
        plt.imshow(images[i])
    else:
        plt.imshow(images[i], cmap="gray")

    plt.title(titles[i])
    plt.axis("off")

plt.tight_layout()

# Save the final result
cv2.imwrite("pencil_sketch_result.png", sketch)

print("Pencil sketch generated successfully!")
print("Saved as pencil_sketch_result.png")

plt.show()
