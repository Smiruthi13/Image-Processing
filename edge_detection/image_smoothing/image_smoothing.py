
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

# ---------------------------------
# 1. LOAD INPUT IMAGE
# ---------------------------------

image = cv2.imread("noisy_image.webp")

if image is None:
    raise FileNotFoundError("Check noisy_image.webp")

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

print("Image loaded successfully:", image.shape)


# ---------------------------------
# 2. GAUSSIAN BLUR
# ---------------------------------

start = time.time()

gaussian = cv2.GaussianBlur(image, (5, 5), 0)

gaussian_time = (time.time() - start) * 1000


# ---------------------------------
# 3. MEDIAN FILTERING
# ---------------------------------

start = time.time()

median = cv2.medianBlur(image, 5)

median_time = (time.time() - start) * 1000


# ---------------------------------
# 4. SPATIAL DOMAIN CONVOLUTION
# AVERAGING FILTER
# ---------------------------------

# Create a 5x5 averaging kernel
kernel = np.ones((5, 5), np.float32) / 25

start = time.time()

averaging = cv2.filter2D(image, -1, kernel)

averaging_time = (time.time() - start) * 1000


# ---------------------------------
# 5. DISPLAY ALL RESULTS
# ---------------------------------

plt.figure(figsize=(14, 8))

plt.subplot(2, 2, 1)
plt.imshow(image)
plt.title("Original Noisy Image")
plt.axis("off")

plt.subplot(2, 2, 2)
plt.imshow(gaussian)
plt.title("Gaussian Blur (5x5)")
plt.axis("off")

plt.subplot(2, 2, 3)
plt.imshow(median)
plt.title("Median Filter (5x5)")
plt.axis("off")

plt.subplot(2, 2, 4)
plt.imshow(averaging)
plt.title("Averaging Filter (5x5)")
plt.axis("off")

plt.tight_layout()

plt.savefig("smoothing_comparison.png")

plt.show()


# ---------------------------------
# 6. PERFORMANCE COMPARISON
# ---------------------------------

print("\n===== IMAGE SMOOTHING PERFORMANCE =====")

print(f"Gaussian Blur:    {gaussian_time:.3f} ms")
print(f"Median Filter:    {median_time:.3f} ms")
print(f"Averaging Filter: {averaging_time:.3f} ms")

print("\nComparison image saved successfully!")
