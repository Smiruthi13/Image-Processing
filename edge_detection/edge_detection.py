
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

# 1. Load the input image
image = cv2.imread("images/input.jpeg")

if image is None:
    raise FileNotFoundError("Image not found! Check images/input.jpg")

print("Image loaded successfully:", image.shape)

# 2. Convert image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 3. Apply Gaussian Blur to reduce noise
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# --------------------------------
# SOBEL EDGE DETECTION
# --------------------------------

start = time.time()

sobel_x = cv2.Sobel(
    blurred, cv2.CV_64F, 1, 0, ksize=3
)

sobel_y = cv2.Sobel(
    blurred, cv2.CV_64F, 0, 1, ksize=3
)

sobel_magnitude = np.sqrt(
    sobel_x**2 + sobel_y**2
)

sobel_edges = cv2.convertScaleAbs(sobel_magnitude)

sobel_time = (time.time() - start) * 1000

# --------------------------------
# LAPLACIAN EDGE DETECTION
# --------------------------------

start = time.time()

laplacian = cv2.Laplacian(
    blurred, cv2.CV_64F, ksize=3
)

laplacian_edges = cv2.convertScaleAbs(laplacian)

laplacian_time = (time.time() - start) * 1000

# --------------------------------
# CANNY EDGE DETECTION
# --------------------------------

start = time.time()

canny_edges = cv2.Canny(
    blurred, 100, 200
)

canny_time = (time.time() - start) * 1000

# --------------------------------
# DISPLAY RESULTS
# --------------------------------

plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("Original Image")
plt.axis("off")

plt.subplot(2, 2, 2)
plt.imshow(sobel_edges, cmap="gray")
plt.title("Sobel Edge Detection")
plt.axis("off")

plt.subplot(2, 2, 3)
plt.imshow(laplacian_edges, cmap="gray")
plt.title("Laplacian Edge Detection")
plt.axis("off")

plt.subplot(2, 2, 4)
plt.imshow(canny_edges, cmap="gray")
plt.title("Canny Edge Detection")
plt.axis("off")

plt.tight_layout()
plt.savefig("edge_detection_results.png")
plt.show()

# --------------------------------
# PERFORMANCE COMPARISON
# --------------------------------

print("\n===== EDGE DETECTION PERFORMANCE =====")

print(f"Sobel Time:     {sobel_time:.3f} ms")
print(f"Laplacian Time: {laplacian_time:.3f} ms")
print(f"Canny Time:     {canny_time:.3f} ms")

print("\nResults saved as edge_detection_results.png")
