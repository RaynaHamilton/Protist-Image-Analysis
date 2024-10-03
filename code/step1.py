# Basic cell recognition: 
    # NOTES: this does a bad job, only identifying dots and words, no cells DUE to blur

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Load the image and convert to grayscale
image = cv2.imread("images/ClumpyLate_AS_C_I_22_2_20240724.jpg")
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Step 2: Apply Gaussian blur to reduce noise
# blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

# Step 3: Apply edge detection (Canny algorithm)
edges = cv2.Canny(image, threshold1=50, threshold2=150, apertureSize=3, L2gradient=False)

# Save plot
plt.imshow(edges, cmap="gray")
plt.title("Canny Edge Detection")
plt.savefig('./image_process/figures/S1_CannyEdgeDectection.jpg')
# plt.show()

