# break down of take 3: Part 1, step 1

import cv2
import numpy as np
from skimage.measure import label, regionprops
from skimage import filters
import skimage
import matplotlib.pyplot as plt

def analyze_image(image_path):
    # Load the image
    image = cv2.imread(image_path)
    
    # Step 1: Convert to grayscale for edge detection and color for Sobel
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # read in RGB version
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


    # Blur & Apply Canny edge detection
    # blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    blurred = cv2.GaussianBlur(gray, (0,0), sigmaX=6, sigmaY=6) # replace 'gray'
    # blurred = cv2.GaussianBlur(image_rgb, (0,0), sigmaX=6, sigmaY=6) # replace 'gray'
    # Step 3: Apply edge detection (Canny algorithm) on gray image and RGB
    edges_bw = cv2.Canny(gray, threshold1=50, threshold2=150, apertureSize=3, L2gradient=False)
    edges_rgb = cv2.Canny(image_rgb, threshold1=50, threshold2=150, apertureSize=3, L2gradient=False)
 
    # Plotting
    fig, ax = plt.subplots(nrows=2, ncols=2) # fix: TypeError: 'Axes' object is not subscriptable
    # set title
    plt.suptitle("Step-wise Comparison")
    ax[0][0].imshow(image_rgb, cmap='magma') # good to show 'image' here
    ax[0][1].imshow(gray, cmap='magma')
    ax[1][0].imshow(edges_rgb, cmap='magma')
    ax[1, 1].imshow(edges_bw, cmap='magma') # THIS
    # ax[1, 1].imshow(thresh, cmap='magma')
    # plt.show() # this may be the issue for running scripts
    # fig, ax = plt.subplots(1,1) # no change after commenting
    # ax.imshow(edges, cmap='magma')

    # set the title to all subplots
    ax[0, 0].set_title("RGB Image")
    ax[0, 1].set_title("Grayscale Image")
    ax[1, 0].set_title("RGB Edges")
    ax[1, 1].set_title("Grayscale Edges")
    # adjust spacing
    fig.tight_layout()

    # Save plot
    # plt.imshow(edges, cmap="gray")
    plt.savefig('./image_process/figures/T3P1S1_ImageComparison.jpg')


# Example usage
image_path = "images/ClumpyLate_AS_C_I_22_2_20240724.jpg"
result = analyze_image(image_path)

