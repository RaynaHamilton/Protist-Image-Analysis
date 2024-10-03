# Breakdown of take 3: Part 1, step 2

import cv2
import numpy as np
from skimage.measure import label, regionprops
from skimage import filters
import skimage
import matplotlib.pyplot as plt

def analyze_image(image_path):
    # Load the image
    image = cv2.imread(image_path)
    
    # Convert to grayscale for edge detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # read in RGB version
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


    # Blur & Apply Canny edge detection
    # blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    blurred = cv2.GaussianBlur(gray, (0,0), sigmaX=6, sigmaY=6) # replace 'gray'
    # blurred = cv2.GaussianBlur(image_rgb, (0,0), sigmaX=6, sigmaY=6) # replace 'gray'

    # edge detection for both image types
    edges_bw = cv2.Canny(gray, threshold1=50, threshold2=150, apertureSize=3, L2gradient=False)
    edges_rgb = cv2.Canny(image_rgb, threshold1=50, threshold2=150, apertureSize=3, L2gradient=False)

    # Step 3: morphology for bw (doesn't work on RGB)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (45,45))
    morph = cv2.morphologyEx(blurred, cv2.MORPH_CLOSE, kernel)

    # threshold
    thresh = cv2.threshold(morph, 0, 255, cv2.THRESH_OTSU)[1] # this threw an error when given 'image_rgb'

    # get contours and filter on size
    # masked1 = gray.copy()
    masked1 = image_rgb.copy()
    meanval = int(np.mean(masked1))
    contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    for cntr in contours:
        area = cv2.contourArea(cntr)
        # NOTE: maybe increase pixel size here--match threshold from before, but is this by total pixels in area of contour..?
        if area > 50 and area < 5000:
            cv2.drawContours(masked1, [cntr], 0, (meanval), -1)

    # stretch
    minval = int(np.amin(masked1))
    maxval = int(np.amax(masked1))
    # NOTE: result1 is currently the same as 'gray'
    result1 = skimage.exposure.rescale_intensity(masked1, in_range=(minval,maxval), out_range=(0,255)).astype(np.uint8)

    edges = filters.sobel(blurred) #replaced 'result1'


    # Find contours from the edges
    # contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Mask to isolate the cells
        # NOTE: 'mask' is same as 'thresh'
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, contours, -1, color=255, thickness=cv2.FILLED)

    fig, ax = plt.subplots(nrows=2, ncols=2)

    # set main title
    plt.suptitle("Step-wise Comparison")
    # ax[0][0].imshow(edges, cmap='magma')
    ax[0][0].imshow(image_rgb, cmap='magma') # good to show 'image' here
    ax[0][1].imshow(gray, cmap='magma')
    # ax[0][1].imshow(result1, cmap='magma') # strange contours in outlying cells
    ax[1][0].imshow(edges_rgb, cmap='magma')
    ax[1, 1].imshow(edges, cmap='magma')
    # ax[1, 1].imshow(thresh, cmap='magma')
    # plt.show() # this may be the issue for running scripts
    # fig, ax = plt.subplots(1,1) # no change after commenting
    # ax.imshow(edges, cmap='magma')

    # set the title to all subplots
    ax[0, 0].set_title("RGB Image")
    ax[0, 1].set_title("Grayscale Image")
    ax[1, 0].set_title("RGB Edges")
    ax[1, 1].set_title("Refined RGB Edges")
    # adjust spacing
    fig.tight_layout()

    # Save plot
    # plt.imshow(edges, cmap="gray")
    # plt.savefig('./image_process/figures/T3P1S2_BWRefinedEdges.jpg')
    plt.savefig('./image_process/figures/T3P1S2_RGBRefinedEdges.jpg')


# Example usage
image_path = "images/ClumpyLate_AS_C_I_22_2_20240724.jpg"
result = analyze_image(image_path)

