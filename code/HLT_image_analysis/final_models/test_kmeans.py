# K-means Automated Image Analysis

'''

'''

# Import Libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from skimage.measure import label, regionprops
import random
import glob
import os
from sklearn.cluster import KMeans
from PIL import Image as im 

# Set seed for K-means clustering reproducibility
random.seed(10)


#-----------------------------------------------------------------


# Define Functions

def remove_large_background(array,window_size,step_size,max_mean):
    '''
    Scans across an image and removes large pixel blobs i.e. blocks where almost all pixels are filled in.
    Parameters:
    array: input 2d array of pixels
    window_size: size of box used when calculating proportion of filled pixels.  Should be larger than desired cell width.
    step_size: movement distance between one window and the next
    max_mean: mean that pixels in a window must reach for them to be removed.
    '''
    clean_array=np.copy(array)
    for i in range(0,len(array)-window_size,step_size):
        for j in range(0,len(array[i])-window_size,step_size):    
            subset=array[i:i+window_size,j:j+window_size]
            if subset.mean()>max_mean:
                clean_array[max(i-step_size,0):min(i+window_size+step_size,len(array)-1),max(j-window_size,0):min(j+window_size+step_size,len(array[i]))]=0 #replace pixels of region passing max_mean threshold with 0
    return clean_array

def make_custom_grayscale(red_channel,blue_channel):
    '''
    Convert red and blue channels to a single grayscale array by taking the max at each coordinate.
    Parameters:
    red_channel: 2d np array of red pixel values
    blue_channel: 2d np array of blue pixel values
    '''
    custom_gray=[]
    for i,row in enumerate(red_channel):
        custom_gray.append([])
        for j,col in enumerate(row):
            if red_channel[i,j]>blue_channel[i,j]:
                custom_gray[-1].append(red_channel[i,j]*255)
            else:
                custom_gray[-1].append(blue_channel[i,j]*255)
    custom_gray=np.array(custom_gray).astype(np.uint8)
    return custom_gray

# K-means approach as a function
def Kmeans_process_image(image_path):
    # 1. Read image and convert to RGB
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Split into RGB channels for thresholding
    red_channel = image_rgb[:, :, 0]
    blue_channel = image_rgb[:, :, 2]
    
    # 2. Define thresholds for red and blue channels based on percentiles
        # try: r-92,b-95
    red_threshold = np.percentile(red_channel, 85)
    blue_threshold = np.percentile(blue_channel, 95)
    
        # Create binary masks for each color channel
            # CHANGED from red/blue_threshold
    _, red_mask = cv2.threshold(red_channel, red_threshold, 255, cv2.THRESH_BINARY)
    _, blue_mask = cv2.threshold(blue_channel, blue_threshold, 255, cv2.THRESH_BINARY)
        # Combine the red and blue masks
    combined_mask = cv2.bitwise_or(red_mask, blue_mask)
        # Apply the mask to isolate high-intensity pixels in the image
    thresholded_image = cv2.bitwise_and(image_rgb, image_rgb, mask=combined_mask)

    # 3. Perform K-means clustering
        # Flatten the masked pixels (ignore dark background)
    masked_pixels = thresholded_image[combined_mask > 0]
    pixels = masked_pixels.reshape((-1, 3)).astype(np.float32)
        # Define K-means criteria and number of clusters
    k = 4  # Number of clusters
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        # Perform K-means clustering--cv2
    # _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        # preform K-means clustering--scikit
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42).fit(pixels)
        # Convert centers to uint8 for display
    centers = np.uint8(kmeans.cluster_centers_)
        # Reconstruct segmented data using labels
    labels = kmeans.labels_
    segmented_data = centers[labels.flatten()]
        # Create an empty array to store the full-sized segmented image
    segmented_image = np.zeros_like(image_rgb)
        # Fill in only the high-intensity pixels in the original mask
    segmented_image[combined_mask > 0] = segmented_data

    # 4. Subset mask by keeping only the clusters corresponding to red and blue
        # Automate cluster characterization based on RBG values
            # instantiate cluster values
    bg_cluster = -1
    red_cluster = -1
    blue_cluster = -1
    lab_cluster = -1
        # instantiate list of color values outside loop
    max_g_val = -1
    max_r_val = -1
    max_b_val = -1
        # 4.1: Identify labels cluster
            # iterate over centers (RGB values)
    for idx, cntr in enumerate(centers):
        print(idx, cntr)
            # set green value
        g_val = cntr[1]
            # compare to max green value
        if g_val > max_g_val:
                # update max green value
            max_g_val = g_val
                # set max green value to background cluster
            lab_cluster = idx
        # identify red cluster
    for idx, cntr in enumerate(centers):
            # exclude cluster previously identified as bg
        if idx != lab_cluster:
            # set red value
            r_val = cntr[0]
            # compare to max red value
            if r_val > max_r_val:
                max_r_val = r_val
                red_cluster = idx
        # 4.3: Identify BLUE cluster
    for idx, cntr in enumerate(centers):
            # exclude clusters previously identified as bg & red
        if idx != lab_cluster and idx != red_cluster:
            # set red value
            b_val = cntr[2]
            # compare to max red value
            if b_val > max_b_val:
                max_b_val = b_val
                blue_cluster = idx
        # 4.2: Identify bg cluster
    for idx, cntr in enumerate(centers):
            # exclude cluster previously identified as bg
        if idx != lab_cluster and idx != red_cluster and idx != blue_cluster:
            bg_cluster = idx
    print("Background cluster is", bg_cluster)
    print("Labeled cluster is", lab_cluster)
    print("Red cluster is", red_cluster)
    print("Blue cluster is", blue_cluster)
        # Reshape labels to match combined_mask shape
    labels_reshaped = np.zeros(combined_mask.shape, dtype=int)
    labels_reshaped[combined_mask > 0] = labels.flatten()
        # Create a mask that includes only the red and blue clusters
    foreground_mask = np.isin(labels_reshaped, [red_cluster, blue_cluster])
    bg_mask = np.isin(labels_reshaped, [bg_cluster, lab_cluster])
        # Initialize the foreground image with black background
    foreground_final = np.zeros_like(image_rgb)
    bground_final = np.zeros_like(image_rgb)
        # Apply the mask to keep only the red and blue clusters
    foreground_final[foreground_mask] = image_rgb[foreground_mask]
    bground_final[bg_mask] = image_rgb[bg_mask]

    # 5. Perform Canny edge detection on the final foreground
        # foreground_final or custom_gray
    edges = cv2.Canny(foreground_final, 100, 200, apertureSize=3, L2gradient=False)

    # 6. Find contours and filter by area
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Set minimum & maximum cell area threshold
    min_area = 500
    max_area = 15000
    
    # Set aspect ratio thresholds (adjust based on cell shapes in your images)
    min_aspect_ratio = 1.0  # Minimum ratio for pencil-shaped cells
    max_aspect_ratio = 20.0  # Maximum ratio for pencil-shaped cells
    
    cell_contours = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if min_area < area < max_area:
            # Calculate bounding rectangle and aspect ratio
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = max(w, h) / min(w, h)  # Ensure the ratio is always >= 1
            # print(w,h,aspect_ratio)
            # Filter by aspect ratio
            if min_aspect_ratio <= aspect_ratio <= max_aspect_ratio:
                cell_contours.append(cnt)
    
        # Initialize metrics for counting and blue:red ratio
    total_cells = len(cell_contours)
    cell_ratios = []
        # Create a copy of the original image for cell-labeling
    labeled_image = foreground_final.copy()
        # Count all filtered edges (cells) & calculate blue:red ratio
    print(f"Total cells detected: {total_cells}")

    # 7. Calculate blue:red ratios within each cell
    for i, contour in enumerate(cell_contours):
        # Create a mask for each cell
        cell_mask = np.zeros(foreground_mask.shape, dtype=np.uint8)
        cv2.drawContours(cell_mask, [contour], -1, 255, -1)  # Fill contour to create mask
        # Extract pixels in red and blue channels for this cell
        red_values = red_channel[cell_mask == 255]
        blue_values = blue_channel[cell_mask == 255]
        # Calculate blue:red ratio for this cell
        red_sum = np.sum(red_values)
        blue_sum = np.sum(blue_values)
        blue_red_ratio = blue_sum / red_sum if red_sum > 0 else 0  # Avoid division by zero
        proportion_infected = blue_sum / (blue_sum + red_sum) if (blue_sum + red_sum) > 0 else 0
        cell_ratios.append({
            'Cell Number': i + 1,
            'Red Intensity Sum': red_sum,
            'Blue Intensity Sum': blue_sum,
            'Blue:Red Ratio': blue_red_ratio,
            'Proportion Infected': proportion_infected
        })
            # Label the detected cell on the image
        M = cv2.moments(contour)  # Get moments of the contour to calculate centroid
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])  # Centroid X
            cY = int(M["m01"] / M["m00"])  # Centroid Y
            cv2.putText(
                labeled_image, 
                str(i + 1),  # Label with cell number
                (cX, cY),  # Position at centroid
                cv2.FONT_HERSHEY_SIMPLEX, 
                3,  # Font size
                (255, 255, 255),  # White text
                2  # Thickness
            )

    # add edges to labeled image
    cv2.drawContours(labeled_image, cell_contours, -1, (0, 255, 0), 2)  # Green contours with thickness 2

    # 8. Convert ratios to DataFrame for easy viewing
    cell_ratios_df = pd.DataFrame(cell_ratios)
        # Calculate & report image overall blue:red ratio
    total_red = np.sum(red_channel[foreground_mask])
    total_blue = np.sum(blue_channel[foreground_mask])
    overall_blue_red_ratio = total_blue / total_red if total_red > 0 else 0
        # Display results
    print("\nIndividual Cell Blue:Red Ratios")
    # print(cell_ratios_df)
    print(f"\nTotal Cells: {total_cells}")
    print(f"Overall Blue:Red Ratio in Image: {overall_blue_red_ratio:.2f}")
    # Convert labeled_image (numpy array) to a Pillow image
    labeled_image_pil = im.fromarray(labeled_image)
 
        # Display the final cell edges
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 4, 1)
    plt.imshow(image_rgb)
    plt.title('Original RGB')
    plt.subplot(1, 4, 2)
    plt.imshow(bground_final, cmap='gray') #edges, bground_final
    plt.title('BG mask') # Canny Edge Detection
    plt.subplot(1, 4, 3)
    plt.imshow(labeled_image)
    plt.title('Foreground with Detected Cells')
    plt.subplot(1, 4, 4)
    plt.imshow(labeled_image_pil)
    plt.title('Returned Image')
    plt.show()
    # print(type(cell_ratios_df))
    # print(type(labeled_image))
    return cell_ratios_df, labeled_image_pil


#-----------------------------------------------------------------


# Call Functions

# Process all images in a folder
image_folder = "../../images/raw/"
image_paths = glob.glob(os.path.join(image_folder, "*.jpg"))
csv_folder = "../../figures/Kmeans_classification/data/"
jpg_folder = "../../figures/Kmeans_classification/classified/"

for image_path in image_paths:
    print("")
    print(f"Processing {os.path.basename(image_path)}")
    df, image_output = Kmeans_process_image(image_path)
    # Generate basename
    base_filename = os.path.splitext(os.path.basename(image_path))[0]
    # Generate .csv output filename based on the input image name
    csv_output_path = os.path.join(csv_folder, f"{base_filename}.csv")
    image_output_path = os.path.join(jpg_folder, f"{base_filename}_labeled.jpg")

    # csv_output_filename = os.path.splitext(os.path.basename(image_path))[0] + ".csv"
    # csv_output_path = os.path.join(csv_folder, csv_output_filename)
    # Generate .jpg output filename based on the input image name
    # jpg_output_filename = os.path.splitext(os.path.basename(image_path))[0] + ".jpg"
    # jpg_output_path = os.path.join(jpg_folder, jpg_output_filename)
    
    # Save the dataframe to a CSV file
    df.to_csv(csv_output_path, index=False)
    # Save the image to a JPG file
    print(f"Saved results to {csv_output_path}")
    # Save the labeled image to a JPG file
    # cv2.imwrite(image_output_path, cv2.cvtColor(image_output, cv2.COLOR_RGB2BGR))  # Convert back to BGR for saving--OLD
    image_output.save(image_output_path)  # Convert back to BGR for saving--NEW
    print(f"Saved labeled image to {image_output_path}")

