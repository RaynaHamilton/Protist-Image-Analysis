# DEPLOY HLT
#-------------------------------------------------------------------------------------------------------------

# Read in libraries
import cv2
print(cv2.__version__)
import numpy as np
from skimage.measure import label, regionprops
from skimage import filters
import skimage
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import shutil
from math import sqrt

# Define variables
original_x_size,original_y_size=1388,1040 #initial size of input images
x_size,y_size=160,120 #downscaled size of images used for hough line detection
infection_threshold=0.2 #proportion of pixels which must be blue for a cell to be declared infected
target_red_quantile=0.92 #quantile that red pixel must pass to be kept
red_minimum=30 #minimum red pixel value to be kept
target_blue_quantile=0.95 #quantile that blue pixel must pass to be kept
blue_minimum=60 #minimum blue pixel value to be kept
    # home of raw images
        # example image name: "AS_C_I_22_3-Image Export-21_c1-2.jpg"
mother_directory = "/Users/kjehickman/Documents/Research/parasites/E3/data/micrographs/" # only process directories that start with "2024"
    # home of analyzed images
child_directory_image = "/Users/kjehickman/Documents/Research/parasites/E3/data/hlt_image_analysis_out/classified/"
child_directory_csv = "/Users/kjehickman/Documents/Research/parasites/E3/data/hlt_image_analysis_out/data/"


if not os.path.isdir(child_directory_csv):
    os.mkdir(child_directory_csv)
if not os.path.isdir(child_directory_image):
    os.mkdir(child_directory_image)

#-------------------------------------------------------------------------------------------------------------

# Read in functions
def remove_overlapping_lines(lines,slope_threshold=0.2,distance_threshold=4):
    '''
    Removes very similar/redundant lines produced by hough line transformation.
    For an array of line coordinates, finds pairs of lines with start or end points within distance_threshold of each otehr and slopes within slope_threshold of each other, then removes the shorter line.
    '''
    line_slopes=[]
    for line in lines:
        line_slopes.append((line[3]-line[1])/(line[2]-line[0])) 
    
    indices_to_remove=[]
    for i,slope1 in enumerate(line_slopes):
        for j,slope2 in enumerate(line_slopes):
            if i<j:
                if abs(slope2-slope1)<=slope_threshold: #examine pairs of lines with slopes within slope threshold
                    line1,line2=lines[i],lines[j]
                    x11,y11,x12,y12=line1 #first x coordinate, first y coordinate, second x coordinate, second y coordinate of line 1
                    x21,y21,x22,y22=line2 #first x coordinate, first y coordinate, second x coordinate, second y coordinate of line 2
                    #if (abs(x11-x21)<=distance_threshold and abs(y11-y21)<=distance_threshold) or (abs(x11-x22)<=distance_threshold and abs(y11-y22)<=distance_threshold) or (abs(x12-x21)<=distance_threshold and abs(y12-y21)<=distance_threshold) or (abs(x12-x22)<=distance_threshold and abs(y12-y22)<=distance_threshold):
                    if (sqrt((x11-x21)**2+(y11-y21)**2)<=distance_threshold) or (sqrt((x11-x22)**2+(y11-y22)**2)<=distance_threshold) or (sqrt((x12-x21)**2+(y12-y21)**2)<=distance_threshold) or (sqrt((x12-x22)**2+(y12-y22)**2)<=distance_threshold):
                        #Keep longer line:
                        if sqrt((x11-x12)**2 + (y11-y12)**2) > sqrt((x21-x22)**2 + (y21-y22)**2):
                            indices_to_remove.append(j)
                        else:
                            indices_to_remove.append(i)
    new_lines=[]
    #image=cv2.resize(backup,(x_size, y_size))
    for i,line in enumerate(lines):
        if i not in indices_to_remove:
            x1,y1,x2,y2=line
            new_lines.append(line)
            #cv2.line(image,(x1,y1),(x2,y2),(np.random.randint(50,255),np.random.randint(50,255),np.random.randint(50,255)),2)
    #plt.imshow(image)
    print(f"Removed {len(set(indices_to_remove))} very similar lines.")
    return new_lines

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

def convert_to_binary(red_channel,blue_channel,target_red_quantile,red_minimum,target_blue_quantile,blue_minimum):
    '''
    Converts red and blue continuous pixel values to 0 or 255 based on whether they are above the desired pixel thresholds.
    Parameters:
    red_channel: 2d np array of red pixel values
    blue_channel: 2d np array of blue pixel values
    target_red_quantile: quantile that red pixel must pass to be kept
    red_minimum: minimum red pixel value to be kept
    target_blue_quantile: quantile that blue pixel must pass to be kept
    blue_minumum: minimum blue pixel value to be kept
    '''
    #apply threshold based on quantile specified above
    target_quantile=max(red_minimum,np.quantile(red_channel,target_red_quantile))
    red_channel[red_channel<target_quantile]=0
    red_channel[red_channel>=target_quantile]=255
    
    target_quantile=max(blue_minimum,np.quantile(blue_channel,target_blue_quantile))
    blue_channel[blue_channel<target_quantile]=0
    blue_channel[blue_channel>=target_quantile]=255
    return red_channel,blue_channel
    
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

#-------------------------------------------------------------------------------------------------------------

# CALL HLT
iterator = 0
for dir in os.listdir(mother_directory):
    # Rlevant directories with images are named after dates the images were taken (e.g. 20240726)
    if dir.startswith("2024"):
        # set path for current directories
        directory_path = os.path.join(mother_directory, dir)
        date_name = dir
        # Create new directories for the same date in child_directory_(csv/image)
        new_dir_path_img = os.path.join(child_directory_image, date_name)
        new_dir_path_csv = os.path.join(child_directory_csv, date_name)
        os.makedirs(new_dir_path_img, exist_ok=True)
        os.makedirs(new_dir_path_csv, exist_ok=True)

        for file in os.listdir(directory_path):
            if file.endswith(".jpg"):
                image_path = os.path.join(directory_path, file)
                print(f"Now processing: {file}")
                iterator += 1
                print("\n\n\n\n\n\n")
                print(f"We are {(iterator/612)*100:.2f}% there! Only {612-iterator:.0f} images to go.")
                print("\n\n\n\n\n\n")
                # Read in image, extract color channels
                image = cv2.imread(image_path) 
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                red_channel = np.copy(image_rgb[:, :, 0])
                blue_channel = np.copy(image_rgb[:, :, 2])

                red_channel,blue_channel=convert_to_binary(red_channel,blue_channel,target_red_quantile,red_minimum,target_blue_quantile,blue_minimum)
                
                # Remove large blobs/noise
                blue_channel=remove_large_background(blue_channel,30,10,250)
                red_channel=remove_large_background(red_channel,30,10,250)

                #convert to grayscale
                custom_gray=make_custom_grayscale(red_channel,blue_channel)

                #make smaller image copies for hough line detection
                backup=np.copy(image_rgb)
                image=cv2.resize(backup,(x_size, y_size))
                img = np.copy(cv2.resize(custom_gray,(x_size, y_size)))
                img=img*255
                edges = cv2.resize(custom_gray,(x_size, y_size))
                
                #run line detection
                
                try:
                    lines = list([list(val[0]) for val in cv2.HoughLinesP(
                            cv2.resize(custom_gray,(x_size,y_size)), # Input edge image
                            1, # Distance resolution in pixels - I haven't played around with this much
                            np.pi/180, # Angle resolution in radians - I haven't played around with this much
                            threshold=15, # Min number of votes for valid line - lower values will give higher detection rate but also result in more-double counting
                            minLineLength=20, # Min allowed length of line - in practice I haven't found changing this parameter either way to help much
                            maxLineGap=5 # Max allowed gap between line for joining them - lower values with result in better detection rate of spotty cells, but also higher probability that background noise will be misclassified as a cell
                            )])
                except Exception:
                    continue

                #remove lines with very similar slopes and start or end points
                lines=remove_overlapping_lines(lines)

                #convert to python list - is this still necessary?
                lines_list =[]
                for k,points in enumerate(lines):
                    x1,y1,x2,y2=points
                    lines_list.append([(x1,y1),(x2,y2)])

                all_coordinates=[]
                img=np.zeros((original_y_size, original_x_size),np.uint8)
                print(f"I counted {len(lines)} cells.")
                print("Infection proportions are:")
                cell_ratios=[]
                total_cells=len(lines)
                annotated_image=np.copy(image_rgb)

                #iterate through lines, calculate infection proportion of each cell and annotate the original image with a line and cell number
                for cell_number,line in enumerate(lines):
                    temp_red=red_channel.copy()
                    temp_blue=blue_channel.copy()

                    #convert coordinates to size of initial image, pre-downsizing
                    x1,y1,x2,y2=int(line[0]*original_y_size/y_size),int(line[1]*original_x_size/x_size),int(line[2]*original_y_size/y_size),int(line[3]*original_x_size/x_size)
                    img=np.zeros((original_y_size, original_x_size),np.uint8)
                    cv2.line(img,(x1,y1),(x2,y2),255,20)
                    coords = np.argwhere(img) #returns coordinates of all non-zero pixels i.e. where the line/cell is
                    all_coordinates.append([f"{val[0]}_{val[1]}" for val in coords])

                    #get an array with pixels filled just at the location of the line, in order to calculate infection proportion
                    #this uses the binary (0/255) pixel values- would the original continuous values be more meaningful?
                    subsetted_rgb=image_rgb.copy()
                    temp_red=red_channel.copy()
                    temp_blue=blue_channel.copy()
                    target_coordinates={val:"" for val in all_coordinates[cell_number]}
                    for i,row in enumerate(temp_red):
                            for j,val in enumerate(row):
                                if f"{i}_{j}" not in target_coordinates.keys():
                                    temp_red[i][j]=0
                                    temp_blue[i][j]=0
                                    subsetted_rgb[i][j][0]=0
                                    subsetted_rgb[i][j][1]=0
                                    subsetted_rgb[i][j][2]=0
                    # Calculate blue:red ratio for this cell
                    red_sum = np.sum(temp_red)
                    blue_sum = np.sum(temp_blue)
                    blue_red_ratio = blue_sum / red_sum if red_sum > 0 else 0  # Avoid division by zero
                    proportion_infected = blue_sum / (blue_sum + red_sum) if (blue_sum + red_sum) > 0 else 0
                    # add to df for csv export
                    cell_ratios.append({
                        'Cell_Number': cell_number,
                        'R_Area': red_sum,
                        'B_Area': blue_sum,
                        'BR_Ratio': blue_red_ratio,
                        'Prop_Inf': proportion_infected
                    })        
                    # proportion_infected.append(np.sum(temp_blue)/(np.sum(temp_red)+np.sum(temp_blue)))
                    is_infected=False if proportion_infected<0.2 else True
                    # Annotate image
                        # draw lines on image
                    cv2.line(annotated_image,(x1,y1),(x2,y2),(255,255,255),20) #white line
                    if is_infected:
                        number_color=(150, 150, 255)
                    else:
                        number_color=(255,150,150)
                    cv2.putText(
                        annotated_image, 
                        str(cell_number),  # Label with cell number
                        (x1,y1),  
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        2,  # Font size
                        number_color,  # White text
                        4  # Thickness
                    )
                # print(cell_ratios)
                # print(f"I counted {len([val for val in proportion_infected if val>=0.2])} infected and {len([val for val in proportion_infected if val<0.2])} uninfected cells.")
                print("Image processing complete.")
            
                # 8. Convert ratios to DataFrame for easy viewing
                cell_ratios_df = pd.DataFrame(cell_ratios)
                    # Calculate & report image overall blue:red ratio
                        # CHANGED from foreground_mask to thresh_combined_mask
                total_red = np.sum(red_channel)
                        # CHANGED from foreground_mask to thresh_combined_mask
                total_blue = np.sum(blue_channel)
                overall_blue_red_ratio = total_blue / total_red if total_red > 0 else 0
                    # Display results
                print("\nIndividual Cell Blue:Red Ratios")
                print(cell_ratios_df)
                print(f"\nTotal Cells: {total_cells}")
                print(f"Overall Blue:Red Ratio in Image: {overall_blue_red_ratio:.2f}")
                
                # Generate basename of image
                # base_filename = os.path.splitext(os.path.basename(file))[0]
                base_name = os.path.basename(file)
                # Trim the basename after the "-" symbol
                trimmed_base = base_name.split("-")[0] + "-"
                # Append the directory name, "_labeled", and ".jpg"
                new_name = f"{trimmed_base}{date_name}-labeled"

                # Generate .csv output filename based on the input image name
                csv_output_path = os.path.join(new_dir_path_csv, f"{new_name}.csv")
                image_output_path = os.path.join(new_dir_path_img, f"{new_name}.jpg")
                print(csv_output_path)
                print(image_output_path)
                # Save the dataframe to a CSV file
                cell_ratios_df.to_csv(csv_output_path, index=False)
                # Save the image to a JPG file
                print(f"Saved results to {csv_output_path}")
                # Save the labeled image to a JPG file
                cv2.imwrite(image_output_path, cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))  # Convert back to BGR for saving
                print(f"Saved labeled image to {image_output_path}")
            
                #uncomment below to get inline images
                '''
                fig, ax = plt.subplots(1, 4, figsize=(13, 10))
                ax[0].imshow(image_rgb, cmap='magma')
                ax[1].imshow(custom_gray, cmap='magma')
                ax[2].imshow(edges, cmap='magma')
                ax[3].imshow(annotated_image, cmap='magma')
                    # set the title to all subplots
                ax[0].set_title(file)
                ax[1].set_title("Grayscale Image")
                ax[2].set_title("Downsized grayscale Image")
                ax[3].set_title(f"{len([val for val in proportion_infected if val>=infection_threshold])} infected, {len([val for val in proportion_infected if val<infection_threshold])} uninfected")
                fig.tight_layout()'''

