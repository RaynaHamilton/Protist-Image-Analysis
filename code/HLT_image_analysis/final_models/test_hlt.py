# Hough Line Transform Automated Image ANalysis

'''
# Automated Image Analysis: Diatoms & Infection Status

Below, I have made five main modifications to the existing micrograph analysis protocol:
1) Conversion of continuous (0-255) blue and red color channel values to binary (0 or 255) values based on 
whether pixel intensities pass a threshold
2) Conversion of RGB (3-dimensional) to grayscale (2-dimensional) array by taking the maximum of the red and 
blue pixel value (the grayscale matrix pixel values are also 0 or 255)
3) Removal of large (i.e. wider than cell width) blobs that look like background noise/stains using a sliding window approach.
4) Use of Hough Line transformation algorithm to model cells as lines, rather than more abstract shapes
5) Downstream filtering of lines with similar slopes and start/end coordinates, as Hough Line transformation 
sometimes returns similar parellel lines.

This approach uses the Hough line transform to model lines, leveraging the long, thin shape of cells. Note that this algorithm 
has an unfortunate tendency to make lots of parallel lines when a thick line is supplied. In order to circumvent this, I am 
downscaling the images to 160*120 pixels before inputting them to the model. Some playing with these dimensions as well as the 
parameters to the HoughLinesP function would be recommended.

I have implemented a quick solution to remove large clumps of pixels. This just involves sliding a (here 30x30) window across 
the image and, if a region if found where almost all of the pixels are filled in, forcibly replacing these pixel values with 0. 
This helps to reduce some of the background noise, but is not a perfect solution - it reduces the true positive rate alongside 
the false positive rate. Also, it is important that the window size (30) is larger than the cell width, to avoid removing cells. 
Also, I am currently using a quantile to set the threshold for the red and blue color channels. This seems to work well for these 
few images, but a more nuanced approach may be needed when the whole dataset is used.

Filtering of similar lines is done by first finding pairs of lines with slopes within a certain threshold of each other. For all these 
pairs, those with a single start or endpoint within a certain threshold from each other are filtered by removing the shorter line.
'''

# Import Libraries
import cv2
import numpy as np
from skimage.measure import label, regionprops
from skimage import filters
import skimage
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from math import sqrt

# Define Variables
original_x_size,original_y_size=1388,1040 #initial size of input images
x_size,y_size=160,120 #downscaled size of images used for hough line detection
infection_threshold=0.2 #proportion of pixels which must be blue for a cell to be declared infected
target_red_quantile=0.92 #quantile that red pixel must pass to be kept
red_minimum=30 #minimum red pixel value to be kept
target_blue_quantile=0.95 #quantile that blue pixel must pass to be kept
blue_minimum=60 #minimum blue pixel value to be kept
target_dir="../../images/raw/"
output_csv_dir="../../figures/Hough_classification/data/"
output_image_dir="../../figures/Hough_classification/classified"

if not os.path.isdir(output_csv_dir):
    os.mkdir(output_csv_dir)
if not os.path.isdir(output_image_dir):
    os.mkdir(output_image_dir)


#-----------------------------------------------------------------


# Define Functions

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


#-----------------------------------------------------------------


# Call Functions

#process the images
for file in os.listdir(target_dir):
    if file.endswith(".jpg"):
        print(file)
        #read in image and get channels
        image = cv2.imread(f"{target_dir}/{file}") 
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        red_channel = np.copy(image_rgb[:, :, 0])
        blue_channel = np.copy(image_rgb[:, :, 2])

        red_channel,blue_channel=convert_to_binary(red_channel,blue_channel,target_red_quantile,red_minimum,target_blue_quantile,blue_minimum)
        
        #remove large blobs/noise
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
                'Cell Number': cell_number,
                'Red Intensity Sum': red_sum,
                'Blue Intensity Sum': blue_sum,
                'Blue:Red Ratio': blue_red_ratio,
                'Proportion Infected': proportion_infected
            })        
            # proportion_infected.append(np.sum(temp_blue)/(np.sum(temp_red)+np.sum(temp_blue)))
            is_infected=False if proportion_infected<0.2 else True
            # is_infected=False if cell_ratios[4][-1] <0.2 else True
            # is_infected=False if cell_ratios.at['Proportion Infected', -1] <0.2 else True
            # is_infected=False if proportion_infected[-1]<0.2 else True
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
        
        # Generate basename
        base_filename = os.path.splitext(os.path.basename(file))[0]
        # Generate .csv output filename based on the input image name
        csv_output_path = os.path.join(output_csv_dir, f"{base_filename}.csv")
        image_output_path = os.path.join(output_image_dir, f"{base_filename}_labeled.jpg")
        
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
    