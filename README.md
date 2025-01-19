# Protist-Image-Analysis
Code, data, and firgures for analyzing protistan parasites of diatoms via micrographs taken under fluorescent light 
with CalcoFluor White (CFW) staining

- Problem (manual analysis): Too many images, not enough time
- Problem (automated analysis): Complicated images characteristics, background noise, strange shapes, inconsistent 
cell boundaries, infection 
dynamics
- Goal: Build a computationsl model that can accurately 1. Count cells, 2. Classify cell infection 
status by proportion of pixels in blue channel versus red channel
- Aproach(es):
	- Normalize RGB color channels so that intensity ranges are comparable
		- Threshold channels 
	- Mask individual color channels from thresholded images
		- Combine red and blue channels for a full cell mask
	- Model cells with Hough Line Transform (HLT)
	- Cluster pixels with K-means clustering
	- Apply Sobel edge detection to identify cell boundaries
	- Cluter pixels using Gaussian Mixture Models
	- Combin HLT & K-means methods

- Model features:
	- Count individual cells
	- Calculate individual cell infected proportion via red & blue pixels within individual cell mask
		- Proportion outcomes determine infection classification

- Eventual Outcomes:
	- Extrapolate population dynamics
		- Changes in infection ratios over time series facilitates inference about shifts in 
mixed-community population dynamics

- Repository organization: 
	- "code": 
		- "benchmarking": includes code for cleaning raw csvs (result of running .ipynb files from 
"../code/final_models") in "data_cleaning"; model_evaluation.py evaluates performance using 
"../../data/master_csvs/*.csv"; model_evaluation_results_weight25-75_nonzero0_classes.csv is output from 
model_evaluation.py
	- "data": 
	- "figures":
	- "deprecated":
