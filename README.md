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

-----------------------------------------------------------------------------------------------------
- Repository organization: 
	- **code**: 
		- **benchmarking**: "data_cleaning" includes code for cleaning raw csvs (result of running .ipynb 
files 
from 
**../code/final_models**); **model_evaluation.py** evaluates performance using 
"./master_csvs/*.csv"; **model_evaluation_statistics.csv** is output from 
model_evaluation_F1.py
		- **final_models**: code for final versions of each 6 models
                - **_deprecated**: outdated code from various stages of model development
	- **data**: 
		- **test_images**: set of ".jpg" files that models were trained on (note, models were only evaluated 
on AS*.csv files)
		- **experimental_metadata.csv**: metadata collected during lab experiment
		- **model_evaluation_statistics.csv**: output from
"../code/benchmarking/model_evaluation_F1.py"
	- **figures**:
		- **model_evaluation_statistics.csv**: output from
"../code/benchmarking/model_evaluation_F1.py"
		- **video_explision.mp4**: downloadable video of active parasitism
	- **_deprecated**: 
		- outdated code and files from various stages of project development 
