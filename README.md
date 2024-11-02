# Protist-Image-Analysis
Code, data, and firgures for analyzing protistan parasites of diatoms

- Problem (manual analysis): Too many images, not enough time
- Problem (automated analysis): Complicated images, background noise, strange shapes, infection dynamics
- Goal: Recognize and count cells; categorize cells by infection ratios
- Aproach(es):
	- Normalize RGB color channels so that intensity ranges are comparable
		- Threshold channels 
	- Mask individual color channels from thresholded images
		- Combine red and blue channels for a full cell mask
	- Count cells
		- Use Hough line transformation to estimate number of individual cells
	- Classify cells
		- Calculate individual cell infection ratio via individual cell mask
		- Report number of cells and individual infection ratios
	- Extrapolate population dynamics
		- Changes in infection ratios over time series facilitates inference about shifts in 
mixed-community population dynamics
