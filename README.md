# Protist-Image-Analysis
Code, data, and firgures for analyzing protistan parasites of diatoms

- Problem (manual analysis): Too many images, not enough time
- Problem (automated analysis): Complicated images, background noise, strange shapes, infection dynamics
- Aproach(es):
	- Start by detecting the cells (via edge detection)
		- BW vs RGB:
			- Canny edge detection: 
				- noise reduction (Gaussian blur)
					- this one caused issues for me, but it's because Guassian is part of Canny()
				- finding the intensity gradient (Sobel kernel)
					- fine-tuning happens by adjusting the aperture (odd between 3-7)--3 is the right level of detail
					- next steps: remove letters from title, remove dots, decrease blur for red
				- non-maximum suppression
				- Hystersis thresholding
	- For RGB: isolate the color channels (red, blue)
		- Detect edges for both red and blue channels and overlap?
			- How will this impact the red:blue ratios?
	- Count cells
	- Calculate ratios of blue:red within cell masks
