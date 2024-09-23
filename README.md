# Protist-Image-Analysis
Code, data, and firgures for analyzing protistan parasites of diatoms

- Problem (manual analysis): Too many images, not enough time
- Problem (automated analysis): Complicated images, background noise, strange shapes, infection dynamics
- Aproach(es):
	- Start by detecting the cells (via edge detection)
		- Sobel edge detection
	- Then isolate the color channels (red, blue)
	- Refine edges by thresholding
		- Hysteresis thresholding
	- Count cells
	- Calculate ratios of blue:red within cell masks
