# Explore master csv data

# load packages
library("dplyr")
library("readr")
library("stringr")
library(tidyverse)

# dfs for reference
man_fp <- "/Users/kjehickman/Documents/Research/parasites/code/image_analysis/Jupyter_ImageAnalysis/figures/master_csvs/master_cleaned_manual.csv"
man_data <- read_csv(man_fp)

str(gmm_data)


# Create count inputs for MAE (counts) ####
  # predicted counts cells per image 
  # manual counts cells per image

manual_counts = list(28, 13, 15, 29, 36, 10, 28, 21, 32, 21, 33, 20, 44, 41) # 14
predicted_counts_comb_w = list(26, 55, 43, 9, 6, 1, 3, 72, 33, 48, 30, 21, 51, 50) # 14
predicted_counts_comb_wo = list(30, 170, 175, 14, 8, 1, 5, 198, 46, 167, 50, 35, 167, 175) # 14
predicted_counts_gmm = list(2, 1, 6, 4, 6, 4, 6, 1, 2, 1, 1, 2, 0, 1) # 13 + 1
predicted_counts_hlt = list(10, 21, 17, 13, 12, 10, 10, 8, 22, 7, 12, 14, 7, 8) # 14
predicted_counts_km = list(1, 5, 11, 8, 8, 4, 6, 6, 7, 2, 4, 5, 12, 18) # 14
predicted_counts_km_new = list(3, 0, 2, 3, 3, 2, 11, 0, 9, 0, 2, 5, 8, 4) # 11 + 3

clean_csvs_fp <- Sys.glob("/Users/kjehickman/Documents/Research/parasites/code/image_analysis/Jupyter_ImageAnalysis/figures/master_csvs/*.csv")

# loop through master files (models)
# for (master_file in clean_csvs_fp){
#   print(master_file)
#   # loop through images
#   uniq_imgs <- master_file$Cell_Source %>% unique()
#   for (img in uniq_imgs){
#     print(img)
#     # calculate how many cells are in each image
#     count <- 0
#     if (master_file$Cell_Source == img)
#       counts <- x
#   }
# }
# count number of cells per image PER model

# Charlie ####

# Define the folder path
folder_path <- "/Users/kjehickman/Documents/Research/parasites/code/image_analysis/Jupyter_ImageAnalysis/figures/master_csvs/"

# Initialize an empty list to store results
results <- list()

# Get all file names in the folder
files <- list.files(folder_path, pattern = "\\.csv$", full.names = TRUE)

# Loop through each file
for (file in files) {
  # Read the file
  df <- read_csv(file)
  
  # Extract the file name without extension
  file_name <- tools::file_path_sans_ext(basename(file))
  
  # Initialize a list for the current file
  file_results <- list()
  
  # Loop through each unique 'Cell_Source'
  for (cell_source in unique(df$Cell_Source)) {
    # Filter data for the current 'Cell_Source'
    source_data <- df %>% filter(Cell_Source == cell_source)
    
    # Get the highest 'Cell_Number' for this 'Cell_Source'
    max_cell_number <- max(source_data$Cell_Number, na.rm = TRUE)
    
    # Add the result to the file_results list
    file_results[[cell_source]] <- max_cell_number
  }
  
  # Add the file_results to the main results list
  results[[file_name]] <- file_results
}

# Check the results
str(results)

results$combined_w_master_cleaned$`AS_A_I_15_4-20240720`

# Save the results as an RDS file (optional)
saveRDS(results, "results.rds")



# Create label inputs for XX (labels) ####
  # characters, infection classes from 1...n manual cells
  # characters, infection classes from 1...n predicted cells

manual_labels = list()
predicted_labels_comb_w = list() # 14
predicted_labels_comb_wo = list() # 14
predicted_labels_gmm = list() # 13 + 1
predicted_labels_hlt = list() # 14
predicted_labels_km = list() # 14
predicted_labels_km_new =list() #

# combine each master file with manual by "Manual_Match" column
  # convert non-strict NA to strict w/new column
  # keep all info in manual
  # keep all info in model, introduce NAs in empty rows for manual
  # keep columns: Cell_Source, Cell_Number, Prop_Infected, Status, F1, Manual_Match
  # drop columns: R_Area, B_Area, BR_Ratio, Unique_id
colnames(gmm_data)
str(gmm_data)

# Clean NAs ####

# GMM
gmm_fp <- "/Users/kjehickman/Documents/Research/parasites/code/image_analysis/Jupyter_ImageAnalysis/figures/master_csvs/gmm_master_cleaned.csv"
gmm_data <- read_csv(gmm_fp)
# Rename Manual_Match to preserve data
df <- gmm_data %>%
  rename(Og_Manual_Match = Manual_Match)

# Create the new "Manual_Match" column
df <- df %>%
  mutate(Manual_Match = if_else(startsWith(Og_Manual_Match, "N"), NA_character_, Og_Manual_Match))

# Save the updated dataframe
write_csv(df, "/Users/kjehickman/Documents/Research/parasites/code/image_analysis/Jupyter_ImageAnalysis/figures/master_csvs/gmm_master_cleaned_na.csv")

# Kmeans (old)
km_fp <- "/Users/kjehickman/Documents/Research/parasites/code/image_analysis/Jupyter_ImageAnalysis/figures/master_csvs/kmeans_master_cleaned.csv"
km_data <- read_csv(km_fp)
# Rename Manual_Match to preserve data
df <- km_data %>%
  rename(Og_Manual_Match = Manual_Match)

# Create the new "Manual_Match" column
df <- df %>%
  mutate(Manual_Match = if_else(startsWith(Og_Manual_Match, "N"), NA_character_, Og_Manual_Match))

# Save the updated dataframe
write_csv(df, "/Users/kjehickman/Documents/Research/parasites/code/image_analysis/Jupyter_ImageAnalysis/figures/master_csvs/kmeans_master_cleaned_na.csv")

# Kmeans (new)
km_new_fp <- "/Users/kjehickman/Documents/Research/parasites/code/image_analysis/Jupyter_ImageAnalysis/figures/master_csvs/kmeans_new_master_cleaned.csv"
km_new_data <- read_csv(km_new_fp)
# Rename Manual_Match to preserve data
df <- km_new_data %>%
  rename(Og_Manual_Match = Manual_Match)

# Create the new "Manual_Match" column
df <- df %>%
  mutate(Manual_Match = if_else(startsWith(Og_Manual_Match, "N"), NA_character_, Og_Manual_Match))

# Save the updated dataframe
write_csv(df, "/Users/kjehickman/Documents/Research/parasites/code/image_analysis/Jupyter_ImageAnalysis/figures/master_csvs/kmeans_new_master_cleaned_na.csv")

# HLT
hlt_fp <- "/Users/kjehickman/Documents/Research/parasites/code/image_analysis/Jupyter_ImageAnalysis/figures/master_csvs/hlt_master_cleaned.csv"
hlt_data <- read_csv(hlt_fp)
# Rename Manual_Match to preserve data
df <- hlt_data %>%
  rename(Og_Manual_Match = Manual_Match)

# Create the new "Manual_Match" column
df <- df %>%
  mutate(Manual_Match = if_else(startsWith(Og_Manual_Match, "N"), NA_character_, Og_Manual_Match))

# Save the updated dataframe
write_csv(df, "/Users/kjehickman/Documents/Research/parasites/code/image_analysis/Jupyter_ImageAnalysis/figures/master_csvs/hlt_master_cleaned_na.csv")

# Combined W
comb_w_fp <- "/Users/kjehickman/Documents/Research/parasites/code/image_analysis/Jupyter_ImageAnalysis/figures/master_csvs/combined_w_master_cleaned.csv"
comb_w_data <- read_csv(comb_w_fp)
# Rename Manual_Match to preserve data
df <- comb_w_data %>%
  rename(Og_Manual_Match = Manual_Match)

# Create the new "Manual_Match" column
df <- df %>%
  mutate(Manual_Match = if_else(startsWith(Og_Manual_Match, "N"), NA_character_, Og_Manual_Match))

# Save the updated dataframe
write_csv(df, "/Users/kjehickman/Documents/Research/parasites/code/image_analysis/Jupyter_ImageAnalysis/figures/master_csvs/combined_w_master_cleaned_na.csv")


# Combined WO
comb_wo_fp <- "/Users/kjehickman/Documents/Research/parasites/code/image_analysis/Jupyter_ImageAnalysis/figures/master_csvs/combined_wo_master_cleaned.csv"
comb_wo_data <- read_csv(comb_wo_fp)
# Rename Manual_Match to preserve data
df <- comb_wo_data %>%
  rename(Og_Manual_Match = Manual_Match)

# Create the new "Manual_Match" column
df <- df %>%
  mutate(Manual_Match = if_else(startsWith(Og_Manual_Match, "N"), NA_character_, Og_Manual_Match))

# Save the updated dataframe
write_csv(df, "/Users/kjehickman/Documents/Research/parasites/code/image_analysis/Jupyter_ImageAnalysis/figures/master_csvs/combined_wo_master_cleaned_na.csv")

