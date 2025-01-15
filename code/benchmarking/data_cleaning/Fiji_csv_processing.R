# Load necessary library
library(tidyverse)
library(tools)

# Function to process a single CSV file
process_csv_simple <- function(filepath) {
  # Read in data
  data <- read_csv(filepath, show_col_types = FALSE)
  # Separate the RGB values (first column) for reference
  rgb <- data[[1]]
  # Remove the first column (RGB identifiers) and transpose data for reshaping
  reshaped_data <- data %>%
    select(-1) %>%
    as.data.frame() %>%
    mutate(RGB = rgb) %>%
    pivot_longer(
      cols = -RGB, #everything(),
      names_to = "Metric", 
      values_to = "Value"
    ) 
  # Add columns for cell number, cell source, area_percent & area_total (for RGB)
  reshaped_data <- reshaped_data %>% 
    mutate(
      Cell_Number=as.integer(ifelse(startsWith(Metric,"A"), str_remove(Metric,"Area"), str_remove(Metric,"%Area"))),
      # Extract filename without extension
      Cell_Source=basename(filepath) %>% str_remove(".csv"),
      # set the R area equal to Value when RGB==1 and no % in Metric (repeat for GB==2,3)
      R_Area=as.integer(ifelse(RGB == 1 & startsWith(Metric, "A"), Value, NA)),
      # set the R percent equal to Value when RGB==1,2,3 and % in Metric (repeat for GB==2,3)
      R_Percent=as.integer(ifelse(RGB == 1 & startsWith(Metric, "%"), Value, NA)),
      G_Area=as.integer(ifelse(RGB == 2 & startsWith(Metric, "A"), Value, NA)),
      G_Percent=as.integer(ifelse(RGB == 2 & startsWith(Metric, "%"), Value, NA)),
      B_Area=as.integer(ifelse(RGB == 3 & startsWith(Metric, "A"), Value, NA)),
      B_Percent=as.integer(ifelse(RGB == 3 & startsWith(Metric, "%"), Value, NA))
    )
  # sort by Cell_Number and remove redundant columns (containing NA)
  as.data.frame(reshaped_data)
  # Consolidate rows by Cell_Number and collapse the data for each column
  tidy_data <- reshaped_data %>%
    # Group by unique cell identifier
    group_by(Cell_Number, Cell_Source) %>% 
    summarize(
      R_Area = na.omit(R_Area)[1],         # Keep the non-NA value for R_Area
      R_Percent = na.omit(R_Percent)[1],   # Keep the non-NA value for R_Percent
      G_Area = na.omit(G_Area)[1],         # Keep the non-NA value for G_Area
      G_Percent = na.omit(G_Percent)[1],   # Keep the non-NA value for G_Percent
      B_Area = na.omit(B_Area)[1],         # Keep the non-NA value for B_Area
      B_Percent = na.omit(B_Percent)[1],   # Keep the non-NA value for B_Percent
      .groups = "drop"                     # Ungroup the result
    )
  # Make column for Blue:Red Ratio
    # THIS NEEDS TO BE X_Area BUT RN ALL THOSE ARE THE SAME...
  tidy_data <- tidy_data %>% 
    mutate(BR_Ratio=ifelse(R_Percent>0,B_Percent/R_Percent,0),
           # Make a unique identifier for each cell--needs to be in subsequent mutate..?
           Unique_id=interaction(Cell_Source, sep="_", Cell_Number))

  # TDL: 
    # reorder: tech rep, cell num, unique_id, status, etc.
    # fix x_area (fix in Fiji)
  return(tidy_data)
}

# write all clean csvs to master csv
  # write individually, rbind clean
dir_csv <- "/Users/kjehickman/Documents/Research/parasites/code/image_analysis/Jupyter_ImageAnalysis/figures/manual_classification/classified/ROI_csv/"
files <- Sys.glob("/Users/kjehickman/Documents/Research/parasites/code/image_analysis/Jupyter_ImageAnalysis/figures/manual_classification/classified/ROI_csv/*.csv")

for (file in files) {
  # Extract file name for saving later
  file_name <- basename(file) %>% str_remove(".csv")
  # Call cleaning function
  tidy_data <- process_csv_simple(file)
  # save object for each csv then concatenate all csv into one--no
  # write each df to csv, using filename specified
  write_csv(tidy_data, file = paste("/Users/kjehickman/Documents/Research/parasites/code/image_analysis/Jupyter_ImageAnalysis/figures/manual_classification/classified/clean_csv/", file_name, ".csv", sep = ""))
  }

# Loop through cleaned files and combine them
clean_files <- Sys.glob("/Users/kjehickman/Documents/Research/parasites/code/image_analysis/Jupyter_ImageAnalysis/figures/manual_classification/classified/clean_csv/*.csv")
# set master to first file in clean_files
master <- data.frame() # read_csv(clean_files[1]) # write.csv(rbind(df1, d32, df3), "filename.csv")
# set index
index = 0
for (file in clean_files) {
  if (index == 0) {
    # set master to first file
    master <- as.data.frame(read_csv(file))
  } else {
    # convert current file to df, csv
    current_file <- as.data.frame(read_csv(file))
    # bind current file to master
    master <- rbind(master, current_file)
  }
  # bump index
  index = index + 1
  print(paste0("On iteration ", index))
}
# write combined df
write_csv(master, file="/Users/kjehickman/Documents/Research/parasites/code/image_analysis/Jupyter_ImageAnalysis/figures/manual_classification/master_cleaned_take2.csv")


#----------------

# Test file
# fp <- "/Users/kjehickman/Documents/Research/parasites/code/image_analysis/Jupyter_ImageAnalysis/figures/manual_classification/classified/ROI_csv/AS_B_I_22_5-20240720.csv"
# 
# # Process test file
# tidy_data <- process_csv_simple(fp)
# 
# # View result
# print(tidy_data)
# str(tidy_data)

# WRITE TO OUTFILE
# write_csv(tidy_data, "/Users/kjehickman/Documents/Research/parasites/code/image_analysis/Jupyter_ImageAnalysis/figures/manual_classification/classified/ROI_csv/fuckup_FijiImages.csv")
# file_test <- Sys.glob("/Users/kjehickman/Documents/Research/parasites/code/image_analysis/Jupyter_ImageAnalysis/figures/manual_classification/classified/ROI_csv/AS_A_I_15_4-20240720.csv")
# fix_data <- process_csv_simple(file_test)
# write_csv(fix_data, "/Users/kjehickman/Documents/Research/parasites/code/image_analysis/Jupyter_ImageAnalysis/figures/manual_classification/classified/clean_csv/AS_A_I_15_4-20240720.csv")


# open master csv--sink() writes superfluous info, separates tables, duplicates header
# take 1
# ?sink
# sink(file="/Users/kjehickman/Documents/Research/parasites/code/image_analysis/Jupyter_ImageAnalysis/figures/manual_classification/classified/sink_master_cleaned.csv", type = "output")

# close master csv
# sink()



# myFiles <- list.files(pattern = "\\.csv$") 
# 
# for(i in myFiles){
#   myDf <- read.csv(i)
#   outputFile <- paste0(tools::file_path_sans_ext(i), ".Rdata")
#   outputFile <- gsub("nE_pT_", "e_", outputFile, fixed = TRUE)
#   save(myDf, file = outputFile)
# }

# # output n individual data sets, containing a single column
# for (i in 1:ncol(mymat)) {
#   a <- data.frame(mymat[, i])
#   mytime <- format(Sys.time(), "%b_%d_%H_%M_%S_%Y")
#   myfile <- file.path(tempdir(), paste0(mytime, "_", i, ".txt"))
#   write.table(a, file = myfile, sep = "", row.names = FALSE, col.names = FALSE,
#               quote = FALSE, append = FALSE)
# }
