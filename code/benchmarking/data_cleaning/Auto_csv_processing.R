# Process automated csvs
library(dplyr)
# mutate columns
  # Cell_Source = image name
  # Status = uninfected (<20%B), early_infected (20-50%B), late_infected (50-80%B), dead (>80%B)
  # Unique_id = 
# clean column names
  # Blue:Red Ratio = BR_Ratio, Red Intensity = R_Area, Blue Intensity = B_Area, Proportion Infected = Prop_inf
# subset to As and Thn..? YES, FOR NOW
# combine all csv into master directory
  # /Users/kjehickman/Documents/Research/parasites/code/image_analysis/Jupyter_ImageAnalysis/figures/master_csvs/

# HLT ####

## HLT Data Cleaning
process_csv_hlt <- function(filepath) {
  # Extract file name for saving later
  file_name <- basename(filepath) %>% str_remove(".csv")
  # Read in data
  data <- read_csv(filepath, show_col_types = FALSE)
  # Fix col names 
  colnames(data) <- c("Cell_Number", "R_Area", "B_Area", "BR_Ratio", "Prop_Infected")
  # adjust cell number to 1-start index (from Python's 0)
  data$Cell_Number <- data$Cell_Number + 1
  # Add columns for cell number, cell source, and Status
  reshaped_data <- data %>%
    mutate(
      Cell_Source=as.character(file_name),
      Unique_id=as.character(interaction(Cell_Source, sep="_", Cell_Number)),
      Status = case_when(
        Prop_Infected < 0.2 ~ "uninfected",
        Prop_Infected >= 0.2 & Prop_Infected < 0.5 ~ "early_infected",
        Prop_Infected >= 0.5 & Prop_Infected < 0.8 ~ "late_infected",
        Prop_Infected >= 0.8 ~ "dead"
      )
    )
  # make new column a factor for easy data analysis
  reshaped_data$Status <- factor(reshaped_data$Status, levels = c("uninfected", "early_infected", "late_infected", "dead"))
  # return clean df
  return(reshaped_data)
}


# call process_csv_hlt() in for loop of directory
hlt_folder <- Sys.glob("/Users/kjehickman/Documents/Research/parasites/code/image_analysis/Jupyter_ImageAnalysis/figures/Hough_classification/data/*.csv")

for (file in hlt_folder) {
  # Extract file name for saving later
  file_name <- basename(file) %>% str_remove(".csv")
  # clean data
  tidy_data <- process_csv_hlt(file)
  # export to clean csv
  write_csv(tidy_data, file = paste("/Users/kjehickman/Documents/Research/parasites/code/image_analysis/Jupyter_ImageAnalysis/figures/Hough_classification/data/clean_csvs/", file_name, ".csv", sep = ""))
}

## HLT Data Combine

# set clean csv directory
  # JUST AS SUBSET
clean_hlt_as <- Sys.glob("/Users/kjehickman/Documents/Research/parasites/code/image_analysis/Jupyter_ImageAnalysis/figures/Hough_classification/data/clean_csvs/A*.csv")
# set master to first file in clean_files
master_hlt <- data.frame() # read_csv(clean_files[1]) # write.csv(rbind(df1, d32, df3), "filename.csv")
# set index
index = 0
for (file in clean_hlt_as) {
  if (index == 0) {
    # set master to first file
    master_hlt <- as.data.frame(read_csv(file))
  } else {
    # convert current file to df, csv
    current_file <- as.data.frame(read_csv(file))
    # bind current file to master
    master_hlt <- rbind(master_hlt, current_file)
  }
  # bump index
  index = index + 1
  print(paste0("On iteration ", index))
}
# write combined df
write_csv(master_hlt, file="/Users/kjehickman/Documents/Research/parasites/code/image_analysis/Jupyter_ImageAnalysis/figures/Hough_classification/hlt_master_cleaned.csv")


# Kmeans ####

## Kmeans Data Cleaning
process_csv_kmeans <- function(filepath) {
  # Extract file name for saving later
  file_name <- basename(filepath) %>% str_remove(".csv")
  # Read in data
  data <- read_csv(filepath, show_col_types = FALSE)
  # Fix col names 
  colnames(data) <- c("Cell_Number", "R_Area", "B_Area", "BR_Ratio", "Prop_Infected")
  # # adjust cell number to 1-start index (from Python's 0)--KMEANS DOESN'T NEED THIS
  # data$Cell_Number <- data$Cell_Number + 1
  # Add columns for cell number, cell source, and Status
  reshaped_data <- data %>%
    mutate(
      Cell_Source=as.character(file_name),
      Unique_id=as.character(interaction(Cell_Source, sep="_", Cell_Number)),
      Status = case_when(
        Prop_Infected < 0.2 ~ "uninfected",
        Prop_Infected >= 0.2 & Prop_Infected < 0.5 ~ "early_infected",
        Prop_Infected >= 0.5 & Prop_Infected < 0.8 ~ "late_infected",
        Prop_Infected >= 0.8 ~ "dead"
      )
    )
  # make new column a factor for easy data analysis
  reshaped_data$Status <- factor(reshaped_data$Status, levels = c("uninfected", "early_infected", "late_infected", "dead"))
  # return clean df
  return(reshaped_data)
}


# call process_csv_kmeans() in for loop of directory
kmeans_folder <- Sys.glob("/Users/kjehickman/Documents/Research/parasites/code/image_analysis/Jupyter_ImageAnalysis/figures/Kmeans_classification/data/*.csv")

for (file in kmeans_folder) {
  # Extract file name for saving later
  file_name <- basename(file) %>% str_remove(".csv")
  # clean data
  tidy_data <- process_csv_kmeans(file)
  # export to clean csv
  write_csv(tidy_data, file = paste("/Users/kjehickman/Documents/Research/parasites/code/image_analysis/Jupyter_ImageAnalysis/figures/Kmeans_classification/data/clean_csvs/", file_name, ".csv", sep = ""))
}

## Kmeans Data Combine

# set clean csv directory
# JUST AS SUBSET
clean_kmeans_as <- Sys.glob("/Users/kjehickman/Documents/Research/parasites/code/image_analysis/Jupyter_ImageAnalysis/figures/Kmeans_classification/data/clean_csvs/A*.csv")
# set master to first file in clean_files
master_kmeans <- data.frame() # read_csv(clean_files[1]) # write.csv(rbind(df1, d32, df3), "filename.csv")
# set index
index = 0
for (file in clean_kmeans_as) {
  if (index == 0) {
    # set master to first file
    master_kmeans <- as.data.frame(read_csv(file))
  } else {
    # convert current file to df, csv
    current_file <- as.data.frame(read_csv(file))
    # bind current file to master
    master_kmeans <- rbind(master_kmeans, current_file)
  }
  # bump index
  index = index + 1
  print(paste0("On iteration ", index))
}
# write combined df
write_csv(master_kmeans, file="/Users/kjehickman/Documents/Research/parasites/code/image_analysis/Jupyter_ImageAnalysis/figures/Kmeans_classification/kmeans_master_cleaned.csv")

# Kmeans_new (strict method, some empty files) ####

## Kmeeans_new Data Cleaning
process_csv_kmeans_new <- function(filepath) {
  # Extract file name for saving later
  file_name <- basename(filepath) %>% str_remove(".csv")
  # Read in data
  data <- read_csv(filepath, show_col_types = FALSE)
  # check to ensure file is not empty
  if (nrow(data) > 0) {
    # Process the data frame if it is not empty
    print(paste("Processing file:", file_name))
    # Fix col names 
    colnames(data) <- c("Cell_Number", "R_Area", "B_Area", "BR_Ratio", "Prop_Infected")
    # # adjust cell number to 1-start index (from Python's 0)--GMM DOESN'T NEED THIS
    # data$Cell_Number <- data$Cell_Number + 1
    # Add columns for cell number, cell source, and Status
    reshaped_data <- data %>%
      mutate(
        Cell_Source=as.character(file_name),
        Unique_id=as.character(interaction(Cell_Source, sep="_", Cell_Number)),
        Status = case_when(
          Prop_Infected < 0.2 ~ "uninfected",
          Prop_Infected >= 0.2 & Prop_Infected < 0.5 ~ "early_infected",
          Prop_Infected >= 0.5 & Prop_Infected < 0.8 ~ "late_infected",
          Prop_Infected >= 0.8 ~ "dead"
        )
      )
    # make new column a factor for easy data analysis
    reshaped_data$Status <- factor(reshaped_data$Status, levels = c("uninfected", "early_infected", "late_infected", "dead"))
    # return clean df
    return(reshaped_data)
  } else {
    # Skip the file if it is empty
    print(paste("Skipping empty file:", file_name))
  }
}



# call process_csv_kmeans() in for loop of directory
kmeans_new_folder <- Sys.glob("/Users/kjehickman/Documents/Research/parasites/code/image_analysis/Jupyter_ImageAnalysis/figures/Kmeans_new_classification/data/*.csv")

for (file in kmeans_new_folder) {
  # Extract file name for saving later
  file_name <- basename(file) %>% str_remove(".csv")
  # clean data
  tidy_data <- process_csv_gmm(file)
  # export to clean csv if file is not empty
  if (is.data.frame(tidy_data) == TRUE) {
    write_csv(tidy_data, file = paste("/Users/kjehickman/Documents/Research/parasites/code/image_analysis/Jupyter_ImageAnalysis/figures/Kmeans_new_classification/data/clean_csvs/", file_name, ".csv", sep = ""))
  } else {
    print(paste(file_name, "is empty."))
  }
}

## Kmeans Data Combine

# set clean csv directory
# JUST AS SUBSET
clean_kmeans_new_as <- Sys.glob("/Users/kjehickman/Documents/Research/parasites/code/image_analysis/Jupyter_ImageAnalysis/figures/Kmeans_new_classification/data/clean_csvs/A*.csv")
# set master to first file in clean_files
master_kmeans_new <- data.frame() # read_csv(clean_files[1]) # write.csv(rbind(df1, d32, df3), "filename.csv")
# set index
index = 0
for (file in clean_kmeans_new_as) {
  if (index == 0) {
    # set master to first file
    master_kmeans_new <- as.data.frame(read_csv(file))
  } else {
    # convert current file to df, csv
    current_file <- as.data.frame(read_csv(file))
    # bind current file to master
    master_kmeans_new <- rbind(master_kmeans_new, current_file)
  }
  # bump index
  index = index + 1
  print(paste0("On iteration ", index))
}
# write combined df
write_csv(master_kmeans_new, file="/Users/kjehickman/Documents/Research/parasites/code/image_analysis/Jupyter_ImageAnalysis/figures/Kmeans_new_classification/kmeans_new_master_cleaned.csv")



# GMM (strict method, some empty files) ####

## GMM Data Cleaning
process_csv_gmm <- function(filepath) {
  # Extract file name for saving later
  file_name <- basename(filepath) %>% str_remove(".csv")
  # Read in data
  data <- read_csv(filepath, show_col_types = FALSE)
  # check to ensure file is not empty
  if (nrow(data) > 0) {
    # Process the data frame if it is not empty
    print(paste("Processing file:", file_name))
    # Fix col names 
    colnames(data) <- c("Cell_Number", "R_Area", "B_Area", "BR_Ratio", "Prop_Infected")
    # # adjust cell number to 1-start index (from Python's 0)--GMM DOESN'T NEED THIS
    # data$Cell_Number <- data$Cell_Number + 1
    # Add columns for cell number, cell source, and Status
    reshaped_data <- data %>%
      mutate(
        Cell_Source=as.character(file_name),
        Unique_id=as.character(interaction(Cell_Source, sep="_", Cell_Number)),
        Status = case_when(
          Prop_Infected < 0.2 ~ "uninfected",
          Prop_Infected >= 0.2 & Prop_Infected < 0.5 ~ "early_infected",
          Prop_Infected >= 0.5 & Prop_Infected < 0.8 ~ "late_infected",
          Prop_Infected >= 0.8 ~ "dead"
        )
      )
    # make new column a factor for easy data analysis
    reshaped_data$Status <- factor(reshaped_data$Status, levels = c("uninfected", "early_infected", "late_infected", "dead"))
    # return clean df
    return(reshaped_data)
  } else {
    # Skip the file if it is empty
    print(paste("Skipping empty file:", file_name))
  }
}



# call process_csv_kmeans() in for loop of directory
gmm_folder <- Sys.glob("/Users/kjehickman/Documents/Research/parasites/code/image_analysis/Jupyter_ImageAnalysis/figures/GMM_classification/data/*.csv")

for (file in gmm_folder) {
  # Extract file name for saving later
  file_name <- basename(file) %>% str_remove(".csv")
  # clean data
  tidy_data <- process_csv_gmm(file)
  # export to clean csv
  if (is.data.frame(tidy_data) == TRUE) {
    write_csv(tidy_data, file = paste("/Users/kjehickman/Documents/Research/parasites/code/image_analysis/Jupyter_ImageAnalysis/figures/GMM_classification/data/clean_csvs/", file_name, ".csv", sep = ""))
  } else {
    print(paste(file_name, "is empty."))
  }
}

## GMM Data Combine

# set clean csv directory
# JUST AS SUBSET
clean_gmm_as <- Sys.glob("/Users/kjehickman/Documents/Research/parasites/code/image_analysis/Jupyter_ImageAnalysis/figures/GMM_classification/data/clean_csvs/A*.csv")
# set master to first file in clean_files
master_gmm <- data.frame() # read_csv(clean_files[1]) # write.csv(rbind(df1, d32, df3), "filename.csv")
# set index
index = 0
for (file in clean_gmm_as) {
  if (index == 0) {
    # set master to first file
    master_gmm <- as.data.frame(read_csv(file))
  } else {
    # convert current file to df, csv
    current_file <- as.data.frame(read_csv(file))
    # bind current file to master
    master_gmm <- rbind(master_gmm, current_file)
  }
  # bump index
  index = index + 1
  print(paste0("On iteration ", index))
}
# write combined df
write_csv(master_gmm, file="/Users/kjehickman/Documents/Research/parasites/code/image_analysis/Jupyter_ImageAnalysis/figures/GMM_classification/gmm_master_cleaned.csv")


# Combined w ####

## Combined_w Data Cleaning
process_csv_combined_w <- function(filepath) {
  # Extract file name for saving later
  file_name <- basename(filepath) %>% str_remove(".csv")
  # Read in data
  data <- read_csv(filepath, show_col_types = FALSE)
  # Fix col names 
  colnames(data) <- c("Cell_Number", "R_Area", "B_Area", "BR_Ratio", "Prop_Infected")
  # adjust cell number to 1-start index (from Python's 0)
  data$Cell_Number <- data$Cell_Number + 1
  # Add columns for cell number, cell source, and Status
  reshaped_data <- data %>%
    mutate(
      Cell_Source=as.character(file_name),
      Unique_id=as.character(interaction(Cell_Source, sep="_", Cell_Number)),
      Status = case_when(
        Prop_Infected < 0.2 ~ "uninfected",
        Prop_Infected >= 0.2 & Prop_Infected < 0.5 ~ "early_infected",
        Prop_Infected >= 0.5 & Prop_Infected < 0.8 ~ "late_infected",
        Prop_Infected >= 0.8 ~ "dead"
      )
    )
  # make new column a factor for easy data analysis
  reshaped_data$Status <- factor(reshaped_data$Status, levels = c("uninfected", "early_infected", "late_infected", "dead"))
  # return clean df
  return(reshaped_data)
}


# call process_csv_combined_w() in for loop of directory
combined_w_folder <- Sys.glob("/Users/kjehickman/Documents/Research/parasites/code/image_analysis/Jupyter_ImageAnalysis/figures/Combined_classification/w_data/*.csv")

for (file in combined_w_folder) {
  # Extract file name for saving later
  file_name <- basename(file) %>% str_remove(".csv")
  # clean data
  tidy_data <- process_csv_combined_w(file)
  # export to clean csv
  write_csv(tidy_data, file = paste("/Users/kjehickman/Documents/Research/parasites/code/image_analysis/Jupyter_ImageAnalysis/figures/Combined_classification/w_data/clean_csvs/", file_name, ".csv", sep = ""))
}

## Combined_w Data Combine

# set clean csv directory
# JUST AS SUBSET
clean_combined_w_as <- Sys.glob("/Users/kjehickman/Documents/Research/parasites/code/image_analysis/Jupyter_ImageAnalysis/figures/Combined_classification/w_data/clean_csvs/A*.csv")
# set master to first file in clean_files
master_combined_w <- data.frame() # read_csv(clean_files[1]) # write.csv(rbind(df1, d32, df3), "filename.csv")
# set index
index = 0
for (file in clean_combined_w_as) {
  if (index == 0) {
    # set master to first file
    master_combined_w <- as.data.frame(read_csv(file))
  } else {
    # convert current file to df, csv
    current_file <- as.data.frame(read_csv(file))
    # bind current file to master
    master_combined_w <- rbind(master_combined_w, current_file)
  }
  # bump index
  index = index + 1
  print(paste0("On iteration ", index))
}
# write combined df
write_csv(master_combined_w, file="/Users/kjehickman/Documents/Research/parasites/code/image_analysis/Jupyter_ImageAnalysis/figures/Combined_classification/combined_w_master_cleaned.csv")


# Combined wo ####


## Combined_wo Data Cleaning
process_csv_combined_wo <- function(filepath) {
  # Extract file name for saving later
  file_name <- basename(filepath) %>% str_remove(".csv")
  # Read in data
  data <- read_csv(filepath, show_col_types = FALSE)
  # Fix col names 
  colnames(data) <- c("Cell_Number", "R_Area", "B_Area", "BR_Ratio", "Prop_Infected")
  # adjust cell number to 1-start index (from Python's 0)
  data$Cell_Number <- data$Cell_Number + 1
  # Add columns for cell number, cell source, and Status
  reshaped_data <- data %>%
    mutate(
      Cell_Source=as.character(file_name),
      Unique_id=as.character(interaction(Cell_Source, sep="_", Cell_Number)),
      Status = case_when(
        Prop_Infected < 0.2 ~ "uninfected",
        Prop_Infected >= 0.2 & Prop_Infected < 0.5 ~ "early_infected",
        Prop_Infected >= 0.5 & Prop_Infected < 0.8 ~ "late_infected",
        Prop_Infected >= 0.8 ~ "dead"
      )
    )
  # make new column a factor for easy data analysis
  reshaped_data$Status <- factor(reshaped_data$Status, levels = c("uninfected", "early_infected", "late_infected", "dead"))
  # return clean df
  return(reshaped_data)
}


# call process_csv_combined_w() in for loop of directory
combined_wo_folder <- Sys.glob("/Users/kjehickman/Documents/Research/parasites/code/image_analysis/Jupyter_ImageAnalysis/figures/Combined_classification/wo_data/*.csv")

for (file in combined_wo_folder) {
  # Extract file name for saving later
  file_name <- basename(file) %>% str_remove(".csv")
  # clean data
  tidy_data <- process_csv_combined_wo(file)
  # export to clean csv
  write_csv(tidy_data, file = paste("/Users/kjehickman/Documents/Research/parasites/code/image_analysis/Jupyter_ImageAnalysis/figures/Combined_classification/wo_data/clean_csvs/", file_name, ".csv", sep = ""))
}

## Combined_w Data Combine

# set clean csv directory
# JUST AS SUBSET
clean_combined_wo_as <- Sys.glob("/Users/kjehickman/Documents/Research/parasites/code/image_analysis/Jupyter_ImageAnalysis/figures/Combined_classification/wo_data/clean_csvs/A*.csv")
# set master to first file in clean_files
master_combined_wo <- data.frame() # read_csv(clean_files[1]) # write.csv(rbind(df1, d32, df3), "filename.csv")
# set index
index = 0
for (file in clean_combined_wo_as) {
  if (index == 0) {
    # set master to first file
    master_combined_wo <- as.data.frame(read_csv(file))
  } else {
    # convert current file to df, csv
    current_file <- as.data.frame(read_csv(file))
    # bind current file to master
    master_combined_wo <- rbind(master_combined_wo, current_file)
  }
  # bump index
  index = index + 1
  print(paste0("On iteration ", index))
}
# write combined df
write_csv(master_combined_wo, file="/Users/kjehickman/Documents/Research/parasites/code/image_analysis/Jupyter_ImageAnalysis/figures/Combined_classification/combined_wo_master_cleaned.csv")


# Notes ####

## HTL testing

# file_test <- Sys.glob("/Users/kjehickman/Documents/Research/parasites/code/image_analysis/Jupyter_ImageAnalysis/figures/Hough_classification/data/AS_A_I_15_4-20240720.csv")
# # Extract file name for saving later
# file_name <- basename(file) %>% str_remove(".csv")
# # clean data
# test_f <- process_csv_hlt(file_test)
# write_csv(test_f, file = paste("/Users/kjehickman/Documents/Research/parasites/code/image_analysis/Jupyter_ImageAnalysis/figures/Hough_classification/data/clean_csvs/", file_name, ".csv", sep = ""))



