# Image analysis model evaluation workflow

# READ IN LIBRARIES
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, r2_score
import matplotlib.pyplot as plt

# READ IN DATA

## Manual & Model Data Frames
manual_df = pd.read_csv("/Users/kjehickman/Documents/Research/parasites/code/image_analysis/Jupyter_ImageAnalysis/figures/master_csvs/master_cleaned_manual.csv") # manual
predicted_df_comb_w = pd.read_csv("/Users/kjehickman/Documents/Research/parasites/code/image_analysis/Jupyter_ImageAnalysis/figures/master_csvs/combined_w_master_cleaned_na.csv") # comb_w
predicted_df_comb_wo = pd.read_csv("/Users/kjehickman/Documents/Research/parasites/code/image_analysis/Jupyter_ImageAnalysis/figures/master_csvs/combined_wo_master_cleaned_na.csv") # comb_wo
predicted_df_gmm = pd.read_csv("/Users/kjehickman/Documents/Research/parasites/code/image_analysis/Jupyter_ImageAnalysis/figures/master_csvs/gmm_master_cleaned_na.csv") # gmm
predicted_df_hlt = pd.read_csv("/Users/kjehickman/Documents/Research/parasites/code/image_analysis/Jupyter_ImageAnalysis/figures/master_csvs/hlt_master_cleaned_na.csv") # hlt
predicted_df_km = pd.read_csv("/Users/kjehickman/Documents/Research/parasites/code/image_analysis/Jupyter_ImageAnalysis/figures/master_csvs/kmeans_master_cleaned_na.csv") # km
predicted_df_km_new = pd.read_csv("/Users/kjehickman/Documents/Research/parasites/code/image_analysis/Jupyter_ImageAnalysis/figures/master_csvs/kmeans_new_master_cleaned_na.csv") # km_new

# predicted_df_comb_w.info()
# manual_df.info()

## Manual & Model Stats
manual_counts = [28, 13, 15, 29, 36, 10, 28, 21, 32, 21, 33, 20, 44, 41] # 14
predicted_counts_comb_w = [26, 55, 43, 9, 6, 1, 3, 72, 33, 48, 30, 21, 51, 50] # 14
predicted_counts_comb_wo = [30, 170, 175, 14, 8, 1, 5, 198, 46, 167, 50, 35, 167, 175] # 14
predicted_counts_gmm = [2, 1, 6, 4, 6, 4, 6, 1, 2, 1, 1, 2, 0, 1] # 13 + 1
predicted_counts_hlt = [10, 21, 17, 13, 12, 10, 10, 8, 22, 7, 12, 14, 7, 8] # 14
predicted_counts_km = [1, 5, 11, 8, 8, 4, 6, 6, 7, 2, 4, 5, 12, 18] # 14
predicted_counts_km_new = [3, 0, 2, 3, 3, 2, 11, 0, 9, 0, 2, 5, 8, 4] # 11 + 3
all_labels = ["uninfected", "early_infected", "late_infected", "dead", "missed", "unlabeled"]


# READ IN FUNCTIONS
    ## ALL FUNCTIONS WORK!--may have to enter in terminal when instantiating functions that return dictionaries

## Data Wrangling: align arrays to make same length
def align_labels(manual_df, predicted_df):
    '''
    This function takes two dataframes with labeled cells and deciphers matched, missed, and misclassified cells. If the two dataframes
    have differing number of observations, this function reconciles those differences by substituting "missed" for cells in the manual 
    set that were not predicted, and "unlabeled" for cells that were erroneously predicted.
    Inputs: 
    manual_df = Manually curated cells with colname, "Matched_Manual", for cell ID and "Status" for classification status.
    predicted_df = Predicted cells with colname, "Matched_Manual", for cell ID and "Status" for classification status.
    '''
    # Initialize arrays for manual and predicted labels
    manual_labels = []
    predicted_labels = []

    # Instatiate matched cells--array of all cells that were MATCHED in predicted
    matched_cells = predicted_df.dropna(subset=["Manual_Match"])
    for _, row in matched_cells.iterrows():
        manual_id = row["Manual_Match"]
        
        # Check if the manual_id exists in the manual dataframe
        if manual_id in manual_df["Manual_Match"].values:
            manual_status = manual_df.loc[manual_df["Manual_Match"] == manual_id, "Status"].values[0]
            manual_labels.append(manual_status)
            predicted_labels.append(row["Status"])
        else:
            # Handle unmatched case (e.g., skip or append a default)
            print(f"Warning: Manual_Match '{manual_id}' not found in manual_df")
            continue
    # print("Manual len after matching:", len(manual_labels))
    # print("Predicted len after matching:", len(predicted_labels))
        
    # Instantiate missed cells--manual cells not represented in the predicted set
        # make a list of all ids that match in both dfs
    manual_matched_ids = manual_df["Manual_Match"].isin(predicted_df["Manual_Match"])
    # print("Matched ids:", manual_matched_ids)
        # missed_cells are all members of manual cells NOT in predicted?
    missed_cells = manual_df[~manual_matched_ids]
    for _, row in missed_cells.iterrows():
        # append missed cell status to manual_labels
        manual_labels.append(row["Status"])
        # append "missed" for status in missed cell index
        predicted_labels.append("missed")
    print("Missed cells:", len(missed_cells))
    # print("Manual len after missing:", len(manual_labels))
    # print("Predicted len after missing:", len(predicted_labels))

    # Instantiate misidentified cells--predicted cells that do not exist in the manual set
        # misidentified cells are set to locations where prediceted df has "NA" in Manual_Match
    misidentified_cells = predicted_df[predicted_df["Manual_Match"].isna()]
    for _, row in misidentified_cells.iterrows():
        # note unlabeled for manual set
        manual_labels.append("unlabeled")
        # add status for misidentified cell in the indexed row of predicted_labels
        predicted_labels.append(row["Status"])
    print("Misidentified cells:", len(misidentified_cells))
    print("Manual len after misidentified:", len(manual_labels))
    print("Predicted len after misidentified:", len(predicted_labels))

    return manual_labels, predicted_labels


## Evaluate individual model scores

def evaluate_cell_count(manual_counts, predicted_counts):
    '''
    This function evaluates the performance of model accuracy in cell counting.
    Inputs:
    manual_count = array of total cells counted across all images e.g. [10, 45, 13, 25, 30]
    predicted_counts = array of total cells predicted across all images e.g. [10, 45, 13, 25, 30]
    '''
    # Calculate Mean Absolute Error (MAE)
    mae = np.mean(np.abs(np.array(predicted_counts) - np.array(manual_counts)))
    # Calculate False Negative Rate (FNR) and False Positive Rate (FPR)
    false_negatives = np.sum(np.array(predicted_counts) < np.array(manual_counts))
    false_positives = np.sum(np.array(predicted_counts) > np.array(manual_counts))
    total_cells = np.sum(manual_counts)
    fnr = false_negatives / total_cells
    fpr = false_positives / total_cells
    r2 = r2_score(manual_counts, predicted_counts)

    return {"MAE": mae, "FNR": fnr, "FPR": fpr, "R^2": r2}


## Evaluate cell classifications using confusion matrix
def evaluate_classification(manual_labels, predicted_labels):
    '''
    This function evaluates the performance of a model's accuracy in cell classification with confusion matrices. Both inputs must be
    arrays of the same length, with each position correlating to a cell described at the same index in the other.
    Inputs: 
    manual_labels = array of total cell classification described across all images, concatenated e.g. ['dead', 'early_infected', 'late_infected', 'missed']
    predicted_labels = array of total cell classification predicted across all images, concatenated e.g. ['unlabeled', 'uninfected', 'late_infected', 'dead']
    '''
    # Confusion Matrix
    cm = confusion_matrix(manual_labels, predicted_labels, labels=all_labels)
    # Classification Report (includes F1 score, precision, recall for each class)
    report = classification_report(manual_labels, predicted_labels, labels=all_labels, output_dict=True, zero_division=1)

    return {"Confusion Matrix": cm, "Classification Report": report}


## 
def evaluate_clustering(manual_labels, predicted_labels):
    '''
    This function returns the Adjusted Rand Index for the two arrays. 
    Inputs: 
    manual_labels = array of total cell classification described across all images, concatenated e.g. ['dead', 'early_infected', 'late_infected', 'missed']
    predicted_labels = array of total cell classification predicted across all images, concatenated e.g. ['unlabeled', 'uninfected', 'late_infected', 'dead']
    '''
    # Calculate Adjusted Rand Index
    ari = adjusted_rand_score(manual_labels, predicted_labels)
    
    return ari


## 
def weighted_score(cell_count_metrics, classification_report, weight_count=0.5, weight_classification=0.5):
    '''
    This function produces a weighted score based on the performance of the model. The score is comprised of weights
    for both count and classification, which must total 1.0.
    Inputs: 
    manual_labels = array of total cell classification described across all images, concatenated e.g. ['dead', 'early_infected', 'late_infected', 'missed']
    predicted_labels = array of total cell classification predicted across all images, concatenated e.g. ['unlabeled', 'uninfected', 'late_infected', 'dead']
    '''
    # Extract relevant metrics
    count_score = 1 - cell_count_metrics["MAE"]  # Assuming lower MAE is better
    f1_scores = [classification_report[class_]['f1-score'] for class_ in classification_report if class_ not in ('accuracy', 'macro avg', 'weighted avg')]
    classification_score = np.mean(f1_scores)  # Average F1 score for all classes

    # Combine scores with weights
    combined_score = (weight_count * count_score) + (weight_classification * classification_score)
    return combined_score


## Combine all functions into a single evaluation function
def evaluate_method(manual_counts, predicted_counts, manual_labels, predicted_labels):
    '''
    This function combines cell count and classification methods to holistically evaluate the performance of a model's accuracy in both metrics.
    It computes MEA, FPR, FNR, Confusion Matrices and Reports, ARI, and produces a weighted score considering count and classification performance.
    Inputs: 
    manual_count = array of total cells counted across all images e.g. [10, 45, 13, 25, 30]
    predicted_counts = array of total cells predicted across all images e.g. [10, 45, 13, 25, 30]
    manual_labels = array of total cell classification described across all images, concatenated e.g. ['dead', 'early_infected', 'late_infected', 'missed']
    predicted_labels = array of total cell classification predicted across all images, concatenated e.g. ['unlabeled', 'uninfected', 'late_infected', 'dead']
    '''
    # Cell Count Metrics
    count_metrics = evaluate_cell_count(manual_counts, predicted_counts)

    # Classification Metrics
    class_metrics = evaluate_classification(manual_labels, predicted_labels)

    # Adjusted Rand Index
    ari = evaluate_clustering(manual_labels, predicted_labels)

    # Weighted Combined Score
    combined_score = weighted_score(count_metrics, class_metrics["Classification Report"], weight_count=0.25, weight_classification=0.75) # 

    return {
        "Count Metrics": count_metrics,
        "Classification Metrics": class_metrics,
        "ARI": ari,
        "Combined Score": combined_score
    }



# EVALUATE MODELS

## align arrays to make same length
    ### call comb_w
manual_labels_comb_w, predicted_labels_comb_w = align_labels(manual_df, predicted_df_comb_w) # missed: 249; misidentified: 324
    ### call comb_wo
manual_labels_comb_wo, predicted_labels_comb_wo = align_labels(manual_df, predicted_df_comb_wo) # missed: 283; misidentified: 1153
    ### call gmm--LOOKING GOOD
manual_labels_gmm, predicted_labels_gmm = align_labels(manual_df, predicted_df_gmm) # missed: 344; misidentified: 10
    ### call hlt--LOOKING GOOD
manual_labels_hlt, predicted_labels_hlt = align_labels(manual_df, predicted_df_hlt) # missed: 244; misidentified: 42
    ### call km
manual_labels_km, predicted_labels_km = align_labels(manual_df, predicted_df_km) # missed: 343; misidentified: 69
    ### call km_new--LOOKING GOOD
manual_labels_km_new, predicted_labels_km_new = align_labels(manual_df, predicted_df_km_new) # missed: 335; misidentified: 16


# Evaluate ALL models

    ## Combined w/remove_overlapping_lines()
        # Count Metrics: 'MAE': 18.214285714285715, 'FNR': 0.016172506738544475, 'FPR': 0.0215633423180593
        # F1 Macro: 0.04192699933089112
        # F1 Weighted: 0.01865182578103027
        # ARI: 0.2751711247064386
        # Combined score: -6.860558086115751
comb_w_evaluation = evaluate_method(manual_counts, predicted_counts_comb_w, manual_labels_comb_w, predicted_labels_comb_w)
print(comb_w_evaluation)
comb_w_disp = ConfusionMatrixDisplay(confusion_matrix=comb_w_evaluation['Classification Metrics']['Confusion Matrix'], display_labels=all_labels)
comb_w_disp.plot()
# plt.show()
plt.suptitle('Combined_w Evaluation (W: 25-75, NZ: 0)', fontsize=15, horizontalalignment='center')
plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig('/Users/kjehickman/Documents/Research/parasites/code/image_analysis/Jupyter_ImageAnalysis/figures/confusion_matrices/comb_w_weighted25-75_nonzero0.png', dpi = (150))

    ## Combined wo/remove_overlapping_lines()--Combined score: -28.73416246418463
        # Count Metrics: 'MAE': 72.85714285714286, 'FNR': 0.01078167115902965, 'FPR': 0.026954177897574125
        # F1 Macro: 0.014491131120857094
        # F1 Weighted: 0.0030776558676839255
        # ARI: 0.29494226779872973
        # Combined score: -28.73416246418463
comb_wo_evaluation = evaluate_method(manual_counts, predicted_counts_comb_wo, manual_labels_comb_wo, predicted_labels_comb_wo)
print(comb_wo_evaluation)
comb_wo_disp = ConfusionMatrixDisplay(confusion_matrix=comb_wo_evaluation['Classification Metrics']['Confusion Matrix'], display_labels=all_labels)
comb_wo_disp.plot()
# plt.show()
plt.suptitle('Combined_wo Evaluation (W: 25-75, NZ: 0)', fontsize=15, horizontalalignment='center')
plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig('/Users/kjehickman/Documents/Research/parasites/code/image_analysis/Jupyter_ImageAnalysis/figures/confusion_matrices/comb_wo_weighted25-75_nonzero0.png', dpi = (150))

    ## Gaussian Mixed Model
        # Count Metrics: 'MAE': 23.857142857142858, 'FNR': 0.03773584905660377, 'FPR': 0.0
        # F1 Macro: 0.04405797101449275
        # F1 Weighted: 0.0333219217163072
        # ARI: 0.06365472758616385
        # Combined score: -9.116422360248448
gmm_evaluation = evaluate_method(manual_counts, predicted_counts_gmm, manual_labels_gmm, predicted_labels_gmm)
print(gmm_evaluation)
gmm_disp = ConfusionMatrixDisplay(confusion_matrix=gmm_evaluation['Classification Metrics']['Confusion Matrix'], display_labels=all_labels)
gmm_disp.plot()
# plt.show()
plt.suptitle('GMM Evaluation (W: 25-75, NZ: 0)', fontsize=15, horizontalalignment='center')
plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig('/Users/kjehickman/Documents/Research/parasites/code/image_analysis/Jupyter_ImageAnalysis/figures/confusion_matrices/gmm_weighted25-75_nonzero0.png', dpi = (150))

    ## Hough Line Transform
        # Count Metrics: 'MAE': 15.714285714285714, 'FNR': 0.029649595687331536, 'FPR': 0.005390835579514825
        # F1 Macro: 0.11871605009593907
        # F1 Weighted: 0.2282488031454518
        # ARI: 0.11616712591550578
        # Combined score: -5.814484655656723
hlt_evaluation = evaluate_method(manual_counts, predicted_counts_hlt, manual_labels_hlt, predicted_labels_hlt)
print(hlt_evaluation)
hlt_disp = ConfusionMatrixDisplay(confusion_matrix=hlt_evaluation['Classification Metrics']['Confusion Matrix'], display_labels=all_labels)
hlt_disp.plot()
# plt.show()
plt.suptitle('HLT Evaluation (W: 25-75, NZ: 0)', fontsize=15, horizontalalignment='center')
plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig('/Users/kjehickman/Documents/Research/parasites/code/image_analysis/Jupyter_ImageAnalysis/figures/confusion_matrices/hlt_weighted25-75_nonzero0.png', dpi = (150))

    ## Kmeans (original)
        # Count Metrics: 'MAE': 19.571428571428573, 'FNR': 0.03773584905660377, 'FPR': 0.0
        # F1 Macro: 0.03214071407140714
        # F1 Weighted: 0.02154692742001473
        # ARI: 0.284305283474167
        # Combined score: -7.409287000128585
km_evaluation = evaluate_method(manual_counts, predicted_counts_km, manual_labels_km, predicted_labels_km)
print(km_evaluation)
km_disp = ConfusionMatrixDisplay(confusion_matrix=km_evaluation['Classification Metrics']['Confusion Matrix'], display_labels=all_labels)
km_disp.plot()
# plt.show()
plt.suptitle('Kmeans Evaluation (W: 25-75, NZ: 0)', fontsize=15, horizontalalignment='center')
plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig('/Users/kjehickman/Documents/Research/parasites/code/image_analysis/Jupyter_ImageAnalysis/figures/confusion_matrices/km_weighted25-75_nonzero0.png', dpi = (150))

    ## Kmeans w/bounded box (new)
        # Count Metrics: 'MAE': 22.785714285714285, 'FNR': 0.03773584905660377, 'FPR': 0.0
        # F1 Macro: 0.07329363261566652
        # F1 Weighted: 0.05382205224538689
        # ARI: 0.10775953384140045
        # Combined score: -8.670309534716313
km_new_evaluation = evaluate_method(manual_counts, predicted_counts_km_new, manual_labels_km_new, predicted_labels_km_new)
print(km_new_evaluation)
km_new_disp = ConfusionMatrixDisplay(confusion_matrix=km_new_evaluation['Classification Metrics']['Confusion Matrix'], display_labels=all_labels)
km_new_disp.plot()
# plt.show()
plt.suptitle('Kmeans_new Evaluation (W: 25-75, NZ: 0)', fontsize=15, horizontalalignment='center')
plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig('/Users/kjehickman/Documents/Research/parasites/code/image_analysis/Jupyter_ImageAnalysis/figures/confusion_matrices/km_new_weighted25-75_nonzero0.png', dpi = (150))



# EXPORT MODEL EVALUATION DATA

## Define the results as dictionaries (placeholders for demonstration)
model_results = {
    "combined_w": comb_w_evaluation,
    "combined_wo": comb_wo_evaluation,
    "gmm": gmm_evaluation,
    "hlt": hlt_evaluation,
    "kmeans": km_evaluation,
    "kmeans_new": km_new_evaluation
}

# Initialize an empty list to store data for the DataFrame
model_data = []

# Extract metrics for each model
for model_name, evaluation in model_results.items():
    # Extract metrics from Count Metrics
    mae = evaluation['Count Metrics']['MAE']
    fnr = evaluation['Count Metrics']['FNR']
    fpr = evaluation['Count Metrics']['FPR']
    r2 = evaluation['Count Metrics']['R^2']
    
    # Extract metrics from Classification Metrics
    # confusion_matrix = evaluation['Classification Metrics']['Confusion Matrix']
        ## Overall scores
    F1_macro = evaluation['Classification Metrics']['Classification Report']['macro avg'].get('f1-score', None)
    precision_macro = evaluation['Classification Metrics']['Classification Report']['macro avg'].get('precision', None)
    recall_macro = evaluation['Classification Metrics']['Classification Report']['macro avg'].get('recall', None)
    F1_weighted = evaluation['Classification Metrics']['Classification Report']['weighted avg'].get('f1-score', None)
    precision_weighted = evaluation['Classification Metrics']['Classification Report']['weighted avg'].get('precision', None)
    recall_weighted = evaluation['Classification Metrics']['Classification Report']['weighted avg'].get('recall', None)
    accuracy = evaluation['Classification Metrics']['Classification Report'].get('accuracy', None)
        ## Classification scores
    f1_uninfected = evaluation['Classification Metrics']['Classification Report']['uninfected'].get('f1-score', None)
    precision_uninfected = evaluation['Classification Metrics']['Classification Report']['uninfected'].get('precision', None)
    recall_uninfected = evaluation['Classification Metrics']['Classification Report']['uninfected'].get('recall', None)
    f1_Einfected = evaluation['Classification Metrics']['Classification Report']['uninfected'].get('f1-score', None)
    precision_Einfected = evaluation['Classification Metrics']['Classification Report']['uninfected'].get('precision', None)
    recall_Einfected = evaluation['Classification Metrics']['Classification Report']['uninfected'].get('recall', None)
    f1_Linfected = evaluation['Classification Metrics']['Classification Report']['uninfected'].get('f1-score', None)
    precision_Linfected = evaluation['Classification Metrics']['Classification Report']['uninfected'].get('precision', None)
    recall_Linfected = evaluation['Classification Metrics']['Classification Report']['uninfected'].get('recall', None)
    f1_dead = evaluation['Classification Metrics']['Classification Report']['uninfected'].get('f1-score', None)
    precision_dead = evaluation['Classification Metrics']['Classification Report']['uninfected'].get('precision', None)
    recall_dead = evaluation['Classification Metrics']['Classification Report']['uninfected'].get('recall', None)

    # Extract Adjusted Rand Index (ARI) and Combined Score
    ari = evaluation.get('ARI', None)
    combined_score = evaluation.get('Combined Score', None)
    
    # Append a row of results for the current model
    model_data.append({
        "Model": model_name,
        "MAE": mae,
        "R^2": r2,
        # "FNR": fnr,
        # "FPR": fpr,
        "Uninf_F1": f1_uninfected,
        "Uninf_precision": precision_uninfected,
        "Uninf_recall": recall_uninfected,
        "EarlyInf_F1": f1_Einfected,
        "EarlyInf_precision": precision_Einfected,
        "EarlyInf_recall": recall_Einfected,
        "LateInf_F1": f1_Linfected,
        "LateInf_precision": precision_Linfected,
        "LateInf_recall": recall_Linfected,
        "Dead_F1": f1_dead,
        "Dead_precision": precision_dead,
        "Dead_recall": recall_dead,
        "F1_Macro": F1_macro,
        "Precision_Macro": precision_macro,
        "Recall_Macro": recall_macro,
        "F1_Weighted": F1_weighted,
        "Precision_Weighted": precision_weighted,
        "Recall_Weighted": recall_weighted,
        "Accuracy": accuracy,
        # "ARI": ari,
        "Combined Score": combined_score
        # "Confusion Matrix": confusion_matrix  # Optional: for human readability, you might omit this
    })

# Create a DataFrame
model_eval_df = pd.DataFrame(model_data)

# Save to CSV
output_file = "/Users/kjehickman/Documents/Research/parasites/code/image_analysis/Jupyter_ImageAnalysis/figures/model_evaluation_results_weight25-75_nonzero1_classes.csv"
model_eval_df.to_csv(output_file, index=False)

print(f"Results saved to {output_file}")

