import numpy as np
import pandas as pd
from nilearn import input_data, connectome

# Set the paths to the fMRI data and parcellation file
fmri_data_path = "path_to_fmri_data.nii.gz"
parcellation_path = "path_to_parcellation.nii.gz"

# Set the high-pass and low-pass filter cutoffs (in seconds)
high_pass = 0.042  # Example: 0.01 Hz
low_pass = 0.125   # Example: 0.1 Hz

# Set the masker settings
smoothing_fwhm = 6  # Smoothing FWHM value in mm
detrend = False      # Whether to detrend the time series signals
standardize = True  # Whether to z-score the time series signals

# Set the output path for the functional connectivity matrix
output_path = "path_to_save_fc_matrix.csv"

# Load the fMRI data and parcellation file
fmri_img = input_data.fmri_dataset.fmri_img(fmri_data_path)
parcellation_img = input_data.parcellations.load_parcellation(parcellation_path)

# Create the NiftiLabelsMasker with the desired settings
masker = input_data.NiftiLabelsMasker(labels_img=parcellation_img,
                                      standardize=standardize,
                                      detrend=detrend,
                                      smoothing_fwhm=smoothing_fwhm,
                                      high_pass=high_pass,
                                      low_pass=low_pass)

# Apply the masker to extract time series signals from the fMRI data
time_series = masker.fit_transform(fmri_img)

# Calculate the functional connectivity matrix
correlation_measure = connectome.ConnectivityMeasure(kind='correlation')
functional_connectivity = correlation_measure.fit_transform([time_series])[0]

# Convert the functional connectivity matrix to a pandas DataFrame
fc_matrix = pd.DataFrame(functional_connectivity)

# Save the functional connectivity matrix to a CSV file
fc_matrix.to_csv(output_path, index=False)

print("Functional connectivity matrix saved successfully!")