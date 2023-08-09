import os
import numpy as np
import pandas as pd
from nilearn import input_data, connectome
from nilearn import image as nimg
from nilearn import plotting as nplot
import nibabel as nib
import bids


# Set the paths to the fMRI data and parcellation file
fmri_data_path = "path_to_fmri_data.nii.gz"
layout = bids.BIDSLayout(fmri_data_path,validate=False,
                        config=['bids','derivatives'])
parcellation_path = "path_to_parcellation.nii.gz"
sub = 'subject' #Specify subject

#Get the different files
func_files = layout.get(subject=sub,
                        datatype='func', task='rest',
                        desc='preproc',
                        space='MNI152NLin2009cAsym',
                        extension='nii.gz',
                       return_type='file')

mask_files = layout.get(subject=sub,
                        datatype='func', task='rest',
                        desc='brain',
                        suffix='mask',
                        space='MNI152NLin2009cAsym',
                        extension="nii.gz",
                       return_type='file')

confound_files = layout.get(subject=sub,
                            datatype='func', task='rest',
                            desc='confounds',
                           extension="tsv",
                           return_type='file')

func_file = func_files[0]
mask_file = mask_files[0]
confound_file = confound_files[0]

#Set up confound df
confound_df = pd.read_csv(confound_file, delimiter='\t')
confound_df.head()

# Select and set up confounds
confound_vars = ['trans_x','trans_y','trans_z',
                 'rot_x','rot_y','rot_z',
                 'global_signal',
                 'csf', 'white_matter']

derivative_columns = ['{}_derivative1'.format(c) for c
                     in confound_vars]

final_confounds = confound_vars + derivative_columns
confound_df = confound_df[final_confounds]
confound_df.head()

#Drop dummy TRs
raw_func_img = nimg.load_img(func_file)
func_img = raw_func_img.slicer[:,:,:,4:]

#Drop confound dummy TRs
drop_confound_df = confound_df.loc[4:]
drop_confound_df.head()


# Set the high-pass and low-pass filter cutoffs (in seconds)
high_pass = 0.042  # Example: 0.01 Hz
low_pass = 0.125   # Example: 0.1 Hz

# Set the masker settings
smoothing_fwhm = 6  # Smoothing FWHM value in mm
detrend = False      # Whether to detrend the time series signals
standardize = True  # Whether to z-score the time series signals

# Set the output path for the functional connectivity matrix
output_path = "path_to_save_fc_matrix.csv"

# Load the parcellation file
parcellation_img = input_data.parcellations.load_parcellation(parcellation_path)

# Create the NiftiLabelsMasker with the desired settings
masker = input_data.NiftiLabelsMasker(labels_img=parcellation_img,
                                      standardize=standardize,
                                      detrend=detrend,
                                      smoothing_fwhm=smoothing_fwhm,
                                      high_pass=high_pass,
                                      low_pass=low_pass)


# Apply the masker to extract  time series signals from the fMRI data
time_series = masker.fit_transform(func_img, confounds=final_confounds)


# Calculate the functional connectivity matrix
correlation_measure = connectome.ConnectivityMeasure(kind='correlation')
functional_connectivity = correlation_measure.fit_transform([time_series])[0]

# Convert the functional connectivity matrix to a pandas DataFrame
fc_matrix = pd.DataFrame(functional_connectivity)

# Save the functional connectivity matrix to a CSV file
fc_matrix.to_csv(output_path, index=False)

print("Functional connectivity matrix saved successfully!")