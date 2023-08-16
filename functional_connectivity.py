import os
import numpy as np
import pandas as pd
from nilearn import input_data, connectome
from nilearn.connectome import ConnectivityMeasure
from nilearn import image as nimg
from nilearn import plotting as nplot
import nibabel as nib
import bids

def get_confounds(confound_tsv,confounds,dt=True):
    '''

    Parameters
    ----------
    confound_tsv : path to confounds.tsv
        
    confounds : list of confounder variables to be extracted
        
    dt : compute temporal derivatives. The default is True.

    Returns
    -------
    confound_mat : matrix of confounds

    '''
    if dt:
        dt_names = ['{}_derivative1'.format(c) for c in confounds]
        confounds = confounds + dt_names
    
    confound_df = pd.read_csv(confound_tsv,delimiter='/t')
    confound_df = counfound_df[confounds]
    confound_mat = confound_df.values
    
    return confound_mat


# Set the paths to the fMRI data, parcellation file, and output directory
#fmri_data_path = "path_to_fmri_data.nii.gz"
layout = bids.BIDSLayout(fmri_data_path,validate=False,
                        config=['bids','derivatives'])
#parcellation_path = "path_to_parcellation.nii.gz"
#output_path = "path_to_save_fc_matrix.csv"
#sub = 'subject' #Specify subject

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

# Load functional image and remove dummy trs
raw_func_img = nimg.load_img(func_file)
tr_drop = 4
func_img = raw_func_img.slicer[:,:,:,tr_drop:]

# Extract confounds
confound_vars = ['trans_x','trans_y','trans_z',
                 'rot_x','rot_y','rot_z',
                 'global_signal',
                 'csf', 'white_matter']
confounds = get_confounds(confound_file, confound_vars)
confounds = confounds[tr_drop:,:]

# Set the high-pass and low-pass filter cutoffs (in seconds)
high_pass = 0.042  # Example: 0.01 Hz
low_pass = 0.125   # Example: 0.1 Hz

# Set the masker settings
smoothing_fwhm = 6  # Smoothing FWHM value in mm
detrend = False      # Whether to detrend the time series signals
standardize = True  # Whether to z-score the time series signals

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
time_series = masker.fit_transform(func_img, confounds=confounds)


# Calculate the functional connectivity matrix
correlation_measure = ConnectivityMeasure(kind='correlation')
functional_connectivity = correlation_measure.fit_transform([time_series])

# Convert the functional connectivity matrix to a pandas DataFrame
fc_matrix = pd.DataFrame(functional_connectivity)

# Save the functional connectivity matrix to a CSV file
fc_matrix.to_csv(output_path, index=False)

print("Functional connectivity matrix saved successfully!")