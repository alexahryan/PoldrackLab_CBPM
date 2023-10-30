#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
from nilearn import input_data, connectome
from nilearn.connectome import ConnectivityMeasure
from nilearn import image as nimg
import nibabel as nib
import bids
from bids import BIDSLayout
import os.path as op
from glob import glob

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

OAK = '/oak/stanford/groups/russpold'
fmri_dir = op.join(OAK, 'data/uh2/aim1/BIDS')
confound_dir = op.join(OAK, 'data/uh2/aim1/derivatives/fmriprep')
subjects = [i.split('/')[-1] for i in glob(op.join(fmri_dir, 'sub-s*'))] 
derivatives = ['trans_x','trans_y','trans_z', 'rot_x','rot_y','rot_z','global_signal','csf', 'white_matter']
#layout = bids.layout.BIDSLayout(root=fmri_dir, validate=False, derivatives=derivatives)
parcellation_path = "/home/users/ahryan/CBPM_PoldrackLab/tpl-MNI152NLin2009cAsym_res-01_atlas-smorgasbord_dseg.nii"
output_files = dict.fromkeys(subjects)
for sub in subjects:
    output_path = '/home/users/ahryan/CBPM_PoldrackLab/fc-data/'+sub+'.csv'
    output_files[sub] = output_path

# Set the high-pass and low-pass filter cutoffs (in seconds)
high_pass = 0.042  
low_pass = 0.12

# Set the masker settings
smoothing_fwhm = 6  # Smoothing FWHM value in mm
detrend = False      # Whether to detrend the time series signals
standardize = True  # Whether to z-score the time series signals

task = 'rest'
sessions = ['ses-1', 'ses-2']
for sub in subjects:
    for session in sessions:
        #for task in tasks:
        fmri_sub_dir = op.join(fmri_dir, sub, session, 'func')
        confound_sub_dir = op.join(confound_dir, sub, session, 'func')    
        func_file_name = sub+'_'+session+'_task-'+task+'_run-1_bold.nii.gz'
        confound_file_name = sub+'_'+session+'_task-'+task+'_'+'run-1_desc-confounds_timeseries.tsv'
        func_file = os.path.join(fmri_sub_dir, func_file_name)
        confound_file = os.path.join(confound_sub_dir, confound_file_name)
        #if os.path.exists(confound_file):
            #confound_df = pd.read_csv(confound_file, delimiter='\t')
        raw_func_img = nimg.load_img(func_file)
        tr_drop = 4
        func_img = raw_func_img.slicer[:,:,:,tr_drop:]
        confounds = get_confounds(confound_file, derivatives)
        confounds = confounds[tr_drop:,:]
        parcellation_img = input_data.parcellations.load_parcellation(parcellation_path)
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
        fc_matrix.to_csv(output_files[sub], index=False)