import os
import numpy as np
import pandas as pd
from nilearn import input_data, connectome, image as nimg
from nilearn.input_data import NiftiLabelsMasker
from nilearn.connectome import ConnectivityMeasure
import nibabel as nib
from glob import glob

def load_node_numbers(mapping_file):
    mapping_df = pd.read_csv(mapping_file)
    node_numbers = list(mapping_df['NodeNo'])
    return node_numbers

def get_confounds(confound_tsv, confounds, dt=True):
    if dt:
        dt_names = [f'{c}_derivative1' for c in confounds]
        confounds += dt_names
    
    confound_df = pd.read_csv(confound_tsv, delimiter='\t')
    # Verify if confound columns exist
    missing_cols = [col for col in confounds if col not in confound_df.columns]
    if missing_cols:
        raise ValueError(f'Missing confounds columns: {missing_cols}')
    confound_df = confound_df[confounds]
    # Ensure there are no entirely NaN columns
    if confound_df.isnull().all().any():
        raise ValueError('One or more confound columns are entirely NaN.')
    confound_mat = confound_df.values
    
    return confound_mat

def process_subject(sub, fmri_dir, fmri_prep, parcellation_path, output_dir, sessions, tasks):
    # Initialize the connectivity measure and masker outside the task loop if they don't depend on task-specific parameters
    try:
        # Attempt to load the parcellation image
        parcellation_img = nimg.load_img(parcellation_path)
    except Exception as e:
        print(f"Error loading parcellation image: {e}")
        return False  # Or handle the error as appropriate
    
    masker = NiftiLabelsMasker(labels_img=parcellation_img, standardize=True, detrend=False, smoothing_fwhm=6, high_pass=0.042, low_pass=0.125, t_r = 1.0)
    correlation_measure = ConnectivityMeasure(kind='correlation')
    
    for session in sessions:
        for task in tasks:
            func_files = glob(f'{fmri_dir}/{sub}/{session}/func/{sub}_{session}_task-{task}_run-1_bold.nii.gz')
            if not func_files:
                print(f'Skipping {sub}: Missing functional files for session {session}, task {task}.')
                continue
            func_file = func_files[0] 

            mask_file = f'{fmri_prep}/{sub}/{session}/func/{sub}_{session}_task-{task}_run-1_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz'
            if not os.path.exists(mask_file):
                print(f'Skipping {sub}: Missing mask file for session {session}, task {task}.')
                continue

            confound_file = f'{fmri_prep}/{sub}/{session}/func/{sub}_{session}_task-{task}_run-1_desc-confounds_timeseries.tsv'
            if not os.path.exists(confound_file):
                print(f'Skipping {sub}: Missing confound files for session {session}, task {task}.')
                continue

            raw_func_img = nimg.load_img(func_file)
            tr_drop = 4
            func_img = raw_func_img.slicer[:,:,:,tr_drop:]

            confound_vars = ['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z', 'global_signal', 'csf', 'white_matter']
            try:
                confounds = get_confounds(confound_file, confound_vars)
                confounds = confounds[tr_drop:, :]
            except ValueError as e:
                print(f'Error processing {sub} in session {session}, task {task}: {e}')
                continue

            time_series = masker.fit_transform(func_img, confounds=confounds)
            if time_series.size == 0:
                print(f'Error: Time series extraction failed for {sub} in session {session}, task {task}.')
                continue

            masker_labels = [int(label) for label in masker.labels_]
            valid_labels = [label for label in masker_labels if label-1 < len(node_numbers) and label > 0]

            # Map column and row indices to node numbers using valid labels
            node_labels = [node_numbers[label - 1] for label in valid_labels]

            functional_connectivity = correlation_measure.fit_transform([time_series])
            if functional_connectivity.size == 0:
                print(f'Error: Functional connectivity computation failed for {sub} in session {session}, task {task}.')
                continue
            fc_matrix = pd.DataFrame(functional_connectivity[0], columns=node_labels, index=node_labels)
            
            output_file = os.path.join(output_dir, f'{sub}_session-{session}_task-{task}.csv')
            fc_matrix.to_csv(output_file, index=True)
            print(f'Processed and saved {sub} for session {session}, task {task} successfully.')



# Main processing loop
if __name__ == "__main__":
    fmri_dir = "/oak/stanford/groups/russpold/data/uh2/aim1/BIDS"
    fmri_prep = "/oak/stanford/groups/russpold/data/uh2/aim1/derivatives/fmriprep"
    parcellation_path = "/home/groups/russpold/ahryan_home/shen_2mm_268_parcellation_adwarpMNI2009.nii"
    output_dir = "/oak/stanford/groups/russpold/users/ahryan/fc_data/matrices"
    mapping_file = "/home/groups/russpold/ahryan_home/shen268_coords.csv"
    node_numbers = load_node_numbers(mapping_file)
    subjects = [i.split('/')[-1] for i in glob(os.path.join(fmri_dir, 'sub-s*'))]
    sessions = ['ses-1','ses-2']
    tasks = ['CCTHot', 'stopSignal', 'twoByTwo', 'WATT3', 'discountFix', 'DPX', 'motorSelectiveStop', 'rest', 'stroop']
    for sub in subjects:
        process_subject(sub, fmri_dir, fmri_prep, parcellation_path, output_dir, sessions, tasks)