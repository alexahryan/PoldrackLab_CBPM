import numpy as np
import pandas as pd
import os
from glob import glob
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

# Directory paths
top_dir = "path/to/directory"
data_dir = "path/to/directory"
output_dir = "path/to/directory"
exclusion_data_path = "path/to/file"
behavioral_data_path = "path/to/file"

# Load data
exclusion_data = pd.read_csv(exclusion_data_path)
exclusion_data.set_index('Subject', inplace=True)
all_behav_data = pd.read_csv(behavioral_data_path, dtype={'ParticipantID': str})
all_behav_data.set_index('ParticipantID', inplace=True)

subj_list = [i.split('/')[-1] for i in glob(os.path.join(fmri_dir, 'sub-s*'))]

def clean_data(data_frame):
    """Clean data by replacing infinities with NaNs and dropping rows with any NaNs."""
    data_frame = data_frame.replace([np.inf, -np.inf], np.nan)
    cleaned_data = data_frame.dropna()
    return cleaned_data


def read_in_matrices(subj_list, condition, data_dir, exclusion_data, zscore=False):
    """Read and preprocess functional connectivity data, excluding subjects as necessary."""
    valid_subj_list = []
    subjects_to_remove = []
    all_fc_data = {}

    for subj in subj_list:
        key = '{}_{}'.format(subj, condition)
        if key not in exclusion_data.index:
            valid_subj_list.append(subj)

    for subj_id in valid_subj_list:
        session_data_found = False
        for session_name in ['ses-1', 'ses-2']:
            pattern = "{}_session-{}_task-{}.csv".format(subj_id, session_name, condition)
            file_path = os.path.join(data_dir, pattern)
            if os.path.exists(file_path):
                session_data_found = True
                df = pd.read_csv(file_path, delimiter=',', index_col=0)
                df = df.apply(pd.to_numeric, errors='coerce')
                # Clean the data right after reading it
                df = clean_data(df)
                all_fc_data[subj_id] = df.values.flatten()
                break
        if not session_data_found:
            subjects_to_remove.append(subj_id)

    for subj_id in subjects_to_remove:
        valid_subj_list.remove(subj_id)
        print("Subject {} removed due to missing session data.".format(subj_id))

    all_fc_data_df = pd.DataFrame.from_dict(all_fc_data, orient='index')
    return all_fc_data_df, valid_subj_list

def filter_behavioral_data(all_behav_data, valid_subj_list):
    """Filter behavioral data to include only entries for valid subjects."""
    valid_subj_list = [sub for sub in valid_subj_list if sub in all_behav_data.index]
    return all_behav_data.loc[valid_subj_list]

def mk_kfold_indices(valid_subj_list, k):
    """Create indices for K-fold cross-validation."""
    n_subs = len(valid_subj_list)
    n_subs_per_fold = n_subs // k
    indices = [[fold_no] * n_subs_per_fold for fold_no in range(k)]
    remainder = n_subs % k
    remainder_inds = list(range(remainder))
    indices = [item for sublist in indices for item in sublist]
    indices.extend(remainder_inds)
    np.random.shuffle(indices)
    return np.array(indices)

def split_train_test(valid_subj_list, indices, test_fold):
    """Split subjects into training and testing groups based on fold indices."""
    train_subs = [sub for i, sub in enumerate(valid_subj_list) if indices[i] != test_fold]
    test_subs = [sub for i, sub in enumerate(valid_subj_list) if indices[i] == test_fold]
    if not train_subs:
        print("Training subset is empty. Check index handling and fold assignment.")
    if not test_subs:
        print("Test subset is empty. Check index handling and fold assignment.")
    return train_subs, test_subs

def get_train_test_data(all_fc_data, train_subs, test_subs, behav_data, behav):
    """Retrieve train and test data sets for functional connectivity and behavioral measures."""
    if behav not in behav_data.columns:
        raise ValueError("The behavior measure '{}' does not exist in the provided behavioral data.".format(behav))
    train_vcts = all_fc_data.loc[train_subs, :]
    test_vcts = all_fc_data.loc[test_subs, :]
    train_behav = behav_data.loc[train_subs, behav]
    return (train_vcts, train_behav, test_vcts)

def select_features(train_vcts, train_behav, r_thresh=0.2, corr_type='pearson', verbose=False):
    """Select features based on correlation with behavior above a threshold."""
    assert train_vcts.index.equals(train_behav.index), "Row indices of FC vcts and behavior don't match!"
    if train_vcts.size == 0 or train_behav.size == 0:
        raise ValueError("Input vectors for correlation are empty. Check data preprocessing and splitting steps.")
    mask_dict = {'pos': [], 'neg': []}
    if corr_type == 'pearson':
        corr = np.corrcoef(train_vcts.T, train_behav)[0:len(train_vcts.T), -1]
        mask_dict['pos'] = corr > r_thresh
        mask_dict['neg'] = corr < -r_thresh
    if verbose:
        print("Positive mask count: {}, Negative mask count: {}".format(np.sum(mask_dict['pos']), np.sum(mask_dict['neg'])))
    return mask_dict

def build_model(train_vcts, mask_dict, train_behav):
    """Build predictive models from training data using the selected features."""
    assert train_vcts.index.equals(train_behav.index), "Row indices of FC vcts and behavior don't match!"

    model_dict = {}
    X_glm = np.zeros((train_vcts.shape[0], len(mask_dict.items())))

    t = 0
    for tail, mask in mask_dict.items():
        X = train_vcts.values[:, mask].sum(axis=1)
        X_glm[:, t] = X
        y = train_behav
        # Skip model building if any NaNs are detected in X or y
        if np.any(np.isnan(X)) or np.any(np.isnan(y)):
            print(f"Skipping model for {tail} due to NaN values.")
            continue
        (slope, intercept) = np.polyfit(X, y, 1)
        model_dict[tail] = (slope, intercept)
        t += 1

    X_glm = np.c_[X_glm, np.ones(X_glm.shape[0])]  # Add intercept for global model
    # Check for NaNs in the design matrix before fitting the global model
    if np.isnan(X_glm).any():
        print("Skipping GLM due to NaN values in design matrix.")
    else:
        model_dict["glm"] = tuple(np.linalg.lstsq(X_glm, y, rcond=None)[0])

    return model_dict


def apply_model(test_vcts, mask_dict, model_dict):
    """Apply built models to test data to generate behavioral predictions."""
    behav_pred = {}
    X_glm = np.zeros((test_vcts.shape[0], len(mask_dict.items())))
    t = 0
    for tail, mask in mask_dict.items():
        X = test_vcts.loc[:, mask].sum(axis=1)
        X_glm[:, t] = X
        slope, intercept = model_dict[tail]
        behav_pred[tail] = slope * X + intercept
        t += 1
    X_glm = np.c_[X_glm, np.ones(X_glm.shape[0])]
    behav_pred["glm"] = np.dot(X_glm, model_dict["glm"])
    return behav_pred

def cpm_wrapper(all_fc_data, all_behav_data, valid_subj_list, behav, k, **cpm_kwargs):
    """Wrapper function to apply Connectome-based Predictive Modeling over multiple folds."""
    behav = "{}_{}".format(task, measure)
    valid_subj_list = all_fc_data.index
    filtered_behav_data = all_behav_data.loc[valid_subj_list]
    assert all_fc_data.index.equals(filtered_behav_data.index), "Row (subject) indices of FC vcts and behavior don't match!"
    indices = mk_kfold_indices(valid_subj_list, k=k)
    col_list = ["{} predicted ({})".format(behav, tail) for tail in ["pos", "neg", "glm"]]
    col_list.append("{} observed".format(behav))
    behav_obs_pred = pd.DataFrame(index=valid_subj_list, columns=col_list)
    n_edges = all_fc_data.shape[1]
    all_masks = {"pos": np.zeros((k, n_edges)), "neg": np.zeros((k, n_edges))}
    for fold in range(k):
        print("doing fold {}".format(fold))
        train_subs, test_subs = split_train_test(valid_subj_list, indices, test_fold=fold)
        train_vcts, train_behav, test_vcts = get_train_test_data(all_fc_data, train_subs, test_subs, filtered_behav_data, behav=behav)
        mask_dict = select_features(train_vcts, train_behav, **cpm_kwargs)
        all_masks["pos"][fold, :] = mask_dict["pos"]
        all_masks["neg"][fold, :] = mask_dict["neg"]
        model_dict = build_model(train_vcts, mask_dict, train_behav)
        behav_pred = apply_model(test_vcts, mask_dict, model_dict)
        for tail, predictions in behav_pred.items():
            behav_obs_pred.loc[test_subs, "{} predicted ({})".format(behav, tail)] = predictions
    behav_obs_pred.loc[valid_subj_list, "{} observed".format(behav)] = filtered_behav_data[behav]
    return behav_obs_pred, all_masks

tasks = ['CCTHot', 'stopSignal', 'twoByTwo', 'discountFix', 'DPX', 'motorSelectiveStop', 'stroop', 'WATT3']
k = 100
cpm_kwargs = {'r_thresh': 0.2, 'corr_type': 'pearson'}

for task in tasks:
    measures = [col.split('_')[1] for col in all_behav_data.columns if task in col]
    for measure in measures:
        print("Processing task: {}, Measure: {}".format(task, measure))
        all_fc_data, valid_subj_list = read_in_matrices(subj_list, task, data_dir, exclusion_data)
        if not valid_subj_list:
            print("No valid subjects for task {}, measure {}. Skipping...".format(task, measure))
        else:
            print("Valid subjects for task {}, measure {}: {}".format(task, measure, valid_subj_list))
            filtered_behav_data = filter_behavioral_data(all_behav_data, valid_subj_list)
            behav_obs_pred, all_masks = cpm_wrapper(all_fc_data, filtered_behav_data, valid_subj_list, measure, k, **cpm_kwargs)
            filtered_filename = 'filtered_behavioral_data_{}_{}_k{}.csv'.format(task, measure, k)
            predictions_filename = 'behavior_observations_predictions_{}_{}_k{}.csv'.format(task, measure, k)
            filtered_behav_data.to_csv(os.path.join(output_dir, filtered_filename))
            behav_obs_pred.to_csv(os.path.join(output_dir, predictions_filename))
            print("Data saved for task {}, measure {}".format(task, measure))
