import numpy as np
from numpy.polynomial import Polynomial as poly
import pandas as pd
import glob
import os
from scipy.stats import t

#
FC_dir = "../output/functional_connectivity"
behavior_dir = "../output/behavior"
taskName = 'discountFix'
#Threshold for feature selection:
thresh = 0.01

FC_dir = os.path.join(FC_dir, '*')
behavior_dir = os.path.join(behavior_dir, '*')

# Get connectivity matrices and behavior vectors csv file paths
conn_mats = sorted(glob.glob(os.path.join(FC_dir,f'*{taskName}*.csv*')))
behav_vects = sorted(glob.glob(os.path.join(behavior_dir,'*behavior*.csv*')))

file_names = [os.path.basename(path) for path in conn_mats]
subject_names = [string.split("sub-")[1].split("_")[0] for string in file_names]

conn_mats = [pd.read_csv(f, delimiter=',', header = None) for f in conn_mats]
behav_vects = [pd.read_csv(f, delimiter=',', usecols = [taskName]) for f in behav_vects]


#INPUTS:
all_mats = np.stack(conn_mats)
all_mats = all_mats.transpose(2,1,0)
all_behav = np.stack(behav_vects)
all_behav = all_behav.squeeze()

shape = all_mats.shape
num_sub = shape[2]
num_node = shape[0]

behav_pred_pos = np.zeros(num_sub)
behav_pred_neg = np.zeros(num_sub)

def pair_corrcoef(X,y):
    # find mean across subjects per node (row)
    Xm = np.reshape(np.mean(X,axis=1),(X.shape[0],1))
    ym = np.reshape(np.mean(y,axis=1),(y.shape[0],1))
    r_num = np.sum((X-Xm)*(y-ym),axis=1)
    r_den = np.sqrt(np.sum((X-Xm)**2,axis=1)*np.sum((y-ym)**2, axis=1))
    r = np.true_divide(r_num, r_den, out=np.zeros_like(r_num), where= r_den!=0)
    for val in r:
        if val > 1 or val < -1:
            print('WARNING: Correlation coefficients should be between -1 and 1')
    return r

def calculate_p(num_sub, r_vector, alpha = 0.05):
    t_numerator = (r_vector * np.sqrt(num_sub-2))
    t_denominator = np.sqrt(1 - r_vector**2)
    t_val = np.true_divide(t_numerator,t_denominator)
    deg_freedom = (num_sub-2)
    p_vector = 2 * (1.0 - t.cdf(abs(t_val), deg_freedom))
    return p_vector

num_train_sub = num_sub - 1
for leftout in range(num_sub):
    print(f"Leaving out subject {subject_names[leftout]}")

    # leave out subject from matrices and behavior

    train_mats = all_mats
    train_mats = np.delete(train_mats, leftout, 2)
    train_vcts = train_mats.reshape(num_node**2, num_train_sub)
    # train_vcts = np.transpose(train_vcts)

    train_behav = all_behav
    train_behav = np.delete(train_behav, leftout)

    # create copies of 1st row into num_node^2 rows to match r matrix dimensions
    train_behav_expanded = np.repeat(train_behav[:, np.newaxis], num_node**2, axis=1)
    train_behav_expanded = np.transpose(train_behav_expanded)

    # correlate edges with behavior
    r_vector = pair_corrcoef(train_vcts, train_behav_expanded)
    p_vector = calculate_p(num_train_sub, r_vector)

    # define masks
    pos_mask = np.logical_and(r_vector > 0, p_vector < thresh)
    neg_mask = np.logical_and(r_vector < 0, p_vector < thresh)

    # get sum of all edges in TRAIN subs (divide by 2 to control for
    # assymetric matrices)

    train_sumpos = np.zeros(num_train_sub)
    train_sumneg = np.zeros(num_train_sub)

    for ss in range(num_train_sub):
        train_sumpos[ss] = sum(train_vcts[:,ss] * pos_mask)/2
        train_sumneg[ss] = sum(train_vcts[:,ss] * neg_mask)/2

    #build model on TRAIN subs

    # numpy poly.fit.coef returns the coefficients in backwards order:
    # 2.499350570324139 + 0.5006460643242153 x**1
    fit_pos = poly.fit(train_sumpos, train_behav, 1).coef
    fit_neg = poly.fit(train_sumneg, train_behav, 1).coef
    
    # run model on TEST sub
    test_mat = all_mats[:,:,leftout]
    test_vct = test_mat.reshape(num_node**2)
    test_sumpos = sum(test_vct * pos_mask)/2
    test_sumneg = sum(test_vct * neg_mask)/2
    
    behav_pred_pos[leftout] = fit_pos[1] * test_sumpos + fit_pos[0]
    behav_pred_neg[leftout] = fit_neg[1] * test_sumneg + fit_neg[0]

#compare predicted and observed scores
print(all_behav)
print(behav_pred_pos)
print(behav_pred_neg)
R_pos = np.corrcoef(behav_pred_pos, all_behav)
R_neg = np.corrcoef(behav_pred_neg, all_behav)

P_pos = calculate_p(num_sub, R_pos)
P_neg = calculate_p(num_sub, R_neg)

print(f"{R_pos=} \n{P_pos=} \n{R_neg=} \n{P_neg=}")



