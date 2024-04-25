# PoldrackLab_CBPM
This code is intended to perform connectome-based predictive modeling on a dataset of fMRI scans collected while participants performed 8 tasks: the stop signal task (Logan & Cowan, 1984), the conditional motor selective stop-signal task (De Jong, Coles, & Logan, 1995), the Stroop task (Stroop, 1935), the Dot Pattern Expectancy task (DPX; Jones et al. 2010), the Attention Network Task (ANT; Fan et al. 2002), the Columbia Card Task (hot version) (CCTHot; Figner et al. 2009), cued task switching (Meiran 1996), the Kirby Delay Discounting task (Kirby & Maraković, 1996), the Delay Discounting Titration, and the Tower of Hanoi. 
To perform this analysis, first functional connectivity must be performed on preprocessed scans. 
The functional connectivity in this repository uses the Shen k=268 atlas. 
Next, the functional connectivity matrices can be inputted to the code for connectome-based predictive modeling.
<br>
<br>
**Directory Contents**
<br>
The code to perform functional connectivity on preprocessed scans: fc-withglob.py
<br>
The atlas label file: shen_2mm_268_parcellation_adwarpMNI2009.nii.gz
<br>
The information about the parcels: shen268_coords.csv
<br>
The code to perform connectome-based predictive modeling: CBPM_tutorial.ipynb
<br>
*The code for cbpm_tutorial.ipynb was adapted from [cpm_tutorial](https://github.com/esfinn/cpm_tutorial), which was prepared for the Georgetown Methods Lab.*
