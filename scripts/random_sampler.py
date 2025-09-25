# %% Load required libraries
import numpy as np
import pandas as pd
import os

os.chdir(r'C:\Users\gbida\Projects\Frog-party_detector')
cwd = os.getcwd()
print("Current working directory:", cwd)
# %% Load your predictions CSV
df = pd.read_csv(
    r'C:\Users\gbida\Projects\anurabird\results\predictions_with_timestamps_and_scores_20250912_142843.csv')
# %% Sample the data frame
samp_df = df.sample(n=300, random_state=42)
samp_df
# %% Locate score and non score columns
selec_spec = samp_df[['score_BOAFAB', 'score_BOAALM', 'score_LEPNOT']]
rest_columns = samp_df.loc[:, ~df.columns.str.contains('score')]
# %% Merge subsets and export to csv
val_df = pd.concat([rest_columns, selec_spec], axis=1)
val_df
val_df.to_csv('results/Validation_subset.csv')
# %%
