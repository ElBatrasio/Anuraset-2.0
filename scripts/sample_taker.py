import numpy as np
import pandas as pd

# replace with your actual column names
species_cols = ['score_species1', 'score_species2', 'score_species3']
n_bins = 10
samples_per_species = 300

validation_samples = {}

for col in species_cols:
    # Bin the scores
    df['score_bin'] = pd.cut(df[col], bins=np.linspace(
        df[col].min(), df[col].max(), n_bins+1), include_lowest=True)
    samples_per_bin = samples_per_species // n_bins
    samples = []
    for bin_value in df['score_bin'].unique():
        bin_df = df[df['score_bin'] == bin_value]
        n = min(samples_per_bin, len(bin_df))
        if n > 0:
            samples.append(bin_df.sample(n=n, random_state=42))
    validation_samples[col] = pd.concat(samples)
    df.drop(columns=['score_bin'], inplace=True)

# Each DataFrame in validation_samples contains 300 samples for that species, spanning the full score range.
