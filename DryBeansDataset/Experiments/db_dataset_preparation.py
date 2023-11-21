"""
Transforms the DryBeansDatset into a format which can be used by the experiments.py script
"""

from pathlib import Path
import pandas as pd

data_folder = Path("../../../data/DryBeanDataset/")
file_to_open = data_folder / "Dry_Bean_Dataset.xlsx"

# load data
data = pd.read_excel(file_to_open)

# transform labels to [0, ..., 6]
labels_dict = {key:value for (value,key) in enumerate(data["Class"].unique())}
data["Class"] = data["Class"].map(labels_dict)

data.to_csv(data_folder / 'DryBeanDataset.csv', index=False)