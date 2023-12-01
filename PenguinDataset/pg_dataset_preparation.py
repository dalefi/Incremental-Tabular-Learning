from pathlib import Path
import pandas as pd

from mylib import helper_funcs
from mylib import class_distributions

data_folder = Path("../../data/PenguinDataset/")
file_to_open = data_folder / "penguins.csv"

data = pd.read_csv(file_to_open)

data.drop(['rowid'], axis=1, inplace=True)
data.drop(['year'], axis=1, inplace=True)
nan_idx = data[data.isna().any(axis=1)].index
data.drop(nan_idx, inplace=True)

data = helper_funcs.create_numbered_categories(data, 'sex')
data = helper_funcs.create_numbered_categories(data, 'island')
data = helper_funcs.create_numbered_categories(data, 'species')
data = data.rename({'species': 'Class'}, axis=1)

data.to_csv(data_folder / 'PenguinDataset.csv', index=False)