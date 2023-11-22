from pathlib import Path
import pandas as pd



# open file

data_folder = Path("../../../data/ForestCoverDataset/")
file_to_open = data_folder / "covtype.data"


# read and prepare data

data = pd.read_csv(file_to_open, delimiter=",", header=None)

# add the header manually
header = {0: "Elevation", 1: "Aspect", 2: "Slope", 3: "Horizontal_Distance_To_Hydrology",
          4: "Vertical_Distance_To_Hydrology", 5: "Horizontal_Distance_To_Roadways",
          6: "Hillshade_9am", 7: "Hillshade_Noon", 8: "Hillshade_3pm", 9: "Horizontal_Distance_To_Fire_Points"}

# add the names of binary columns
for i in range(1, 5):
    header[9+i] = f"Wilderness_Area_{i}"

for i in range(1, 41):
    header[13+i] = f"Soil_Type_{i}"

header[54] = "Class"

#data = data.drop(range(10,54), axis=1)

data = data.rename(header, axis=1)
data["Class"] = data["Class"] - 1   # want 0-based index
data.to_csv(data_folder / 'ForestCoverDataset.csv', index=False)