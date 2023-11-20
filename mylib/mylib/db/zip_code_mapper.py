import pandas as pd

ZIP_CODES_PATH = "gs://vf-de-ca-lab-dev/notebooks/jupyter/home/christopher.orlowicz/data/zipcodes_de.csv"


class ZipCodeMapper:
    """Class for mapping zip codes to (longitude, latitude) pairs."""

    def __init__(self):
        """
        Reads the zip code table, keeps only unique zip codes and reduces the table to
        the three columns (zipcode, latitude, longitude).
        """
        zip_map_df = pd.read_csv(ZIP_CODES_PATH, dtype={"zipcode": str})
        # keep unique zip codes (some (smaller) towns can have the same zip code)
        zip_map_df.drop_duplicates(subset=["zipcode"], inplace=True)
        # keep relevant columns
        self.zip_code_map = zip_map_df[["zipcode", "longitude", "latitude"]].set_index("zipcode")

    def map_single_zip_code_to_coord(self, zip_code: str):
        """
        Maps a single zip code to a (longitude, latitude) tuple.
        :param zip_code: a valid zip code
        :return: latitude, longitude
        """
        long, lat = self.zip_code_map.loc[zip_code]
        return long, lat

    def map_zip_codes_to_coords(self, zip_codes: pd.Series) -> pd.DataFrame:
        """
        Maps a series of zip_codes to their corresponding (latitude, longitude) pairs.
        :param zip_codes: series of zip codes
        :return: a dataframe of the zip codes df with two additional columns lat and long
        """
        coords = pd.DataFrame(zip_codes, columns=["adr_zip"])
        coords[["long", "lat"]] = self.zip_code_map.loc[zip_codes].reset_index(drop=True)
        return coords