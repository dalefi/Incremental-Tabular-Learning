from csv import reader

import pandas as pd
import typing

import mylib.db.big_query as bq
from mylib.db import constants
from mylib.db.zip_code_mapper import ZipCodeMapper

DATA_PATH = "vf-de-datahub.vfde_dh_lake_de_mob_datamart_s.car"


class CarPreprocessingException(Exception):

    def __init__(self, msg):
        super(CarPreprocessingException, self).__init__(msg)


class Preprocessor:
    """
    Class for preprocessing the CAR dataset. Reads the data and removes categorical,
    non-numerical and other unsuitable columns.
    Adapted from
    https://github.vodafone.com/VFDE-CloudAnalytics/helios/blob/develop/core/modeloperations/preprocessing/car.py.
    """

    def __init__(self,
                 from_date,
                 to_date,
                 limit: int = None,
                 features: typing.List[str] = constants.RELEVANT_COLUMNS, 
                 data: pd.DataFrame = None,
                 remove_cat_columns = True,
                 normalization = True,
                 verbose=True):
        """
        Reads the CAR dataset (if not provided by the param 'data') and removes categorical,
        non-numerical and manually selected variables.
        Imputes missing values using the methods specified in [src/db/resources/imputation_dictionary.csv].
        Converts all columns to float data type. Finally, normalizes all columns (zero-mean and std. dev. of 1).
        :param from_date: start of queried timeframe
        :param to_date: end of queried timeframe
        :param limit: maximum number of rows in the query result (only if data is None)
        :param features: (optional) list of features to select
        :param data: DataFrame of the CAR dataset to be preprocessed
                (if None, the specified timeframe will be queried)
        :param remove_cat_columns: whether to remove categorical columns
        :param normalization: whether to normalize the data
        :param verbose: whether to write the processing steps to the standard output
        """
        if data is not None:
            self.car_df = data
        else:
            print("Reading CAR dataset...") if verbose else None
            if limit is not None:
                assert limit > 0
                self.car_df = bq.car_query_timeframe_sample(from_date, to_date, limit)
            else:
                self.car_df = bq.car_query_timeframe(from_date, to_date, features)

        self.client_ids = self.car_df["client_id"]
        self.adr_zips = self.car_df["adr_zip"]
        
        if remove_cat_columns:
            # remove categorical columns
            print("\nRemoving categorical columns...") if verbose else None
            self.remove_columns(constants.CATEGORICAL_COLUMNS, verbose)
        
        # remove unsuitable columns
        print("\nRemoving unsuitable columns...") if verbose else None
        self.remove_columns(constants.VARIABLES_TO_EXCLUDE, verbose)

        # remove protected columns
        print("\nRemoving protected columns...") if verbose else None
        self.remove_columns(constants.PROTECTED_COLUMNS, verbose)

        # remove object columns
        print("\nRemoving non-numeric columns...") if verbose else None
        object_cols = self.car_df.select_dtypes('object').columns.tolist()
        self.remove_columns(object_cols, verbose)
        
        # impute missing data
        print("\nImputing missing data...") if verbose else None
        self._impute_missing_values(ignore_missing_columns=True, ignore_unknown_columns=False)

        print("\nCleaning zip codes...") if verbose else None
        self._clean_adr_zips()

        # convert all columns to numeric type
        print("\nConverting object columns to numeric type...") if verbose else None
        self._convert_obj_cols_to_num()
        
        if normalization:
            # normalize the data
            print("\nNormalizing features...") if verbose else None
            self.car_df = self.normalize()
            
        print("Preprocessing successful.")

    def remove_columns(self, columns: list, verbose=True):
        """
        Removes columns of the internal dataframe inplace.
        :param columns: columns to remove
        :param verbose: whether to write to standard output which column was removed
        :return:
        """
        for column in columns:
            if column in self.car_df.columns:
                print(f"Removing column '{column}'") if verbose else None
                self.car_df.drop(column, axis=1, inplace=True)

    def _clean_adr_zips(self):
        """
        Performs preprocessing on the zip codes. Pads too short zip codes with zeros from the left
        and removes unknown (i.e., unmappable with ZipCodeMapper) zip codes.
        """
        # pad zip codes length is less than 5
        self.adr_zips = self.pad_zip_codes_with_zeros()

        # load mapper for zip_code -> (longitude, latitude)
        zip_mapper = ZipCodeMapper()
        # load zip codes of customers into a DataFrame to apply isin() function
        adr_zip_df = pd.DataFrame(self.adr_zips, dtype=str)
        # remove unknown (unmappable) zip codes
        known_zips = adr_zip_df.adr_zip.isin(zip_mapper.zip_code_map.index)
        # apply mask to all three Dataframes
        self.adr_zips = self.adr_zips[known_zips].reset_index(drop=True)
        self.car_df = self.car_df[known_zips].reset_index(drop=True)
        self.client_ids = self.client_ids[known_zips].reset_index(drop=True)

    def pad_zip_codes_with_zeros(self) -> pd.Series:
        """
        Fills invalid zip codes (i.e., with less than 5 digits) with zeros from the left.
        :return: padded zip codes
        """
        padded_zip_codes = self.adr_zips.copy()
        invalid_zip_codes = ~self.validate_zip_codes()
        # prepend leading zeros
        padded_zip_codes[invalid_zip_codes] = padded_zip_codes[invalid_zip_codes].apply(
            lambda zip_code: zip_code.zfill(5)
        )
        return padded_zip_codes

    def validate_zip_codes(self) -> pd.Series:
        """
        Checks if the length of each zip code equals exactly five.
        :return: boolean array
        """
        return self.adr_zips.apply(len) == 5

    def _impute_missing_values(self, ignore_missing_columns=False, ignore_unknown_columns=False):
        """
        Imputes missing values of a dataframe using an imputation dictionary.
        :param ignore_missing_columns: whether to ignore columns that are in the dict but not in the Dataframe
        :param ignore_unknown_columns: whether to ignore columns that are in the Dataframe but not in the dict
        """
        imputation_dict = self._read_imputation_dict()
        self._impute_values(imputation_dict, ignore_missing_columns, ignore_unknown_columns)

    def _impute_values(self, imputation_dict, ignore_missing_columns=False, ignore_unknown_columns=False):
        """
        Returns a new CAR pandas Dataframe with imputed values.
        :param imputation_dict: imputation dictionary
        :param ignore_missing_columns: if True no exception is raised when the provided dataframe misses columns
        :param ignore_unknown_columns: if True no exception is raised when the provided dataframe has columns
        for which no imputations are defined.
        :return: Dataframe with imputed values
        """
        if type(self.car_df) is not pd.DataFrame:
            raise CarPreprocessingException("Only pandas DataFrames are supported.")

        known_columns = imputation_dict.keys()
        columns = set(self.car_df.columns) & set(known_columns)
        missing_columns = set(known_columns) - set(self.car_df.columns)
        unknown_columns = set(self.car_df.columns) - set(known_columns)

        if len(missing_columns) > 0 and not ignore_missing_columns:
            raise CarPreprocessingException(
                "Warning: provided dataframe is missing columns: {}".format(missing_columns))
        if len(unknown_columns) > 0 and not ignore_unknown_columns:
            raise CarPreprocessingException(
                "Warning: provided dataframe has unknown columns: {}".format(unknown_columns))

        for column in columns:
            self._impute_value(column, imputation_dict)

    def _impute_value(self, column_name, imputation_dict):
        """
        Imputes the default value for a particular column in a dataframe. Returns a new CAR Dataframe
        with the imputed values.
        :param column_name: column name
        :param imputation_dict: imputation dictionary of functions to apply to each column
        """
        if type(self.car_df) is not pd.DataFrame:
            raise CarPreprocessingException("Only Pandas DataFrames are supported.")
        if column_name not in imputation_dict.keys():
            raise CarPreprocessingException("No imputations defined for column '{}'.".format(column_name))
        elif column_name not in self.car_df.columns:
            raise CarPreprocessingException("Column name '{}' not found in provided dataframe.".format(column_name))

        f = imputation_dict[column_name]
        f(column_name)

    def _read_imputation_dict(self):
        """Returns the imputation dictionary. For each column (keys) it stores the
        corresponding imputation method."""
        imputation_dict = {}
        imputation_file = self._get_dict_path()
        
        with open(imputation_file, 'r') as file:
            csv_file = reader(file, delimiter=';')
            # skip header
            next(csv_file)
            for row in csv_file:
                column, dtype, value = row

                if dtype != "string" and not _is_numeric(value) and value not in ["median", "mean", "median<0"]:
                    raise CarPreprocessingException(
                        "Encountered a non-numerical value for a numerical field '{}'.".format(column))

                if value == "median":
                    imputation_dict[column] = self._median
                elif value == "median<0":
                    imputation_dict[column] = self._replace_less_than_with_median
                else:
                    imputation_dict[column] = self._get_fillna(value)
        return imputation_dict

    @staticmethod
    def _get_dict_path():
        """Returns the path to the imputation dictionary."""
        return "home/dataproc/Students/2023_Fischer_Incremental_Tabular_Learning/mylib/db/resources/imputation_dictionary.csv"

    def _median(self, name):
        """
        Replaces NaN values with approximation of Median (inplace).
        :param name: column name
        :return: column with filled in default value
        """
        col = self.car_df[name]
        col.fillna(col.median(), inplace=True)

    def _mean(self, name):
        """
        Replaces NaN values with mean of a column (inplace).
        :param name: column name
        :return: column with filled in default value
        """
        col = self.car_df[name]
        col.fillna(col.mean(), inplace=True)

    def _get_fillna(self, value):
        """
        Returns a function that fills NaN with 'value'.
        :param value: fill value
        :return: imputation function that expects a dataframe and a column name
        """
        return lambda name: self.car_df[name].fillna(value, inplace=True)

    def _replace_less_than_with_median(self, name):
        """
        Replaces all values less than zero with the median of the positive values (inplace).
        :param name: column name
        :return: column with filled in default value
        """
        col = self.car_df[name]
        negative_values = col[col < 0]
        median_positive = col[col >= 0].median()
        return col.replace(negative_values.unique(), median_positive, inplace=True)

    def _convert_obj_cols_to_num(self):
        """Converts all object columns to float type."""
        # convert all columns to numeric type
        obj_cols = self.car_df.select_dtypes('object').columns.tolist()
        self.car_df[obj_cols] = self.car_df[obj_cols].astype(float)

    def normalize(self):
        """
        Normalizes (i.e., zero-centers & scales) a column or whole dataframe.
        :return: normalized DataFrame
        """
        return (self.car_df - self.car_df.mean(axis=0)) / (self.car_df.std(axis=0) + 1e-8)


def _is_numeric(value):
    """
    Checks whether a given value is numeric.
    :param value: some value
    :return: True if the value is numeric
    """
    try:
        float(value)
        return True
    except ValueError:
        return False