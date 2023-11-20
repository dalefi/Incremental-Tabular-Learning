import typing

import pandas
import pandas as pd
from google.cloud import bigquery

PROJECT_ID = "vf-de-ca-lab"


def query(sql_query: str, query_config: bigquery.QueryJobConfig = None) -> pd.DataFrame:
    """
    Sends an SQL query with a given configuration to Google BigQuery.
    :param sql_query: query in SQL
    :param query_config: query configuration containing e.g. parameters that shall be passed to the query
    :return: DataFrame containing the query result
    """
    client = bigquery.Client()
    df = client.query(sql_query, job_config=query_config).to_dataframe()
    return df


def car_query_timeframe(from_date, to_date, features: typing.List[str] = None):
    """
    Queries a given list (or all) features of the CAR dataset in a given timeframe.
    :param from_date: start of timeframe
    :param to_date: end of timeframe
    :param features: list of columns to select
    :return: DataFrame of the query result
    """
    sql = """
        SELECT {features}
        FROM vf-de-datahub.vfde_dh_lake_de_mob_datamart_s.car
        WHERE 
            CAST(adr_zip AS INT) > 0
            AND partition_year_month_day BETWEEN @from_date AND @to_date
        """
    if features is not None:
        sql = sql.replace("{features}", features_to_string(features))
    else:
        sql = sql.replace("{features}", "*")
    query_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("from_date", "DATE", from_date),
            bigquery.ScalarQueryParameter("to_date", "DATE", to_date)
        ]
    )
    return query(sql, query_config)


def car_query_timeframe_sample(from_date, to_date, limit: int) -> pd.DataFrame:
    """
    Queries a random sample of the CAR dataset in a given timeframe. Returns all columns.
    :param from_date: start of the timeframe
    :param to_date: end of the timeframe
    :param limit: number of tuples to return
    :return: DataFrame of a random CAR sample
    """
    assert limit > 0
    sql = """
        SELECT *
        FROM vf-de-datahub.vfde_dh_lake_de_mob_datamart_s.car
        WHERE 
            partition_year_month_day BETWEEN @from_date AND @to_date
            AND CAST(adr_zip AS INT) > 0
            AND RAND() < 0.25
        LIMIT @limit
        """
    query_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("from_date", "DATE", from_date),
            bigquery.ScalarQueryParameter("to_date", "DATE", to_date),
            bigquery.ScalarQueryParameter("limit", "INTEGER", limit)
        ]
    )
    return query(sql, query_config)


def nps_query_timeframe(from_date, to_date) -> pandas.DataFrame:
    """
    Queries unique (client_id, answer_value) tuples from the NPS table.
    In case of multiple answers per customer, the highest given answer_value is returned.
    :param from_date: start of timeframe
    :param to_date: end of timeframe
    :return: DataFrame containing client ids and their highest answer value
    """
    sql = """
        SELECT client_id, MAX(answer_value) AS answer_value
        FROM vf-de-datahub.vfde_dh_lake_dsl_customer_rawprepared_s.nps_cs_base
        WHERE
            touchpoint_new = "Customer Base (All)"
            AND question_name = "NPS"
            AND client_id IS NOT NULL
            AND answer_value IS NOT NULL
            AND contactdate BETWEEN @from_date AND @to_date
        GROUP BY client_id
        """
    query_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("from_date", "DATE", from_date),
            bigquery.ScalarQueryParameter("to_date", "DATE", to_date)
        ]
    )
    return query(sql, query_config)


def join_car_nps(from_date, to_date) -> pd.DataFrame:
    """
    Queries the distinct customers of the joint CAR and NPS table in a bounded timeframe.
    Returns all columns of the CAR table.
    :param from_date: start of timeframe
    :param to_date: end of timeframe
    :return: DataFrame containing unique client ids
    """
    sql = f"""
        SELECT 
            DISTINCT car.*
        FROM 
            vf-de-datahub.vfde_dh_lake_de_mob_datamart_s.car AS car, 
            vf-de-datahub.vfde_dh_lake_dsl_customer_rawprepared_s.nps_cs_base AS nps
        WHERE
            car.client_id = nps.client_id
            AND nps.touchpoint_new = "Customer Base (All)"
            AND nps.question_name = "NPS"
            AND nps.client_id IS NOT NULL
            AND nps.answer_value IS NOT NULL
            AND nps.contactdate BETWEEN @from_date AND @to_date
            AND CAST(car.adr_zip AS INT) > 0
            AND car.partition_year_month_day BETWEEN @from_date AND @to_date
        """
    query_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("from_date", "DATE", from_date),
            bigquery.ScalarQueryParameter("to_date", "DATE", to_date)
        ]
    )
    return query(sql, query_config)


def features_to_string(features: list) -> str:
    return ",".join(features)