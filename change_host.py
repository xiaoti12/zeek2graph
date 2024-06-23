import pandas as pd
from pandas import DataFrame
import numpy as np
from Extractor import Extractor
from typing import List

host_column = "id.orig_h"
ts_column = "ts"


def replace_malware_hosts(malware_df: DataFrame, normal_df: DataFrame) -> None:
    normal_hosts = normal_df[host_column].unique()
    replacement = {}

    for host in malware_df[host_column].unique():
        replacement[host] = np.random.choice(normal_hosts)
    malware_df[host_column] = malware_df[host_column].map(replacement)


def change_malware_ts(malware_df: DataFrame, normal_df: DataFrame) -> None:
    base_timestamp = np.random.choice(normal_df[ts_column])
    time_diffs = malware_df[ts_column] - malware_df[ts_column].iloc[0]

    malware_df[ts_column] = base_timestamp + time_diffs


def combine(malware_df: DataFrame, normal_df: DataFrame) -> DataFrame:
    df_combined = pd.concat([malware_df, normal_df])
    df_combined.sort_values(by=ts_column, inplace=True)
    df_combined.reset_index(drop=True, inplace=True)
    return df_combined


if __name__ == "__main__":
    malware_df=Extractor.log2df("tls.log")
    normal_df=Extractor.log2df("campus1.log")

    replace_malware_hosts(malware_df, normal_df)
    change_malware_ts(malware_df, normal_df)

    df_combined = combine(malware_df, normal_df)
    df_combined.to_csv("combined.csv", index=False)
