import pandas as pd
from pandas import DataFrame
import numpy as np
from typing import List
from Constants import *



class Combiner:
    def __init__(self, malware_df: DataFrame, normal_df: DataFrame):
        self.malware = malware_df
        self.normal = normal_df
        self.malware_list = self.split_df_by_hosts(malware_df)

    def replace_malware_hosts(self) -> None:
        normal_hosts = self.normal[COLUMN.HOST].unique()

        for malware_df in self.malware_list:
            replacement = {}

            for host in malware_df[COLUMN.HOST].unique():
                replacement[host] = np.random.choice(normal_hosts)
            malware_df[COLUMN.HOST] = malware_df[COLUMN.HOST].map(replacement)

    def change_malware_ts(self) -> None:
        base_timestamp = np.random.choice(self.normal[COLUMN.TIMESTAMP])
        for malware_df in self.malware_list:
            time_diffs = malware_df[COLUMN.TIMESTAMP] - malware_df[COLUMN.TIMESTAMP].iloc[0]

            malware_df[COLUMN.TIMESTAMP] = base_timestamp + time_diffs

    def get_combined(self) -> DataFrame:
        self.replace_malware_hosts()
        self.change_malware_ts()

        df_combined = pd.concat(self.malware_list + [self.normal])
        df_combined.sort_values(by=COLUMN.TIMESTAMP, inplace=True)
        df_combined.reset_index(drop=True, inplace=True)
        return df_combined

    def split_df_by_hosts(self, df: DataFrame) -> List[DataFrame]:
        grouped = df.groupby(["id.orig_h", "id.resp_h"])
        return [group for _, group in grouped]
