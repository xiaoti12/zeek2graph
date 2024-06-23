from zat.log_to_dataframe import LogToDataFrame
from pandas import DataFrame
import pandas as pd
import os
from os import path
from typing import List, Dict
import numpy as np
from utils import get_node_attribute
import json


BLACK_LABEL = 1
WHITE_LABEL = 0

node_info_file = path.join("raw", "node_info.json")


class Extractor:
    def __init__(self, log_path: str):
        self.log_path = log_path

        self.df: DataFrame = self.log2df()
        self.graph_id = self.get_current_graph_id()

        self.node_infos: List[Dict] = []
        self.host2node = dict()
        self.edges: np.ndarray = None


    def get_current_graph_id(self) -> int:
        # create file if not exist
        if not os.path.exists(node_info_file):
            with open(node_info_file, "w") as f:
                pass
            return 0
        with open(node_info_file, "r") as f:
            f.seek(0, os.SEEK_END)
            if f.tell() == 0:
                return 0
        data = self.load_node_infos()
        return data[-1]["graph_id"] + 1

    def log2df(self) -> DataFrame:
        if not os.path.exists(self.log_path):
            print("File not found")
            return None
        log_reader = LogToDataFrame()
        df = log_reader.create_dataframe(self.log_path, ts_index=False, aggressive_category=False)
        # delete ts and duration column
        df.replace([pd.NA, pd.NaT, np.nan], 0, inplace=True)
        df.infer_objects(copy=False)
        return df

    def add_edge(self, host1: str, host2: str, node_id: int):
        if host1 not in self.host2node:
            self.host2node[host1] = [node_id]
        else:
            self.host2node[host1].append(node_id)

        if host2 not in self.host2node:
            self.host2node[host2] = [node_id]
        else:
            self.host2node[host2].append(node_id)

        for node in self.host2node[host1]:
            self.edges[node][node_id] = 1
            self.edges[node_id][node] = 1
        for node in self.host2node[host2]:
            self.edges[node][node_id] = 1
            self.edges[node_id][node] = 1

    # 更新node_infos和edges邻接矩阵
    def extract(self):
        if self.df is None:
            print("data not found")
            return
        df = self.df
        graph_id = self.graph_id
        node_total = df.shape[0]
        self.edges = np.zeros((node_total, node_total), dtype=int)
        for node_id, row in df.iterrows():
            # 每一行为一条流，代表一个节点
            node_info = dict()
            node_info["graph_id"] = graph_id
            node_info["node_id"] = node_id
            node_info["flow_uid"] = row["uid"]
            node_info["attribute"] = get_node_attribute(row)
            node_info["label"] = BLACK_LABEL

            host1 = row["id.orig_h"]
            host2 = row["id.resp_h"]
            self.add_edge(host1, host2, node_id)

            self.node_infos.append(node_info)

        self.save()
    
    def save_node_infos(self):
        pre_data = self.load_node_infos()
        with open(node_info_file, "w") as f:
            json.dump(pre_data + self.node_infos, f)

    def load_node_infos(self) -> List[Dict]:
        with open(node_info_file, "r") as f:
            data = json.load(f)
        return data

    def save_edges(self):
        edges_file = path.join("raw", f"edges_{self.graph_id}.npy")
        np.save(edges_file, self.edges)

    def save(self):
        self.save_node_infos()
        self.save_edges()

    