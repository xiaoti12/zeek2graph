from zat.log_to_dataframe import LogToDataFrame
from pandas import DataFrame, Series
import pandas as pd
import os
from os import path
from typing import List, Dict
import numpy as np
from utils import get_node_attribute, replace_source_ip_randomly
import json
from Constants import *


class Extractor:
    def __init__(self, log_path: str = None, df: DataFrame = None, label: int = None):
        if log_path is not None:
            self.log_path = log_path
            self.df: DataFrame = self.log2df()
        elif df is not None:
            df.reset_index(inplace=True)
            self.df = df

        self.label = label

        self.graph_id = self.get_current_graph_id()

        self.node_infos: List[Dict] = []
        # 记录每个主机对应的节点，即在哪些流中
        self.host2node: Dict[str, List[int]] = dict()
        self.edges: np.ndarray = None
        self.edge_attr: np.ndarray = None

    def get_current_graph_id(self) -> int:
        # 找出node_info_file中，最后一行数据的graph_id，做递增
        # 每次提取，会将df中所有数据标为同一个graph
        # create file if not exist
        if not os.path.exists(NODE_INFO_FILE):
            with open(NODE_INFO_FILE, "w") as f:
                pass
            return 0
        with open(NODE_INFO_FILE, "r") as f:
            f.seek(0, os.SEEK_END)
            if f.tell() == 0:
                return 0
        data = self.load_node_infos()
        return data[-1]["graph_id"] + 1

    @classmethod
    def log2df(self, log_path: str = None, label: int = None, replace_src: bool = False) -> DataFrame:
        log_path = path.join(LOG_DIR, log_path)
        if log_path is not None:
            self.log_path = log_path
        if not os.path.exists(self.log_path):
            print("File not found")
            return None
        log_reader = LogToDataFrame()
        df = log_reader.create_dataframe(self.log_path, ts_index=False, aggressive_category=False)
        if label is not None:
            df[COLUMN.LABEL] = label
        # delete ts and duration column
        df.replace([pd.NA, pd.NaT, np.nan], 0, inplace=True)
        df.infer_objects(copy=False)
        if replace_src:
            replace_source_ip_randomly(df)
        return df

    def add_edge(self, host1: str, host2: str, node_id: int):
        # 更新主机对应节点、邻接矩阵
        if host1 not in self.host2node:
            self.host2node[host1] = [node_id]
        else:
            self.host2node[host1].append(node_id)

        if host2 not in self.host2node:
            self.host2node[host2] = [node_id]
        else:
            self.host2node[host2].append(node_id)

        for node in self.host2node[host1]:
            if node == node_id:
                continue
            self.edges[node][node_id] = 1
            self.edges[node_id][node] = 1

            attr = self.get_edge_attr(node, node_id)
            self.edge_attr[node][node_id] = attr
            self.edge_attr[node_id][node] = attr
        for node in self.host2node[host2]:
            if node == node_id:
                continue
            self.edges[node][node_id] = 1
            self.edges[node_id][node] = 1

            attr = self.get_edge_attr(node, node_id)
            self.edge_attr[node][node_id] = attr
            self.edge_attr[node_id][node] = attr

    # 更新node_infos和edges邻接矩阵
    def extract(self):
        if self.df is None:
            print("data not found")
            return
        df = self.df
        graph_id = self.graph_id
        node_total = df.shape[0]
        self.edges = np.zeros((node_total, node_total), dtype=int)
        self.edge_attr = np.zeros((node_total, node_total), dtype=float)
        for node_id, row in df.iterrows():
            # 每一行为一条流，代表一个节点
            node_info = dict()
            node_info["graph_id"] = graph_id
            node_info["node_id"] = node_id
            node_info["attribute"] = get_node_attribute(row)
            if self.label is None:
                node_info["label"] = row[COLUMN.LABEL]
            else:
                node_info["label"] = self.label

            host1 = row["id.orig_h"]
            host2 = row["id.resp_h"]
            self.add_edge(host1, host2, node_id)

            self.node_infos.append(node_info)

        self.save()

    def save_node_infos(self):
        pre_data = self.load_node_infos()
        with open(NODE_INFO_FILE, "w") as f:
            json.dump(pre_data + self.node_infos, f)

    @classmethod
    def load_node_infos(self) -> List[Dict]:
        with open(NODE_INFO_FILE, "r") as f:
            content = f.read().strip()

        if len(content) == 0:
            data = []
        else:
            data = json.loads(content)
        return data

    def save_edges(self):
        edges_file = path.join("raw", f"edges_{self.graph_id}.npy")
        np.save(edges_file, self.edges)

        edge_attr_file = path.join("raw", f"edge_attr_{self.graph_id}.npy")
        np.save(edge_attr_file, self.edge_attr)

    @classmethod
    def load_edges(self, graph_id: int) -> np.ndarray:
        edges_file = path.join("raw", f"edges_{graph_id}.npy")
        return np.load(edges_file)

    def get_edge_attr(self, node1: int, node2: int) -> float:
        # 边权重为时间差的倒数
        node1_col = self.df.iloc[node1]
        node2_col = self.df.iloc[node2]
        time_diff: pd.Timedelta = node2_col[COLUMN.TIMESTAMP] - node1_col[COLUMN.TIMESTAMP]
        time_diff = time_diff.seconds
        return abs(1 / (time_diff + 0.01))

    def save(self):
        self.save_node_infos()
        self.save_edges()
