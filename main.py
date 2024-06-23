from zat.log_to_dataframe import LogToDataFrame
from pandas import DataFrame
import os
from typing import List, Dict
import numpy as np
from utils import *

host2node = dict()
node_infos = []

BLACK_LABEL = 1
WHITE_LABEL = 0


def log2df(path: str) -> DataFrame:
    if not os.path.exists(path):
        print("File not found")
        return None
    log_reader = LogToDataFrame()
    df = log_reader.create_dataframe(path, ts_index=False, aggressive_category=False)
    # delete ts and duration column
    df.replace([pd.NA, pd.NaT, np.nan], 0, inplace=True)
    df.infer_objects(copy=False)
    return df


# 更新邻接矩阵
def add_edge(host1: str, host2: str, node_id: int, edges: np.ndarray):
    if host1 not in host2node:
        host2node[host1] = [node_id]
    else:
        host2node[host1].append(node_id)

    if host2 not in host2node:
        host2node[host2] = [node_id]
    else:
        host2node[host2].append(node_id)

    for node in host2node[host1]:
        edges[node][node_id] = 1
        edges[node_id][node] = 1
    for node in host2node[host2]:
        edges[node][node_id] = 1
        edges[node_id][node] = 1


# 更新node_infos
# 返回邻接矩阵
def extract(df: DataFrame) -> np.ndarray:
    graph_id = get_current_graph_id()
    node_total = df.shape[0]
    edges = np.zeros((node_total, node_total), dtype=int)
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
        add_edge(host1, host2, node_id, edges)

        node_infos.append(node_info)

    return edges


if __name__ == "__main__":
    df = log2df("tls.log")
    edge_info = extract(df)
    save_edges(get_current_graph_id(), edge_info)
    save_node_infos(node_infos)
