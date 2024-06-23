from typing import List, Dict
import os
from os import path
import csv
import json
import numpy as np
import torch
from torch_geometric.utils import to_networkx
import pandas as pd

node_info_file = path.join("raw", "node_info.json")


def get_current_graph_id() -> int:
    # create file if not exist
    if not os.path.exists(node_info_file):
        with open(node_info_file, "w") as f:
            pass
        return 0
    with open(node_info_file, "r") as f:
        if f.tell() == 0:
            return 0
    data = load_node_infos()
    return data[-1]["graph_id"] + 1


def save_node_infos(node_infos: List[Dict]):
    with open(node_info_file, "w") as f:
        json.dump(node_infos, f)


def load_node_infos() -> pd.DataFrame:
    with open(node_info_file, "r") as f:
        data = json.load(f)
    return pd.DataFrame(data)


def get_node_attribute(row: pd.Series) -> List:
    attr = []
    attr.append(row["san_num"])
    attr.append(row["ext_num"])
    return attr


def save_edges(graph_id: int, edges: np.ndarray):
    edges_file = path.join("raw", f"edges_{graph_id}.npy")
    np.save(edges_file, edges)


def load_edges(graph_id: int) -> np.ndarray:
    edges_file = path.join("raw", f"edges_{graph_id}.npy")
    return np.load(edges_file)


def dense_matrix_to_coo(adj_matrix: np.ndarray) -> torch.Tensor:
    row, col = np.nonzero(adj_matrix)
    values = adj_matrix[row, col]

    # 创建 COO 格式的索引和数据
    indices = np.vstack((row, col))
    values = values.astype(np.int16)
    # 转换为 PyTorch 张量
    indices = torch.tensor(indices, dtype=torch.long)
    values = torch.tensor(values, dtype=torch.float32)
    shape = adj_matrix.shape

    # 创建 COO 格式的稀疏张量
    coo_tensor = torch.sparse_coo_tensor(indices, values, torch.Size(shape))

    return coo_tensor


def visualize_graph(data):
    G = to_networkx(data, to_undirected=True)
    visualize_graph(G, color=data.y)

if __name__ == "__main__":
    df=load_node_infos()
    attr=df["attribute"]
