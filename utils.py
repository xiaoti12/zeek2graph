from typing import List, Dict
import numpy as np
import torch
from torch_geometric.utils import to_networkx
import pandas as pd
from Constants import *


def get_node_attribute(row: pd.Series) -> List:
    attr = []
    attr.append(row["san_num"])
    attr.append(row["ext_num"])
    return attr


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

def get_edge_attr(coo_tensor: torch.Tensor, graph_id: int) -> torch.Tensor:
    edge_attr_file = path.join("raw", f"edge_attr_{graph_id}.npy")
    edge_attr_matrix: np.ndarray = np.load(edge_attr_file)

    edge_attr = torch.empty(0,1)

    for index in range(edge_attr_matrix.shape[0]):
        start = coo_tensor[0, index].item()
        end = coo_tensor[1, index].item()
        cur_attr = edge_attr_matrix[start][end]

        value = torch.tensor([[cur_attr]])
        edge_attr = torch.cat((edge_attr, value), dim=0)

    return edge_attr


def visualize_graph(data):
    G = to_networkx(data, to_undirected=True)
    visualize_graph(G, color=data.y)


def split_df_by_time(df: pd.DataFrame, time_interval: str) -> List[pd.DataFrame]:
    if not time_interval.endswith(("s", "min", "H")):
        print("Invalid time interval")
        return [df]
    
    df.set_index(COLUMN.TIMESTAMP, inplace=True)

    resampled_groups = df.resample(time_interval)
    dfs = [group for _, group in resampled_groups if group.shape[0] > 0]

    return dfs


if __name__ == "__main__":
    pass
