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
