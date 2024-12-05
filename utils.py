from typing import List, Dict, Tuple
import numpy as np
import torch
from torch_geometric.utils import to_networkx
import pandas as pd
from Constants import *
import networkx as nx
import matplotlib.pyplot as plt
import random


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


def load_edge_attr(coo_tensor: torch.Tensor, graph_id: int) -> torch.Tensor:
    # 从COO向量读取边的起终节点，读取权重，返回[n, 1]的向量
    edge_attr_file = path.join("raw", f"edge_attr_{graph_id}.npy")
    edge_attr_matrix: np.ndarray = np.load(edge_attr_file)

    # 获取索引
    indices = coo_tensor._indices()
    num_edges = indices.shape[1]

    # 预分配结果张量
    edge_attr = torch.empty(num_edges, 1)

    # 使用向量化操作
    start_indices = indices[0].numpy()
    end_indices = indices[1].numpy()
    edge_attr[:, 0] = torch.from_numpy(edge_attr_matrix[start_indices, end_indices])

    return edge_attr


def visualize_graph(data):
    G = to_networkx(data, to_undirected=True)

    # 创建颜色映射
    color_map = {0: 'green', 1: 'red'}
    node_colors = [color_map[label.item()] for label in data.y.cpu().numpy()]

    plt.figure(figsize=(15, 8))
    nx.draw(G, node_color=node_colors)

    # 添加自定义标签的图例
    label_names = {0: 'normal', 1: 'malware'}
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color_map[i], label=label_names[i], markersize=10) for i in [0, 1]
    ]

    plt.legend(handles=legend_elements, loc='upper right')
    plt.savefig(f"graph.png")
    # plt.show()
    # plt.close()


def split_df_by_time(df: pd.DataFrame, time_interval: str) -> List[pd.DataFrame]:
    # 按时间拆分为dataframe的列表
    if not time_interval.endswith(("s", "min", "H")):
        print("Invalid time interval")
        return [df]

    df.set_index(COLUMN.TIMESTAMP, inplace=True)

    resampled_groups = df.resample(time_interval)
    dfs = [group for _, group in resampled_groups if group.shape[0] > 0]

    return dfs


def get_graph_edge_num(data):
    return len(data.edge_attr)


def generate_random_ip() -> str:
    return ".".join(str(random.randint(0, 255)) for _ in range(4))


def replace_source_ip_randomly(df: pd.DataFrame):
    # 将连续的相同<源IP,目的IP>行中，源IP替换为随机IP，避免log文件中源IP数量较少

    random_ips = []
    random_count = max(3, int(len(df) / 100))
    for _ in range(random_count):
        random_ips.append(generate_random_ip())

    previous_pair = None
    random_ip = None

    for idx in range(len(df)):
        current_pair = (df.at[idx, COLUMN.SRC_HOST], df.at[idx, COLUMN.DST_HOST])

        if current_pair == previous_pair:
            df.at[idx, COLUMN.SRC_HOST] = random_ip
        else:
            random_ip = random.choice(random_ips)
            df.at[idx, COLUMN.SRC_HOST] = random_ip

        previous_pair = current_pair


def replace_dest_ip_randomly(df: pd.DataFrame):
    sni_ip = {}
    for sni in df["sni"].unique():
        sni_ip[sni] = generate_random_ip()
    df[COLUMN.DST_HOST] = df["sni"].map(sni_ip)


if __name__ == "__main__":
    pass
