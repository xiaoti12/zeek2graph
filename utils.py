from typing import List, Dict, Tuple
import numpy as np
import torch
from torch_geometric.utils import to_networkx
import pandas as pd
from Constants import *
import networkx as nx
import matplotlib.pyplot as plt
import random

cipher_list = [5, 47, 61, 4865, 4866, 49171, 49191, 49199, 49200]
cipher_std = {c: i for i, c in enumerate(cipher_list)}


def get_node_attribute(row: pd.Series) -> List:
    attr = []
    attr.append(row["up_bytes"])
    attr.append(row["down_bytes"])
    attr.append(row["up_bytes"] / (row["down_bytes"] + 0.1))
    attr.append(row["up_pkts"] + row["down_pkts"])
    attr.append(row["up_pkts"] / (row["down_pkts"] + 0.1))
    attr.append(row["san_num"])
    attr.append(row["ext_num"])
    attr.append(get_duration(row))
    attr.append(get_tls_version(row))
    attr.append(is_self_sighed(row))
    attr.append(get_valid_time(row))
    attr.append(get_ciphers_len(row))
    attr = attr + get_packet_len_bin(row["or_spl"])
    attr = attr + get_cipher(row)

    return attr


def get_cipher(row: pd.Series) -> List[int]:
    ciphers = [0] * len(cipher_list)
    server_cipher = int(row["cipher"])
    if server_cipher in cipher_std:
        ciphers[cipher_std[server_cipher]] = 1
    return ciphers


def get_packet_len_bin(packet_len: str) -> List:
    packet_len = packet_len.split(",")
    bins = [0 for i in range(10)]
    for l in packet_len:
        index = int(abs(int(l)) / 150)
        bins[min(index, 9)] += 1
    return bins


def get_tls_version(series):
    version_map = {
        (771, 772): 7,
        766: 1,
        767: 2,
        768: 3,
        769: 4,
        770: 5,
        771: 6,
    }
    version = version_map.get((series['server_version'], series['server_supported_version']))
    if version is None:
        version = version_map.get(series['server_version'])
    return version if version is not None else 0


def get_duration(row: pd.Series) -> float:
    # in format of seconds
    delta = row["duration"]
    return delta.total_seconds()


def is_self_sighed(row: pd.Series) -> int:
    if row["subject"] != 0 and row["issuer"] != 0:
        if row["subject"] == row["issuer"]:
            return 1
        else:
            return 0
    else:
        return -1


def get_valid_time(row: pd.Series) -> int:
    if row["valid_time"] == 0:
        return 0
    valid_times = row["valid_time"].split(",")
    valid_times = [float(i) for i in valid_times]
    avg_valid_time = sum(valid_times) / len(valid_times)
    return int(avg_valid_time / 365.0)


def get_ciphers_len(row: pd.Series) -> int:
    ciphers = row["client_ciphers"]
    if ciphers == 0:
        return 0
    ciphers = ciphers.split(",")
    return len(ciphers)


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
    # 从COO向量读取边的起终节点，读取权重，返回[n, 1]的向量
    edge_attr_file = path.join("raw", f"edge_attr_{graph_id}.npy")
    edge_attr_matrix: np.ndarray = np.load(edge_attr_file)

    edge_attr = torch.empty(0, 1)

    coo_tensor = coo_tensor._indices()

    for index in range(coo_tensor.shape[1]):
        start = coo_tensor[0, index].item()
        end = coo_tensor[1, index].item()
        cur_attr = edge_attr_matrix[start][end]

        value = torch.tensor([[cur_attr]])
        edge_attr = torch.cat((edge_attr, value), dim=0)

    return edge_attr


def visualize_graph(data):
    G = to_networkx(data, to_undirected=True)
    nx.draw(G, node_color=data.y, with_labels=True)
    plt.savefig("graph.png")
    plt.show()


def split_df_by_time(df: pd.DataFrame, time_interval: str) -> List[pd.DataFrame]:
    # 按时间拆分为dataframe的列表
    if not time_interval.endswith(("s", "min", "H")):
        print("Invalid time interval")
        return [df]

    df.set_index(COLUMN.TIMESTAMP, inplace=True)

    resampled_groups = df.resample(time_interval)
    dfs = [group for _, group in resampled_groups if group.shape[0] > 0]

    return dfs


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
