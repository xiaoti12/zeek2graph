import torch
from torch_geometric.data import InMemoryDataset, Data
import os
from utils import *
from Extractor import Extractor
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
import numpy as np


class MyDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        return os.path.join(self.root, "raw")

    @property
    def processed_dir(self):
        return os.path.join(self.root, "processed")

    @property
    def raw_file_names(self):
        return ["node_info.json"]

    @property
    def processed_file_names(self):
        return ["data.pt"]

    def download(self):
        pass

    def process(self):
        # processed_file_names exists
        if os.path.exists(self.processed_paths[0]):
            return

        data_list = []
        node_info_df = pd.DataFrame.from_dict(Extractor.load_node_infos())
        unique_ids = node_info_df["graph_id"].unique()

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 边权重归一化
        all_edge_attrs = []
        print("Collecting all edge attributes for scaling...")
        for graph_id in tqdm(unique_ids):
            edges = Extractor.load_edges(graph_id)
            edges_index = dense_matrix_to_coo(edges)
            edge_attr = load_edge_attr(edges_index, graph_id).numpy()
            all_edge_attrs.append(edge_attr)

        all_edge_attrs = np.concatenate(all_edge_attrs)

        edge_min = all_edge_attrs.min()
        edge_max = all_edge_attrs.max()

        # 节点特征归一化
        all_attrs = []
        print("Collecting all node attributes for scaling...")
        for graph_id in tqdm(unique_ids):
            current_data = node_info_df.loc[node_info_df["graph_id"] == graph_id]
            attrs = np.array(current_data["attribute"].to_list(), dtype=np.float32)
            all_attrs.append(attrs)

        # Fit the scaler on all attributes
        all_attrs = np.vstack(all_attrs)
        scaler = StandardScaler()
        scaler.fit(all_attrs)

        for graph_id in tqdm(unique_ids):
            edges = Extractor.load_edges(graph_id)
            current_data = node_info_df.loc[node_info_df["graph_id"] == graph_id]

            attrs = np.array(current_data["attribute"].to_list(), dtype=np.float32)
            attrs = scaler.transform(attrs)
            attrs = torch.from_numpy(attrs)

            labels = np.array(current_data["label"].to_list(), dtype=np.int64)
            y = torch.from_numpy(labels)

            edges_index = dense_matrix_to_coo(edges)
            edge_attr = load_edge_attr(edges_index, graph_id).numpy()
            edge_attr_scaled = (edge_attr - edge_min) / (edge_max - edge_min)
            edge_attr_scaled = torch.from_numpy(edge_attr_scaled).float()

            data = Data(x=attrs, y=y, edge_index=edges_index, edge_attr=edge_attr_scaled).to(device)
            data_list.append(data)

        self.save(data_list, self.processed_paths[0])


class MyDatasetCuda1(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        return os.path.join(self.root, "raw")

    @property
    def processed_dir(self):
        return os.path.join(self.root, "processed2")

    @property
    def processed_file_names(self):
        return ["data.pt"]


def find_graph(dataset: MyDataset) -> Data:
    # 寻找最小节点数的图，适合用于可视化
    z = dataset[0]
    min_node_num = z.num_nodes
    for data in dataset:
        if data.num_nodes < min_node_num and data.num_nodes > 100 and get_graph_edge_num(data) > 100:
            min_node_num = data.num_nodes
            z = data
    return z


if __name__ == "__main__":
    # dataset = torch.load(os.path.join("processed", "data.pt"))
    dataset = MyDataset(root="./")
    z = dataset[61]
    print(f'Dataset: {dataset}:')
    print('==============================')
    print(f'Number of graphs: {len(dataset)}')
    print(f'Number of features: {dataset.num_features}')
    print(f'Number of classes: {dataset.num_classes}')

    if z.x.is_sparse:
        z.x = z.x.to_dense()  # 将特征转为密集张量
    if z.edge_index.is_sparse:
        z.edge_index = z.edge_index._indices()

    visualize_graph(z)
