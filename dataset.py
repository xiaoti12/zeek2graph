import torch
from torch_geometric.data import InMemoryDataset, Data
import os
from utils import *
from Extractor import Extractor
from tqdm import tqdm


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

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        for graph_id in tqdm(node_info_df["graph_id"].unique()):
            edges = Extractor.load_edges(graph_id)
            current_data = node_info_df.loc[node_info_df["graph_id"] == graph_id]

            attrs = np.array(current_data["attribute"].to_list(), dtype=np.float32)
            attrs = torch.from_numpy(attrs)

            labels = np.array(current_data["label"].to_list(), dtype=np.int64)
            y = torch.from_numpy(labels)

            edges_index = dense_matrix_to_coo(edges)
            edge_attr = get_edge_attr(edges_index, graph_id)

            data = Data(x=attrs, y=y, edge_index=edges_index, edge_attr=edge_attr).to(device)
            data_list.append(data)

        self.save(data_list, self.processed_paths[0])


if __name__ == "__main__":
    # dataset = torch.load(os.path.join("processed", "data.pt"))
    dataset = MyDataset(root="./")
    z = dataset[0]
    print(f'Dataset: {dataset}:')
    print('==============================')
    print(f'Number of graphs: {len(dataset)}')
    print(f'Number of features: {dataset.num_features}')
    print(f'Number of edges: {z.num_edges}')
    print(f'Number of classes: {dataset.num_classes}')

    if z.x.is_sparse:
        z.x = z.x.to_dense()  # 将特征转为密集张量
    if z.edge_index.is_sparse:
        z.edge_index = z.edge_index._indices()

    visualize_graph(z)
