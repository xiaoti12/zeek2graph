import torch
from torch_geometric.data import Data

data = [[0, 1], [1, 0], [1, 2], [2, 1]]

# 将数据展平为一维列表
flattened_data = [item for sublist in data for item in sublist]

edge_index = torch.tensor([[0, 1, 1, 2],
                           [1, 0, 2, 1]], dtype=torch.long)
# 节点的特征                           
x = torch.tensor([[-1], [0], [1]], dtype=torch.float)

data = Data(x=x, edge_index=edge_index)

print(data)