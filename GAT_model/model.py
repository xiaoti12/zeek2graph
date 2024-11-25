import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, BatchNorm
from mygat import *


class GATModel(torch.nn.Module):
    def __init__(self, in_channels, out_channels, dropout):
        heads = 4
        super(GATModel, self).__init__()
        self.conv1 = GATConv(in_channels, 128, heads=heads)
        self.bn1 = BatchNorm(128 * heads)

        self.conv2 = GATConv(128 * heads, 128, heads=heads)
        self.bn2 = BatchNorm(128 * heads)

        self.lin1 = torch.nn.Linear(128 * heads, 128)  # Linear layer with a reasonable size
        self.lin2 = torch.nn.Linear(128, 64)  # Second layer for reducing dimensionality
        self.lin3 = torch.nn.Linear(64, out_channels)  # Final layer outputting two classes

        self.dropout = dropout

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # First GAT layer
        x = F.elu(self.conv1(x, edge_index))
        x = self.bn1(x)

        # Second GAT layer
        x = F.elu(self.conv2(x, edge_index))
        x = self.bn2(x)

        # Final linear layers
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin2(x))
        x = F.log_softmax(self.lin3(x), dim=-1)

        return x


class CustomGATModel(torch.nn.Module):
    def __init__(self, in_channels, out_channels, dropout):
        super(CustomGATModel, self).__init__()
        heads = 4
        self.conv1 = MyGAT(in_channels, 128, heads=heads)
        # self.conv1 = OptimizedCustomGATConv(in_channels, 128, heads=heads)
        self.bn1 = BatchNorm(128 * heads)

        self.conv2 = MyGAT(128 * heads, 128, heads=heads)
        self.bn2 = BatchNorm(128 * heads)

        self.lin1 = torch.nn.Linear(128 * heads, 128)
        self.lin2 = torch.nn.Linear(128, 64)
        self.lin3 = torch.nn.Linear(64, out_channels)

        self.dropout = dropout

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        # First GAT layer
        x = F.elu(self.conv1(x, edge_index))
        x = self.bn1(x)

        # Second GAT layer
        x = F.elu(self.conv2(x, edge_index))
        x = self.bn2(x)

        # Final linear layers
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin2(x))
        x = F.log_softmax(self.lin3(x), dim=-1)

        return x


if __name__ == "__main__":
    import torch.nn as nn
    from torch_geometric.utils import remove_self_loops

    class SimpleGAT(nn.Module):
        def __init__(self, num_node_features, hidden_channels, num_classes, heads=2, dropout=0.6):
            super(SimpleGAT, self).__init__()
            self.conv1 = MyGAT(in_channels=num_node_features, out_channels=hidden_channels, heads=heads, concat=True, dropout=dropout)
            self.conv2 = MyGAT(
                in_channels=hidden_channels * heads,
                out_channels=num_classes,
                heads=1,
                concat=False,
                dropout=dropout,
            )

        def forward(self, x, edge_index):
            x = self.conv1(x, edge_index)  # Output shape: [10, 16] (heads=2 * hidden_channels=8)
            x = F.elu(x)
            x = F.dropout(x, p=0.6, training=self.training)
            x = self.conv2(x, edge_index)  # Output shape: [10, 3]
            return F.log_softmax(x, dim=1)

    num_node_features = 16
    hidden_channels = 8
    num_classes = 3
    heads = 2

    model = SimpleGAT(num_node_features, hidden_channels, num_classes, heads=heads).to('cuda')

    # Example data
    x = torch.rand((10, num_node_features)).to('cuda')  # 10 nodes with num_node_features features each
    edge_index = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]], dtype=torch.long).to('cuda')

    # Add self-loops
    edge_index, _ = remove_self_loops(edge_index)
    edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

    # Forward pass
    try:
        out = model(x, edge_index)
        print(out)
    except RuntimeError as e:
        print(f'RuntimeError: {e}')
