import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, BatchNorm


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
