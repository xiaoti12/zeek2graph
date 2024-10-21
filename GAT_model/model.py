import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, BatchNorm


class GATModel(torch.nn.Module):
    def __init__(self, in_channels, out_channels, dropout):
        super(GATModel, self).__init__()
        self.conv1 = GATConv(in_channels, 128, heads=4)
        self.bn1 = BatchNorm(128 * 4)

        self.conv2 = GATConv(128 * 4, 128, heads=4)
        self.bn2 = BatchNorm(128 * 4)

        self.conv3 = GATConv(128 * 4, 128, heads=4)
        self.bn3 = BatchNorm(128 * 4)

        self.conv4 = GATConv(128 * 4, 128, heads=4)
        self.bn4 = BatchNorm(128 * 4)

        self.lin1 = torch.nn.Linear(128 * 4, 128)  # Linear layer with a reasonable size
        self.lin2 = torch.nn.Linear(128, 64)  # Second layer for reducing dimensionality
        self.lin3 = torch.nn.Linear(64, out_channels)  # Final layer outputting two classes

        self.dropout = dropout

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # First GAT layer
        x = F.elu(self.conv1(x, edge_index))
        x = self.bn1(x)

        # Second GAT layer
        x = F.elu(self.conv2(x, edge_index))
        x = self.bn2(x)

        # Third GAT layer
        x = F.elu(self.conv3(x, edge_index))
        x = self.bn3(x)

        # Fourth GAT layer
        x = F.elu(self.conv4(x, edge_index))
        x = self.bn4(x)

        # Final linear layers
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin2(x))
        x = F.log_softmax(self.lin3(x), dim=-1)

        return x
