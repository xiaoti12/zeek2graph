import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Linear, Parameter
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax, add_self_loops


class MyGAT(MessagePassing):
    def __init__(self, in_channels, out_channels, heads=1, concat=True, negative_slope=0.2, dropout=0.7):
        super(MyGAT, self).__init__(aggr='add')  # "Add" aggregation.
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout

        # Define the learnable parameters
        self.lin = nn.Linear(in_channels, heads * out_channels, bias=False)
        self.att = nn.Parameter(torch.Tensor(1, heads, 2 * out_channels))
        self.bias = nn.Parameter(torch.Tensor(heads * out_channels)) if concat else nn.Parameter(torch.Tensor(out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.xavier_uniform_(self.att)
        nn.init.zeros_(self.bias)

    def forward(self, x, edge_index):
        if edge_index.is_sparse:
            edge_index = edge_index.coalesce().indices()
        # Linear transformation
        x = self.lin(x)  # Shape: [num_nodes, heads * out_channels]

        # Reshape x to [num_nodes, heads, out_channels]
        x = x.view(-1, self.heads, self.out_channels)

        # Start propagating messages
        # Flatten the heads into the feature dimension for propagate
        x = x.view(-1, self.heads * self.out_channels)  # Shape: [num_nodes, heads * out_channels]
        return self.propagate(edge_index, x=x)

    def message(self, x_i, x_j, edge_index, size_i):
        # Reshape x_i and x_j to [num_edges, heads, out_channels]
        x_i = x_i.view(-1, self.heads, self.out_channels)
        x_j = x_j.view(-1, self.heads, self.out_channels)

        # Compute attention coefficients
        # Concatenate for attention: [num_edges, heads, 2 * out_channels]
        x_cat = torch.cat([x_i, x_j], dim=-1)

        # Compute attention scores
        alpha = (x_cat * self.att).sum(dim=-1)  # Shape: [num_edges, heads]
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, edge_index[0], num_nodes=size_i)

        # Apply dropout to attention coefficients
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)  # Shape: [num_edges, heads]

        # Multiply by attention coefficients
        # Expand alpha for multiplication: [num_edges, heads, 1]
        alpha = alpha.unsqueeze(-1)
        out = x_j * alpha  # Shape: [num_edges, heads, out_channels]

        # Reshape back to [num_edges, heads * out_channels]
        out = out.view(-1, self.heads * self.out_channels)
        return out

    def update(self, aggr_out):
        if self.concat:
            aggr_out = aggr_out + self.bias  # Shape: [num_nodes, heads * out_channels]
        else:
            aggr_out = aggr_out.view(-1, self.out_channels)
            aggr_out = aggr_out + self.bias  # Shape: [num_nodes, out_channels]
        return aggr_out


class MyGATWithEdge(MessagePassing):
    def __init__(self, in_channels, out_channels, heads=1, concat=True, negative_slope=0.2, dropout=0.7):
        super(MyGATWithEdge, self).__init__(aggr='add')  # "Add" aggregation.
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout

        # Define the learnable parameters
        self.lin = nn.Linear(in_channels, heads * out_channels, bias=False)
        self.att = nn.Parameter(torch.Tensor(1, heads, 2 * out_channels + 1))  # add 1 to inclued edge_attr
        self.bias = nn.Parameter(torch.Tensor(heads * out_channels)) if concat else nn.Parameter(torch.Tensor(out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.xavier_uniform_(self.att)
        nn.init.zeros_(self.bias)

    def forward(self, x, edge_index, edge_attr):
        if edge_index.is_sparse:
            edge_index = edge_index.coalesce().indices()
        # Linear transformation
        x = self.lin(x)  # Shape: [num_nodes, heads * out_channels]

        # Reshape x to [num_nodes, heads, out_channels]
        x = x.view(-1, self.heads, self.out_channels)

        # Start propagating messages
        # Flatten the heads into the feature dimension for propagate
        x = x.view(-1, self.heads * self.out_channels)  # Shape: [num_nodes, heads * out_channels]
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_index, size_i, edge_attr):
        # Reshape x_i and x_j to [num_edges, heads, out_channels]
        x_i = x_i.view(-1, self.heads, self.out_channels)
        x_j = x_j.view(-1, self.heads, self.out_channels)

        # concat node features and edge attr
        edge_attr = edge_attr.view(-1, 1).repeat(1, self.heads)  # shape [num_edges, heads]
        x_cat = torch.cat([x_i, x_j, edge_attr.unsqueeze(-1)], dim=-1)
        # now shape [num_edges, heads, 2 * out_channels + 1]

        # Compute attention scores
        alpha = (x_cat * self.att).sum(dim=-1)  # Shape: [num_edges, heads]
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, edge_index[0], num_nodes=size_i)

        # Apply dropout to attention coefficients
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)  # Shape: [num_edges, heads]

        # Multiply by attention coefficients
        # Expand alpha for multiplication: [num_edges, heads, 1]
        alpha = alpha.unsqueeze(-1)
        out = x_j * alpha  # Shape: [num_edges, heads, out_channels]

        # Reshape back to [num_edges, heads * out_channels]
        out = out.view(-1, self.heads * self.out_channels)
        return out

    def update(self, aggr_out):
        if self.concat:
            aggr_out = aggr_out + self.bias  # Shape: [num_nodes, heads * out_channels]
        else:
            aggr_out = aggr_out.view(-1, self.out_channels)
            aggr_out = aggr_out + self.bias  # Shape: [num_nodes, out_channels]
        return aggr_out
