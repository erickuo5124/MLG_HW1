import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree
from torch.nn import Linear, GRUCell, ReLU, Sigmoid
import torch.nn.functional as F
from torch.nn import Sequential

class GCNConv(MessagePassing):
  def __init__(self, in_channels, out_channels):
    super(GCNConv, self).__init__(aggr='add')

  def forward(self, x, edge_index):
    # Compute normalization.
    row, col = edge_index
    deg = degree(col, x.size(0), dtype=x.dtype) + 1
    deg_inv_sqrt = deg.pow(-0.5)
    norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

    # Start propagating messages.
    return self.propagate(edge_index, x=x, norm=norm)

  def message(self, x_j, norm):
    # Normalize node features.
    return norm.view(-1, 1) * x_j

class DrBC(torch.nn.Module):
  def __init__(self, num_L):
    super(DrBC, self).__init__()
    self.num_L = num_L

    self.data_in = Sequential(Linear(3, 128), ReLU())
    self.Aggregation = GCNConv(128, 128)
    self.Combine = GRUCell(128, 128, bias=False)
    self.data_out = Sequential(Linear(128, 64), ReLU(), Linear(64,1), ReLU())

  def forward(self, graph):
    # encoder
    h = self.data_in(graph.x)
    h = F.normalize(h,dim=-1,p=2)
    layer = torch.tensor([])
    layer = torch.cat((layer, torch.unsqueeze(h, 0)))
    for _ in range(self.num_L):
      hn = self.Aggregation(h, graph.edge_index)
      h = self.Combine(h, hn)
      h = F.normalize(h,dim=-1,p=2)
      layer = torch.cat((layer, torch.unsqueeze(h, 0)))

    h, _ = torch.max(layer, 0)
    z = F.normalize(h,dim=-1,p=2)

    # decoder
    pred = self.data_out(z)
    pred = torch.squeeze(pred)

    # loss
    label = graph.y
    pair_ids_src = torch.randint(graph.num_nodes, (graph.num_nodes * 5,))
    pair_ids_tgt = torch.randint(graph.num_nodes, (graph.num_nodes * 5,))

    preds = torch.index_select(pred, 0, pair_ids_src) - torch.index_select(pred, 0, pair_ids_tgt)
    labels = torch.index_select(label, 0, pair_ids_src) - torch.index_select(label, 0, pair_ids_tgt)
    preds, labels = torch.sigmoid(preds), torch.sigmoid(labels)
    loss = torch.nn.BCELoss()(preds, labels)
    return pred, loss