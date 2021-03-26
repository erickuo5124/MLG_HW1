import torch
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx

from model import DrBC 

test_path = '/content/drive/MyDrive/MLG/hw1/hw1_data/youtube/com-youtube.txt'
testans_path = '/content/drive/MyDrive/MLG/hw1/hw1_data/youtube/com-youtube_score.txt'

test_ans = {}
with open(testans_path) as f:
  line = f.readline()
  while line:
    key, value = line.split(':')
    key = eval(key)
    test_ans[key] = eval(value)
    line = f.readline()

G = nx.read_edgelist(test_path, nodetype = int)
edge_index = from_networkx(G).edge_index
x = torch.tensor([[G.degree(node),1,1] for node in G.nodes], dtype=torch.float)
y = torch.tensor([test_ans[node] for node in G.nodes], dtype=torch.float)
test_data = Data(edge_index=edge_index, x=x, y=y)

model = DrBC(5)
model.load_state_dict(torch.load('/content/drive/MyDrive/MLG/hw1/net_params.pkl'))
prediction, loss = model(test_data)

print(f'Loss: {loss:.6f}, Accuracy: {top_n_accuracy(out, 5, test_ans)}')
print(prediction)