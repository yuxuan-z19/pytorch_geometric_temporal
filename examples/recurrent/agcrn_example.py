try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable):
        return iterable

import torch
import torch.nn.functional as F
from torch_geometric_temporal.nn.recurrent import AGCRN

from torch_geometric_temporal.dataset import ChickenpoxDatasetLoader
from torch_geometric_temporal.signal import temporal_signal_split

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

loader = ChickenpoxDatasetLoader("../../dataset")

dataset = loader.get_dataset(lags=8)

train_dataset, test_dataset = temporal_signal_split(dataset, train_ratio=0.2)

class RecurrentGCN(torch.nn.Module):
    def __init__(self, node_features):
        super(RecurrentGCN, self).__init__()
        self.recurrent = AGCRN(number_of_nodes = 20,
                              in_channels = node_features,
                              out_channels = 2,
                              K = 2,
                              embedding_dimensions = 4)
        self.linear = torch.nn.Linear(2, 1)
        self.h = None

    def forward(self, x, e, _):
        self.h = self.recurrent(x, e, self.h)
        y = F.relu(self.h)
        y = self.linear(y)
        self.h = self.h.detach()
        return y
        
model = RecurrentGCN(node_features = 8).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

model.train()

e = torch.empty(20, 4).to(device)

torch.nn.init.xavier_uniform_(e)

for epoch in tqdm(range(200)):
    cost = 0
    h = None
    for time, snapshot in enumerate(train_dataset):
        snapshot = snapshot.to(device)
        x = snapshot.x.view(1, 20, 8)
        y_hat = model(x, e, None)
        cost = cost + torch.mean((y_hat-snapshot.y)**2)
    cost = cost / (time+1)
    cost.backward()
    optimizer.step()
    optimizer.zero_grad()
    
model.eval()
cost = 0
for time, snapshot in enumerate(test_dataset):
    snapshot = snapshot.to(device)
    x = snapshot.x.view(1, 20, 8)
    y_hat = model(x, e, None)
    cost = cost + torch.mean((y_hat-snapshot.y)**2)
cost = cost / (time+1)
cost = cost.item()
print("MSE: {:.4f}".format(cost))
