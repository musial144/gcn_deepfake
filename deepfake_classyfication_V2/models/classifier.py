import torch
from torch import nn
from .CLIP import CLIPExtractor 
from .GATv2 import GATv2
from graph.builder import build_graph_from_patches_coords
from torch_geometric.data import Batch


class GraphClassifier(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.device =  torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
        self.clip_extr =  CLIPExtractor(cfg.model.clip_name, cfg.model.clip_trainable)
        in_dim = self.clip_extr.out_dim + 2  # doklejamy 2 wartości dla coord's
        self.gnn = GATv2(in_dim, cfg.model.gat.hidden_dim, cfg.model.gat.out_dim, cfg.model.gat.heads1, cfg.model.gat.heads2, cfg.model.gat.dropout)
        self.head = nn.Sequential (
            nn.Linear(self.gnn.out_dim, 256), nn.GELU(), nn.Dropout(0.2), nn.Linear(256,2)
        )
        self.cfg = cfg

    def forward_single(self, patches, coords):
        with torch.set_grad_enabled(self.cfg.model.clip_trainable):
            emb = self.clip_extr.extract_features(patches)

        data = build_graph_from_patches_coords(emb, coords, k=self.cfg.graph.knn_k)
        return data

    def forward(self, batch):
        outputs = []
        for sample in batch:
            patches = sample["patches"].to(self.device)
            coords = sample["coords"].to(self.device)
            outputs.append(self.forward_single(patches,coords))
        batch = Batch.from_data_list(outputs).to(self.device)

        logits = self.gnn(batch)
        return logits