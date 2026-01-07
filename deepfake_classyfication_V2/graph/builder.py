import torch
from torch_geometric.data import Data
from torch_cluster import knn_graph 

"""Na podstawie koordynatów patchy budujemy graf, łącząc patche według metody knn - innymisłowy łączymy patche z ich sąsiadami lub przekątnymi

PARAMETRY:
    embeddings - tensor cech patchy o rozmiarze [num_patches, emb_dim]. Pozyskiwane podczas ekstrakcji cech w modelu.classifier.GraphClassifier 
                 przy ekstrakcji cech z patchy za pomocą modelu CLIP
    coords     - tensor koordynatów patchy o rozmiarze [num_patches, 2] (x,y). Pozyskiwane w dataset.PatchDataset przy podziale obrazu na patche 
    k          - liczba sąsiadów do połączenia w grafie (w tym samym patchu też jest połączenie). Konfigurowalne w pliku config.yaml"""
def build_graph_from_patches_coords(embeddings: torch.Tensor, coords: torch.Tensor, k: int = 6) -> Data:
    coords = coords.float()
    # coords = coords / coords.max()
    coords = (coords - coords.mean(0)) / (coords.std(0) + 1e-6)
    
    edge_index = knn_graph(coords, k=k, loop=True)

    x=torch.cat([embeddings, coords], dim=1)
    data = Data(x=x, edge_index=edge_index)

    return data

