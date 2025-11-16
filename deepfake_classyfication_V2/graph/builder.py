import torch
from torch_geometric.data import Data
from torch_cluster import knn_graph 

"""Na podstawie koordynatów patchy budujemy graf, łącząc patche według metody knn - innymisłowy łączymy patche z ich sąsiadami lub przekątnymi"""
def build_graph_from_patches_coords(embeddings: torch.Tensor, coords: torch.Tensor, k: int = 6) -> Data:
    edge_index = knn_graph(coords, k=k, loop=False)

    x=torch.cat([embeddings, coords], dim=1)
    data = Data(x=x, edge_index=edge_index)

    return data



"""Na podstawie cech z ekstrachowanych patchy budujemy graf używając metody knn"""
def build_graph_from_features() -> Data:
    None


"""Na podstawie cech z patchy przed ekstrakcją cech budujemy graf używając metody knn"""
def build_graph_from_similar_patches() -> Data:
    None
