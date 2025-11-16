
from torch.utils.data import Dataset
from PIL import Image
from typing import List, Tuple, Dict
from pathlib import Path
import torchvision.transforms as T
import torch

"""
Klasa obsługująca wczytywanie zdjęć, przechowuje listę zdjęć i zwraca zdjęcie
o zadanym indeksie 
"""
class ImageDataset(Dataset):
    def __init__(self, path: str, max_img_no: int):
        self.path = Path(path)
        self.max_img_no = max_img_no
        self.img_list = []

        for label_name in ("Real", "Fake"):
            label_dir = self.path / label_name 
            label = 0 if label_name == 'Real' else 1
            img_no = 0

            print("Rozpoczynamy wczytywanie ścieżki " + str(label_dir))

            for p in sorted(label_dir.rglob("*")):
                # print("Rozpoczynamy wczytywanie ścieżki " + str(label_dir))
                img_no += 1
                if img_no > max_img_no: 
                    break
                self.img_list.append((str(p), label))

    
    def __len__(self):
       return len(self.img_list)

    def __getitem__(self, idx):
        path, label = self.img_list[idx]
        img = Image.open(path).convert("RGB")
        return img, label, path
        

"""
Przechowuje dataset obrazów w formie klasy ImageDataset + przechowuje metadane (pozycje patchy, labelki, id obrazu ...),
wymiary obrazów do jakich należy je przeskalować
Zawiera również procedurę do obsługi podziału wskazanych zdjęć na patche. """
class PatchDataset(Dataset):
    
    def __init__(self, dataset: ImageDataset, img_Height: int, img_Width: int, patch_size: int, mean, std):
        self.dataset = dataset
        # self.img_size = {"H": img_Height, 'W': img_Width}
        self.patch_size = patch_size
        self.to_tensor = T.Compose([
            T.Resize((img_Height, img_Width)),
            T.ToTensor(),
            T.Normalize(mean = mean, std = std)
        ])

    def __len__(self):
        return len(self.dataset)

    def _extract_patches(self, img: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        C, H, W = img.shape
        patch_size = self.patch_size
        patches = []
        coords = []

        for y in range(0, H, patch_size):
            for x in range(0, W, patch_size):
               patch = img[:, y:y+patch_size, x:x+patch_size]
               patches.append(patch)
               coords.append([x, y])
        
        return torch.stack(patches), torch.tensor(coords, dtype=torch.float32)
        
    def __getitem__(self, idx):
        img, label, path = self.dataset[idx]
        img = self.to_tensor(img)
        patches, coords = self._extract_patches(img)
        return {
            "patches": patches,
            "coords": coords,
            "label": torch.tensor(label).long(),
            "patch": path
        }

