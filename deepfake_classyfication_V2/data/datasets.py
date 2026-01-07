
from torch.utils.data import Dataset
from PIL import Image
from typing import List, Tuple, Dict
from pathlib import Path
import torchvision.transforms as T
import torch

"""
Klasa obsługująca wczytywanie zdjęć, przechowuje listę zdjęć i zwraca zdjęcie
o zadanym indeksie. 

PARAMETRY:
    path       - ścieżka do katalogu ze zdjęciami do wczytania. Zakładamy, że katalog zawiera
                 podkatalogi "Real" oraz "Fake", przeznaczone dla posegregowanych zbiorów zdjęć
                 prawdziwych i fałszywych. Parametr jest konfigurowalny w configu
    max_img_no - liczba oznaczająca ile zdjęć wczytamy. Liczba ta dotyczy obu podkatalogów 
                 ("Fake", "Real") osobno. Parametr konfugurowalny w configu

PROCEDURY:
    _len_()     - zwracamy liczbę przechowywanych ścieżek do zdjęć
    _getitem_() - dla wskazanego indeksu zdjęcia, pobiera jego ścieżkę z listym, wczytuje
                  i zwraca zdjęcie 
"""
class ImageDataset(Dataset):
    def __init__(self, path: str, max_img_no: int):
        self.path = Path(path)          # ścieżka do pliku
        self.max_img_no = max_img_no    # max liczba zdjęć do wczytania
        self.img_list = []              # lista ścieżek do zdjęć 

        for label_name in ("Real", "Fake"):
            label_dir = self.path / label_name 
            label = 0 if label_name == 'Real' else 1
            img_no = 0

            print("Rozpoczynamy wczytywanie ścieżki " + str(label_dir))

            for p in sorted(label_dir.rglob("*")):
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
Przechowuje dataset obrazów w formie klasy ImageDataset

PARAMETRY:
    dataset    - lista plików ze zdjęciami do analizy - klasa ImageDataset
    img_Height - wysokość pliku do jakiej chcemy go przeskalować. Do zmiany w configu
    img_Width  - szerokość pliku do jakiej chcemy go przeskalować. Do zmiany w configu
    patch_size - rozmiar patcha, czyli fragmentu obrazu, który zostanie poddany ekstrakcji cech i podany jako wierzchołek grafu. Do zmiany w configu
    mean       - średnia potrzebna do normalizacji obrazu. Do zmiany w configu
    std        - odchylenie standardowe potrzebne do normalizacji obrazu. Do zmiany w configu

PROCEDURY:
        _len_() - zwraca liczbę zdjęć przechowywanych w zmiennej "dataset"
        _extract_patches() - dzieli wskazany obraz na fragmenty, zapisuje je jako listę wraz z informacją o ich położeniu na zdjęciu (coordy). 
                             Listę fragmentów i coordów zwraca osobno 
        _getitem_() - pobiera obraz z datasetu, ekstrachuje jego cechy za pomocą _estract_patches() i zwraca słownik z patchami, coordami, etykietą oraz ścieżką do obrazu
        """
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

