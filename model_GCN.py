import torch 
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
import torch.nn as nn
import glob 
from PIL import Image
from torchvision import transforms, models
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import logging
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from collections import Counter
from sklearn.model_selection import train_test_split
import os




FEATURES_NUM  = 512 # liczba cech każdego węzłe - tutaj jest to liczba kolorów każdego piksela
FIRST_LAYER_OUT_CH = 256 # Liczba cech na wyjściu pierwszej 
SECOUND_LAYER_OUT_CH = 128
THIRD_LAYER_OUT_CH = 64
FORTH_LAYER_OUT_CH = 32
FIFTH_LAYER_OUT_CH = 16
SIXTH_LAYER_OUT_CH = 8
SEVENTH_LAYER_OUT_CH = 4
LAST_LAYER_OUT_CH = 2 # Liczba cech na wyjściu drugiej (ostatniej) warstwy

#----------------------------
# Definicja modelu
#----------------------------
class GCN(torch.nn.Module):
   def __init__(self):
       super().__init__()
       self.conv1 = GCNConv(in_channels = FEATURES_NUM, out_channels = 128) 
    #    self.conv2 = GCNConv(in_channels = 256, out_channels = 128) 
       self.conv3 = GCNConv(in_channels = 128, out_channels = 32) 
       self.conv4 = GCNConv(in_channels = 32, out_channels = LAST_LAYER_OUT_CH) 
   
   def forward(self, x, edge_index, batch):
       x = self.conv1(x, edge_index)
       x = F.relu(x)
       x = F.dropout(x, p=0.3, training=self.training)

    #    x = self.conv2(x, edge_index)
    #    x = F.relu(x)
    #    x = F.dropout(x, p=0.3, training=self.training)

       x = self.conv3(x, edge_index)
       x = F.relu(x)
       x = F.dropout(x, p=0.3, training=self.training)

       x = self.conv4(x, edge_index)
       x = global_mean_pool(x, batch) # labelki bez tego były tylko do jednego obrazu. Tutaj przypisujemy tę samą labelkę to całego obrazu
       
       return F.log_softmax(x, dim=1) # można spróbować zwrócić goły x + zastosować crossentropy zamiast NLLLoss
   
# model = GCN()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = 'cpu'
model = GCN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
criterion = torch.nn.NLLLoss()
logger = logging.getLogger(__name__)
logging.basicConfig(filename='example.log', encoding='utf-8', level=logging.DEBUG)

resnet18 = models.resnet18(weights=models.ResNet18_Weights.DEFAULT) #definiujemy model, ustawiamy domyślne wagi - przy ustawieniu pretrained = True był warning o nowszym podejściu z defaultowymi wagami
resnet18.fc = nn.Identity() # ostatnia warstwa (klasyfikacyjna) zamiast predykcji, będzie kopiować wyniki z poprzedniej warstwy uśredniającej (avgpool - uśrednienie przestrzenne)
resnet18 = resnet18.to(device)
resnet18.eval() # przechodzimy w tryb "predykcji" - na dalszym etapie wyskakiwał błąd, ponieważ niektóre operacje nie są dozwolone w trybie (treningu?? - chyba taki tryb jest domyślnie włączony)
torch.backends.cudnn.benchmark = True

# ------------------------
# trenowanie modelu
# ----------------------------

def train(data):
   model.train()
   optimizer.zero_grad()
   out = model(data.x, data.edge_index, data.batch)
   loss = criterion(out, data.y)
   loss.backward()
   optimizer.step()
   
   return loss.item()

def evaluate(data):
    model.eval()
    
    y_true = []
    y_pred = []

    correct  = 0 
    total = 0

    for batch in data:
        batch = batch.to(device)
        _, pred =  model(batch.x, batch.edge_index, batch.batch).max(dim=1)

        y_true.extend(batch.y.detach().cpu().numpy())
        y_pred.extend(pred.detach().cpu().numpy())
        correct += int((pred == batch.y).sum())
        total += batch.y.size(0)

    precision = precision_score(y_true, y_pred, average='binary', zero_division=0)
    recall = recall_score(y_true, y_pred, average='binary', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='binary', zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp)
    acc = correct / total


    print("Pred:", y_pred[:20])
    print("True:", y_true[:20])

    print("correct: {}, total: {}:".format(correct, total))
    print("tn: {}, fp: {}, fn: {}, tp: {}".format(tn, fp, fn, tp))
    print(f"Dokładność klasyfikacji: {acc:.2%}")
    print(f"precyzja klasyfikacji: {precision:.2%}")
    print(f"recall klasyfikacji: {recall:.2%}")
    print(f"f1 klasyfikacji: {f1:.2%}")
    print(f"specificity klasyfikacji: {specificity:.2%}")

    return [acc, precision, recall, f1, specificity]

def save_model():
    torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    }, 'gcn_model.pth')

def load_model():
    if os.path.exists('gcn_model.pth'):
        checkpoint = torch.load('gcn_model.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print('Załądowano model z pliku')
    else:
        print('Plik modelu nie istnieje')

def create_edges(image_size, patch_size):
    edge_index = []
    h = image_size // patch_size
    w = image_size // patch_size

    for row in range(h):
        for col in range(w):
            idx = row*w + col 
            for dx, dy in [(-1, -1), (-1, 0), (-1, -1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]: # łączenie pikseli krawędziami boki + po skosie 
            # for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]: # łączenie pikseli krawędziami same boki 
                nrow, ncol = row + dx, col + dy 
                if 0 <= nrow < h and 0 <= ncol < w:
                    neighbour_idx = nrow*w + ncol
                    edge_index.append([idx, neighbour_idx])

    edge_index = torch.tensor(edge_index).t().contiguous()
    edge_index = edge_index.to(device)

    return edge_index

def load_pictures(dir_path, transform, image_size, patch_size, start_pic, end_pic):

    #czyścimy cache, co by się nam blue screen nie pokazał
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    all_img_patches = []

    iter = 0
    for pic_path in glob.glob(dir_path):
        iter += 1
        # if iter %1000 == 0:
        if iter %1000 == 0:
            print("wczytano obraz nr {}".format(iter))
            # break
        if iter < start_pic:
            continue
        if iter > end_pic:
            break

        image = Image.open(pic_path).convert("RGB")
        image = transform(image)
        
        patches = []
        for i in range(0, image_size, patch_size):
            for j in range(0, image_size, patch_size):
               patch = image[:, i:i+patch_size, j:j+patch_size]
               patches.append(patch)
        
        patches = torch.stack(patches)
        all_img_patches.append(patches)
    
    # patches = torch.stack(patches)

    return all_img_patches #patches

# używamy RESNET18 do ekstrachowania cech z patchy
@torch.no_grad()
def extract_features(all_img_patches, patch_size, batch_size):
    # przenoszę eval jako jedno wywołanie - globalne - nigdzie nie zmieniamy na .train(), więc nie powinno być problemu
    # resnet18.eval() # przechodzimy w tryb "predykcji" - na dalszym etapie wyskakiwał błąd, ponieważ niektóre operacje nie są dozwolone w trybie (treningu?? - chyba taki tryb jest domyślnie włączony)

    # with torch.no_grad():
    all_img_patch_features = []

    for patches in all_img_patches:
        patches = patches.to(device, non_blocking = True)

        # patches = F.interpolate(patches, size=(patch_size, patch_size), mode='bilinear', align_corners=False)  # tutaj w zasadzie ten resize_ chyba nie potrzebny, bo już mamy [1,3,16,16], ale nie wiem czy nie będzie trzeba zmienić wymiarów inputu na 254x254 - do sprawdzenia
        # patches = F.interpolate(patches, size=(224, 224), mode='bilinear', align_corners=False)  # tutaj w zasadzie ten resize_ chyba nie potrzebny, bo już mamy [1,3,16,16], ale nie wiem czy nie będzie trzeba zmienić wymiarów inputu na 254x254 - do sprawdzenia

        patch_features = []
        # for patch in patches:
        for i in range(0, patches.size(0), batch_size):
            # tu musimy jakoś dostosować input do tego, który przyjmie model
            # input = # input_tensor = preprocess(image).unsqueeze(0)
            # input_tensor = patch.unsqueeze(0).resize_(1, 3, patch_size, patch_size)  # tutaj w zasadzie ten resize_ chyba nie potrzebny, bo już mamy [1,3,16,16], ale nie wiem czy nie będzie trzeba zmienić wymiarów inputu na 254x254 - do sprawdzenia
            # input_tensor = F.interpolate(patch.unsqueeze(0), size=(patch_size, patch_size), mode='bilinear', align_corners=False)  # tutaj w zasadzie ten resize_ chyba nie potrzebny, bo już mamy [1,3,16,16], ale nie wiem czy nie będzie trzeba zmienić wymiarów inputu na 254x254 - do sprawdzenia
            # output = resnet18(input_tensor)
            # patch_features.append(output)
            patch = patches[i:i+batch_size]
            if device == 'cuda':
                with torch.cuda.amp.autocast(device_type = 'cuda'):
                    output = resnet18(patch)
            else:
                output = resnet18(patch)

            patch_features.append(output)
          
        patch_features = torch.cat(patch_features, dim=0).to(device)    # zamieniamy listę tensorów na jeden tensor [N, 512]
        all_img_patch_features.append(patch_features)

    # x = torch.cat(patch_features, dim=0) # zamieniamy listę tensorów na jeden tensor [N, 512]
    # x = x.to(device)
    
    return all_img_patch_features

def create_dataset(patches_fake, patches_real, PATCH_SIZE, IMAGE_SIZE, FAKE_LABEL, REAL_LABEL, edge_index, split_ratio):
    
    print("2. Ekstrachujemy cechy za pomocą resnet".center(60, '-'))
    # używamy RESNET18 do ekstrachowania cech z patchy
    batch_size = (IMAGE_SIZE // PATCH_SIZE) ** 2
    all_img_x_fake = extract_features(patches_fake, PATCH_SIZE, batch_size)
    all_img_x_real = extract_features(patches_real, PATCH_SIZE, batch_size)

    # img = transforms.ToTensor()(img) 

    print("4. Tworzymy dataset".center(60, '-'))
    data_parted = []

    for x in all_img_x_fake:
        data_parted.append(Data(x=x, edge_index=edge_index, y=torch.tensor([FAKE_LABEL])).to(device))

    for x in all_img_x_real:
        data_parted.append(Data(x=x, edge_index=edge_index, y=torch.tensor([REAL_LABEL])).to(device))


    labels = [d.y.item() for d in data_parted]
    print("Label count:", Counter(labels))

    # split_ratio = 0.8
    # split_data_idx = int(len(data_parted) * split_ratio)
    # if split_ratio == 1.0:
    #     data_parted_train = []
    #     data_parted_test = data_parted
    # elif split_ratio == 0.0:
    #     data_parted_train = data_parted
    #     data_parted_test = []
    # else:
    #     data_parted_train = data_parted[:split_data_idx]
    #     data_parted_test = data_parted[split_data_idx:]

    labels = [d.y.item() for d in data_parted]
    data_parted_train, data_parted_test = train_test_split(data_parted, test_size=1-split_ratio, stratify=labels, random_state=42)


    data_train, data_test = [], []

    if data_parted_train:
        data_train = DataLoader(data_parted_train, batch_size=16, shuffle=True)
    if data_parted_test:
        # data_test = DataLoader(data_parted_test, batch_size=16, shuffle=True)
        data_test = DataLoader(data_parted_test, batch_size=16, shuffle=False)

    return [data_train, data_test]