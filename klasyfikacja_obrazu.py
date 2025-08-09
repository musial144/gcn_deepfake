import torch 
import model_GCN as gcn
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image, ImageDraw
from datetime import datetime

import glob 


#-----------------------------------------------------------------------------------------------------------------------------
# instrukcja na GCN dla obrazów - https://www.kaggle.com/discussions/questions-and-answers/388425                   
# 1. Extract the pixel values for each image from the loaded image data set.

# 2. Reorganize each image's pixel values into a grid.

# 3. To generate the picture grid graph, specify a set of edges between neighbouring pixels in the grid.

# 4. To guarantee the stability of the graph convolution process, normalise the adjacency matrix for the picture grid graph.

# 5. Supply the GCN model with the picture grid graph and pixel attributes for training and classification.
#-----------------------------------------------------------------------------------------------------------------------------


#-----------------------------------------------------------------------------------------------------------------------------
# 1. Extract the pixel values for each image from the loaded image data set.

# stałe
# FAKE_DIR = '../../kod_pracy/pictures/Deep-vs-Real/Deepfake/*'#.jpg'
# REAL_DIR = '../../kod_pracy/pictures/Deep-vs-Real/Real/*'#.jpg'

# FAKE_DIR = '../../kod_pracy/pictures/Final_Dataset/Fake/*'#.jpg'
# REAL_DIR = '../../kod_pracy/pictures/Final_Dataset/Real/*'#.jpg'

# FAKE_DIR = '../../kod_pracy/pictures/My_Dataset/Fake/*'#.jpg'
# REAL_DIR = '../../kod_pracy/pictures/My_Dataset/Real/*'#.jpg'

REAL_DIR = '../../kod_pracy/pictures/Dataset/Train/Real/*'#.jpg'
FAKE_DIR = '../../kod_pracy/pictures/Dataset/Train/Fake/*'#.jpg'

FAKE_LABEL = 0
REAL_LABEL = 1

PATCH_SIZE = 32 #16
IMAGE_SIZE = 256#512 #128

# ładujemy obraz i dzielimy go na patche
img = Image.open("../../kod_pracy/pictures/fake/1_(14).jpg", "r")

print("Liczba FAKE obrazów:", len(glob.glob(FAKE_DIR)))
print("Liczba REAL obrazów:", len(glob.glob(REAL_DIR)))

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

transform_patch = transforms.Compose([
    # transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
])

model_mode = input('Chcesz: \n 1. trenować \n 2. evaluować?\n')

print("1. Tworzymy krawędzie".center(60, '-'))

edge_index = gcn.create_edges(IMAGE_SIZE, PATCH_SIZE)


for i in range(0, 100):
    # if i > 0:
    print ('i: {}, czas: {}'.format(i, datetime.now()))
    gcn.load_model()

    print("2. wczytujemy obrazy".center(60, '-'))
    start_pic, end_pic = 1000*i, 1000*i+999
    patches_fake = gcn.load_pictures(FAKE_DIR, transform, IMAGE_SIZE, PATCH_SIZE, start_pic, end_pic)
    patches_real = gcn.load_pictures(REAL_DIR, transform, IMAGE_SIZE, PATCH_SIZE, start_pic, end_pic)

    # undersampling 
    if not patches_fake or not patches_real:
        break

    split_ratio = 0.8
    data_train, data_test = gcn.create_dataset(patches_fake, patches_real, PATCH_SIZE, IMAGE_SIZE, FAKE_LABEL, REAL_LABEL, edge_index, split_ratio)
    # data_train, data_test = gcn.create_dataset(patches_fake, patches_real, PATCH_SIZE, FAKE_LABEL, REAL_LABEL, edge_index)

    total_loss = 0 

    if model_mode == '1':
        print("5. Trenujemy model".center(60, '-'))
        for epoch in range(200):
            total_loss = 0 
            for batch in data_train:
                loss = gcn.train(batch)
                total_loss += loss

            if epoch % 20 == 0:
                print(f"Epoch {epoch} | Loss: {total_loss:.4f}")
            # if epoch % 50 == 0:
            #     gcn.save_model()
        gcn.save_model()

    print("5.ewaluujemy model".center(60, '-'))

    acc, precision, recall, f1, specificity = gcn.evaluate(data_test)

    with open("metrics_log.csv", "a") as f:
        f.write(f"{i},{acc},{precision},{recall},{f1},{specificity},{total_loss:.4f}\n")

# gcn.model.eval()
# _, pred =  gcn.model(data.x, data.edge_index).max(dim=1)
# correct = int((pred == data.y).sum())
# acc = correct / data.num_nodes
# print(f"Dokładność klasyfikacji: {acc:.2%}")


# print("Inne testy na pojedyńczym obrazie".center(300, '-'))
# #-----------------------------------------------------------------------------------------------------------------------------
# # 2. Reorganize each image's pixel values into a grid.
# print("2. Wypłaszczamy obraz".center(60, '-'))
# color, w, h = img.shape 
# h_max,h_max = 512, 512 # 16*32, 16*32
# patch_size = 32 # jeszcze myśłałem nad 16 - do sprawdzenie, które lepsze

#  # dostajemy kształy [num_pixels, 3], np [[0.5882, 0.5804, 0.5725], [...]] - poszczególne wartości [x1, x2, x3] 
#  # odpowiadają kolejnym pixelom tego samego koloru z pierwotnego kształtu 
#  # (np. [[R1, R2, R3], [R4, R5, R6], ..., [G1, G2, G3], [G4, G5, G6], ..., [B1, B2, B3], [B4, B5, B6], ...])
#  # nie jest to dla nas odpowiednie rozwiązanie
# # x = img.reshape(w*h, color)

#  # dostajemy kształy [num_pixels, 3], np [[0.5882, 0.6196, 0.7647], [...]] - poszczególne wartości [x1, x2, x3] odpowiadają 
#  # kolejnym wartością RGB danego pixela z pierwotnego kształtu
#  # np. [[R1, G1, B1], [R2, G2, B2], ...]
#  # chcemy właśnie to rozwiązanie
# x = img.permute(1,2,0).reshape(w*h, color)

# # tutaj testowałem sobie kolejne przykłady
# #print(img[0]) print(img[1]) print(img[2]) print(x[0]) print(x2[0]) print(x.shape) print(x2.shape)

# # Normalizacja
# # ...

# #-----------------------------------------------------------------------------------------------------------------------------
# # 3. To generate the picture grid graph, specify a set of edges between neighbouring pixels in the grid.
# print("3. Tworzymy krawędzie".center(60, '-'))

# edge_index = []
# for row in range(h):
#     for col in range(w):
#         idx = row*w + col 
#         # for dx, dy in [(-1, -1), (-1, 0), (-1, -1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]: # łączenie pikseli krawędziami boki + po skosie 
#         for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]: # łączenie pikseli krawędziami same boki 
#             nrow, ncol = row + dx, col + dy 
#             if 0 <= nrow < h and 0 <= ncol < w:
#                 neighbour_idx = row*w + ncol
#                 edge_index.append([idx, neighbour_idx])

# edge_index = torch.tensor(edge_index).t().contiguous()

# # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# data = Data(x=x, edge_index=edge_index).to(gcn.device)

# num_classes = 2
# data.y = torch.randint(0, num_classes, (x.size(0),), dtype=torch.long).to(gcn.device)

# #-----------------------------------------------------------------------------------------------------------------------------
# # 4. To guarantee the stability of the graph convolution process, normalise the adjacency matrix for the picture grid graph.



# #-----------------------------------------------------------------------------------------------------------------------------
# # 5. Supply the GCN model with the picture grid graph and pixel attributes for training and classification.


# # pixels = list(myImage.getdata())
# # width, height = myImage.size
# # pixels = [pixels[i * 4:(i + 1) * 4] for i in range(4)]
# print("4. trenujemy model".center(60, '-'))

# for epoch in range(200):
#    loss = gcn.train(data)
#    if epoch % 20 == 0:
#        print(f"Epoch {epoch} | Loss: {loss:.4f}")

# # ------------------
# # Ewaluacja
# # -------------------
# print("5.ewaluujemy model".center(60, '-'))

# gcn.model.eval()
# _, pred =  gcn.model(data.x, data.edge_index).max(dim=1)
# correct = int((pred == data.y).sum())
# acc = correct / data.num_nodes
# print(f"Dokładność klasyfikacji: {acc:.2%}")