import torch 
import model_GCN as gcn
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image, ImageDraw
from datetime import datetime
import glob 
import gc
import settings as set


FAKE_DIR, REAL_DIR = set.FAKE_DIR, set.REAL_DIR
FAKE_DIR_out, REAL_DIR_out = set.FAKE_DIR_out, set.REAL_DIR_out
FAKE_DIR_TRAIN_out, FAKE_DIR_TEST_out = set.FAKE_DIR_TRAIN_out, set.FAKE_DIR_TEST_out
REAL_DIR_TRAIN_out, REAL_DIR_TEST_out = set.REAL_DIR_TRAIN_out, set.REAL_DIR_TEST_out

FAKE_LABEL = 0
REAL_LABEL = 1

PATCH_SIZE = 16 #16 # 32
IMAGE_SIZE = 256#512 #128

print("Liczba FAKE obrazów:", len(glob.glob(FAKE_DIR)))
print("Liczba REAL obrazów:", len(glob.glob(REAL_DIR)))

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

model_mode = input('Chcesz: \n 1. trenować \n 2. evaluować \n 3. ekstrachować cechy za pomocą resnet18 \n 4. Train pretrained? \n')

print("1. Tworzymy krawędzie".center(60, '-'))

edge_index = gcn.create_edges(IMAGE_SIZE, PATCH_SIZE)

gcn.load_model()

if model_mode == '3':
    print("2. wczytujemy obrazy".center(60, '-'))
    i = 0
    while True:
        start_pic, end_pic = 100*i, 100*i+99
        patches_fake = gcn.load_pictures(FAKE_DIR, transform, IMAGE_SIZE, PATCH_SIZE, start_pic, end_pic)
        patches_real = gcn.load_pictures(REAL_DIR, transform, IMAGE_SIZE, PATCH_SIZE, start_pic, end_pic)

        # undersampling 
        if not patches_fake or not patches_real:
            break

        all_img_x_fake = gcn.extract_features(patches_fake, IMAGE_SIZE, PATCH_SIZE)
        all_img_x_real = gcn.extract_features(patches_real, IMAGE_SIZE, PATCH_SIZE)

        gcn.save_features(all_img_x_fake, FAKE_DIR_out, start_pic)
        gcn.save_features(all_img_x_real, REAL_DIR_out, start_pic)

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        i += 1

min_loss_glob = 1000000
min_val_loss = 1000000

for i in range(0, 100):
    # if i > 0:
    print ('i: {}, czas: {}'.format(i, datetime.now()))
    print (f"i: {i}, lr: {gcn.optimizer.param_groups[0]['lr']:.6e}")

    if model_mode == '3':
        break

    start_pic_idx, end_pic_idx = 1000*i, 1000*i+999

    train_img_x_fake = gcn.load_features(FAKE_DIR_TRAIN_out, start_pic_idx, end_pic_idx)
    train_img_x_real = gcn.load_features(REAL_DIR_TRAIN_out, start_pic_idx, end_pic_idx)

    test_img_x_fake = gcn.load_features(FAKE_DIR_TEST_out, 0, 999)
    test_img_x_real = gcn.load_features(REAL_DIR_TEST_out, 0, 999)

    if not train_img_x_fake or not train_img_x_real:
       break
    
    data_train = gcn.create_dataset(train_img_x_fake, train_img_x_real, PATCH_SIZE, IMAGE_SIZE, FAKE_LABEL, REAL_LABEL, edge_index, True)
    data_test =  gcn.create_dataset(test_img_x_fake, test_img_x_real, PATCH_SIZE, IMAGE_SIZE, FAKE_LABEL, REAL_LABEL, edge_index, False)
    
    for g in gcn.optimizer.param_groups:
        g['lr'] = gcn.init_lr

    total_loss = 0 
    min_loss = 1000000000
    # patience = 10

    if model_mode in ('1', '4'):
        print("5. Trenujemy model".center(60, '-'))
        
        patience = 11

        epoch_num = 41
        if model_mode == '4':
            epoch_num = 21

        for epoch in range(epoch_num):
            total_loss = 0 
            loss = 0
            for batch in data_train:
                loss = gcn.train(batch)
                total_loss += loss

            acc, precision, recall, f1, specificity, val_loss = gcn.evaluate(data_test)
            gcn.scheduler.step(val_loss)

            if val_loss < min_loss:
            # if val_loss < min_val_loss:
                min_loss = val_loss
                # min_val_loss = val_loss
                patience = 11
                # gcn.save_model()
            else:
                patience -= 1
            
            if patience == 0:
                break

            if epoch % 10 == 0:
                print(f"Epoch {epoch} | total_Loss: {total_loss:.5f} | loss: {loss:.5f}")
                print(f"evaluacja: {i},{acc:.2%},{precision:.2%},{recall:.2%},{f1:.2%},{specificity:.2%},{val_loss:.5f}\n")

    acc_avg, precision_avg, recall_avg, f1_avg, specificity_avg, val_loss_avg = [0.0] * 6
    max_j = 0
    
    print("5.ewaluujemy model".center(60, '-'))
    for j in range(0, 100):
        max_j = j + 1

        start_pic_idx, end_pic_idx = 1000*j, 1000*j+999

        test_img_x_fake = gcn.load_features(FAKE_DIR_TEST_out, start_pic_idx, end_pic_idx)
        test_img_x_real = gcn.load_features(REAL_DIR_TEST_out, start_pic_idx, end_pic_idx)

        if not test_img_x_fake or not test_img_x_real:
            break

        data_test = gcn.create_dataset(test_img_x_fake, test_img_x_real, PATCH_SIZE, IMAGE_SIZE, FAKE_LABEL, REAL_LABEL, edge_index, False)

        acc, precision, recall, f1, specificity, val_loss = gcn.evaluate(data_test)
        
        acc_avg         += acc
        precision_avg   += precision
        recall_avg      += recall
        f1_avg          += f1 
        specificity_avg += specificity
        val_loss_avg    += val_loss

    acc_avg         /= max_j
    precision_avg   /= max_j
    recall_avg      /= max_j
    f1_avg          /= max_j
    specificity_avg /= max_j
    val_loss_avg    /= max_j

    if val_loss_avg < min_val_loss:
        min_val_loss = val_loss_avg
        gcn.save_model()

    with open("metrics_log.csv", "a") as f:
        print(f"evaluacja zbioru: {i},{acc_avg:.2%},{precision_avg:.2%},{recall_avg:.2%},{f1_avg:.2%},{specificity_avg:.2%},{val_loss_avg:.5f}\n")
        f.write(f"{i},{acc_avg:.2%},{precision_avg:.2%},{recall_avg:.2%},{f1_avg:.2%},{specificity_avg:.2%},{val_loss_avg:.5f}\n")
    
    gcn.clear_memory()
