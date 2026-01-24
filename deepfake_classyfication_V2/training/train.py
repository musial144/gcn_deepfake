import torch
from omegaconf import OmegaConf, DictConfig
from torch.utils.data import DataLoader
from utils.seed import set_seed
from data.datasets import ImageDataset, PatchDataset
from training.Trainer import Trainer
from models.classifier import GraphClassifier
from utils.logging import get_logger
from tqdm import tqdm
from pathlib import Path
import os


# funkcja do collate w DataLoader - zwraca listę batchy bez zmian
def collate_list(batch):
    return batch

def main(cfg_path="configs/default.yaml"):
    # ładujemy ustawienia początkowe - plik config, logger, device
    cfg = OmegaConf.load(cfg_path) 
    set_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger = get_logger(cfg.log_output_path)

    logger.info("-".ljust(49, 'x'))
    logger.info("CONFIG:\n" + OmegaConf.to_yaml(cfg))

    # ładujemy dane - obrazy prawidzwych i fałszywych
    base_train = ImageDataset(cfg.data.root.train, cfg.data.max_img_no)
    base_val = ImageDataset(cfg.data.root.val, cfg.data.max_img_no)
    base_test = ImageDataset(cfg.data.root.test, cfg.data.max_img_no)


    # zamieniamy obrazy na listę patchy i koordynatów patchy
    train_ds = PatchDataset(base_train, cfg.data.img_Height, cfg.data.img_Width, cfg.data.patch_size, cfg.data.mean, cfg.data.std)
    val_ds = PatchDataset(base_val, cfg.data.img_Height, cfg.data.img_Width, cfg.data.patch_size, cfg.data.mean, cfg.data.std)
    test_ds = PatchDataset(base_test, cfg.data.img_Height, cfg.data.img_Width, cfg.data.patch_size, cfg.data.mean, cfg.data.std)


    # liste patchy zamieniamy na dataloadery
    train_loader = DataLoader(train_ds, batch_size=cfg.train.batch_size_images, shuffle=True, collate_fn=collate_list)
    val_loader = DataLoader(val_ds, batch_size=cfg.train.batch_size_images, shuffle=False, collate_fn=collate_list)
    test_loader = DataLoader(test_ds, batch_size=cfg.train.batch_size_images, shuffle=False, collate_fn=collate_list)

    logger.info(f"len(train_loader)={len(train_loader)} len(val_loader)={len(val_loader)}")

    # definiujemy model, optymalizator, scheduler, kryterium straty i inne zmienne pomocnicze   
    model = GraphClassifier(cfg).to(device)

    steps_per_epoch = len(train_loader)

    optim = torch.optim.AdamW(filter(lambda x: x.requires_grad, model.parameters()), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=len(train_loader)*cfg.train.epochs)
    scheduler =torch.optim.lr_scheduler.CyclicLR(optim, base_lr=cfg.train.lr/10, max_lr=cfg.train.lr, step_size_up=steps_per_epoch*5, mode="triangular2", cycle_momentum=False, )
    criterion = torch.nn.CrossEntropyLoss()

    best_auc = 0.0
    start_epoch = 0
    save_model_path = Path(cfg.output_dir) / cfg.model_to_save_path
    save_model_path.parent.mkdir(parents=True, exist_ok=True)

    # jeżeli istnieje zapisany model, ładujemy go i kontynuujemy trenowanie od zapisanego epoch'a
    try:
        # model.load_state_dict(torch.load(Path(cfg.output_dir) / cfg.saved_model_path, map_location="cpu"))
        ckpt = torch.load(save_model_path, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        optim.load_state_dict(ckpt["optim"])
        scheduler.load_state_dict(ckpt["scheduler"])
        best_auc = ckpt.get("best_auc", 0.0)
        start_epoch = ckpt.get("epoch", -1) + 1
        logger.info(f"Resumed from {save_model_path} at epoch={start_epoch}, best_auc={best_auc:.4f}")
    except:
        logger.info(f"Error during loading model from {save_model_path}. Training from scratch.")

    # ładujemy model do trenera. Dopiero ewentualnym wczytaniu starego modelu
    trainer = Trainer(model, optim, scheduler)

    # rozpoczynamy trenowanie modelu
    logger.info(f"START TRAIN")
    # for epoch in range(cfg.train.epochs):
    for epoch in range(start_epoch, cfg.train.epochs):
        # jeżeli w confugi ustawiliśmy, że nie chcemy treningu to opuszczamy pętlę trenowania
        if not cfg.is_train_available:
            logger.info(f"STOP TRAIN - cfg.is_train_available <> True")
            break

        epoch_loss = 0.0
        n_batches = 0

        # trenowanie jednego epoch'a. Progress bar wskazuje postęp trenowania w danym epoch'u (tqdm)
        for batch in tqdm(train_loader):
            loss = trainer.step_batch(batch, criterion)

            scheduler.step()

            epoch_loss += loss
            n_batches += 1
            
            # do debugowania, gdy na początku mi się nie uczyło - grad był cały czas = 0
            # total_norm = 0.0
            # for p in model.parameters():
            #     if p.grad is not None:
            #         total_norm += p.grad.data.norm(2).item()
            # print(f"[DEBUG] grad_norm={total_norm:.4f}")

        # liczymy metryki po epoch'u trenowania
        avg_loss = epoch_loss / n_batches
        print(f"[DEBUG] epoch {epoch} train loss = {avg_loss:.4f}")

        metrics = trainer.evaluate(val_loader)
        logger.info(f"Epoch {epoch}: acc={metrics['acc']:.4f}, auc={metrics['auc']:.4f}, f1={metrics['f1']:.4f}")

        # zapisujemy model jeżeli uzyskał najlepszy wynik AUC na walidacji
        if metrics["auc"] > best_auc:
                best_auc = metrics["auc"]
                # torch.save(model.state_dict(), save_model_path)
                torch.save({
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optim": optim.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "best_auc": best_auc,
                    "cfg_path": cfg_path,
                }, save_model_path)
                logger.info(f"Saved best model (AUC={best_auc:.4f}) to {save_model_path}")

    # po zakończeniu trenowania wykonujemy ewaluację na teście
    if not cfg.is_train_available:
        logger.info(f"START TEST")
        metrics = trainer.evaluate(test_loader)
        logger.info(f"acc={metrics['acc']:.4f}, auc={metrics['auc']:.4f}, f1={metrics['f1']:.4f}")

if __name__ == "__main__":
    main()