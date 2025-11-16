import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from utils.seed import set_seed
from data.datasets import ImageDataset, PatchDataset
from training.Trainer import Trainer
from models.classifier import GraphClassifier
from utils.logging import get_logger
from tqdm import tqdm

def collate_list(batch):
    return batch

def main(cfg_path="configs/default.yaml"):
    cfg = OmegaConf.load(cfg_path) 
    set_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger = get_logger(cfg.log_output_path)


    base_train = ImageDataset(cfg.data.root.train, cfg.data.max_img_no)
    base_val = ImageDataset(cfg.data.root.val, cfg.data.max_img_no)


    train_ds = PatchDataset(base_train, cfg.data.img_Height, cfg.data.img_Width, cfg.data.patch_size, cfg.data.mean, cfg.data.std)
    val_ds = PatchDataset(base_val, cfg.data.img_Height, cfg.data.img_Width, cfg.data.patch_size, cfg.data.mean, cfg.data.std)


    train_loader = DataLoader(train_ds, batch_size=cfg.train.batch_size_images, shuffle=True, collate_fn=collate_list)
    val_loader = DataLoader(val_ds, batch_size=cfg.train.batch_size_images, shuffle=False, collate_fn=collate_list)


    model = GraphClassifier(cfg).to(device)


    optim = torch.optim.AdamW(filter(lambda x: x.requires_grad, model.parameters()), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=len(train_loader)*cfg.train.epochs)
    criterion = torch.nn.CrossEntropyLoss()


    trainer = Trainer(model, optim, scheduler)

    logger.info(f"START TRAIN")
    for epoch in range(cfg.train.epochs):
        model.train()
        for batch in tqdm(train_loader):
            loss = trainer.step_batch(batch, criterion)
        metrics = trainer.evaluate(val_loader)
        logger.info(f"Epoch {epoch}: acc={metrics['acc']:.4f}, auc={metrics['auc']:.4f}, f1={metrics['f1']:.4f}")


if __name__ == "__main__":
    main()