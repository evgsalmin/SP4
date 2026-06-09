import os
import random
from functools import partial

import numpy as np

import timm
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

import torchmetrics

from transformers import AutoModel, AutoTokenizer

from scripts.dataset import MultimodalDataset, collate_fn, get_transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau


def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = True


def set_requires_grad(module: nn.Module, unfreeze_pattern="", verbose=False):
    if len(unfreeze_pattern) == 0:
        for _, param in module.named_parameters():
            param.requires_grad = False
        return

    pattern = unfreeze_pattern.split("|")

    for name, param in module.named_parameters():
        if any([name.startswith(p) for p in pattern]):
            param.requires_grad = True
            if verbose:
                print(f"Разморожен слой: {name}")
        else:
            param.requires_grad = False


class MultimodalModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.text_model = AutoModel.from_pretrained(config.TEXT_MODEL_NAME)
        self.image_model = timm.create_model(
            config.IMAGE_MODEL_NAME,
            pretrained=True,
            num_classes=0
        )

        self.text_proj = nn.Linear(self.text_model.config.hidden_size, config.HIDDEN_DIM)
        self.image_proj = nn.Linear(self.image_model.num_features, config.HIDDEN_DIM)

        self.head = nn.Linear((config.HIDDEN_DIM+1), 1)

    def forward(self, input_ids, attention_mask, image, mass):
        text_features = self.text_model(input_ids, attention_mask).last_hidden_state[:,  0, :]
        image_features = self.image_model(image)

        text_emb = self.text_proj(text_features)
        image_emb = self.image_proj(image_features)
        fused_emb = text_emb * image_emb
        
        if mass.dim() == 1:
            mass = mass.unsqueeze(1)
        
        x = torch.cat((fused_emb, mass), dim=1)

        logits = self.head(x)
        return logits


def train(config, device):
    seed_everything(config.SEED)

    # Инициализация модели
    model = MultimodalModel(config).to(device)
    tokenizer = AutoTokenizer.from_pretrained(config.TEXT_MODEL_NAME)

    set_requires_grad(model.text_model,
                      unfreeze_pattern=config.TEXT_MODEL_UNFREEZE, verbose=True)
    set_requires_grad(model.image_model,
                      unfreeze_pattern=config.IMAGE_MODEL_UNFREEZE, verbose=True)

    # Оптимизатор с разными LR
    optimizer = AdamW([
                {"params": model.text_model.parameters(), "lr": config.TEXT_LR},
                {"params": model.image_model.parameters(), "lr": config.IMAGE_LR},
                {"params": model.text_proj.parameters(), "lr": config.CLASSIFIER_LR},
                {"params": model.image_proj.parameters(), "lr": config.CLASSIFIER_LR},
                {"params": model.head.parameters(), "lr": config.CLASSIFIER_LR},
    ])

    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    criterion_mean = nn.L1Loss(reduction='mean')
    criterion_sum = nn.L1Loss(reduction='sum')

    # Загрузка данных
    transforms = get_transforms(config)
    val_transforms = get_transforms(config, ds_type="val")
    train_dataset = MultimodalDataset(config, transforms)
    val_dataset = MultimodalDataset(config, val_transforms, ds_type="val")
    train_loader = DataLoader(train_dataset,
                              batch_size=config.BATCH_SIZE,
                              shuffle=True,
                              collate_fn=partial(collate_fn,
                                                 tokenizer=tokenizer))
    val_loader = DataLoader(val_dataset,
                            batch_size=config.BATCH_SIZE,
                            shuffle=False,
                            collate_fn=partial(collate_fn,
                                               tokenizer=tokenizer))

    # инициализируем метрику
    
    best_mae_val = float('inf')

    print("training started")
    for epoch in range(config.EPOCHS):
        model.train()
        
        total_train_loss = 0.0
        total_train_samples = 0

        for batch in train_loader:
            # Подготовка данных
            inputs = {
                'input_ids': batch['input_ids'].to(device),
                'attention_mask': batch['attention_mask'].to(device),
                'image': batch['image'].to(device),
                'mass': batch['mass'].to(device)
            }
            labels = batch['label'].to(device)

            # Forward
            optimizer.zero_grad()
            logits = model(**inputs)
            loss = criterion_mean(logits.view(-1), labels.view(-1))

            # Backward
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                loss_sum = criterion_sum(logits.view(-1), labels.view(-1))
                total_train_loss += loss_sum.item()
            total_train_samples += labels.size(0)

            

        # Валидация
        val_mae = validate(model, val_loader, device)

        scheduler.step(val_mae)
        
        avg_train_loss = total_train_loss / total_train_samples
        
        print(
            f"Epoch {epoch}/{config.EPOCHS-1} | avg_Loss: {avg_train_loss:.4f} | Val MAE: {val_mae:.4f}"
        )

        if val_mae < best_mae_val:
            print(f"New best model, epoch: {epoch}")
            best_mae_val = val_mae
            torch.save(model.state_dict(), config.SAVE_PATH)

        if val_mae < 50.0:
            print(f"\n[ЦЕЛЕВАЯ МЕТРИКА ДОСТИГНУТА]: Обучение остановлено досрочно на эпохе {epoch}.")
            print(f"Итоговый Val MAE: {val_mae:.4f}")
            break


def validate(model, val_loader, device):
    model.eval()
    criterion_sum = nn.L1Loss(reduction='sum')
    
    total_absolute_error = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch in val_loader:
            inputs = {
                'input_ids': batch['input_ids'].to(device),
                'attention_mask': batch['attention_mask'].to(device),
                'image': batch['image'].to(device),
                'mass': batch['mass'].to(device)
            }
            labels = batch['label'].to(device)

            logits = model(**inputs)
            loss_sum = criterion_sum(logits.squeeze(), labels.squeeze())
            
            total_absolute_error += loss_sum.item()
            total_samples += labels.size(0)
            

    return total_absolute_error / total_samples

