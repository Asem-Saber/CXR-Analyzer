import torch
import torch.nn as nn
from tqdm import tqdm


class LossFunction(nn.Module): 
    def __init__(self, mask_weight=1.0, cls_weight=1.0): 
        super(LossFunction, self).__init__()
        self.mask_weight = mask_weight
        self.cls_weight = cls_weight

        self.mask_loss_fn = nn.BCEWithLogitsLoss()
        self.cls_loss_fn = nn.CrossEntropyLoss()

    def forward(self, mask_preds, mask_targets, cls_preds, cls_targets): 
        mask_loss = self.mask_loss_fn(mask_preds, mask_targets)
        cls_loss = self.cls_loss_fn(cls_preds, cls_targets)
        
        total_loss = (self.mask_weight * mask_loss) + (self.cls_weight * cls_loss)
        
        return total_loss, mask_loss, cls_loss

def train_per_epoch(model, data_loader, optimizer, loss_fn, device): 
    # TRAINING
    model.train()
    epoch_loss, epoch_mask_loss, epoch_cls_loss = 0.0, 0.0, 0.0

    for images, masks, labels in data_loader: 
        optimizer.zero_grad()

        images, masks, labels = images.to(device), masks.to(device), labels.to(device)
        mask_preds , label_preds = model(images)

        total_loss, mask_loss, cls_loss = loss_fn(mask_preds, masks, label_preds, labels)

        total_loss.backward()
        optimizer.step()

        epoch_loss += total_loss.item()
        epoch_mask_loss += mask_loss.item()
        epoch_cls_loss += cls_loss.item()

    epoch_loss /= len(data_loader)
    epoch_mask_loss /= len(data_loader)
    epoch_cls_loss /= len(data_loader)
    return epoch_loss, epoch_mask_loss, epoch_cls_loss

