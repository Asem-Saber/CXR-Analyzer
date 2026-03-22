import torch
from tqdm import tqdm

def evaluation_per_epoch(model, data_loader, loss_fn, metric, device): 
    model.eval()
    metric.reset()
    with torch.inference_mode(): 
        eval_loss, epoch_mask_loss, epoch_cls_loss = 0.0, 0.0, 0.0
        for images, masks, labels in data_loader: 
            images, masks, labels = images.to(device), masks.to(device), labels.to(device)
            mask_preds , label_preds = model(images)

            total_loss, mask_loss, cls_loss = loss_fn(mask_preds, masks, label_preds, labels)

            eval_loss += total_loss.item()
            epoch_mask_loss += mask_loss.item()
            epoch_cls_loss += cls_loss.item()

            mask_preds = (torch.sigmoid(mask_preds) > 0.5).float()
            metric.update(mask_preds, masks)

    eval_loss /= len(data_loader)
    epoch_mask_loss /= len(data_loader)
    epoch_cls_loss /= len(data_loader)
    eval_iou = metric.compute()
    eval_iou = eval_iou.item()
    return eval_loss, epoch_mask_loss, epoch_cls_loss, eval_iou