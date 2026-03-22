import numpy as np
from tqdm import tqdm
import torch
import torchvision as tv
from torchvision import tv_tensors
from torchvision.transforms import v2 as T

def inference(model, img_path, transforms, id2label, device): 
    image = tv.io.read_image(img_path, mode=tv.io.ImageReadMode.RGB)
    image = tv_tensors.Image(image)
    
    image = transforms(image)
    image_tensor = image.unsqueeze(0).to(device)

    model.eval()
    with torch.inference_mode(): 
        mask_prediction , class_prediction = model(image_tensor)

    # class prediction probability
    class_probs = torch.softmax(class_prediction, dim=1)

    # class prediction highest probability
    pred_class_id = torch.argmax(class_probs, dim=1).item()

    confidence = class_probs[0, pred_class_id].item()
    pred_class_name = id2label[pred_class_id]

    mask_prob = torch.sigmoid(mask_prediction)
    mask_pred = (mask_prob > 0.5).float()
    mask_pred = mask_pred.squeeze().cpu().numpy()

    image_uint8 = (image * 255).to(torch.uint8)
    mask = torch.from_numpy(mask_pred)
    mask = mask.to(dtype = torch.bool)

    overlay = tv.utils.draw_segmentation_masks(
        image = image_uint8 , 
        masks = mask , 
        alpha = 0.5 , 
        colors = ["blue"]
    )

    overlay = overlay.permute(1, 2, 0).numpy()

    return pred_class_name, confidence, mask_pred, overlay