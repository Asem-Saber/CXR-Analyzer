import numpy as np
import pandas as pd
import os 
import glob as gb
from tqdm import tqdm

import torch
from torch.utils.data import Dataset , DataLoader
import torchvision as tv
from torchvision import tv_tensors
from torchvision.transforms import v2 as T

def read_dataset(data_dir) : 
    class_folders = [f for f in os.listdir(data_dir)]

    id2label = {k:v for k,v in enumerate(class_folders)}
    label2id = {v:k for k,v in enumerate(class_folders)}

    data = []
    for class_name in tqdm(class_folders):
        images_dir = os.path.join(data_dir, class_name, 'images')
        masks_dir = os.path.join(data_dir, class_name, 'masks')
        
        if os.path.exists(images_dir) and os.path.exists(masks_dir):
            image_filenames = [f for f in os.listdir(images_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
            
            for img_name in image_filenames:
                img_path = os.path.join(images_dir, img_name)
                mask_path = os.path.join(masks_dir, img_name) 
                
                if os.path.exists(mask_path):
                    data.append({
                        'img_path': img_path,
                        'mask_path': mask_path,
                        'class': class_name,
                        'class_id': label2id[class_name]
                    }) 

    return pd.DataFrame(data)



def get_transform(img_size): 
    transforms = []
    transforms.append(T.Resize((img_size, img_size), antialias=True))
    transforms.append(T.ToDtype(torch.float, scale=True))
    transforms.append(T.ToPureTensor()) 
    
    return T.Compose(transforms)


class CustomDataset(Dataset): 
    def __init__(self, df, transforms = None): 
        self.df = df
        self.transforms = transforms

    def __len__(self): 
        return len(self.df)

    def __getitem__(self, idx): 
        row = self.df.iloc[idx]

        image = tv.io.read_image(row['img_path'], mode = tv.io.ImageReadMode.RGB)
        mask = tv.io.read_image(row['mask_path'], mode = tv.io.ImageReadMode.GRAY)

        image = tv_tensors.Image(image)
        mask = tv_tensors.Mask(mask)

        if self.transforms is not None:
            image, mask = self.transforms(image, mask)

        mask = (mask > 127).float()

        label = torch.tensor(row['class_id'], dtype=torch.long)
        
        return image, mask, label


def get_dataloaders(data_dir, img_size, batch_size = 16):
    df = read_dataset(data_dir)
    ds = CustomDataset(df , get_transform(img_size))

    indices = torch.randperm(len(ds)).tolist()
    num_train = int(0.8 * len(indices))
    train_ds = torch.utils.data.Subset(ds , indices[:num_train])
    val_ds = torch.utils.data.Subset(ds , indices[num_train:])

    train_loader = DataLoader(
        train_ds,
        batch_size = batch_size,
        shuffle = True,
        pin_memory = True,
    )

    val_loader = DataLoader(
        val_ds, 
        batch_size = batch_size,
        shuffle = False,
        pin_memory = True,
    )

    return train_loader, val_loader