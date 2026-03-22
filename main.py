import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torchmetrics as tm
from dataset import get_transform, get_dataloaders
from model import UNET
from train import train_per_epoch, LossFunction
from evaluate import evaluation_per_epoch
from inference import inference


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    train_loader, val_loader = get_dataloaders(data_dir= args.data_dir, img_size = args.img_size, batch_size=args.batch_size)
    model = UNET(num_classes = args.num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = LossFunction()
    iou_metric = tm.classification.BinaryJaccardIndex().to(device)
    
    best_loss = float('inf')
    
    for epoch in tqdm(range(args.epochs)):
        epoch_loss, train_mask_loss, train_cls_loss = train_per_epoch(model, train_loader, optimizer, criterion, device)
        
        print(f"\n>>> Running evaluation ...")
        eval_loss, eval_mask_loss, eval_cls_loss, eval_iou = evaluation_per_epoch(model, val_loader, criterion, iou_metric, device)
        
        print(f"""\nEpoch: [{epoch+1}/{args.epochs}]
        Loss: {epoch_loss:.4f}
        Mask Loss: {train_mask_loss:.4f}
        Class Loss: {train_cls_loss:.4f}
        Eval Loss: {eval_loss:.4f}
        Eval Mask Loss: {eval_mask_loss:.4f}
        Eval Class Loss: {eval_cls_loss:.4f}
        Eval IOU: {eval_iou:.4f}
        """)
        
        if eval_loss < best_loss:
            best_loss = eval_loss
            torch.save(model.state_dict(), 'best_model.pth')
            print("Saved best model!")

def infer(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    id2label = {0: 'Normal', 1: 'Lung_Opacity', 2: 'Viral Pneumonia', 3: 'COVID'}
    transforms = get_transform(args.img_size)
    
    model = UNET(in_channels=3, num_classes=4).to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    
    pred_class, conf, mask, overlay = inference(model, args.image, transforms, id2label, device)
    
    print(f"Prediction: {pred_class} (Confidence: {conf*100:.2f}%)")
    
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(overlay)
    ax[0].set_title(f"Overlay: {pred_class}")
    ax[1].imshow(mask, cmap='gray')
    ax[1].set_title("Predicted Mask")
    plt.show()


BASE_DIR = r'COVID-19_Radiography_Dataset'
IMG_SIZE = 256
BATCH_SIZE = 16
NUM_CLASSES = 4
LR = 0.005
EPOCHS = 10
WEIGHTS_PATH = 'best_model.pth'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="COVID-19 Dual Task Pipeline")
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'infer'], help="Mode to run")
    
    # Train arguments
    parser.add_argument('--data_dir', type=str, default=BASE_DIR, help="Path to COVID-19 dataset")
    parser.add_argument('--img_size', type=int, default=IMG_SIZE, help= "img size")
    parser.add_argument('--epochs', type=int, default=EPOCHS, help="Number of training epochs")
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help="Batch size")
    parser.add_argument('--lr', type=float, default=LR, help="Learning rate")
    parser.add_argument('--num_classes', type=str, default= NUM_CLASSES, help= "number of classes")
    
    # Inference arguments
    parser.add_argument('--image', type=str, help="Path to image for inference")
    parser.add_argument('--weights', type=str, default=WEIGHTS_PATH, help="Path to model weights")
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        train(args)
    elif args.mode == 'infer':
        if not args.image:
            print("Error: --image argument is required for inference mode.")
        else:
            infer(args)