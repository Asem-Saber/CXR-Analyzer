import torch
import torch.nn as nn

class UNET(nn.Module) : 
    def __init__(self , num_classes) : 
        super(UNET , self).__init__() 
        
        self.enc_layer = nn.Sequential(
            nn.Conv2d(3 , 32 , kernel_size = 3 , stride = 1 , padding = 1) , 
            nn.BatchNorm2d(32) , 
            nn.ReLU(inplace = True)
        )
        
        self.encoders = nn.ModuleList([
            self.ConvBlock(32 , 64) , 
            self.ConvBlock(64 , 128) , 
            self.ConvBlock(128 , 256) , 
            self.ConvBlock(256 , 512)
        ])

        self.pool = nn.MaxPool2d(2) 

        self.bottleneck = self.ConvBlock(512, 1024) 

        self.decoders = nn.ModuleList([
            self.ConvBlock(1024, 512) , 
            self.ConvBlock(512, 256) , 
            self.ConvBlock(256, 128) , 
            self.ConvBlock(128, 64)
        ])

        self.UpSamples = nn.ModuleList([
            nn.ConvTranspose2d(1024 , 512 , kernel_size = 2 , stride = 2) , 
            nn.ConvTranspose2d(512 , 256 , kernel_size = 2 , stride = 2) , 
            nn.ConvTranspose2d(256 , 128 , kernel_size = 2 , stride = 2) , 
            nn.ConvTranspose2d(128 , 64 , kernel_size = 2 , stride = 2)
        ])

        self.mask_fc = nn.Conv2d(64, 1, kernel_size=1, stride=1)
        self.cls_fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),         
            nn.Flatten(),                  
            nn.Linear(1024, 256),       
            nn.ReLU(inplace=True),           
            nn.Linear(256, num_classes)      
        )

    def ConvBlock(self , in_channels , out_channels) : 
        return nn.Sequential(
            nn.Conv2d(in_channels , out_channels , kernel_size = 3 , stride = 1 , padding = 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace = True),

            nn.Conv2d(out_channels , out_channels , kernel_size = 3 , stride = 1 , padding = 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace = True)
        )

    def forward(self , x) : 
        resduials = [] 

        # Encoder Block
        x = self.enc_layer(x) 
        
        for enc in self.encoders : 
            x = enc(x)
            resduials.append(x)
            x = self.pool(x)

        x = self.bottleneck(x) 
        x_botttleneck = x

        # Decoder Block 
        for idx , dec in enumerate(self.decoders) : 
            x = self.UpSamples[idx](x)
            resduial = resduials.pop() 
            x = torch.cat([x, resduial], dim=1) 
            x = dec(x) 

        mask_output = self.mask_fc(x) 
        cls_output = self.cls_fc(x_botttleneck)

        return mask_output, cls_output


DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(DEVICE)
NUM_CLASSES = 4

torch.cuda.empty_cache()
model = UNET(num_classes=NUM_CLASSES).to(DEVICE)

# Print model summary
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters     : {total_params:,}")
print(f"Trainable parameters : {trainable_params:,}")


# test
dummy_input = torch.randn(16, 3, 256, 256 , device = DEVICE)
mask_preds, cls_preds = model(dummy_input)

print(f"\nMask Predictions Shape: {mask_preds.shape}") 
print(f"Class Predictions Shape: {cls_preds.shape}") 