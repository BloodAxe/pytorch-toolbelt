import cv2
import numpy as np
import torch
import albumentations as A


def canny_edges(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    edges = cv2.canny(gray)
    return edges


class CannyModel(nn.Module):
    def __init__(self, input_channels=3, features=16):
        
        self.smooth1 = nn.Conv2d(input_channels, features, kernel_size=3, padding=1)
        self.smooth2 = nn.Conv2d(features, features, kernel_size=3, padding=1)
        
    def forward(self, x):
        x = self.smooth1(x)
        x = self.smooth1(x)
        x = F.selu(x)


class EdgesDataset(Dataset):
    def __init__(self, images_dir, image_size=(224,224)):
        self.images = find_images_in_dir(images_dir)
        self.transform = A.Compose([
            A.RandomCrop(image_size[0], image_size[1])
        ])
        self.normalize = A.Normalize()
        
     def __getitem__(self, index):
        image = read_rgb_image(self.images[index])
        data = self.transform(image=image)        
        data['mask'] = canny_edges(image)
        data = self.normalize(**data)
        return data
        
def main():
    canny_cnn = CannyModel()
    optimizer = Adam(canny_cnn.parameters(), lr=1e-4)
    loss = nn.BCEWithLogits()
    
    train_loader = DataLoader(EdgesDataset(train_images), batch_size=bs, workers=num_workers, shuffle=True, drop_last=True, pin_memory=True)
    valid_loader = DataLoader(EdgesDataset(valid_images), batch_size=bs, workers=num_workers)
    
    for epoch in range(max_epochs):
        pass
        
if __name__ == '__main__':
    main()
