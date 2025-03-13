import gradio as gr
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import requests
from io import BytesIO

class VisibleStyleTransfer:
    def __init__(self):
        # Device configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Load pre-trained VGG model
        self.vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features[:16].to(self.device).eval()
        
        # Freeze VGG parameters
        for param in self.vgg.parameters():
            param.requires_grad = False
        
    def preprocess_image(self, image, size=512):
        """Preprocess image for style transfer"""
        # Convert numpy array to PIL if needed
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # Resize image maintaining aspect ratio
        width, height = image.size
        if width > height:
            new_width = size
            new_height = int(height * size / width)
        else:
            new_height = size
            new_width = int(width * size / height)
        
        image = image.resize((new_width, new_height))
        
        # Transform to tensor
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])
        ])
        
        return transform(image).unsqueeze(0).to(self.device)
    
    def gram_matrix(self, input_tensor):
        """Compute Gram matrix for style representation"""
        batch_size, channels, height, width = input_tensor.size()
        features = input_tensor.view(batch_size * channels, height * width)
        G = torch.mm(features, features.t())
        return G.div(batch_size * channels * height * width)
    
    def transfer_style(self, content_img, style_img, steps=100):
        """Perform neural style transfer with increased visibility"""
        # Preprocess images
        content_tensor = self.preprocess_image(content_img)
        style_tensor = self.preprocess_image(style_img)
        
        # Initialize target as content image
        target = content_tensor.clone().requires_grad_(True)
        
        # Setup optimizer with higher learning rate
        optimizer = optim.Adam([target], lr=0.05)
        
        # Emphasize style over content
        content_weight = 1e4  # Reduced
        style_weight = 1e7    # Increased dramatically
        
        # Extract features
        content_features = self.vgg(content_tensor)
        style_features = self.vgg(style_tensor)
        
        # Compute style Gram matrices
        style_gram = self.gram_matrix(style_features)
        
        # Optimization loop with more steps
        for i in range(steps):
            optimizer.zero_grad()
            
            # Current features
            current_features = self.vgg(target)
            
            # Content loss
            content_loss = F.mse_loss(current_features, content_features)
            
            # Style loss using Gram matrix
            current_gram = self.gram_matrix(current_features)
            style_loss = F.mse_loss(current_gram, style_gram)
            
            # Total loss with higher style emphasis
            total_loss = content_weight * content_loss + style_we