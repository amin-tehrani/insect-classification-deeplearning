import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from transformers import AutoTokenizer, AutoModel, ViTModel, ViTConfig
from transformers import AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import scipy.io
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm
import matplotlib.pyplot as plt

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class MultiModalDataset(Dataset):
    """Dataset for DNA sequences and corresponding images"""
    
    def __init__(self, dna_sequences, images, labels, tokenizer, transform=None, max_length=512):
        self.dna_sequences = dna_sequences
        self.images = images
        self.labels = labels
        self.tokenizer = tokenizer
        self.transform = transform
        self.max_length = max_length
        
    def __len__(self):
        return len(self.dna_sequences)
    
    def __getitem__(self, idx):
        # Process DNA sequence
        dna_seq = str(self.dna_sequences[idx])
        # Tokenize DNA sequence
        encoded = self.tokenizer(
            dna_seq,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Process image
        image = self.images[idx]  # Shape: (3, 64, 64)
        image = torch.FloatTensor(image)
        
        # Normalize image to [0, 1] if needed
        if image.max() > 1.0:
            image = image / 255.0
        
        # Apply transforms if provided
        if self.transform:
            image = self.transform(image)
        
        return {
            'input_ids': encoded['input_ids'].squeeze(),
            'attention_mask': encoded['attention_mask'].squeeze(),
            'image': image,
            'labels': torch.LongTensor([self.labels[idx]])
        }

class DNAEncoder(nn.Module):
    """DNA sequence encoder using pre-trained DNA BERT"""
    
    def __init__(self, model_name='zhihan1996/DNA_bert_6', freeze_bert=False):
        super(DNAEncoder, self).__init__()
        self.dna_bert = AutoModel.from_pretrained(model_name)
        
        # Freeze BERT parameters if specified
        if freeze_bert:
            for param in self.dna_bert.parameters():
                param.requires_grad = False
        
        # Add a projection layer to get 768 dimensions
        self.projection = nn.Linear(self.dna_bert.config.hidden_size, 768)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.dna_bert(input_ids=input_ids, attention_mask=attention_mask)
        # Use [CLS] token representation
        pooled_output = outputs.pooler_output
        # Project to desired dimension
        projected = self.projection(pooled_output)
        return self.dropout(projected)

class ImageEncoder(nn.Module):
    """Image encoder using Vision Transformer"""
    
    def __init__(self, image_size=64, patch_size=8, num_channels=3, 
                 hidden_size=768, num_hidden_layers=6, num_attention_heads=12):
        super(ImageEncoder, self).__init__()
        
        # Create ViT configuration for small images
        config = ViTConfig(
            image_size=image_size,
            patch_size=patch_size,
            num_channels=num_channels,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=3072,
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1
        )
        
        self.vit = ViTModel(config)
        
        # Projection to get 1000 dimensions
        self.projection = nn.Linear(hidden_size, 1000)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, pixel_values):
        outputs = self.vit(pixel_values=pixel_values)
        # Use [CLS] token representation
        pooled_output = outputs.pooler_output
        # Project to desired dimension
        projected = self.projection(pooled_output)
        return self.dropout(projected)

class MultiModalClassifier(nn.Module):
    """Multi-modal classifier combining DNA and image features"""
    
    def __init__(self, num_classes, dna_dim=768, image_dim=1000, hidden_dim=512):
        super(MultiModalClassifier, self).__init__()
        
        self.dna_encoder = DNAEncoder()
        self.image_encoder = ImageEncoder()
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(dna_dim + image_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Classification head
        self.classifier = nn.Linear(hidden_dim // 2, num_classes)
        
    def forward(self, input_ids, attention_mask, images):
        # Get DNA embeddings
        dna_features = self.dna_encoder(input_ids, attention_mask)
        
        # Get image embeddings
        image_features = self.image_encoder(images)
        
        # Concatenate features
        combined_features = torch.cat([dna_features, image_features], dim=1)
        
        # Apply fusion layers
        fused_features = self.fusion(combined_features)
        
        # Get predictions
        logits = self.classifier(fused_features)
        
        return logits, dna_features, image_features

def load_data(mat_file_path):
    """Load data from .mat file"""
    mat = scipy.io.loadmat(mat_file_path)
    dna_sequences = mat['all_string_dnas'].flatten()
    images = mat['all_images']  # Shape: (32424, 3, 64, 64)
    
    # Create dummy labels for demonstration (replace with actual labels)
    # Assuming you have labels in your dataset
    num_classes = 10  # Adjust based on your actual number of species
    labels = np.random.randint(0, num_classes, size=len(dna_sequences))
    
    return dna_sequences, images, labels, num_classes

def train_model(model, train_loader, val_loader, num_epochs=10, lr=2e-5):
    """Train the multi-modal model"""
    
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    
    # Calculate total steps for scheduler
    total_steps = len(train_loader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=0.1 * total_steps,
        num_training_steps=total_steps
    )
    
    criterion = nn.CrossEntropyLoss()
    
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        total_train_loss = 0
        correct_train = 0
        total_train = 0
        
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        
        for batch in train_pbar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            images = batch['image'].to(device)
            labels = batch['labels'].squeeze().to(device)
            
            optimizer.zero_grad()
            
            logits, dna_features, image_features = model(input_ids, attention_mask, images)
            loss = criterion(logits, labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            total_train_loss += loss.item()
            _, predicted = torch.max(logits.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            
            train_pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100*correct_train/total_train:.2f}%'
            })
        
        avg_train_loss = total_train_loss / len(train_loader)
        train_accuracy = 100 * correct_train / total_train
        
        # Validation phase
        model.eval()
        total_val_loss = 0
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]')
            
            for batch in val_pbar:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                images = batch['image'].to(device)
                labels = batch['labels'].squeeze().to(device)
                
                logits, dna_features, image_features = model(input_ids, attention_mask, images)
                loss = criterion(logits, labels)
                
                total_val_loss += loss.item()
                _, predicted = torch.max(logits.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
                
                val_pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{100*correct_val/total_val:.2f}%'
                })
        
        avg_val_loss = total_val_loss / len(val_loader)
        val_accuracy = 100 * correct_val / total_val
        
        # Store metrics
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2f}%')
        print(f'Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.2f}%')
        print('-' * 60)
    
    return train_losses, val_losses, train_accuracies, val_accuracies

def plot_training_history(train_losses, val_losses, train_accuracies, val_accuracies):
    """Plot training history"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot losses
    ax1.plot(train_losses, label='Train Loss')
    ax1.plot(val_losses, label='Validation Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot accuracies
    ax2.plot(train_accuracies, label='Train Accuracy')
    ax2.plot(val_accuracies, label='Validation Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

def main():
    # Load data
    print("Loading data...")
    # Replace 'your_data.mat' with the actual path to your .mat file
    dna_sequences, images, labels, num_classes = load_data('your_data.mat')
    
    print(f"Loaded {len(dna_sequences)} samples with {num_classes} classes")
    print(f"DNA sequences shape: {dna_sequences.shape}")
    print(f"Images shape: {images.shape}")
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained('zhihan1996/DNA_bert_6')
    
    # Split data
    train_dna, val_dna, train_images, val_images, train_labels, val_labels = train_test_split(
        dna_sequences, images, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    # Create datasets
    train_dataset = MultiModalDataset(train_dna, train_images, train_labels, tokenizer)
    val_dataset = MultiModalDataset(val_dna, val_images, val_labels, tokenizer)
    
    # Create data loaders
    batch_size = 16  # Adjust based on your GPU memory
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # Initialize model
    model = MultiModalClassifier(num_classes=num_classes)
    model.to(device)
    
    print(f"Model initialized with {sum(p.numel() for p in model.parameters())} parameters")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    
    # Train model
    print("Starting training...")
    train_losses, val_losses, train_accuracies, val_accuracies = train_model(
        model, train_loader, val_loader, num_epochs=10, lr=2e-5
    )
    
    # Plot results
    plot_training_history(train_losses, val_losses, train_accuracies, val_accuracies)
    
    # Save model
    torch.save(model.state_dict(), 'multimodal_species_classifier.pth')
    print("Model saved as 'multimodal_species_classifier.pth'")
    
    return model

def inference_example(model, tokenizer, dna_sequence, image, device):
    """Example inference function"""
    model.eval()
    
    with torch.no_grad():
        # Process DNA sequence
        encoded = tokenizer(
            dna_sequence,
            truncation=True,
            padding='max_length',
            max_length=512,
            return_tensors='pt'
        )
        
        input_ids = encoded['input_ids'].to(device)
        attention_mask = encoded['attention_mask'].to(device)
        
        # Process image
        image_tensor = torch.FloatTensor(image).unsqueeze(0).to(device)
        if image_tensor.max() > 1.0:
            image_tensor = image_tensor / 255.0
        
        # Get predictions
        logits, dna_features, image_features = model(input_ids, attention_mask, image_tensor)
        probabilities = torch.softmax(logits, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1)
        
        return predicted_class.cpu().numpy()[0], probabilities.cpu().numpy()[0]

if __name__ == "__main__":
    # Run the main training pipeline
    trained_model = main()
    
    # Example of how to use the trained model for inference
    # tokenizer = AutoTokenizer.from_pretrained('zhihan1996/DNA_bert_6')
    # example_dna = "ATCGATCGATCG..."  # Your DNA sequence
    # example_image = np.random.rand(3, 64, 64)  # Your image
    # predicted_class, probabilities = inference_example(trained_model, tokenizer, example_dna, example_image, device)
    # print(f"Predicted class: {predicted_class}")
    # print(f"Class probabilities: {probabilities}")