import torch
import time
import os
import sys
import evaluate
from torch.utils.data import Dataset
from transformers import (
    AutoImageProcessor, 
    AutoModelForImageClassification,
    AutoModel,
    TrainingArguments, 
    Trainer,
    DefaultDataCollator
)
import numpy as np

org_model_name = "google/vit-base-patch16-224"

accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return accuracy.compute(predictions=preds, references=labels)

# Custom dataset class
class ImageDataset(Dataset):
    def __init__(self, images, labels, processor, device):
        self.images = images
        self.labels = labels
        self.processor = processor
        self.device = device
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # Convert tensor to numpy array (HWC format, 0-255 range)
        image = self.images[idx].permute(1, 2, 0).cpu().numpy()  # CHW -> HWC
        if image.max() <= 1.0:  # If normalized to [0,1], scale to [0,255]
            image = (image * 255).astype(np.uint8)
        
        # Process image
        encoding = self.processor(images=image, return_tensors="pt")
        
        return {
            'pixel_values': encoding['pixel_values'].squeeze(0),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }


def finetune_on_species(mat: dict, outdir="./vit-finetuned"+str(int(time.time())), model_name=org_model_name, device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):

    # Check if outdir exists, then another outdir
    _i = 2
    while os.path.exists(outdir):
        outdir = "./vit-finetuned"+str(_i)
        _i+=1

    print("Outdir: ", outdir)

    # Load your data
    # Assuming mat is already loaded (e.g., from scipy.io.loadmat or similar)
    all_images = torch.tensor(mat['all_images']).to(device)  # Shape: (32424, 3, 64, 64)
    all_labels = torch.tensor(mat['all_labels'].squeeze()).to(device) 
    train_indices = (mat['train_loc']-1).flatten()  # Get train indices
    val_indices = (mat['val_seen_loc']-1).flatten()      # Get validation indices

    # If labels are one-hot encoded or multi-dimensional, convert to class indices
    if all_labels.dim() > 1:
        labels = torch.argmax(all_labels.view(all_labels.size(0), -1), dim=1)
    else:
        labels = all_labels

    # Setup
    
    num_classes = int(labels.max().item()) + 1  # Automatically determine number of classes

    # Load processor and model
    processor = AutoImageProcessor.from_pretrained(org_model_name)
    model = AutoModelForImageClassification.from_pretrained(
        org_model_name, 
        num_labels=num_classes,
        ignore_mismatched_sizes=True
    ).to(device)

    # Split data using your indices
    train_images = all_images[train_indices]
    train_labels = labels[train_indices].cpu().numpy()
    val_images = all_images[val_indices]
    val_labels = labels[val_indices].cpu().numpy()

    # Create datasets
    train_dataset = ImageDataset(train_images, train_labels, processor, device)
    val_dataset = ImageDataset(val_images, val_labels, processor, device)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=outdir,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=10,
        learning_rate=1e-4,
        logging_steps=50,
        save_strategy="steps",
        save_steps=500,
        eval_strategy="steps",  # Changed from evaluation_strategy
        eval_steps=500,
        save_total_limit=2,
        remove_unused_columns=False,
        push_to_hub=False,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
    )
    # Data collator
    data_collator = DefaultDataCollator()

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,  # Added validation dataset
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    # Fine-tune the model
    print("Starting fine-tuning...")
    trainer.train()

    results = trainer.evaluate()
    print(results)

    # Save the model
    trainer.save_model(outdir+"-final")
    processor.save_pretrained(outdir+"-final")

    print("Fine-tuning completed!")




def evaluate_model(mat, model_name="./vit-finetuned-final", device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    # Load
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModelForImageClassification.from_pretrained(model_name).to(device)

    # Load your data
    # Assuming mat is already loaded (e.g., from scipy.io.loadmat or similar)
    all_images = torch.tensor(mat['all_images']).to(device)  # Shape: (32424, 3, 64, 64)
    all_labels = torch.tensor(mat['all_labels'].squeeze()).to(device) 
    val_indices = (mat['val_seen_loc']-1).flatten()      # Get validation indices

    # If labels are one-hot encoded or multi-dimensional, convert to class indices
    if all_labels.dim() > 1:
        labels = torch.argmax(all_labels.view(all_labels.size(0), -1), dim=1)
    else:
        labels = all_labels

    # Setup

    num_classes = int(labels.max().item()) + 1  # Automatically determine number of classes

    # Split data using your indices
    val_images = all_images[val_indices]
    val_labels = labels[val_indices].cpu().numpy()

    # Create datasets again
    val_dataset = ImageDataset(val_images, val_labels, processor, device)

    training_args = TrainingArguments(
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=3,
        learning_rate=1e-4,
        logging_steps=50,
        save_strategy="steps",
        save_steps=500,
        eval_strategy="steps",  # Changed from evaluation_strategy
        eval_steps=500,
        save_total_limit=2,
        remove_unused_columns=False,
        push_to_hub=False,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
    )

    data_collator = DefaultDataCollator()

    # Create trainer just for evaluation
    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    # Evaluate
    results = trainer.evaluate()
    return results



if __name__ == "__main__":
    assert len(sys.argv) > 1, "Please provide an argument (finetune, evaluate)"
    if sys.argv[1].strip().lower() == "finetune":
        print("Task: finetune ViT")
        print("Loading mat file...")
        from mat import mat
        print("Mat loaded")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Device: ", device)


        finetune_on_species(mat, device=device)
    elif sys.argv[1].strip().lower() == "evaluate":
        if len(sys.argv) > 2:
            model_name = sys.argv[2]
        else:
            model_name = "./vit-finetuned-final"

        print("Task: evaluate ViT, model:", model_name)

        print("Loading mat file...")
        from mat import mat
        print("Mat loaded")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Device: ", device)


        res = evaluate_model(mat, model_name=model_name, device=device)
        print(res)
    else:
        print("Unknown task:", sys.argv[1])
