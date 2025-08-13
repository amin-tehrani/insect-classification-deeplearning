import torch
import time
import os
import sys
import evaluate
import gc
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModel,
    AutoModelForSequenceClassification,
    TrainingArguments, 
    Trainer,
    DefaultDataCollator
)
import numpy as np
from dotenv import load_dotenv

load_dotenv()  # load environment variables

org_model_name = "InstaDeepAI/nucleotide-transformer-v2-100m-multi-species"

accuracy = evaluate.load("accuracy")

def clear_memory():
    """Clear GPU memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return accuracy.compute(predictions=preds, references=labels)

# Custom dataset class for DNA sequences
class DNADataset(Dataset):
    def __init__(self, dna_strings, labels, tokenizer, max_length=1600):
        self.dna_strings = dna_strings
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length or tokenizer.model_max_length
    
    def __len__(self):
        return len(self.dna_strings)
    
    def __getitem__(self, idx):
        dna_sequence = self.dna_strings[idx].strip()
        label = self.labels[idx]
        
        # Tokenize the DNA sequence
        encoding = self.tokenizer(
            dna_sequence,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long)
        }


def finetune_on_species(mat: dict, outdir="./dnaencoder-finetuned"+str(int(time.time())), model_name=org_model_name, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    
    # Check if outdir exists, then create another outdir
    _i = 2
    while os.path.exists(outdir):
        outdir = "./dnabert-finetuned" + str(_i)
        _i += 1

    print("Outdir: ", outdir)

    # Get HuggingFace token
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        print("Warning: HF_TOKEN not found in environment variables")

    # Load your data
    dna_strings = mat['all_string_dnas'].squeeze()
    all_labels = mat['all_labels'].squeeze()
    train_indices = (mat['train_loc'] - 1).flatten()  # Get train indices
    val_indices = (mat['val_seen_loc'] - 1).flatten()  # Get validation indices

    # Convert labels to class indices if needed
    if len(all_labels.shape) > 1:
        labels = np.argmax(all_labels, axis=1) if isinstance(all_labels, np.ndarray) else torch.argmax(all_labels, dim=1).cpu().numpy()
    else:
        labels = all_labels

    # Setup
    num_classes = int(labels.max()) + 1  # Automatically determine number of classes
    print(f"Number of classes: {num_classes}")

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(org_model_name, trust_remote_code=True, token=hf_token)
    model = AutoModelForSequenceClassification.from_pretrained(
        org_model_name, 
        trust_remote_code=True,
        token=hf_token,
        num_labels=num_classes,
        ignore_mismatched_sizes=True
    ).to(device)

    clear_memory()

    # Split data using your indices
    train_dna_strings = dna_strings[train_indices]
    train_labels = labels[train_indices]
    val_dna_strings = dna_strings[val_indices]
    val_labels = labels[val_indices]

    print(f"Train samples: {len(train_dna_strings)}")
    print(f"Validation samples: {len(val_dna_strings)}")

    # Create datasets
    train_dataset = DNADataset(train_dna_strings, train_labels, tokenizer)
    val_dataset = DNADataset(val_dna_strings, val_labels, tokenizer)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=outdir,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=10,
        learning_rate=2e-5,
        weight_decay=0.01,
        logging_steps=50,
        save_strategy="steps",
        save_steps=500,
        eval_strategy="steps",
        eval_steps=100,
        save_total_limit=2,
        remove_unused_columns=False,
        push_to_hub=False,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        logging_dir=os.path.join(outdir, 'logs'),
        gradient_accumulation_steps=4,
        fp16=True,
    )

    # Data collator
    data_collator = DefaultDataCollator()

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    # Fine-tune the model
    print("Starting fine-tuning...")
    trainer.train()

    results = trainer.evaluate()
    print("Final evaluation results:", results)

    # Save the model
    final_outdir = outdir + "-final"
    trainer.save_model(final_outdir)
    tokenizer.save_pretrained(final_outdir)

    print("Fine-tuning completed!")
    print(f"Model saved to: {final_outdir}")


# Load accuracy metric



def evaluate_model(mat, model_name="./dnaencoder-finetuned-final", device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    
    print(f"Evaluating model: {model_name}")
    
    # Get HuggingFace token
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        print("Warning: HF_TOKEN not found in environment variables")

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, token=hf_token)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, trust_remote_code=True, token=hf_token).to(device)

    # Load your data
    dna_strings = mat['all_string_dnas'].squeeze()
    all_labels = mat['all_labels'].squeeze()
    val_indices = (mat['val_seen_loc'] - 1).flatten()  # Get validation indices

    # Convert labels to class indices if needed
    if len(all_labels.shape) > 1:
        labels = np.argmax(all_labels, axis=1) if isinstance(all_labels, np.ndarray) else torch.argmax(all_labels, dim=1).cpu().numpy()
    else:
        labels = all_labels

    # Setup
    num_classes = int(labels.max()) + 1
    print(f"Number of classes: {num_classes}")

    # Split data using your indices
    val_dna_strings = dna_strings[val_indices]
    val_labels = labels[val_indices]

    print(f"Validation samples: {len(val_dna_strings)}")

    # Create dataset
    val_dataset = DNADataset(val_dna_strings, val_labels, tokenizer)

    # Training arguments (needed for Trainer even in evaluation mode)
    training_args = TrainingArguments(
        output_dir="./eval_temp",
        per_device_eval_batch_size=16,
        remove_unused_columns=False,
        dataloader_pin_memory=False,
    )

    # Data collator
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
    print("Starting evaluation...")
    results = trainer.evaluate()
    
    return results


if __name__ == "__main__":
    # set export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    # set export CUDA_VISIBLE_DEVICES=0,1,2,3
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    # os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0,1,2,3")
    

    assert len(sys.argv) > 1, "Please provide an argument (finetune, evaluate)"
    
    if sys.argv[1].strip().lower() == "finetune":
        print("Task: finetune DNABERT")
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
            model_name = "./dnabert-finetuned-final"

        print("Task: evaluate DNABERT, model:", model_name)

        print("Loading mat file...")
        from mat import mat
        print("Mat loaded")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Device: ", device)

        res = evaluate_model(mat, model_name=model_name, device=device)
        print("Evaluation results:")
        print(res)
        
    else:
        print("Unknown task:", sys.argv[1])
        print("Available tasks: finetune, evaluate")