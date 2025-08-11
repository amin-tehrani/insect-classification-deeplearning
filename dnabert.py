
# Load model directly
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification, Trainer, TrainingArguments
import os
from dataset import DNADataset
import torch 

load_dotenv()  # take environment variables

hf_token = os.getenv("HF_TOKEN")
tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNA_bert_6", trust_remote_code=True, token=hf_token)
model_org = AutoModel.from_pretrained("zhihan1996/DNA_bert_6", trust_remote_code=True, token=hf_token)


# dna = "ACGTAGCATCGGATCTATCTATCGACACTTGGTTATCGATCTACGAGCATCTCGTTAGC"
# inputs = tokenizer(dna, return_tensors='pt',)


# print(res)

def finetune(mat: dict, num_classes: int, device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    classifier_model = AutoModelForSequenceClassification.from_pretrained("zhihan1996/DNA_bert_6", trust_remote_code=True, token=hf_token, num_labels=num_classes)

    dna_strings = mat['all_string_dnas'].squeeze()
    labels = mat['all_labels'].squeeze()

    train_indices = mat['train_loc'].squeeze() - 1
    val_indices = mat['val_seen_loc'].squeeze() - 1
    print(train_indices.shape)
    
    train_labels = labels[train_indices]
    val_labels = labels[val_indices]

    train_dna_strings = dna_strings[train_indices]
    val_dna_strings = dna_strings[val_indices]

    train_dataset = DNADataset(train_dna_strings, train_labels, tokenizer)
    val_dataset = DNADataset(val_dna_strings, val_labels, tokenizer)

    training_args = TrainingArguments(
        output_dir="./dna_bert_finetuned",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=5,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10
    )

    trainer = Trainer(
        model=classifier_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    return trainer.train()