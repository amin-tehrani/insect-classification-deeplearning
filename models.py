import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from sklearn.metrics import accuracy_score, adjusted_rand_score, normalized_mutual_info_score
import json
import time
from torch.nn import functional as F
from transformers import AutoModel, Trainer, TrainingArguments, DefaultDataCollator
import numpy as np
import evaluate

accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred

    # Handle logits if it’s a tuple
    if isinstance(logits, (tuple, list)):
        logits = logits[0]

    # Flatten labels if shape is (batch, 1)
    labels = np.array(labels).reshape(-1)

    preds = np.argmax(logits, axis=-1)

    return accuracy.compute(predictions=preds, references=labels)

    
class AttentionFusion(nn.Module):
    def __init__(self, dna_dim, img_dim, fused_dim=None, dna_len_dim=16, num_distinct_dna_len=120, proj_dna_dim=None, proj_img_dim=None, num_heads=None):
        super().__init__()

        self.dna_dim=dna_dim
        self.img_dim=img_dim
        self._fused_dim = fused_dim
        self.dna_len_dim=dna_len_dim
        self.num_heads=num_heads
        self.proj_dna_dim=proj_dna_dim
        self.proj_img_dim=proj_img_dim

        # Project both embeddings into same space
        self.dna_len_emb = nn.Embedding(num_embeddings=num_distinct_dna_len, embedding_dim=dna_len_dim)
        if proj_dna_dim is not None:
            self.proj_dna = nn.Linear(dna_dim, proj_dna_dim)
        if proj_img_dim is not None:
            self.proj_img = nn.Linear(img_dim, proj_img_dim)

        # self.proj_dna = nn.Linear(dna_dim, fused_dim-dna_len_dim)
        # self.proj_img = nn.Linear(img_dim, fused_dim)
        
        self.weight_dna_len = nn.Parameter(torch.ones(1))
        self.weight_dna = nn.Parameter(torch.ones(1))
        self.weight_img = nn.Parameter(torch.ones(1))

        # Attention layer
        if num_heads:
            self.attn = nn.MultiheadAttention(embed_dim=self.concatenated_dim, num_heads=num_heads, batch_first=True)
        
        
        # Optional: feed-forward layer after attention
        self.ffn = nn.Sequential(
            nn.Linear(self.concatenated_dim, self.fused_dim),
            # nn.ReLU(),
            # nn.Linear(fused_dim, fused_dim)
        )

    @property
    def concatenated_dim(self):
        if self.proj_dna_dim is not None:
            out_dna_dim = self.proj_dna_dim
        else:
            out_dna_dim = self.dna_dim

        if self.proj_img_dim is not None:
            out_img_dim = self.proj_img_dim
        else:
            out_img_dim = self.img_dim

        return self.dna_len_dim + out_dna_dim + out_img_dim
        
    @property
    def fused_dim(self):
        if self._fused_dim is not None:
            return self._fused_dim
        else:
            return self.concatenated_dim
        
    def forward(self, dna_len_tokens, dna_emb, img_emb):
        # dna_emb: [batch, dna_dim]
        # img_emb: [batch, img_dim]
        # Project into same space
        # print("dna_len_tokens.shape",dna_len_tokens.shape)
        
        dna_len_emb = self.dna_len_emb(dna_len_tokens).squeeze(1)  # [batch, dna_len_dim]


        # print("dna_len_emb.shape",dna_len_emb.shape)

        if len(dna_len_emb.shape) == 2:
            dna_len_emb = dna_len_emb

        # print("dna_len_emb.shape",dna_len_emb.shape)

        if self.proj_dna_dim is not None:
            dna_proj = self.proj_dna(dna_emb)  # [batch, 1, proj_dna_dim]
        else:
            dna_proj = dna_emb

        if self.proj_img_dim is not None:
            img_proj = self.proj_img(img_emb)  # [batch, 1, proj_img_dim]
        else:
            img_proj = img_emb

        # print(dna_len_emb.shape, dna_proj.shape, img_proj.shape)
        
        
        
        dna_final_emb = torch.cat([self.weight_dna_len * dna_len_emb, self.weight_dna * dna_proj], dim=1)  # [batch, 1, dna_len_dim + dna_dim]

        seq = torch.cat([dna_final_emb, self.weight_img *img_proj], dim=1)  # [batch, concat_dim]
        
        if self.num_heads:
            # Self-attention
            fused, _ = self.attn(seq, seq, seq)  # [batch, concat_dim]
        else:
            fused = seq
        
        
        
        # Aggregate embelddings (mean pooling)
        # fused = attn_out.mean(dim=1)  # [batch, D]
        
        # Optional FFN
        fused = self.ffn(fused)
        return fused


class Decoder(nn.Module):
    def __init__(self, fused_dim, hidden_dim, num_classes):
        super().__init__()
        self.fused_dim = fused_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        self.ffn = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.Sigmoid(),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, fused):
        x = self.ffn(fused)
        logits = torch.softmax(x, dim=1)
        return logits
    
class GenusClassifier(nn.Module):

    def __init__(self, fused_dim: int, hidden_dim=744,  genus_n_classes=372):
        super().__init__()
        self.genus_n_classes = genus_n_classes

        self.fused_dim = fused_dim
        self.decoder = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.Sigmoid(),
            nn.Linear(hidden_dim, genus_n_classes)
        )

    def forward(self, fused_emb):
        # fused = self.fused_dim(dna_len_tokens, dna_emb, img_emb)
        x = self.decoder(fused_emb)
        logits = torch.softmax(x, dim=1)
        return logits
    


class LocalSpecieClassfier(nn.Module):

    def __init__(self, fused_dim: int, 
                 genus_n_classes=372,
                 reduced_fused_dim=128,
                 max_specie_in_genus=23,
                 genus_embedding_dim=64,
                 specie_decoder_hidden_dim=64
                 ):
        super().__init__()
        # self.species2genus = torch.tensor(species2genus-1, dtype=torch.long)
        # self.genus_species = genus_species
        self.fused_dim = fused_dim
        self.genus_n_classes = genus_embedding_dim
        self.reduced_fused_dim = reduced_fused_dim
        self.max_specie_in_genus = max_specie_in_genus
        self.genus_embedding_dim = genus_embedding_dim
        # self.num_of_species = num_of_species

        self.fused_proj = nn.Linear(fused_dim, reduced_fused_dim)
        self.genus_embedder = nn.Linear(genus_n_classes, genus_embedding_dim)
        self.specie_decoder = nn.Sequential(
            nn.Linear(reduced_fused_dim+genus_embedding_dim, specie_decoder_hidden_dim),
            nn.Sigmoid(),
            nn.Linear(specie_decoder_hidden_dim, max_specie_in_genus)
        )

        # self.optimizer = torch.optim.Adam(self.parameters(), lr=0.01)
        # self.genus_criterion = nn.CrossEntropyLoss()
        # self.specie_criterion = nn.CrossEntropyLoss()

        # self.alpha = alpha
        # self.beta = beta


    def forward(self, fused_emb, genus_logits):
        # genus_logits, fused_emb = self.genus_predictor(dna_len_tokens, dna_emb, img_emb)

        # if teacher_genus is not None:
        #     print(teacher_genus)
        #     genus_logits = F.one_hot(teacher_genus.to(torch.long), num_classes=self.genus_predictor.genus_n_classes).to(torch.float32)


        reduced_fused = self.fused_proj(fused_emb)
        genus_emb = self.genus_embedder(genus_logits)


        # print(genus_logits.shape, "Genus Embedding Shape:", genus_emb.shape, "Reduced Fused Shape:", reduced_fused.shape)

        genus_fused = torch.cat([genus_emb, reduced_fused], dim=1)
        return self.specie_decoder(genus_fused) # size: [batch_size, max_specie_in_genus]

        # return self._flatten_species_probs_batch(prob_matrix), local_specie_logits, genus_logits, fused_emb, reduced_fused


def multimodal_collector(features):
    batch = {}
    keys = features[0].keys()
    
    if 'dna_len_tokens' in keys:
        batch['dna_len_tokens'] = torch.cat([f['dna_len_tokens'] for f in features], dim=0)
    if 'image_inputs' in keys:
        batch['image_inputs'] = {
            "pixel_values": torch.cat([f['image_inputs']['pixel_values'] for f in features], dim=0)
        }
    if 'genus' in keys:
        batch['genus'] = torch.cat([f['genus'] for f in features], dim=0)
    if 'labels' in keys:
        batch['labels'] = torch.cat([f['labels'] for f in features], dim=0)
    if 'dna_inputs' in keys:
        batch['dna_inputs'] = {
            "input_ids": torch.cat([f['dna_inputs']['input_ids'] for f in features], dim=0),
            "attention_mask": torch.cat([f['dna_inputs']['attention_mask'] for f in features], dim=0)
        }
    if 'img_emb' in keys:
        batch['img_emb'] = torch.cat([f['img_emb'] for f in features], dim=0)
    if 'dna_emb' in keys:
        batch['dna_emb'] = torch.cat([f['dna_emb'] for f in features], dim=0)

    # print(batch.keys())
    return batch

class MainClassifier(nn.Module):
    def __init__(self, species2genus: list[int], genus_species: dict[int,list], 
                dna_embedder: AutoModel,
                img_embedder: AutoModel,
                fusion_embedder: AttentionFusion,
                genus_classifier: GenusClassifier,
                local_specie_classifier: LocalSpecieClassfier,
                alpha=0.7,
                beta=0.3):
        super().__init__()
        self.species2genus = torch.tensor(species2genus-1, dtype=torch.long)
        self.genus_species = genus_species

        self.dna_embedder = dna_embedder
        self.img_embedder = img_embedder
        
        assert fusion_embedder.fused_dim == genus_classifier.fused_dim, "Fusion embedder and genus classifier must have the same fused dimension"

        self.fusion_embedder = fusion_embedder
        self.genus_classifier = genus_classifier
        self.local_specie_classifier = local_specie_classifier

        self.num_of_species = len(species2genus)

        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.01)
        self.genus_criterion = nn.CrossEntropyLoss()
        self.specie_criterion = nn.CrossEntropyLoss()

        self.alpha = alpha
        self.beta = beta

        # self.local_specie_labels = []
        # for label in labels:
        #     g = self.species2genus[label]
        #     self.local_specie_labels.append(self.genus_species[g.item()].index(label))
        #     if teacher_force:
        #         true_genus.append(g.item())

    def _calculate_loss(self, logits, labels, genus_logits, genus_labels):
        # print("X", logits.shape, labels)
        # return self.specie_criterion(logits, labels.reshape(-1))
        

        if labels.ndim < 1:
            labels = labels.unsqueeze(0)
        if genus_labels.ndim < 1:
            genus_labels = genus_labels.unsqueeze(0)

        specie_loss = self.specie_criterion(logits, labels.squeeze(1) if labels.ndim > 1 else labels)
        # if freeze_genus:
        #     return specie_loss
        
        genus_loss = self.genus_criterion(genus_logits, genus_labels.squeeze(1) if genus_labels.ndim > 1 else genus_labels)
        return self.alpha * genus_loss + self.beta * specie_loss
    

    def _flatten_species_probs_batch(self, prob_matrix):
        # print("prob_matrix shape:", prob_matrix.shape)
        B = prob_matrix.size(0)
        result = torch.zeros(B, self.num_of_species, dtype=prob_matrix.dtype, device=prob_matrix.device)
        # print("result shape:", result)
        # print("Species2Genus Shape:", self.species2genus.shape, "Genus Species Shape:", len(self.genus_species))
        for genus_id, species_ids in self.genus_species.items():
            n = len(species_ids)
            # print("\t", n, genus_id, species_ids)
            result[:, species_ids] = prob_matrix[:, genus_id, :n]
    
        return result


    def forward(self, dna_len_tokens, image_inputs=None, dna_inputs=None, dna_emb=None, img_emb=None ,genus=None, labels=None):
        # print("MainClassifier input:",'dna_len_tokens', dna_len_tokens.shape,
        #     #   'image_inputs', image_inputs['pixel_values'].shape if image_inputs is not None else None,
        #       'genus', genus.shape if genus is not None else None,
        #       'labels', labels.shape if labels is not None else None,
        #     #   'dna_inputs', dna_inputs['input_ids'].shape if dna_inputs is not None else None,
        #       'img_emb', img_emb.shape if img_emb is not None else None,
        #       'dna_emb', dna_emb.shape if dna_emb is not None else None)
        
        dna_len_tokens = dna_len_tokens.unsqueeze(0) if dna_len_tokens.ndim == 0 else dna_len_tokens

        with torch.no_grad():  # extra safety, avoids computing grads
            if dna_emb is None:
                dna_emb = self.dna_embedder(**dna_inputs).last_hidden_state[:, 0, :]
            if img_emb is None:
                img_emb = self.img_embedder(**image_inputs).last_hidden_state[:, 0, :]

        
        # print("DNA Embedding Shape:", dna_emb.shape, "Image Embedding Shape:", img_emb.shape)

        fused_emb = self.fusion_embedder(dna_len_tokens, dna_emb, img_emb)  # [batch_size, fused_dim]

        # print("Fused Embedding Shape:", fused_emb.shape)
        genus_logits = self.genus_classifier(fused_emb)

        # print("genus_logits1 shape:", genus_logits.shape)

        if genus is not None: # Teacher forcing
            # print(genus)
            genus_logits = F.one_hot(genus, num_classes=self.genus_classifier.genus_n_classes).squeeze().to(torch.float32)


        if len(genus_logits.shape) <= 1:
            genus_logits = genus_logits.unsqueeze(0)

        # print("genus_logits2 shape:", genus_logits.shape)

        local_specie_logits = self.local_specie_classifier(fused_emb, genus_logits)
        # print("local_specie_logits shape:", local_specie_logits.shape)


        prob_matrix = genus_logits.unsqueeze(2) * local_specie_logits.unsqueeze(1)     # size: [batch_size, max_specie_in_genus]

        # print("prob_matrix shape:", prob_matrix.shape)
        logits = self._flatten_species_probs_batch(prob_matrix)
        
        loss = None
        if labels is not None:
            if genus is None:
                genus = self.species2genus[labels]
            loss = self._calculate_loss(logits, labels, genus_logits, genus)

        return {
            "logits": logits,
            "genus_logits": genus_logits,
            "fused_emb": fused_emb,
            "prob_matrix": prob_matrix,
            "loss": loss
        }

    def fit(self, train_dataset, eval_dataset=None, output_dir=f"./results_{time.strftime('%Y%m%d%H%M%S')}", batch_size=8, epochs=3, lr=5e-5, eval_steps=50):

        if self.dna_embedder:
            for param in self.dna_embedder.parameters():
                param.requires_grad = False
        if self.img_embedder:
            for param in self.img_embedder.parameters():
                param.requires_grad = False

        # HuggingFace TrainingArguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            eval_strategy="epoch" if eval_dataset is not None else "no",
            save_strategy="epoch",
            learning_rate=lr,
            eval
            eval_steps=eval_steps if eval_dataset is not None else None,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=epochs,
            weight_decay=0.01,
            logging_dir=f"{output_dir}/logs",
            logging_steps=50,
            load_best_model_at_end=True if eval_dataset is not None else False,
            report_to="none",   # disable W&B unless you want it
        )

        # Data collatordef compute_metrics(eval_pred):

    
        # Define Trainer
        trainer = Trainer(
            model=self,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=multimodal_collector,
            compute_metrics=compute_metrics
        )

        # Train
        trainer.train()
        return trainer

    # def fit(self, train_dataset, val_dataset=None,  epochs=100, lr=0.01, eval_frequency=20, freeze_genus=False, teacher_force=False, batch_size=2):

    #     start_time = time.strftime("%Y%m%d-%H%M%S")

    #     if freeze_genus:
    #         for param in self.genus_classifier.parameters():
    #             param.requires_grad = False

    #     # self.local_specie_labels = []
    #     # true_genus = []
    #     # for label in labels:
    #     #     g = self.species2genus[label]
    #     #     self.local_specie_labels.append(self.genus_species[g.item()].index(label))
    #     #     if teacher_force:
    #     #         true_genus.append(g.item())


    #     log_file = open(f'output/{start_time}_log.txt', "w")
    #     history_file = open(f'output/{start_time}_history.json', "w")

    #     print(f"Fitting MainClassifier with {epochs=}, {lr=}", file=log_file, flush=True)

    #     assert self.optimizer is not None, "optimizer must be set before fitting"

    #     for param_group in self.optimizer.param_groups:
    #         param_group['lr'] = lr

    #     self.scheduler = lr_scheduler.OneCycleLR(self.optimizer, max_lr=lr, total_steps=epochs)

    #     best_val_loss = float('inf')

    #     history = {
    #         'train_loss': [],
    #         'val_loss': [],
    #         'val_accuracy': [],
    #         'val_ari': [],
    #         'val_nmi': []
    #     }
        
    #     self.train()

    #     for epoch in range(epochs):
    #         try:
    #             self.optimizer.zero_grad()

    #             batch_indices = torch.randperm(len(train_dataset))[:batch_size if batch_size is not None else len(train_dataset)]
    #             train_data = train_dataset[batch_indices]
    #             train_labels = train_data['labels']
    #             train_result = self.forward(**train_data)
    #             train_loss = train_result['loss']
    #             train_outputs_logits = train_result['logits']
                
    #             train_loss.backward()
    #             self.optimizer.step()

    #             if hasattr(self, 'scheduler') and self.scheduler is not None:
    #                 self.scheduler.step()

    #             history['train_loss'].append(train_loss.item())

    #             train_missclass = (train_labels.cpu().numpy() != torch.argmax(train_outputs_logits, dim=-1).cpu().numpy()).sum().item()

                
    #             if val_dataset and (epoch + 1) % eval_frequency == 0:
    #                 self.eval()
                    
    #                 batch_indices = torch.randperm(len(val_dataset))[:batch_size if batch_size is not None else len(val_dataset)]
    #                 val_data = val_dataset[batch_indices]
    #                 val_labels = val_data['labels']
    #                 val_result = self.forward(**val_data)
    #                 val_loss = val_result['loss']
    #                 val_outputs_logits = val_result['logits']
        
    #                 # Get predictions
    #                 predictions = torch.argmax(val_outputs_logits, dim=-1)
                    
    #                 # Convert to numpy for sklearn metrics
    #                 val_truth_np = val_labels.cpu().squeeze().numpy()
    #                 predictions_np = predictions.cpu().numpy()
                    
    #                 print(val_truth_np.shape, predictions_np.shape, "Val Truth and Predictions Shape")
    #                 # Calculate metrics
    #                 accuracy = accuracy_score(val_truth_np, predictions_np)
    #                 ari = adjusted_rand_score(val_truth_np, predictions_np)
    #                 nmi = normalized_mutual_info_score(val_truth_np, predictions_np)
                    
    #                 val_metrics = {
    #                     'loss': val_loss.item(),
    #                     'accuracy': accuracy,
    #                     'ari': ari,
    #                     'nmi': nmi
    #                 }
                    
    #                 self.train()

    #                 history['val_loss'].append(val_metrics['loss'])
    #                 history['val_accuracy'].append(val_metrics['accuracy'])
    #                 history['val_ari'].append(val_metrics['ari'])
    #                 history['val_nmi'].append(val_metrics['nmi'])

    #                 if val_loss < best_val_loss:
    #                     best_val_loss = val_metrics['loss']
    #                     torch.save(self.state_dict(), f'output/{start_time}_best_specie_predictor.pt')

    #                 misclassification = (val_truth_np != predictions_np).sum()
    #                 print(f"Epoch {epoch+1}/{epochs}, Loss: {train_loss.item():.5f}, LR: {self.optimizer.param_groups[0]['lr']:.5f}, Train Miss: {train_missclass}, Val Loss: {val_loss:.5f}, Val Miss: {misclassification}, Val Metrics: {val_metrics}", file=log_file, flush=True)
    #             else:
    #                 print(f"Epoch {epoch+1}/{epochs}, Loss: {train_loss.item():.5f}, LR: {self.optimizer.param_groups[0]['lr']:.5f}, Train Miss: {train_missclass}", file=log_file, flush=True)
    #         except KeyboardInterrupt:
    #             print("KeyboardInterrupt detected, stopping fit", file=log_file, flush=True)
    #             break

    #     log_file.close()

    #     json.dump(history, fp=history_file)
    #     history_file.close()

    #     return history