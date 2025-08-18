import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from sklearn.metrics import accuracy_score, adjusted_rand_score, normalized_mutual_info_score
import json
import time
from torch.nn import functional as F
from transformers import AutoModel, Trainer, TrainingArguments, DefaultDataCollator

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
        
        dna_len_emb = self.dna_len_emb(dna_len_tokens)  # [batch, dna_len_dim]


        # print("dna_len_emb.shape",dna_len_emb.shape)

        if len(dna_len_emb.shape) == 2:
            dna_len_emb = dna_len_emb.unsqueeze(1)

        # print("dna_len_emb.shape",dna_len_emb.shape)

        if self.proj_dna_dim is not None:
            dna_proj = self.proj_dna(dna_emb).unsqueeze(1)  # [batch, 1, proj_dna_dim]
        else:
            dna_proj = dna_emb.unsqueeze(1)

        if self.proj_img_dim is not None:
            img_proj = self.proj_img(img_emb).unsqueeze(1)  # [batch, 1, proj_img_dim]
        else:
            img_proj = img_emb.unsqueeze(1)

        # print(dna_len_emb.shape, dna_proj.shape, img_proj.shape)
        
        
        
        dna_final_emb = torch.cat([self.weight_dna_len * dna_len_emb, self.weight_dna * dna_proj], dim=2)  # [batch, 1, dna_len_dim + dna_dim]

        seq = torch.cat([dna_final_emb, self.weight_img *img_proj], dim=2).squeeze(1)  # [batch, concat_dim]
        
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
         # Decoder(fused_dim, 744 ,genus_n_classes)

        # self.optimizer = torch.optim.Adam(self.parameters(), lr=0.01)
        # self.criterion = nn.CrossEntropyLoss()

    def forward(self, fused_emb):
        # fused = self.fused_dim(dna_len_tokens, dna_emb, img_emb)
        x = self.decoder(fused_emb)
        logits = torch.softmax(x, dim=1)
        return logits
    
    # def fit(self, dna_len_tokens, dna_emb, img_emb, specie_labels, val_indices, train_indices, 
    #         epochs=100, lr=0.01, eval_frequency=20, evaluate=True, device=None):

    #     start_time = time.ctime()
    #     train_indices = train_indices - 1
    #     val_indices = val_indices - 1
    #     specie_labels = specie_labels - 1
    #     genus_labels = self.specie2genus[specie_labels]

    #     log_file = open(f'output/{start_time}_log_genus.txt', "w")
    #     history_file = open(f'output/{start_time}_history_genus.json', "w")

    #     print(f"Fitting GenusPredictor with {epochs=}, {lr=}", file=log_file, flush=True)

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

    #     # Send tensors to device
    #     dna_emb = torch.tensor(dna_emb).to(device)
    #     img_emb = torch.tensor(img_emb).to(device)
    #     specie_labels = torch.tensor(specie_labels).to(device)
    #     genus_labels = torch.tensor(genus_labels).squeeze(1).to(device)
    #     dna_len_tokens = torch.tensor(dna_len_tokens).to(device)


    #     train_dna_embs = dna_emb[train_indices]
    #     train_img_embs = img_emb[train_indices]
    #     train_labels = genus_labels[train_indices]
    #     train_dna_len_tokens = dna_len_tokens[train_indices]

    #     val_dna_embs = dna_emb[val_indices]
    #     val_img_embs = img_emb[val_indices]
    #     val_labels = genus_labels[val_indices]
    #     val_dna_len_tokens = dna_len_tokens[val_indices]

    #     self.train()

    #     for epoch in range(epochs):
    #         try:
    #             self.optimizer.zero_grad()

    #             # Forward pass
    #             train_logits, _ = self.forward(train_dna_len_tokens, train_dna_embs, train_img_embs)

    #             # Loss
    #             train_loss = self.criterion(train_logits, train_labels)
    #             train_loss.backward()
    #             self.optimizer.step()

    #             if hasattr(self, 'scheduler') and self.scheduler is not None:
    #                 self.scheduler.step()

    #             history['train_loss'].append(train_loss.item())

    #             train_missclass = (train_labels.cpu().numpy() != torch.argmax(train_logits, dim=-1).cpu().numpy()).sum().item()
                
    #             # Validation
    #             if evaluate and (epoch + 1) % eval_frequency == 0:
    #                 self.eval()

    #                 val_logits, _ = self.forward(val_dna_len_tokens, val_dna_embs, val_img_embs)
    #                 val_loss = self.criterion(val_logits, val_labels.squeeze(1) if val_labels.ndim > 1 else val_labels)

    #                 predictions = torch.argmax(val_logits, dim=-1)
    #                 val_truth_np = val_labels.cpu().numpy()
    #                 predictions_np = predictions.cpu().numpy()

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
    #                     torch.save(self.state_dict(), f'output/{start_time}_best_genus_predictor.pt')

    #                 misclassification = (val_truth_np != predictions_np).sum()
    #                 print(
    #                     f"Epoch {epoch+1}/{epochs}, Loss: {train_loss.item():.5f}, "
    #                     f"LR: {self.optimizer.param_groups[0]['lr']:.5f}, Train Miss: {train_missclass}, "
    #                     f"Val Loss: {val_loss:.5f}, Val Miss: {misclassification}, Val Metrics: {val_metrics}",
    #                     file=log_file, flush=True
    #                 )
    #             else:
    #                 print(
    #                     f"Epoch {epoch+1}/{epochs}, Loss: {train_loss.item():.5f}, "
    #                     f"LR: {self.optimizer.param_groups[0]['lr']:.5f}, Train Miss: {train_missclass}",
    #                     file=log_file, flush=True
    #                 )

    #         except KeyboardInterrupt:
    #             print("KeyboardInterrupt detected, stopping fit", file=log_file, flush=True)
    #             break

    #     log_file.close()
    #     json.dump(history, fp=history_file)
    #     history_file.close()

    #     return history


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
def multimodal_collator(features):
    batch = {}
    batch["dna_len_tokens"] = torch.stack([f["dna_len_tokens"] for f in features])
    batch["labels"] = torch.stack([f["labels"] for f in features])

    if "genus" in features[0]:
        batch["genus"] = torch.stack([f["genus"] for f in features])

    # Combine dna_inputs correctly
    dna_keys = features[0]["dna_inputs"].keys()
    batch["dna_inputs"] = {k: torch.cat([f["dna_inputs"][k] for f in features], dim=0) for k in dna_keys}

    # Combine image_inputs correctly
    image_keys = features[0]["image_inputs"].keys()
    batch["image_inputs"] = {k: torch.cat([f["image_inputs"][k] for f in features], dim=0) for k in image_keys}

    if "dna_emb" in features[0]:
        batch["dna_emb"] = torch.stack([f["dna_emb"] for f in features])

    # Combine img encoding if available
    if "img_emb" in features[0]:
        batch["img_emb"] = torch.stack([f["img_emb"] for f in features])

    

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
        # local_specie_label = torch.tensor(self.local_specie_labels, dtype=torch.long)[specie_labels]
        # print("Logits shape:", logits.shape, "Labels shape:", labels.shape, "Genus logits shape:", genus_logits.shape, "Genus labels shape:", genus_labels.shape if genus_labels is not None else None)
        
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
        print("prob_matrix shape:", prob_matrix.shape)
        B = prob_matrix.size(0)
        result = torch.zeros(B, self.num_of_species, dtype=prob_matrix.dtype, device=prob_matrix.device)
        print("result shape:", result)
        print("Species2Genus Shape:", self.species2genus.shape, "Genus Species Shape:", len(self.genus_species))
        for genus_id, species_ids in self.genus_species.items():
            n = len(species_ids)
            print("\t", n, genus_id, species_ids)
            result[:, species_ids] = prob_matrix[:, genus_id, :n]
    
        return result


    def forward(self, dna_len_tokens, image_inputs, dna_inputs, dna_emb=None, img_emb=None ,genus=None, labels=None):
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

    def fit(self, train_dataset, eval_dataset=None, output_dir="./results", batch_size=8, epochs=3, lr=5e-5):

        for param in self.dna_embedder.parameters():
            param.requires_grad = False
        for param in self.img_embedder.parameters():
            param.requires_grad = False

        # HuggingFace TrainingArguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            eval_strategy="epoch" if eval_dataset is not None else "no",
            save_strategy="epoch",
            learning_rate=lr,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=epochs,
            weight_decay=0.01,
            logging_dir=f"{output_dir}/logs",
            logging_steps=50,
            load_best_model_at_end=True if eval_dataset is not None else False,
            report_to="none",   # disable W&B unless you want it
        )

        # Define Trainer
        trainer = Trainer(
            model=self,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=multimodal_collator
        )

        # Train
        trainer.train()
        return trainer

    # def fit(self, dna_len_tokens, dna_emb, img_emb, labels, val_indices, train_indices, epochs=100, lr=0.01, eval_frequency=20, evaluate=True, freeze_genus=False, teacher_force=False, device=None):

    #     self.species2genus.to(device)

    #     start_time = time.ctime()
    #     train_indices = train_indices - 1
    #     val_indices = val_indices - 1
    #     labels = labels - 1

    #     if freeze_genus:
    #         for param in self.genus_classifier.parameters():
    #             param.requires_grad = False

    #     self.local_specie_labels = []
    #     true_genus = []
    #     for label in labels:
    #         g = self.species2genus[label]
    #         self.local_specie_labels.append(self.genus_species[g.item()].index(label))
    #         if teacher_force:
    #             true_genus.append(g.item())


    #     log_file = open(f'output/{start_time}_log.txt', "w")
    #     history_file = open(f'output/{start_time}_history.json', "w")

    #     print(f"Fitting Predictor with {epochs=}, {lr=}", file=log_file, flush=True)

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

    #     dna_emb = torch.tensor(dna_emb).to(device)
    #     img_emb = torch.tensor(img_emb).to(device)
    #     labels = torch.tensor(labels).to(device)
    #     dna_len_tokens = torch.tensor(dna_len_tokens).to(device)

    #     train_dna_embs = dna_emb[train_indices]
    #     train_img_embs = img_emb[train_indices]
    #     train_labels = labels[train_indices]
    #     train_dna_len_tokens = dna_len_tokens[train_indices]

    #     val_dna_embs = dna_emb[val_indices]
    #     val_img_embs = img_emb[val_indices]
    #     val_labels = labels[val_indices]
    #     val_dna_len_tokens = dna_len_tokens[val_indices]

    #     if teacher_force:
    #         train_true_genus = torch.tensor(true_genus, dtype=torch.float)[train_indices].to(device)
    #         val_true_genus = torch.tensor(true_genus, dtype=torch.float)[val_indices].to(device)

        
    #     self.train()

    #     for epoch in range(epochs):
    #         try:
    #             self.optimizer.zero_grad()

    #             train_outputs_logits, train_local_specie_logits, train_genus_logits, train_fused_emb, train_reduced_fused = self.forward(train_dna_len_tokens, train_dna_embs, train_img_embs, train_true_genus if teacher_force else None)

    #             train_loss = self._calculate_loss(train_labels, train_genus_logits, train_local_specie_logits, freeze_genus)
    #             train_loss.backward()
    #             self.optimizer.step()

    #             if hasattr(self, 'scheduler') and self.scheduler is not None:
    #                 self.scheduler.step()

    #             history['train_loss'].append(train_loss.item())

    #             train_missclass = (train_labels.cpu().numpy() != torch.argmax(train_outputs_logits, dim=-1).cpu().numpy()).sum().item()
                
    #             if evaluate and (epoch + 1) % eval_frequency == 0:
    #                 self.eval()
                    

    #                 val_outputs_logits, val_local_specie_logits, val_genus_logits, val_fused_emb, val_reduced_fused = self.forward(val_dna_len_tokens, val_dna_embs, val_img_embs, val_true_genus if teacher_force else None)
                    
    #                 val_loss = self._calculate_loss(val_labels, val_genus_logits, val_local_specie_logits, freeze_genus)
        
    #                 # Get predictions
    #                 predictions = torch.argmax(val_outputs_logits, dim=-1)
                    
    #                 # Convert to numpy for sklearn metrics
    #                 val_truth_np = val_labels.cpu().numpy()
    #                 predictions_np = predictions.cpu().numpy()
                    
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