import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from sklearn.metrics import accuracy_score, adjusted_rand_score, normalized_mutual_info_score
import json
import time
from torch.nn import functional as F

class AttentionFusion(nn.Module):
    def __init__(self, dna_dim, img_dim, fused_dim, dna_len_dim=16, num_distinct_dna_len=120, num_heads=4):
        super().__init__()

        self.dna_dim=dna_dim
        self.img_dim=img_dim
        self.fused_dim=fused_dim
        self.dna_len_dim=dna_len_dim
        self.num_heads=num_heads

        # Project both embeddings into same space
        self.dna_len_emb = nn.Embedding(num_embeddings=num_distinct_dna_len, embedding_dim=dna_len_dim)
        self.proj_dna = nn.Linear(dna_dim, fused_dim-dna_len_dim)
        self.proj_img = nn.Linear(img_dim, fused_dim)
        
        # Attention layer
        # self.attn = nn.MultiheadAttention(embed_dim=fused_dim, num_heads=num_heads, batch_first=True)

        self.weight_dna_len = nn.Parameter(torch.ones(1))
        self.weight_dna = nn.Parameter(torch.ones(1))
        self.weight_img = nn.Parameter(torch.ones(1))
        
        # Optional: feed-forward layer after attention
        self.ffn = nn.Sequential(
            nn.Linear(fused_dim*2, fused_dim),
            # nn.ReLU(),
            # nn.Linear(fused_dim, fused_dim)
        )

    def forward(self, dna_len_tokens, dna_emb, img_emb):
        # dna_emb: [batch, dna_dim]
        # img_emb: [batch, img_dim]
        # Project into same space
        dna_len_emb = self.dna_len_emb(dna_len_tokens).unsqueeze(1)  # [batch, dna_len_dim]
        dna_proj = self.proj_dna(dna_emb).unsqueeze(1)  # [batch, 1, D - dna_len_dim]
        img_proj = self.proj_img(img_emb).unsqueeze(1)  # [batch, 1, D]
        
        dna_final_emb = torch.cat([self.weight_dna_len * dna_len_emb, self.weight_dna * dna_proj], dim=2)  # [batch, 1, D]

        # Concatenate as sequence
        fused = seq = torch.cat([dna_final_emb, self.weight_img *img_proj], dim=2).squeeze(1)  # [batch, 2, D]
        
        # Self-attention
        # attn_out, _ = self.attn(seq, seq, seq)  # [batch, 2, D]
        
        # Aggregate embeddings (mean pooling)
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
    
class GenusPredictor(nn.Module):

    def __init__(self, specie2genus, fusion_encoder: AttentionFusion, genus_n_classes=372):
        super().__init__()
        self.genus_n_classes = genus_n_classes
        self.specie2genus = specie2genus - 1

        self.fusion_encoder = fusion_encoder
        self.decoder = Decoder(fusion_encoder.fused_dim, 744 ,genus_n_classes)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.01)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, dna_len_tokens, dna_emb, img_emb):
        fused = self.fusion_encoder(dna_len_tokens, dna_emb, img_emb)
        return self.decoder(fused), fused
    
    def fit(self, dna_len_tokens, dna_emb, img_emb, specie_labels, val_indices, train_indices, 
            epochs=100, lr=0.01, eval_frequency=20, evaluate=True, device=None):

        start_time = time.ctime()
        train_indices = train_indices - 1
        val_indices = val_indices - 1
        specie_labels = specie_labels - 1
        genus_labels = self.specie2genus[specie_labels]

        log_file = open(f'output/{start_time}_log_genus.txt', "w")
        history_file = open(f'output/{start_time}_history_genus.json', "w")

        print(f"Fitting GenusPredictor with {epochs=}, {lr=}", file=log_file, flush=True)

        assert self.optimizer is not None, "optimizer must be set before fitting"

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        self.scheduler = lr_scheduler.OneCycleLR(self.optimizer, max_lr=lr, total_steps=epochs)

        best_val_loss = float('inf')

        history = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': [],
            'val_ari': [],
            'val_nmi': []
        }

        # Send tensors to device
        dna_emb = torch.tensor(dna_emb).to(device)
        img_emb = torch.tensor(img_emb).to(device)
        specie_labels = torch.tensor(specie_labels).to(device)
        genus_labels = torch.tensor(genus_labels).squeeze(1).to(device)
        dna_len_tokens = torch.tensor(dna_len_tokens).to(device)


        train_dna_embs = dna_emb[train_indices]
        train_img_embs = img_emb[train_indices]
        train_labels = genus_labels[train_indices]
        train_dna_len_tokens = dna_len_tokens[train_indices]

        val_dna_embs = dna_emb[val_indices]
        val_img_embs = img_emb[val_indices]
        val_labels = genus_labels[val_indices]
        val_dna_len_tokens = dna_len_tokens[val_indices]

        self.train()

        for epoch in range(epochs):
            try:
                self.optimizer.zero_grad()

                # Forward pass
                train_logits, _ = self.forward(train_dna_len_tokens, train_dna_embs, train_img_embs)

                # Loss
                train_loss = self.criterion(train_logits, train_labels)
                train_loss.backward()
                self.optimizer.step()

                if hasattr(self, 'scheduler') and self.scheduler is not None:
                    self.scheduler.step()

                history['train_loss'].append(train_loss.item())

                train_missclass = (train_labels.cpu().numpy() != torch.argmax(train_logits, dim=-1).cpu().numpy()).sum().item()
                
                # Validation
                if evaluate and (epoch + 1) % eval_frequency == 0:
                    self.eval()

                    val_logits, _ = self.forward(val_dna_len_tokens, val_dna_embs, val_img_embs)
                    val_loss = self.criterion(val_logits, val_labels.squeeze(1) if val_labels.ndim > 1 else val_labels)

                    predictions = torch.argmax(val_logits, dim=-1)
                    val_truth_np = val_labels.cpu().numpy()
                    predictions_np = predictions.cpu().numpy()

                    accuracy = accuracy_score(val_truth_np, predictions_np)
                    ari = adjusted_rand_score(val_truth_np, predictions_np)
                    nmi = normalized_mutual_info_score(val_truth_np, predictions_np)

                    val_metrics = {
                        'loss': val_loss.item(),
                        'accuracy': accuracy,
                        'ari': ari,
                        'nmi': nmi
                    }

                    self.train()

                    history['val_loss'].append(val_metrics['loss'])
                    history['val_accuracy'].append(val_metrics['accuracy'])
                    history['val_ari'].append(val_metrics['ari'])
                    history['val_nmi'].append(val_metrics['nmi'])

                    if val_loss < best_val_loss:
                        best_val_loss = val_metrics['loss']
                        torch.save(self.state_dict(), f'output/{start_time}_best_genus_predictor.pt')

                    misclassification = (val_truth_np != predictions_np).sum()
                    print(
                        f"Epoch {epoch+1}/{epochs}, Loss: {train_loss.item():.5f}, "
                        f"LR: {self.optimizer.param_groups[0]['lr']:.5f}, Train Miss: {train_missclass}, "
                        f"Val Loss: {val_loss:.5f}, Val Miss: {misclassification}, Val Metrics: {val_metrics}",
                        file=log_file, flush=True
                    )
                else:
                    print(
                        f"Epoch {epoch+1}/{epochs}, Loss: {train_loss.item():.5f}, "
                        f"LR: {self.optimizer.param_groups[0]['lr']:.5f}, Train Miss: {train_missclass}",
                        file=log_file, flush=True
                    )

            except KeyboardInterrupt:
                print("KeyboardInterrupt detected, stopping fit", file=log_file, flush=True)
                break

        log_file.close()
        json.dump(history, fp=history_file)
        history_file.close()

        return history


class SpeciePredictor(nn.Module):
    def __init__(self, species2genus: list[int], genus_species: dict[int,list], 
                 genus_predictor: GenusPredictor, 
                 reduced_fused_dim=128,
                 max_specie_in_genus=23,
                 genus_embedding_dim=64,
                 specie_decoder_hidden_dim=64,
                 num_of_species=1050,
                 alpha=2,
                 beta=1
                 ):
        super().__init__()
        self.species2genus = torch.tensor(species2genus-1, dtype=torch.long)
        self.genus_species = genus_species
        self.genus_predictor = genus_predictor
        self.reduced_fused_dim = reduced_fused_dim
        self.max_specie_in_genus = max_specie_in_genus
        self.genus_embedding_dim = genus_embedding_dim
        self.num_of_species = num_of_species

        self.fused_proj = nn.Linear(genus_predictor.fusion_encoder.fused_dim, reduced_fused_dim)
        self.genus_embedder = nn.Linear(genus_predictor.genus_n_classes, genus_embedding_dim)
        self.specie_decoder = nn.Sequential(
            nn.Linear(reduced_fused_dim+genus_embedding_dim, specie_decoder_hidden_dim),
            nn.Sigmoid(),
            nn.Linear(specie_decoder_hidden_dim, max_specie_in_genus)
        )

        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.01)
        self.genus_criterion = nn.CrossEntropyLoss()
        self.specie_criterion = nn.CrossEntropyLoss()

        self.alpha = alpha
        self.beta = beta

    def _calculate_loss(self, specie_labels, genus_logits, local_specie_logits, freeze_genus=False):
        local_specie_label = torch.tensor(self.local_specie_labels, dtype=torch.long)[specie_labels]
        specie_loss = self.specie_criterion(local_specie_logits, local_specie_label)
        if freeze_genus:
            return specie_loss
        
        genus_labels = self.species2genus[specie_labels]
        genus_loss = self.genus_criterion(genus_logits, genus_labels.squeeze(1))
        return self.alpha * genus_loss + self.beta * specie_loss
    

    def _flatten_species_probs_batch(self, prob_matrix):
        B = prob_matrix.size(0)
        result = torch.zeros(B, self.num_of_species, dtype=prob_matrix.dtype, device=prob_matrix.device)
    
        for genus_id, species_ids in self.genus_species.items():
            n = len(species_ids)
            result[:, species_ids] = prob_matrix[:, genus_id, :n]
    
        return result


    def forward(self, dna_len_tokens, dna_emb, img_emb, teacher_genus=None):
        genus_logits, fused_emb = self.genus_predictor(dna_len_tokens, dna_emb, img_emb)

        if teacher_genus is not None:
            print(teacher_genus)
            genus_logits = F.one_hot(teacher_genus.to(torch.long), num_classes=self.genus_predictor.genus_n_classes).to(torch.float32)


        reduced_fused = self.fused_proj(fused_emb)
        genus_emb = self.genus_embedder(genus_logits)

        genus_fused = torch.cat([genus_emb, reduced_fused, ], dim=1)
        local_specie_logits = self.specie_decoder(genus_fused) # size: [batch_size, max_specie_in_genus]
    
        prob_matrix = genus_logits.unsqueeze(2) * local_specie_logits.unsqueeze(1)    

        return self._flatten_species_probs_batch(prob_matrix), local_specie_logits, genus_logits, fused_emb, reduced_fused

    def fit(self, dna_len_tokens, dna_emb, img_emb, labels, val_indices, train_indices, epochs=100, lr=0.01, eval_frequency=20, evaluate=True, freeze_genus=False, teacher_force=False, device=None):

        self.species2genus.to(device)

        start_time = time.ctime()
        train_indices = train_indices - 1
        val_indices = val_indices - 1
        labels = labels - 1

        if freeze_genus:
            for param in self.genus_predictor.parameters():
                param.requires_grad = False

        self.local_specie_labels = []
        true_genus = []
        for label in labels:
            g = self.species2genus[label]
            self.local_specie_labels.append(self.genus_species[g.item()].index(label))
            if teacher_force:
                true_genus.append(g.item())


        log_file = open(f'output/{start_time}_log.txt', "w")
        history_file = open(f'output/{start_time}_history.json', "w")

        print(f"Fitting Predictor with {epochs=}, {lr=}", file=log_file, flush=True)

        assert self.optimizer is not None, "optimizer must be set before fitting"

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        self.scheduler = lr_scheduler.OneCycleLR(self.optimizer, max_lr=lr, total_steps=epochs)

        best_val_loss = float('inf')

        history = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': [],
            'val_ari': [],
            'val_nmi': []
        }

        dna_emb = torch.tensor(dna_emb).to(device)
        img_emb = torch.tensor(img_emb).to(device)
        labels = torch.tensor(labels).to(device)
        dna_len_tokens = torch.tensor(dna_len_tokens).to(device)

        train_dna_embs = dna_emb[train_indices]
        train_img_embs = img_emb[train_indices]
        train_labels = labels[train_indices]
        train_dna_len_tokens = dna_len_tokens[train_indices]

        val_dna_embs = dna_emb[val_indices]
        val_img_embs = img_emb[val_indices]
        val_labels = labels[val_indices]
        val_dna_len_tokens = dna_len_tokens[val_indices]

        if teacher_force:
            train_true_genus = torch.tensor(true_genus, dtype=torch.float)[train_indices].to(device)
            val_true_genus = torch.tensor(true_genus, dtype=torch.float)[val_indices].to(device)

        
        self.train()

        for epoch in range(epochs):
            try:
                self.optimizer.zero_grad()

                train_outputs_logits, train_local_specie_logits, train_genus_logits, train_fused_emb, train_reduced_fused = self.forward(train_dna_len_tokens, train_dna_embs, train_img_embs, train_true_genus if teacher_force else None)

                train_loss = self._calculate_loss(train_labels, train_genus_logits, train_local_specie_logits, freeze_genus)
                train_loss.backward()
                self.optimizer.step()

                if hasattr(self, 'scheduler') and self.scheduler is not None:
                    self.scheduler.step()

                history['train_loss'].append(train_loss.item())

                train_missclass = (train_labels.cpu().numpy() != torch.argmax(train_outputs_logits, dim=-1).cpu().numpy()).sum().item()
                
                if evaluate and (epoch + 1) % eval_frequency == 0:
                    self.eval()
                    

                    val_outputs_logits, val_local_specie_logits, val_genus_logits, val_fused_emb, val_reduced_fused = self.forward(val_dna_len_tokens, val_dna_embs, val_img_embs, val_true_genus if teacher_force else None)
                    
                    val_loss = self._calculate_loss(val_labels, val_genus_logits, val_local_specie_logits, freeze_genus)
        
                    # Get predictions
                    predictions = torch.argmax(val_outputs_logits, dim=-1)
                    
                    # Convert to numpy for sklearn metrics
                    val_truth_np = val_labels.cpu().numpy()
                    predictions_np = predictions.cpu().numpy()
                    
                    # Calculate metrics
                    accuracy = accuracy_score(val_truth_np, predictions_np)
                    ari = adjusted_rand_score(val_truth_np, predictions_np)
                    nmi = normalized_mutual_info_score(val_truth_np, predictions_np)
                    
                    val_metrics = {
                        'loss': val_loss.item(),
                        'accuracy': accuracy,
                        'ari': ari,
                        'nmi': nmi
                    }
                    
                    self.train()

                    history['val_loss'].append(val_metrics['loss'])
                    history['val_accuracy'].append(val_metrics['accuracy'])
                    history['val_ari'].append(val_metrics['ari'])
                    history['val_nmi'].append(val_metrics['nmi'])

                    if val_loss < best_val_loss:
                        best_val_loss = val_metrics['loss']
                        torch.save(self.state_dict(), f'output/{start_time}_best_specie_predictor.pt')

                    misclassification = (val_truth_np != predictions_np).sum()
                    print(f"Epoch {epoch+1}/{epochs}, Loss: {train_loss.item():.5f}, LR: {self.optimizer.param_groups[0]['lr']:.5f}, Train Miss: {train_missclass}, Val Loss: {val_loss:.5f}, Val Miss: {misclassification}, Val Metrics: {val_metrics}", file=log_file, flush=True)
                else:
                    print(f"Epoch {epoch+1}/{epochs}, Loss: {train_loss.item():.5f}, LR: {self.optimizer.param_groups[0]['lr']:.5f}, Train Miss: {train_missclass}", file=log_file, flush=True)
            except KeyboardInterrupt:
                print("KeyboardInterrupt detected, stopping fit", file=log_file, flush=True)
                break

        log_file.close()

        json.dump(history, fp=history_file)
        history_file.close()

        return history