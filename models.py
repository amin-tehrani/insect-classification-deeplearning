import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from sklearn.metrics import accuracy_score, adjusted_rand_score, normalized_mutual_info_score
import json
import time

class AttentionFusion(nn.Module):
    def __init__(self, dna_dim, img_dim, fused_dim, num_heads=4):
        super().__init__()
        # Project both embeddings into same space
        self.proj_dna = nn.Linear(dna_dim, fused_dim)
        self.proj_img = nn.Linear(img_dim, fused_dim)
        
        # Attention layer
        self.attn = nn.MultiheadAttention(embed_dim=fused_dim, num_heads=num_heads, batch_first=True)
        
        # Optional: feed-forward layer after attention
        self.ffn = nn.Sequential(
            nn.Linear(fused_dim, fused_dim),
            nn.ReLU(),
            nn.Linear(fused_dim, fused_dim)
        )

    def forward(self, dna_emb, img_emb):
        # dna_emb: [batch, dna_dim]
        # img_emb: [batch, img_dim]
        
        # Project into same space
        dna_proj = self.proj_dna(dna_emb).unsqueeze(1)  # [batch, 1, D]
        img_proj = self.proj_img(img_emb).unsqueeze(1)  # [batch, 1, D]
        
        # Concatenate as sequence
        seq = torch.cat([dna_proj, img_proj], dim=1)  # [batch, 2, D]
        
        # Self-attention
        attn_out, _ = self.attn(seq, seq, seq)  # [batch, 2, D]
        
        # Aggregate embeddings (mean pooling)
        fused = attn_out.mean(dim=1)  # [batch, D]
        
        # Optional FFN
        fused = self.ffn(fused)
        return fused


class Decoder(nn.Module):
    def __init__(self, fused_dim, num_classes):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(fused_dim, num_classes)
        )

    def forward(self, fused):
        x = self.ffn(fused)
        logits = torch.softmax(x, dim=1)
        return logits
    

class Predictor(nn.Module):
    def __init__(self, fusion_encoder: AttentionFusion, decoder: Decoder):
        super().__init__()
        self.fusion_encoder = fusion_encoder
        self.decoder = decoder

        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.01)
        self.criterion = nn.CrossEntropyLoss()

    def fit(self, dna_emb, img_emb, labels, val_indices, train_indices, epochs=100, lr=0.01, eval_frequency=1, evaluate=True,device=None):

        start_time = time.ctime()
        train_indices = train_indices - 1
        val_indices = val_indices - 1
        labels = labels - 1

        log_file = open(f'{start_time}_log.txt', "w")
        history_file = open(f'{start_time}_history.json', "w")

        print(f"Fitting Predictor with {epochs=}, {lr=}", file=log_file, flush=True)

        assert self.optimizer is not None, "optimizer must be set before fitting"
        assert self.criterion is not None, "criterion must be set before fitting"

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

        train_dna_embs = dna_emb[train_indices]
        train_img_embs = img_emb[train_indices]
        train_labels = labels[train_indices]

        val_dna_embs = dna_emb[val_indices]
        val_img_embs = img_emb[val_indices]
        val_labels = labels[val_indices]

        self.fusion_encoder.train()
        self.decoder.train()


        for epoch in range(epochs):
            try:
                self.optimizer.zero_grad()

                train_output_embs = self.fusion_encoder(train_dna_embs, train_img_embs)
                train_outputs = self.decoder(train_output_embs)

                train_loss = self.criterion(train_outputs, train_labels)
                train_loss.backward()
                self.optimizer.step()

                if hasattr(self, 'scheduler') and self.scheduler is not None:
                    self.scheduler.step()

                history['train_loss'].append(train_loss.item())
                
                if evaluate and (epoch + 1) % eval_frequency == 0:
                    self.fusion_encoder.eval()
                    self.decoder.eval()

                    val_output_embs = self.fusion_encoder(val_dna_embs, val_img_embs)
                    val_outputs = self.decoder(val_output_embs)
                    val_loss = self.criterion(val_outputs, val_labels)
        
                    # Get predictions
                    predictions = torch.argmax(val_outputs, dim=-1)
                    
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
                    
                    self.fusion_encoder.train()
                    self.decoder.train()

                    history['val_loss'].append(val_metrics['loss'])
                    history['val_accuracy'].append(val_metrics['accuracy'])
                    history['val_ari'].append(val_metrics['ari'])
                    history['val_nmi'].append(val_metrics['nmi'])

                    if val_loss < best_val_loss:
                        best_val_loss = val_metrics['loss']
                        torch.save(self.fusion_encoder.state_dict(), f'{start_time}_best_fusion_encoder.pt')
                        torch.save(self.decoder.state_dict(), f'{start_time}_best_decoder.pt')

                    misclassification = (val_truth_np != predictions_np).sum()
                    print(f"Epoch {epoch+1}/{epochs}, Loss: {train_loss.item():.5f}, LR: {self.optimizer.param_groups[0]['lr']:.5f}, Val Loss: {val_loss:.5f}, Missclassification: {misclassification}, Val Metrics: {val_metrics}", file=log_file, flush=True)
                else:
                    print(f"Epoch {epoch+1}/{epochs}, Loss: {train_loss.item():.5f}, LR: {self.optimizer.param_groups[0]['lr']:.5f}", file=log_file, flush=True)
            except KeyboardInterrupt:
                print("KeyboardInterrupt detected, stopping fit", file=log_file, flush=True)
                break

        log_file.close()

        json.dump(history, fp=history_file)
        history_file.close()

        return history