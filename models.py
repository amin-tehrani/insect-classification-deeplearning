import os
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
from safetensors.torch import load_file  # pip install safetensors

accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    if isinstance(logits, (tuple, list)):
        logits = logits[0]
    labels = np.array(labels).reshape(-1)
    preds = np.argmax(logits, axis=-1)
    return {"accuracy": accuracy.compute(predictions=preds, references=labels)["accuracy"]}



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
    if 'local_specie_lbl' in keys:
        batch['local_specie_lbl'] = torch.cat([f['local_specie_lbl'] for f in features], dim=0)
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

class AttentionFusion(nn.Module):
    def __init__(self, dna_dim, img_dim, fused_dim=None, dna_len_dim=16, num_distinct_dna_len=120, proj_dna_dim=None, proj_img_dim=None, dropout=0.1):
        super().__init__()

        self._config = {
            "dna_dim": dna_dim,
            "img_dim": img_dim,
            "fused_dim": fused_dim,
            "dna_len_dim": dna_len_dim,
            "num_distinct_dna_len": num_distinct_dna_len,
            "proj_dna_dim": proj_dna_dim,
            "proj_img_dim": proj_img_dim,
            "dropout": dropout,
        }

        dna_dim=int(dna_dim)
        img_dim=int(img_dim)
        fused_dim=int(fused_dim) if fused_dim is not None else None
        dna_len_dim=int(dna_len_dim)
        proj_dna_dim=int(proj_dna_dim) if proj_dna_dim is not None else None
        proj_img_dim=int(proj_img_dim) if proj_img_dim is not None else None

        self.dna_dim=dna_dim
        self.img_dim=img_dim
        self._fused_dim =fused_dim
        self.dna_len_dim=dna_len_dim
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
        
        self.weight_dna_len = nn.Linear(dna_len_dim, 1, bias=False)
        self.weight_dna = nn.Linear(dna_dim, 1, bias=False)
        self.weight_img = nn.Linear(img_dim, 1, bias=False)

        self.dropout = dropout

        # Optional: feed-forward layer after attention
        self.ffn = nn.Sequential(
            nn.Linear(self.concatenated_dim, self.fused_dim),
            # nn.ReLU(),
            # nn.Dropout(p=0.1),
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

        # print("dna_len_emb.shape",dna_len_emb.shape)

        if self.proj_dna_dim is not None:
            dna_proj = F.dropout(self.proj_dna(dna_emb), self.dropout, training=self.training)  # [batch, 1, proj_dna_dim]
        else:
            dna_proj = dna_emb

        if self.proj_img_dim is not None:
            img_proj = F.dropout(self.proj_img(img_emb), self.dropout, training=self.training)  # [batch, 1, proj_img_dim]
        else:
            img_proj = img_emb

        # print(dna_len_emb.shape, dna_proj.shape, img_proj.shape)
        
        dna_len_attn = self.weight_dna_len(dna_len_emb)
        dna_attn = self.weight_dna(dna_emb)
        img_attn = self.weight_img(img_emb)
        
        dna_final_emb = torch.cat([dna_len_attn * dna_len_emb, dna_attn * dna_proj], dim=1)  # [batch, 1, dna_len_dim + dna_dim]

        seq = torch.cat([dna_final_emb, img_attn *img_proj], dim=1)  # [batch, concat_dim]
        
                
        
        
        # Aggregate embelddings (mean pooling)
        # fused = attn_out.mean(dim=1)  # [batch, D]
        
        # Optional FFN
        fused = self.ffn(seq)
        return fused, dna_len_emb

class AttentionFusionV2(nn.Module):
    """
    Transformer-style fusion with a learnable [FUSE] token that attends to:
      - DNA length embedding
      - DNA embedding
      - Image embedding
    Returns fused vector and the dna_len embedding (like your original).
    """
    def __init__(
        self,
        dna_dim,
        img_dim,
        num_distinct_dna_len=120,
        dna_len_dim=16,
        fused_dim=256,
        n_heads=4,
        ff_mult=4,
        dropout=0.1,
        num_layers=1,          # you can set 2 for a bit more capacity
    ):
        super().__init__()
        self._config = {
            "num_distinct_dna_len": num_distinct_dna_len,
            "dna_len_dim": dna_len_dim,
            "fused_dim": fused_dim,
            "n_heads": n_heads,
            "ff_mult": ff_mult,
            "dropout": dropout,
            "num_layers": num_layers
        }

        self.fused_dim = fused_dim
        self.dna_len_emb = nn.Embedding(num_distinct_dna_len, dna_len_dim)


        # Project all modalities to common d_model
        self.proj_dna = nn.Linear(dna_dim, fused_dim)
        self.proj_img = nn.Linear(img_dim, fused_dim)
        self.proj_len = nn.Linear(dna_len_dim, fused_dim)

        # learnable fusion token
        self.fuse_token = nn.Parameter(torch.randn(1, 1, fused_dim) * 0.02)

        # Transformer encoder blocks (pre-norm)
        layers = []
        for _ in range(num_layers):
            layers.append(nn.ModuleDict({
                "ln1": nn.LayerNorm(fused_dim),
                "attn": nn.MultiheadAttention(fused_dim, n_heads, dropout=dropout, batch_first=True),
                "ln2": nn.LayerNorm(fused_dim),
                "ff": nn.Sequential(
                    nn.Linear(fused_dim, ff_mult * fused_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(ff_mult * fused_dim, fused_dim),
                ),
                "drop": nn.Dropout(dropout),
            }))
        self.blocks = nn.ModuleList(layers)

        # Optional post-fusion gate (helps zero-shot)
        self.gate = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Linear(fused_dim, fused_dim),
            nn.Sigmoid()
        )

        self.out_dim = fused_dim

    def forward(self, dna_len_tokens, dna_emb, img_emb):
        """
        dna_len_tokens: [B, 1] (long)
        dna_emb:        [B, dna_dim]
        img_emb:        [B, img_dim]
        """
        B = dna_emb.size(0)

        # Build tokens
        len_tok = self.dna_len_emb(dna_len_tokens.squeeze(1))   # [B, dna_len_dim]
        len_tok = self.proj_len(len_tok)                        # [B, d_model]
        dna_tok = self.proj_dna(dna_emb)                        # [B, d_model]
        img_tok = self.proj_img(img_emb)                        # [B, d_model]

        # Sequence: [FUSE, LEN, DNA, IMG]
        fuse = self.fuse_token.expand(B, -1, -1)                # [B, 1, d_model]
        seq = torch.stack([len_tok, dna_tok, img_tok], dim=1)   # [B, 3, d_model]
        x = torch.cat([fuse, seq], dim=1)                       # [B, 4, d_model]

        # Transformer encoder
        for blk in self.blocks:
            # Self-attention (pre-norm)
            y = blk.ln1(x)
            attn_out, _ = blk.attn(y, y, y, need_weights=False)
            x = x + blk.drop(attn_out)
            # FFN
            y = blk.ln2(x)
            x = x + blk.drop(blk.ff(y))

        # Take fused representation from [CLS]-like token
        fused = x[:, 0, :]                                      # [B, d_model]

        # Gate (stabilizes zero-shot by shrinking out-of-distribution activations)
        g = self.gate(fused)
        fused = fused * g

        # For parity with your API, also return dna_len embedding (pre-projection)
        return fused, self.dna_len_emb(dna_len_tokens).squeeze(1)

    
class GenusClassifier(nn.Module):

    def __init__(self, fused_dim: int, hidden_dim=744, genus_n_classes=372, dropout=0.1, dna_len_dim=16):
        super().__init__()

        self._config = {
            "fused_dim": fused_dim,
            "hidden_dim": hidden_dim,
            "genus_n_classes": genus_n_classes,
            "dropout": dropout,
            "dna_len_dim": dna_len_dim
        }

        fused_dim = int(fused_dim)
        hidden_dim = int(hidden_dim)
        genus_n_classes = int(genus_n_classes)
        dropout = float(dropout)
        dna_len_dim = int(dna_len_dim)

        self.genus_n_classes = genus_n_classes

        self.fused_dim = fused_dim
        self.dna_len_dim = dna_len_dim
        self.dropout = dropout
        self.decoder = nn.Sequential(
            nn.LayerNorm(dna_len_dim+fused_dim),
            nn.Linear(dna_len_dim+fused_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, genus_n_classes),
        )

    def forward(self, fused_emb, dna_len_token):
        # fused = self.fused_dim(dna_len_tokens, dna_emb, img_emb)
        # print("dna_len_token shape:", dna_len_token.shape, "fused_emb shape:", fused_emb.shape)
        x = torch.cat([dna_len_token, fused_emb], dim=1)  # [batch, dna_len_dim + fused_dim]
        x = self.decoder(x)
        logits = torch.softmax(x, dim=1)
        return logits
    


class LocalSpecieClassfier(nn.Module):

    def __init__(self, fused_dim: int, 
                 genus_n_classes=372,
                 reduced_fused_dim=128,
                 max_specie_in_genus=23,
                 genus_embedding_dim=64,
                 specie_decoder_hidden_dim=64,
                 dropout=0.1,
                 dna_len_dim=16
                 ):
        super().__init__()

        fused_dim = int(fused_dim)
        genus_n_classes = int(genus_n_classes)
        reduced_fused_dim = int(reduced_fused_dim)
        max_specie_in_genus = int(max_specie_in_genus)
        genus_embedding_dim = int(genus_embedding_dim)
        specie_decoder_hidden_dim = int(specie_decoder_hidden_dim)
        dropout = float(dropout)
        dna_len_dim = int(dna_len_dim)


        self._config = {
            "fused_dim": fused_dim,
            "genus_n_classes": genus_n_classes,
            "reduced_fused_dim": reduced_fused_dim,
            "max_specie_in_genus": max_specie_in_genus,
            "genus_embedding_dim": genus_embedding_dim,
            "specie_decoder_hidden_dim": specie_decoder_hidden_dim,
            "dropout": dropout,
            "dna_len_dim": dna_len_dim
        }

        # self.species2genus = torch.tensor(species2genus-1, dtype=torch.long)
        # self.genus_species = genus_species
        self.fused_dim = fused_dim
        self.genus_n_classes = genus_embedding_dim
        self.reduced_fused_dim = reduced_fused_dim
        self.max_specie_in_genus = max_specie_in_genus
        self.genus_embedding_dim = genus_embedding_dim
        self.dropout = dropout
        self.dna_len_dim = dna_len_dim
        # self.num_of_species = num_of_species

        self.fused_proj = nn.Linear(fused_dim, reduced_fused_dim)
        self.genus_embedder = nn.Linear(genus_n_classes, genus_embedding_dim)
        self.specie_decoder = nn.Sequential(
            nn.LayerNorm(dna_len_dim+reduced_fused_dim+genus_embedding_dim),
            nn.Linear(dna_len_dim+reduced_fused_dim+genus_embedding_dim, specie_decoder_hidden_dim),
            nn.GELU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(specie_decoder_hidden_dim, max_specie_in_genus),
            nn.LayerNorm(max_specie_in_genus)
        )

        # self.optimizer = torch.optim.Adam(self.parameters(), lr=0.01)
        # self.genus_criterion = nn.CrossEntropyLoss()
        # self.specie_criterion = nn.CrossEntropyLoss()

        # self.alpha = alpha
        # self.beta = beta


    def forward(self, fused_emb, dna_len_token, genus_logits):
        # genus_logits, fused_emb = self.genus_predictor(dna_len_tokens, dna_emb, img_emb)

        # if teacher_genus is not None:
        #     print(teacher_genus)
        #     genus_logits = F.one_hot(teacher_genus.to(torch.long), num_classes=self.genus_predictor.genus_n_classes).to(torch.float32)


        reduced_fused = F.dropout(self.fused_proj(fused_emb), self.dropout, training=self.training)
        genus_emb = F.dropout(self.genus_embedder(genus_logits), self.dropout, training=self.training)


        # print(genus_logits.shape, "Genus Embedding Shape:", genus_emb.shape, "Reduced Fused Shape:", reduced_fused.shape)

        genus_fused = torch.cat([dna_len_token, genus_emb, reduced_fused], dim=1)
        return self.specie_decoder(genus_fused) # size: [batch_size, max_specie_in_genus]

        # return self._flatten_species_probs_batch(prob_matrix), local_specie_logits, genus_logits, fused_emb, reduced_fused



class MainClassifier(nn.Module):
    def __init__(self, species2genus: list[int], genus_species: dict[int,list], 
                dna_embedder: AutoModel,
                img_embedder: AutoModel,
                fusion_embedder: AttentionFusion,
                genus_classifier: GenusClassifier,
                local_specie_classifier: LocalSpecieClassfier,
                alpha=3,
                beta=2,
                theta=0.5,):
        super().__init__()
        self.species2genus = torch.tensor(species2genus-1, dtype=torch.long)
        self.genus_species = genus_species

        self.dna_embedder = dna_embedder
        self.img_embedder = img_embedder
        
        # print(fusion_embedder, fusion_embedder.fused_dim, genus_classifier, genus_classifier.fused_dim)
        assert fusion_embedder.fused_dim == genus_classifier.fused_dim, "Fusion embedder and genus classifier must have the same fused dimension"

        self.fusion_embedder = fusion_embedder
        self.genus_classifier = genus_classifier
        self.local_specie_classifier = local_specie_classifier

        self.num_of_species = len(species2genus)

        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.01)
        self.genus_criterion = nn.CrossEntropyLoss()
        self.specie_criterion = nn.CrossEntropyLoss()
        self.total_criterion = nn.CrossEntropyLoss()

        self.alpha = alpha
        self.beta = beta
        self.theta = theta

        # self.local_specie_labels = []
        # for label in labels:
        #     g = self.species2genus[label]
        #     self.local_specie_labels.append(self.genus_species[g.item()].index(label))
        #     if teacher_force:
        #         true_genus.append(g.item())

    def _calculate_loss(self, logits, labels, genus_logits, genus_labels, local_specie_logits, local_specie_labels):
        # print("X", logits.shape, labels)
        # return self.specie_criterion(logits, labels.reshape(-1))
        
        # if labels.ndim < 1:
        #     labels = labels.unsqueeze(0)
        # if genus_labels.ndim < 1:
        #     genus_labels = genus_labels.unsqueeze(0)
        # if local_specie_labels.ndim < 1:
        #     local_specie_labels = local_specie_labels.unsqueeze(0)

        if self.alpha == 0:
            total_loss = 0
        else:
            total_loss = self.total_criterion(logits, labels.squeeze())
        
        
        if self.beta == 0:
            genus_loss = 0
        else:
            genus_loss = self.genus_criterion(genus_logits, genus_labels.squeeze())

        if self.theta == 0:
            local_specie_loss = 0
        else:
            local_specie_loss = self.specie_criterion(local_specie_logits, local_specie_labels.squeeze())

        return self.alpha * total_loss + self.beta * genus_loss + self.theta * local_specie_loss
    

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

    def predict_genus(self, dna_len_tokens, image_inputs=None, dna_inputs=None, dna_emb=None, img_emb=None, *args, **kwargs):
        dna_len_tokens = dna_len_tokens.unsqueeze(0) if dna_len_tokens.ndim == 0 else dna_len_tokens

        with torch.no_grad():  # extra safety, avoids computing grads
            if dna_emb is None:
                dna_emb = self.dna_embedder(**dna_inputs).last_hidden_state[:, 0, :]
            if img_emb is None:
                img_emb = self.img_embedder(**image_inputs).last_hidden_state[:, 0, :]

        
        # print("DNA Embedding Shape:", dna_emb.shape, "Image Embedding Shape:", img_emb.shape)

        fused_emb, dna_len_emb = self.fusion_embedder(dna_len_tokens, dna_emb, img_emb)  # [batch_size, fused_dim]

        # print("Fused Embedding Shape:", fused_emb.shape)
        return self.genus_classifier(fused_emb, dna_len_emb), fused_emb, dna_len_emb


    def forward(self, dna_len_tokens, image_inputs=None, dna_inputs=None, dna_emb=None, img_emb=None ,genus=None, local_specie_lbl=None, labels=None):

        genus_logits, fused_emb, dna_len_emb = self.predict_genus(dna_len_tokens, image_inputs, dna_inputs, dna_emb, img_emb)
        # print("OnlyGenus", self._only_genus)

        if getattr(self, '_only_genus', False):
            
            # if genus is None:
            #     genus = self.species2genus[labels.clone().cpu()].to(next(self.parameters()).device).squeeze()

            g_loss = self.genus_criterion(genus_logits, genus.squeeze())
            # print(g_loss)
            return {
                "logits": genus_logits,
                "loss": g_loss
            }

        teacher_genus_logits = F.one_hot(genus, num_classes=self.genus_classifier.genus_n_classes).squeeze().to(torch.float32)

        if getattr(self, '_freeze_genus', False):
            # Detach genus path so gradients don't flow back via genus head
            genus_logits = genus_logits.detach()
            genus_logits = teacher_genus_logits

        if getattr(self, '_freeze_fusion', False):
            # Detach genus path so gradients don't flow back via genus head
            fused_emb = fused_emb.detach()
            dna_len_emb = dna_len_emb.detach()


        if len(genus_logits.shape) <= 1:
            genus_logits = genus_logits.unsqueeze(0)

        if len(teacher_genus_logits.shape) <= 1:
            teacher_genus_logits = teacher_genus_logits.unsqueeze(0)

        # print("genus_logits2 shape:", genus_logits.shape)

        local_specie_logits = self.local_specie_classifier(fused_emb, dna_len_emb, genus_logits if genus is None else teacher_genus_logits)
        # print("local_specie_logits shape:", local_specie_logits.shape)


        prob_matrix = genus_logits.unsqueeze(2) * local_specie_logits.unsqueeze(1)     # size: [batch_size, max_specie_in_genus]

        # print("prob_matrix shape:", prob_matrix.shape)
        logits = self._flatten_species_probs_batch(prob_matrix)
        
        loss = None
        if labels is not None:
            if genus is None:
                genus = self.species2genus[labels]
            loss = self._calculate_loss(logits, labels, genus_logits, genus, local_specie_logits, local_specie_lbl)

        return {
            "logits": logits,
            "genus_logits": genus_logits,
            "fused_emb": fused_emb,
            "prob_matrix": prob_matrix,
            "loss": loss
        }

    def fit(self, train_dataset, eval_dataset, output_dir=f"./model_trained_{time.strftime('%Y%m%dT%H%M%S')}", batch_size=8, epochs=3, lr=5e-5, eval_steps=50, save_steps=200, only_genus=False, freeze_genus=False, freeze_fusion=False):

        self._only_genus = only_genus
        self._freeze_genus = freeze_genus
        self._freeze_fusion = freeze_fusion

        if only_genus:
            output_dir += "_onlygenus"
        if freeze_genus:
            output_dir += "_freezegenus"
        if freeze_fusion:
            output_dir += "_freezefusion"

        final_outdir = output_dir + "-final"

        print("Training:", output_dir)

        config = {
            "fusion_embedder": self.fusion_embedder._config,
            "genus_classifier": self.genus_classifier._config,
            "local_specie_classifier": self.local_specie_classifier._config,
        }

        if self.dna_embedder:
            for param in self.dna_embedder.parameters():
                param.requires_grad = False
        if self.img_embedder:
            for param in self.img_embedder.parameters():
                param.requires_grad = False

        # HuggingFace TrainingArguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            eval_strategy="steps" if eval_dataset is not None else "no",
            save_strategy="steps",
            learning_rate=lr,
            eval_steps=eval_steps if eval_dataset is not None else None,
            save_steps=save_steps,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=epochs,
            weight_decay=0.01,
            logging_dir=f"{output_dir}/logs",
            logging_steps=50,
            load_best_model_at_end=True,
            do_train=True,
            # load_best_model_at_end=True if eval_dataset is not None else False,
            # metric_for_best_model="accuracy",   
            greater_is_better=True,
            lr_scheduler_type="linear",
            warmup_steps=200,
            report_to=["tensorboard"],        # or ["wandb"], ["mlflow"], ["comet_ml"]
        )
        print(f"Training arguments: {training_args}")
    

        if only_genus:
            def compute_metrics_genus(eval_pred):
                logits, slabels = eval_pred
                if isinstance(logits, (tuple, list)):
                    logits = logits[0]
                slabels = np.array(slabels).reshape(-1)
                preds = np.argmax(logits, axis=-1)
                labels = self.species2genus[slabels].cpu().numpy()
                return {"accuracy": accuracy.compute(predictions=preds, references=labels)["accuracy"]}

        
        # Define Trainer
        trainer = Trainer(
            model=self,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics_genus if only_genus else compute_metrics,
            data_collator=multimodal_collector,
        )

        # Train
        trainer.train()

        trainer.save_model(final_outdir)

        eval_results = trainer.evaluate()
        print(eval_results)
        
        try:
            with open(f"{final_outdir}/my_model_config.json", "w") as f:
                json.dump(config, f)
        except:
            try:
                with open(f"{final_outdir}_my_model_config.json", "w") as f:
                    json.dump(config, f)
            except:
                pass

        print(f"Model saved to: {final_outdir}")

        return final_outdir, trainer, eval_results

    def load_model_weights(self, model_path):
        state_dict = load_file(f"{model_path}/model.safetensors")

        return self.load_state_dict(state_dict, strict=False)
    
    @classmethod
    def load_model(cls, model_path, species2genus, genus_species, dna_embedder=None, img_embedder=None, device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        with open(f"{model_path}/my_model_config.json", "r") as f:
            config = json.load(f)
        print("Loaded model config:", config)
        
        fusion_embedder = AttentionFusion(**config["fusion_embedder"])
        genus_classifier = GenusClassifier(**config["genus_classifier"])
        local_specie_classifier = LocalSpecieClassfier(**config["local_specie_classifier"])

        # print(fusion_embedder, fusion_embedder.fused_dim)
        # print(genus_classifier, genus_classifier.fused_dim)


        state_dict = load_file(f"{model_path}/model.safetensors")

        self = MainClassifier(species2genus, genus_species, dna_embedder, img_embedder, fusion_embedder, genus_classifier, local_specie_classifier).to(device)
        print("Loaded model state:", self.load_state_dict(state_dict, strict=False))

        return self

    @classmethod
    def load_modelV2(cls, model_path, species2genus, genus_species, dna_embedder=None, img_embedder=None, device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        with open(f"{model_path}/my_model_config.json", "r") as f:
            config = json.load(f)
        print("Loaded model config:", config)
        
        fusion_embedder = AttentionFusionV2(**config["fusion_embedder"], dna_dim=512, img_dim=768)
        genus_classifier = GenusClassifier(**config["genus_classifier"])
        local_specie_classifier = LocalSpecieClassfier(**config["local_specie_classifier"])

        # print(fusion_embedder, fusion_embedder.fused_dim)
        # print(genus_classifier, genus_classifier.fused_dim)


        state_dict = load_file(f"{model_path}/model.safetensors")

        self = MainClassifier(species2genus, genus_species, dna_embedder, img_embedder, fusion_embedder, genus_classifier, local_specie_classifier).to(device)
        print("Loaded model state:", self.load_state_dict(state_dict, strict=False))

        return self


        

    def evaluate(self, datasets: dict, only_genus=False):
        
        training_args = TrainingArguments(
            # per_device_train_batch_size=32,
            per_device_eval_batch_size=32,
            # num_train_epochs=3,
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


        for (data_name,dataset) in datasets.items():

            print(f"Evaluating {data_name}...")
            # Split data using your indices

            self._only_genus = only_genus

            if only_genus:
                def compute_metrics_genus(eval_pred):
                    logits, slabels = eval_pred
                    if isinstance(logits, (tuple, list)):
                        logits = logits[0]
                    slabels = np.array(slabels).reshape(-1)
                    preds = np.argmax(logits, axis=-1)
                    labels = self.species2genus[slabels].cpu().numpy()
                    return {"accuracy": accuracy.compute(predictions=preds, references=labels)["accuracy"]}
            
            trainer = Trainer(
                model=self,
                args=training_args,
                eval_dataset=dataset,
                data_collator=multimodal_collector,
                compute_metrics=compute_metrics_genus if only_genus else compute_metrics
            )
            # Evaluate
            results = trainer.evaluate()

            yield (data_name, results)


    def evaluate_genus(self, datasets: dict, batch=256):
        def evaluate_dataset(d):
            data_type, dataset = d
            print("Evaluating genus for", data_type)
            device = next(self.parameters()).device
            all_predicted = torch.tensor([], dtype=torch.long, device=device)
            all_true = torch.tensor([], dtype=torch.long, device=device)
            for i in range(len(dataset) // batch):
                batch_data = multimodal_collector([dataset[di] for di in range(i * batch, (i + 1) * batch)])
                # Move all tensors in batch_data to the model's device
                for k, v in batch_data.items():
                    if isinstance(v, dict):
                        for kk, vv in v.items():
                            batch_data[k][kk] = vv.to(device)
                    else:
                        batch_data[k] = v.to(device)
                genus_logits, *_ = self.predict_genus(**batch_data)
                predicted_genus = genus_logits.argmax(dim=-1)
                true_genus = batch_data['genus'].squeeze()
                # print(predicted_genus, true_genus)
                all_predicted = torch.cat((all_predicted, predicted_genus), dim=0)
                all_true = torch.cat((all_true, true_genus), dim=0)
            # print(predicted_genus.shape, true_genus.shape, predicted_genus[:2], true_genus[:2])

            # print({
            #     # "predicted_genus": all_predicted,
            #     # "true_genus": all_true,
            #     "accuracy": (all_predicted == all_true).float().mean().item()
            # }, data_type, all_predicted[:20], all_true[:20])
            
            return data_type, {
                # "predicted_genus": all_predicted,
                # "true_genus": all_true,
                "accuracy": (all_predicted == all_true).float().mean().item()
            }

        return map(evaluate_dataset, datasets.items())







def evaluate_model(mat, model_path="./results_20250819T030148-final", device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        
    all_images = mat['all_images']
    all_labels = mat['all_labels'].squeeze()

    val_seen_indices = mat['val_seen_loc'].squeeze()
    val_unseen_indices = mat['val_unseen_loc'].squeeze()
    test_seen_indices = mat['test_seen_loc'].squeeze()
    test_unseen_indices = mat['test_unseen_loc'].squeeze()

    # val_indices = np.concatenate((val_seen_indices, val_unseen_indices))
    # test_indices = np.concatenate((test_seen_indices, test_unseen_indices))

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


    for (data_name,indices) in {
        "val_seen_indices":val_seen_indices,
        "val_unseen_indices":val_unseen_indices,
        "test_seen_indices":test_seen_indices,
        "test_unseen_indices":test_unseen_indices
        }.items():

        print(f"Evaluating {data_name}...")
        # Split data using your indices
        images = torch.tensor(all_images[indices])
        labels = all_labels[indices]

        dataset = ImageDataset(images, labels, processor, device)

        trainer = Trainer(
            model=model,
            args=training_args,
            eval_dataset=dataset,
            data_collator=data_collator,
            compute_metrics=compute_metrics
        )
        # Evaluate
        results = trainer.evaluate()

        yield (data_name, results)