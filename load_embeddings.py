import sys
import torch
import os 
import pickle



def save_dna():
    all_dnas = mat['all_string_dnas']

    for i in range(0, len(all_dnas), B):
        device = torch.device(f"cuda:2")
        print("Processing batch:", i, "to", i+B, "of", len(all_dnas), f"{i/len(all_dnas)*100:4f}%", "on device:", device)
        batch = all_dnas[i:i+B]
        # compute embeddings on GPU
        batch_emb = get_dna_embedding(batch, dna_tokenizer, dna_encoder.to(device), device)
        
        # detach and move to CPU if you don’t want GPU memory to explode
        batch_emb = batch_emb.detach().cpu()
        with open(os.path.join(save_dir, f"batch_dna_{i//B}.pkl"), 'wb') as f:
            pickle.dump(batch_emb, f)

def save_img():
    all_images = mat['all_images']

    for i in range(0, len(all_images), B):
        print("Processing batch:", i, "to", i+B, "of", len(all_images), f"{i/len(all_images)*100:4f}%")
        batch = all_images[i:i+B]
        # compute embeddings on GPU
        batch_emb = get_img_embedding(batch, img_processor, img_encoder.to(device), device)
        
        # detach and move to CPU if you don’t want GPU memory to explode
        batch_emb = batch_emb.detach().cpu()
        with open(os.path.join(save_dir, f"batch_img_{i//B}.pkl"), 'wb') as f:
            pickle.dump(batch_emb, f)

def load_img_embeddings(path="./output_embeddings/img_embeddings"):
    all_images_features = []
    for file in os.listdir(path):
        print("Loading file:", file)
        if not file.endswith('.pkl'):
            continue
        with open(os.path.join(path,file), 'rb') as f:
            batch_emb = pickle.load(f)
            all_images_features.append(batch_emb)
    return torch.cat(all_images_features, dim=0).to(torch.float16)


def load_dna_embeddings(path="./output_embeddings/dna_embeddings"):
    all_images_features = []
    for file in os.listdir(path):
        print("Loading file:", file)
        if not file.endswith('.pkl'):
            continue
        with open(os.path.join(path,file), 'rb') as f:
            batch_emb = pickle.load(f)
            all_images_features.append(batch_emb)
    return torch.cat(all_images_features, dim=0).to(torch.float16)


if __name__ == "__main__":
    from mat import mat

    assert len(sys.argv) >= 4, "Please provide the device and save directory as command line arguments."

    device = torch.device(sys.argv[1]) if len(sys.argv) > 1 else torch.device("cpu")

    save_dir = sys.argv[2] if len(sys.argv) > 2 else "./embeddings"

    os.mkdir(save_dir) if not os.path.exists(save_dir) else None

    from vit import get_processor_encoder, get_img_embedding
    img_processor, img_encoder = get_processor_encoder("./vit-finetuned7-final", device)

    from dnaencoder import get_tokenizer_encoder, get_dna_embedding
    dna_tokenizer, dna_encoder = get_tokenizer_encoder("./dnaencoder-finetuned1755100772-final", device)

    B = int(sys.argv[3] )   # batch size