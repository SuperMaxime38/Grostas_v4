import torch
import torch.nn as nn
import torch.optim as optim
import model as mdl
from fastbpe import Tokenizer
import data_loader as dl
import os
import numpy as np

from inference import generate_decoder_only

# ============================
# CONFIGURATION DE BASE
# ============================

last_saved_epoch = 0

def save_model(model, optimizer, epoch, loss, path="checkpoints/transformer.pt"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, path)
    print(f"✅ Modèle sauvegardé dans {path}")

def load_model(checkpoint_path, device, **model_kwargs):
    if os.path.exists(checkpoint_path) == False:
        return mdl.TransformerDecoderOnly(**model_kwargs)
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Reconstruit un modèle vierge avec la même architecture
    model = mdl.TransformerDecoderOnly(**model_kwargs)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.train()  # ou .train() si tu veux continuer l'entraînement

    global last_saved_epoch
    last_saved_epoch = checkpoint['epoch']

    print(f"✅ Modèle chargé depuis {checkpoint_path} (epoch {checkpoint['epoch']}, loss={checkpoint['loss']:.4f})")
    return model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("using device:", device)

vocab_size = 24576                  # Ton vocabulaire
seq_len = 256                       # Longueur max de séquence
embedding_dim = 1024
batch_size = 8
num_epochs = 80
lr = 1e-5

# Construction du modèle
model = mdl.TransformerDecoderOnly(vocab_size, embedding_dim, num_heads=8, num_layers=6, dropout=0.1, d_ff=2048, max_seq_len=128)
model = load_model("checkpoints/transformer.pt", device, vocab_size=vocab_size, max_seq_len=seq_len, embedding_dim=embedding_dim)
model = model.to(device)

tokenizer = Tokenizer(vocab_size)

criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.special_tokens_map.get("<|pad|>"))  # supposer que token 0 = padding
optimizer = optim.AdamW(model.parameters(), lr=lr)


def train_model():
    
    dataset_tokens = dl.get_data_tokens_as_list()

    print("Dataset size:", len(dataset_tokens))
    print("dataset sample:", dataset_tokens[:50])

    model.train()

    for epoch in range(num_epochs):
        total_loss = 0
        for batch in batchify(dataset_tokens, seq_len, batch_size):
            optimizer.zero_grad()

            batch = torch.tensor(batch, dtype=torch.long, device=device)
            input_seq = batch[:, :-1]
            target_seq = batch[:, 1:]

            # masque causal
            tgt_mask = torch.tril(torch.ones((seq_len-1, seq_len-1), device=device)).unsqueeze(0).unsqueeze(0)

            logits = model(input_seq, tgt_mask)
            loss = criterion(logits.view(-1, logits.size(-1)), target_seq.reshape(-1))

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(list(batchify(dataset_tokens, seq_len, batch_size)))
        print(f"Epoch {last_saved_epoch + epoch+1}/{num_epochs + last_saved_epoch} | Loss: {avg_loss:.4f}")

        if((epoch+last_saved_epoch+1) % 50 == 0 and epoch != num_epochs - 1):
            save_model(model, optimizer, epoch + last_saved_epoch + 1, avg_loss, "checkpoints/transformer.pt")

            # shuffle dataset
            dataset_tokens = dl.convert_datas_to_tokens_with_shuffle(tokenizer=tokenizer, random_val=5)
            print("Shuffled dataset")
    
    save_model(model, optimizer, epoch + last_saved_epoch + 1, avg_loss, "checkpoints/transformer.pt")
    

def batchify(tokens, seq_len, batch_size):
    """
    Crée des batches rectangulaires à partir d'une liste de tokens.
    Coupe les séquences à seq_len et forme des batches homogènes.
    """
    # enlève les derniers tokens non divisibles
    total_len = (len(tokens) // (seq_len * batch_size)) * (seq_len * batch_size)
    tokens = tokens[:total_len]

    # transforme en numpy array
    tokens_np = np.array(tokens, dtype=np.int64)
    tokens_np = tokens_np.reshape(batch_size, -1)  # (batch_size, N)

    for i in range(0, tokens_np.shape[1] - seq_len, seq_len):
        batch = tokens_np[:, i:i+seq_len]
        yield batch

if __name__ == "__main__":

    train_model()

    # Après l’entraînement
    eos_id = tokenizer.special_tokens_map.get("<|eos|>")
    bos_tokens = tokenizer.encode("<|who_i_am|>canward<|end_who_i_am|><|bos|>gros")
    print(bos_tokens)

    output_ids = generate_decoder_only(model=model, start_tokens=bos_tokens, tokenizer=tokenizer, device=device, max_len=128, temperature=1.0, top_k=20, eos_id=eos_id)

    print("Generated IDs:", output_ids)
    print("Generated text:", tokenizer.decode(output_ids))


