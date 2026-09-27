import torch
import torch.nn as nn
import torch.optim as optim
import model as mdl
from fastbpe import Tokenizer
import data_loader as dl
import os
import numpy as np
import threading
import time

from inference import generate_decoder_only

# ============================
# CONFIGURATION DE BASE
# ============================

last_saved_epoch = 0
torch.cuda.set_per_process_memory_fraction(0.95, device=0)

def save_model(model, optimizer, epoch, loss, path="checkpoints/transformer.pt"):
    torch.cuda.empty_cache()
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
num_epochs = 5
lr = 1e-5

# Construction du modèle
model = load_model("checkpoints/transformer.pt", device, vocab_size=vocab_size, max_seq_len=seq_len, embedding_dim=embedding_dim)
model = model.to(device)

tokenizer = Tokenizer(vocab_size)

criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.special_tokens_map.get("<|pad|>"))  # supposer que token 0 = padding
optimizer = optim.AdamW(model.parameters(), lr=lr)

n_params = sum(p.numel() for p in model.parameters())
print(f"{n_params:,} paramètres ({n_params/1e6:.1f}M)")

CONTROL_FILE = "training_control.txt"

def check_control(epoch, avg_loss):
    """
    Lit un fichier de contrôle pour réguler l'entraînement en direct.
    Contenu possible dans le fichier :
      "pause"        -> met en pause jusqu'à changement du fichier
      un nombre (ex: "0.5") -> attend ce délai (en secondes) après le step
      "run" ou fichier absent -> vitesse normale
    """
    if not os.path.exists(CONTROL_FILE):
        return

    while True:
        try:
            with open(CONTROL_FILE, "r") as f:
                content = f.read().strip().lower()
        except (FileNotFoundError, PermissionError):
            return

        if content == "pause":
            time.sleep(1)
            continue  # revérifie après 1s, reste en pause tant que le fichier dit "pause"
        elif content == "save":
            with open(CONTROL_FILE, "w+") as fw:
                save_model(model, optimizer, epoch + last_saved_epoch + 1, avg_loss, "checkpoints/transformer.pt")
                fw.write("pause")
                fw.close()
            return
        elif content in ("run", ""):
            return
        else:
            try:
                time.sleep(float(content))
            except ValueError:
                pass
            return
def train_model():
    model.train()
    scaler = torch.amp.GradScaler('cuda')  # créé une seule fois

    accum_steps = 4  # batch effectif = micro_batch_size * accum_steps = 8

    for epoch in range(num_epochs):
        total_loss = 0.0
        n_micro_batches = 0
        optimizer.zero_grad()

        for i, batch in enumerate(batchify(os.path.join("datas\\tokenized", "tokens.bin"), seq_len, batch_size // accum_steps, dtype=np.uint32)):
            batch = torch.tensor(batch, dtype=torch.long, device=device)
            input_seq = batch[:, :-1]
            target_seq = batch[:, 1:]
            tgt_mask = torch.tril(torch.ones((seq_len-1, seq_len-1), device=device)).unsqueeze(0).unsqueeze(0)

            with torch.autocast(device_type='cuda', dtype=torch.float16):
                logits = model(input_seq, tgt_mask)
                loss = criterion(logits.view(-1, logits.size(-1)), target_seq.reshape(-1))

            scaler.scale(loss / accum_steps).backward()

            total_loss += loss.item()  # loss réel, non divisé -> logging correct
            n_micro_batches += 1

            if (i + 1) % accum_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                check_control(epoch, total_loss / n_micro_batches)

        # flush du dernier groupe incomplet, pour ne pas polluer l'epoch suivante
        if n_micro_batches % accum_steps != 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        avg_loss = total_loss / n_micro_batches
        print(f"Epoch {last_saved_epoch + epoch+1}/{num_epochs + last_saved_epoch} | Loss: {avg_loss:.4f}")

        if (epoch + last_saved_epoch + 1) % 10 == 0 and epoch != num_epochs - 1:
            save_model(model, optimizer, epoch + last_saved_epoch + 1, avg_loss, "checkpoints/transformer.pt")

    save_model(model, optimizer, epoch + last_saved_epoch + 1, avg_loss, "checkpoints/transformer.pt")
    

def batchify(path, seq_len, batch_size, dtype=np.uint32):
    """
    Crée des batches rectangulaires à partir d'un fichier de tokens
    trop gros pour tenir en RAM, via memory-mapping.

    Le fichier doit contenir des tokens en binaire brut, stockés
    avec `dtype` (ex: np.uint16 si vocab < 65536).
    """
    itemsize = np.dtype(dtype).itemsize
    file_size = os.path.getsize(path)
    total_tokens = file_size // itemsize

    # nombre de tokens utilisables (divisible par seq_len * batch_size)
    total_len = (total_tokens // (seq_len * batch_size)) * (seq_len * batch_size)
    n_per_row = total_len // batch_size

    # mmap : ne charge rien en RAM tant qu'on n'accède pas aux données
    mm = np.memmap(path, dtype=dtype, mode='r', shape=(total_tokens,))

    # vue reshape (batch_size, n_per_row) - pas de copie, toujours mmap
    tokens_view = mm[:total_len].reshape(batch_size, n_per_row)

    for i in range(0, n_per_row - seq_len, seq_len):
        # ce slice déclenche la lecture disque des SEULES pages concernées
        batch = np.array(tokens_view[:, i:i+seq_len])  # copie -> vrai ndarray en RAM
        yield batch

if __name__ == "__main__":

    train_model()

    # Après l’entraînement
    eos_id = tokenizer.special_tokens_map.get("<|eos|>")
    # bos_tokens = tokenizer.encode("<|who_i_am|>maxime38<|end_who_i_am|><|bos|>")
    bos_tokens = tokenizer.encode("<|who_i_am|>canward<|end_who_i_am|><|bos|>")
    print(bos_tokens)

    for i in range(4):

        output_ids = generate_decoder_only(model=model, start_tokens=bos_tokens, tokenizer=tokenizer, device=device, max_len=128, temperature=1.0, top_k=20, eos_id=eos_id)

        # print("Generated IDs:", output_ids)
        print("Generated text:", tokenizer.decode(output_ids))


