import torch
import torch.nn as nn
import torch.optim as optim
import model as mdl
import tokenizer as tkn
import data_loader as dl

from inference import generate

# ============================
# CONFIGURATION DE BASE
# ============================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("using device:", device)

vocab_size = 16384                  # Ton vocabulaire
seq_len = 256                       # Longueur max de séquence
embedding_dim = 1024
batch_size = 8
num_epochs = 10
lr = 1e-5

# Construction du modèle
model = mdl.get_model()
model = model.to(device)

criterion = nn.CrossEntropyLoss(ignore_index=0)  # supposer que token 0 = padding
optimizer = optim.AdamW(model.parameters(), lr=lr)

tokenizer = tkn.Tokenizer(vocab_size)

def train_model():
    
    dataset_tokens = tokenizer.encode(dl.gather_datas())

    print("Dataset size:", len(dataset_tokens))

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        for src, tgt in batchify(dataset_tokens, seq_len, batch_size):
            if (src < 0).any() or (src >= vocab_size).any():
                print("src out of range")
            if (tgt < 0).any() or (tgt >= vocab_size).any():
                print("tgt out of range")
            src_mask = None
            tgt_len = tgt.size(1)
            # Crée un masque causal 1 (autorisée) / 0 (bloquée)
            tgt_mask = torch.tril(torch.ones((tgt_len, tgt_len), device=device)).unsqueeze(0).unsqueeze(0)
            # (1, 1, tgt_len, tgt_len)

            # Forward
            enc_out = model.encode(src, src_mask)
            dec_out = model.decode(enc_out, src_mask, tgt, tgt_mask)
            logits = model.project(dec_out)

            # Calcul de la perte
            loss = criterion(logits.view(-1, vocab_size), tgt.view(-1))

            if torch.isnan(loss) or torch.isinf(loss):
                print("Loss NaN or Inf detected before backward")
                print("Logits:", logits[0, 0, :5])
                break

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()

            

            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(list(batchify(dataset_tokens, seq_len, batch_size)))
        print(f"Epoch {epoch+1}/{num_epochs} | Loss: {avg_loss:.4f}")
    

def batchify(tokens, seq_len, batch_size):
    """
    Coupe les tokens en séquences de longueur fixe et crée des batchs.
    """
    chunks = [tokens[i:i+seq_len+1] for i in range(0, len(tokens)-seq_len, seq_len)]
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        src = [torch.tensor(x[:-1]) for x in batch]
        tgt = [torch.tensor(x[1:]) for x in batch]
        src = nn.utils.rnn.pad_sequence(src, batch_first=True)
        tgt = nn.utils.rnn.pad_sequence(tgt, batch_first=True)
        yield src.to(device), tgt.to(device)

if __name__ == "__main__":

    train_model()

    src_tokens = tokenizer.encode("Quel est ton nom ?")
    bos_id = tokenizer.special_tokens_map.get("<|bos|>")
    eos_id = tokenizer.special_tokens_map.get("<|eos|>")

    out_ids = generate(
        model=model,
        tokenizer=tokenizer,
        start_tokens=[bos_id],   # ce que le décodeur va générer
        max_len=50,
        device=device,
        eos_id=eos_id
    )

    text = tokenizer.decode(out_ids)
    print(text)


