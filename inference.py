import torch
import torch.nn.functional as F
import os

@torch.no_grad()
def generate(model, tokenizer, prompt=None, start_tokens=None,
             max_len=128, device=None, eos_id=None, temperature=1.2, top_k=20, greedy=False):

    model.eval()
    with torch.no_grad():
        # === Cas 1 : modèle seq2seq
        if prompt is not None:
            src = torch.tensor(tokenizer.encode(prompt), dtype=torch.long, device=device).unsqueeze(0)
            enc_out = model.encode(src, None)
        else:
            enc_out = None

        # === Cas 2 : modèle decoder-only
        generated = torch.tensor(start_tokens, dtype=torch.long, device=device).unsqueeze(0)

        for _ in range(max_len - generated.size(1)):
            cur_len = generated.size(1)
            tgt_mask = torch.tril(torch.ones((cur_len, cur_len), device=device)).unsqueeze(0).unsqueeze(0)

            dec_out = model.decode(enc_out, None, generated, tgt_mask)
            logits = model.project(dec_out)[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)

            if top_k:
                topv, topi = torch.topk(probs, top_k)
                probs = torch.zeros_like(probs).scatter_(-1, topi, F.softmax(topv, dim=-1))

            next_id = int(torch.multinomial(probs, 1))
            generated = torch.cat([generated, torch.tensor([[next_id]], device=device)], dim=1)

            if eos_id is not None and next_id == eos_id:
                break

        return generated.squeeze(0).tolist()

@torch.no_grad()
def generate_decoder_only(model, start_tokens, tokenizer, device, max_len=128,
                          temperature=1.0, top_k=20, eos_id=None):
    model.eval()
    generated = torch.tensor(start_tokens, dtype=torch.long, device=device).unsqueeze(0)

    for _ in range(max_len - len(start_tokens)):
        cur_len = generated.size(1)
        mask = torch.tril(torch.ones((cur_len, cur_len), device=device)).unsqueeze(0).unsqueeze(0)

        logits = model(generated, mask)[:, -1, :] / temperature
        probs = torch.softmax(logits, dim=-1)

        if top_k:
            topv, topi = torch.topk(probs, top_k)
            probs = torch.zeros_like(probs).scatter_(-1, topi, torch.softmax(topv, dim=-1))

        next_id = int(torch.multinomial(probs, 1))
        generated = torch.cat([generated, torch.tensor([[next_id]], device=device)], dim=1)

        if eos_id is not None and next_id == eos_id:
            break

    return generated.squeeze(0).tolist()

def save_model(model, optimizer, epoch, loss, path="checkpoints/transformer.pt"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, path)
    print(f"✅ Modèle sauvegardé dans {path}")

def load_model(model_class, checkpoint_path, device, **model_kwargs):
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Reconstruit un modèle vierge avec la même architecture
    model = model_class(**model_kwargs)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()  # ou .train() si tu veux continuer l'entraînement

    print(f"✅ Modèle chargé depuis {checkpoint_path} (epoch {checkpoint['epoch']}, loss={checkpoint['loss']:.4f})")
    return model