import torch
import torch.nn.functional as F
import os


@torch.no_grad()
def generate(
    model,
    tokenizer,
    prompt=None,                # texte source à encoder (optionnel, mais recommandé)
    start_tokens=None,          # liste d’IDs de départ (target BOS). Ex: [bos_id]
    max_len=50,
    device=None,
    temperature=1.3,
    top_k=None,
    eos_id=None,
    greedy=True
):
    model.eval()
    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

    # Vérifications et préparation des tokens
    if prompt is not None:
        src_tokens = tokenizer.encode(prompt)
    else:
        src_tokens = []

    if start_tokens is None:
        bos_id = tokenizer.special_tokens_map.get("<|bos|>")
        if bos_id is None:
            raise ValueError("start_tokens n'est pas fourni et <|bos|> introuvable dans tokenizer.")
        start_tokens = [bos_id]

    # Debug : inspecte ce que tu vas encoder
    print("DEBUG: src_tokens len =", len(src_tokens), "sample:", src_tokens[:30])
    print("DEBUG: start_tokens =", start_tokens)

    # Construire les tenseurs d'entrée (batch=1)
    if len(src_tokens) == 0:
        # Pas de prompt : on encode juste le BOS comme "source" minimal
        src = torch.tensor(start_tokens, dtype=torch.long, device=device).unsqueeze(0)  # (1, src_len)
    else:
        src = torch.tensor(src_tokens, dtype=torch.long, device=device).unsqueeze(0)   # (1, src_len)

    tgt = torch.tensor(start_tokens, dtype=torch.long, device=device).unsqueeze(0)     # (1, cur_len)

    # Encode ONCE la source (ne pas réencoder dans la boucle)
    enc_out = model.encode(src, None)   # (1, src_len, emb)
    src_len = enc_out.size(1)

    # debug shapes
    print("DEBUG: enc_out.shape:", enc_out.shape)
    print("DEBUG: initial tgt.shape:", tgt.shape)

    # Génération autoregressive
    for _ in range(max_len - tgt.size(1)):
        cur_len = tgt.size(1)

        # Masque causal pour la target (autoregressive) : (1,1,cur_len,cur_len)
        tgt_mask = torch.tril(torch.ones((cur_len, cur_len), device=device))
        tgt_mask = tgt_mask.unsqueeze(0).unsqueeze(0)

        # Mask pour la cross-attention : (1,1,cur_len,src_len)
        src_mask = torch.ones((1, 1, cur_len, src_len), device=device)

        # Décode
        dec_out = model.decode(enc_out, src_mask, tgt, tgt_mask)   # (1, cur_len, emb)
        logits = model.project(dec_out)                           # (1, cur_len, vocab_size)
        next_token_logits = logits[:, -1, :] / max(temperature, 1e-8)

        # Choix du token suivant
        if greedy:
            next_id = int(torch.argmax(next_token_logits, dim=-1).item())
        else:
            if top_k is not None and top_k > 0:
                values, indices = torch.topk(next_token_logits, top_k)
                probs = F.softmax(values, dim=-1)
                idx = torch.multinomial(probs, 1).item()
                next_id = int(indices[0, idx].item())
            else:
                probs = F.softmax(next_token_logits, dim=-1)
                next_id = int(torch.multinomial(probs, 1).item())

        # Append
        next_token = torch.tensor([[next_id]], dtype=torch.long, device=device)
        tgt = torch.cat([tgt, next_token], dim=1)

        # Stop sur EOS
        if eos_id is not None and next_id == eos_id:
            break

    return tgt.squeeze(0).tolist()

# @torch.no_grad()
# def generate(model, tokenizer, prompt=None, start_tokens=None, max_len=50, device=None, temperature=1.0, top_k=None, eos_id=None, greedy=True):
#     model.eval()

#     if prompt is None:
#         prompt = "<|who_i_am|>AgentAI<|end_who_i_am|><|bos|>"

#     source = tokenizer.encode(prompt)
#     src = torch.tensor(source, dtype=torch.long, device=device).unsqueeze(0)   # (1, src_len)

#     source_mask = (source != tokenizer.special_tokens_map.get("<|pad|>")).unsqueeze(0).unsqueeze(0).int().to(device)

#     enc_out = model.encode(src, source_mask)   # (1, src_len, emb)

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