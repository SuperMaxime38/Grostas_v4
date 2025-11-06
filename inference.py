import torch
import torch.nn.functional as F

def generate(model, tokenizer, start_tokens, max_len=50, device=None,
             temperature=1.0, top_k=None, eos_id=None, greedy=True):
    """
    Génère une suite de tokens à partir d'un contexte `start_tokens` (liste d'entiers).
    - model : ton Transformer (déjà sur device)
    - tokenizer : instance de ton Tokenizer (pour récupérer special tokens si besoin)
    - start_tokens : liste[int] (le contexte initial, ex: [bos_id, ...])
    - max_len : nombre max de tokens à générer (inclut start_tokens)
    - device : torch.device (par défaut prend model device si None)
    - temperature : float (>0) pour sampling
    - top_k : int ou None (si défini, sampling limité aux top_k logits)
    - eos_id : id entier du token de fin (optionnel)
    - greedy : si True, on fait argmax; sinon sampling stochastique
    """
    model.eval()
    if device is None:
        # si model est sur cuda, récupère device du premier paramètre
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

    with torch.no_grad():
        # batch dimension = 1
        src = torch.tensor(start_tokens, dtype=torch.long, device=device).unsqueeze(0)  # (1, L0)
        # encode le contexte (si tu veux générer sans contexte mettre src=torch.tensor([[bos_id]])

        enc_out = model.encode(src, None)  # (1, L0, emb)
        src_mask = torch.ones((1, 1, 1, enc_out.size(1)), device=device)

        # generated commence par le même start tokens (on suit ton training)
        generated = src.clone()  # (1, cur_len)

        for _ in range(max_len - generated.size(1)):
            cur_len = generated.size(1)
            # Mask causal : 1 for allowed positions, 0 for masked (upper triangle)
            tgt_mask = torch.tril(torch.ones((cur_len, cur_len), device=device))
            tgt_mask = tgt_mask.unsqueeze(0).unsqueeze(0)  # (1,1,cur_len,cur_len)

            # décodage autoregressif
            print("enc_out:", enc_out.shape)
            print("generated:", generated.shape)

            dec_out = model.decode(enc_out, src_mask, generated, tgt_mask)
            logits = model.project(dec_out)  # (1, cur_len, vocab_size)
            next_logits = logits[:, -1, :]  # (1, vocab)

            if greedy:
                # argmax déterministe
                next_id = int(torch.argmax(next_logits, dim=-1).item())
            else:
                # sampling temperature + top_k
                logits_scaled = next_logits / max(temperature, 1e-8)
                if top_k is not None and top_k > 0:
                    values, indices = torch.topk(logits_scaled, top_k)
                    probs = torch.zeros_like(logits_scaled).scatter_(-1, indices, F.softmax(values, dim=-1))
                else:
                    probs = F.softmax(logits_scaled, dim=-1)
                next_id = int(torch.multinomial(probs.squeeze(0), num_samples=1).item())

            # append
            next_token = torch.tensor([[next_id]], dtype=torch.long, device=device)
            generated = torch.cat([generated, next_token], dim=1)

            # stop if eos
            if eos_id is not None and next_id == eos_id:
                break

        return generated.squeeze(0).tolist()