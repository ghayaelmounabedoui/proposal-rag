"""Lab : un mini Transformer (decoder-only) et LoRA, écrits à la main avec PyTorch.

Ce fichier accompagne la formation `docs/formation-transformers-lora.md`.
Il est volontairement petit et lisible, pas performant.

Exécution (depuis la racine du projet, venv activé) :

    python docs/labs/mini_transformer_lora.py

Chaque `check_*` correspond à une propriété qu'il faut savoir expliquer :
causalité, équivalence du KV cache, LoRA nul à l'initialisation, fusion exacte,
seuls les paramètres LoRA sont entraînés.
"""

import math
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------
# 1. Briques du Transformer
# --------------------------------------------------------------------------
class RMSNorm(nn.Module):
    """Normalisation de Llama : pas de moyenne soustraite, pas de biais."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms * self.weight


class CausalSelfAttention(nn.Module):
    """Attention multi-têtes causale, avec KV cache optionnel."""

    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        assert d_model % n_heads == 0, "d_model doit être divisible par n_heads"
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, past=None):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)

        def heads(t):  # (B, T, C) -> (B, h, T, head_dim)
            return t.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        q, k, v = heads(q), heads(k), heads(v)

        if past is not None:  # on ajoute les nouvelles clés/valeurs au cache
            k = torch.cat([past[0], k], dim=2)
            v = torch.cat([past[1], v], dim=2)
        present = (k, v)

        S = k.size(2)  # longueur totale vue par l'attention
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B, h, T, S)

        # La requête i correspond à la position absolue S - T + i : elle voit les clés j <= S - T + i.
        mask = torch.ones(T, S, dtype=torch.bool, device=x.device).tril(diagonal=S - T)
        scores = scores.masked_fill(~mask, float("-inf"))

        weights = scores.softmax(dim=-1)
        out = (weights @ v).transpose(1, 2).reshape(B, T, C)
        return self.proj(out), present


class SwiGLU(nn.Module):
    """Réseau feed-forward de Llama : (SiLU(x W_gate) * x W_up) W_down."""

    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.gate = nn.Linear(d_model, d_ff, bias=False)
        self.up = nn.Linear(d_model, d_ff, bias=False)
        self.down = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    """Bloc « pre-norm » : x + Attn(Norm(x)), puis x + FFN(Norm(x))."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int) -> None:
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.norm2 = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, d_ff)

    def forward(self, x, past=None):
        a, present = self.attn(self.norm1(x), past)
        x = x + a  # connexion résiduelle
        x = x + self.ffn(self.norm2(x))
        return x, present


class TinyGPT(nn.Module):
    def __init__(self, vocab_size, d_model=64, n_heads=4, n_layers=2, d_ff=172, max_len=256):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)  # positions apprises (simple)
        self.blocks = nn.ModuleList(Block(d_model, n_heads, d_ff) for _ in range(n_layers))
        self.norm = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight  # poids partagés (weight tying)
        # Petite initialisation : au départ le modèle est « ignorant », donc sa perte vaut ≈ ln(V).
        nn.init.normal_(self.tok_emb.weight, std=0.02)
        nn.init.normal_(self.pos_emb.weight, std=0.02)

    def forward(self, idx, past=None):
        B, T = idx.shape
        offset = 0 if past is None else past[0][0].size(2)
        pos = torch.arange(offset, offset + T, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        presents = []
        for i, block in enumerate(self.blocks):
            x, present = block(x, None if past is None else past[i])
            presents.append(present)
        return self.lm_head(self.norm(x)), presents


# --------------------------------------------------------------------------
# 2. Génération : température, top-k, top-p, KV cache
# --------------------------------------------------------------------------
def filter_logits(logits, temperature=1.0, top_k=0, top_p=1.0):
    """Applique température, top-k puis top-p (noyau) à un vecteur de logits (V,)."""
    logits = logits / max(temperature, 1e-8)
    if top_k > 0:
        kth = torch.topk(logits, min(top_k, logits.numel())).values[-1]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    if top_p < 1.0:
        sorted_logits, order = torch.sort(logits, descending=True)
        probs = sorted_logits.softmax(-1)
        cumulative = probs.cumsum(-1)
        remove = cumulative - probs > top_p  # garde le plus petit ensemble dont la masse dépasse top_p
        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
        logits = torch.full_like(logits, float("-inf")).scatter(0, order, sorted_logits)
    return logits


@torch.no_grad()
def generate(model, idx, max_new_tokens, temperature=1.0, top_k=0, top_p=1.0, use_cache=True):
    model.eval()
    past = None
    for _ in range(max_new_tokens):
        # Avec cache : le « prefill » traite tout le prompt, puis on ne passe qu'UN token à chaque pas.
        inp = idx if (past is None or not use_cache) else idx[:, -1:]
        logits, presents = model(inp, past if use_cache else None)
        past = presents
        if temperature == 0:
            nxt = logits[:, -1].argmax(-1, keepdim=True)  # décodage glouton
        else:
            probs = filter_logits(logits[0, -1], temperature, top_k, top_p).softmax(-1)
            nxt = torch.multinomial(probs, 1).unsqueeze(0)
        idx = torch.cat([idx, nxt], dim=1)
    return idx


# --------------------------------------------------------------------------
# 3. RoPE (positions rotatives)
# --------------------------------------------------------------------------
def rope(x, positions, base=10000.0):
    """Fait tourner chaque paire de dimensions de x (..., T, d) selon sa position."""
    d = x.size(-1)
    freqs = base ** (-torch.arange(0, d, 2, dtype=torch.float32) / d)  # (d/2,)
    angles = positions[:, None].float() * freqs[None, :]  # (T, d/2)
    cos, sin = angles.cos(), angles.sin()
    x1, x2 = x[..., 0::2], x[..., 1::2]
    out = torch.empty_like(x)
    out[..., 0::2] = x1 * cos - x2 * sin
    out[..., 1::2] = x1 * sin + x2 * cos
    return out


# --------------------------------------------------------------------------
# 4. LoRA
# --------------------------------------------------------------------------
class LoRALinear(nn.Module):
    """y = W x + (alpha / r) * B (A x), avec W gelé, A (r x in) et B (out x r) entraînés.

    B est initialisé à zéro : au départ, le modèle est strictement identique à l'original.
    """

    def __init__(self, base: nn.Linear, r: int = 8, alpha: float = 16.0, dropout: float = 0.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.r, self.scale = r, alpha / r
        self.A = nn.Parameter(torch.empty(r, base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))  # comme la bibliothèque PEFT
        self.dropout = nn.Dropout(dropout)
        self.merged = False

    def forward(self, x):
        out = self.base(x)
        if self.merged:
            return out
        return out + self.scale * (self.dropout(x) @ self.A.T @ self.B.T)

    @torch.no_grad()
    def merge(self):
        """W <- W + (alpha/r) B A : plus aucun surcoût à l'inférence."""
        if not self.merged:
            self.base.weight += self.scale * (self.B @ self.A)
            self.merged = True

    @torch.no_grad()
    def unmerge(self):
        if self.merged:
            self.base.weight -= self.scale * (self.B @ self.A)
            self.merged = False


def apply_lora(model: nn.Module, targets=("qkv", "proj"), r=8, alpha=16.0, dropout=0.0):
    """Gèle tout le modèle puis remplace les `nn.Linear` ciblés par des `LoRALinear`."""
    for p in model.parameters():
        p.requires_grad_(False)
    for module in list(model.modules()):
        for name, child in list(module.named_children()):
            if isinstance(child, nn.Linear) and name in targets:
                setattr(module, name, LoRALinear(child, r, alpha, dropout))
    return model


def count_params(model: nn.Module):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


# --------------------------------------------------------------------------
# 5. Calculs de mémoire (formules du cours)
# --------------------------------------------------------------------------
def kv_cache_bytes_per_token(n_layers, n_kv_heads, head_dim, bytes_per_value=2):
    """2 (K et V) x couches x têtes KV x dimension de tête x octets."""
    return 2 * n_layers * n_kv_heads * head_dim * bytes_per_value


def lora_params(d_in, d_out, r):
    return r * (d_in + d_out)



# --------------------------------------------------------------------------
# 6. Tokenisation BPE (au niveau des octets) et perte DPO
# --------------------------------------------------------------------------
def merge_pair(tokens, pair, new_id):
    """Remplace chaque occurrence de `pair` par `new_id`."""
    out, i = [], 0
    while i < len(tokens):
        if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == pair:
            out.append(new_id)
            i += 2
        else:
            out.append(tokens[i])
            i += 1
    return out


def train_bpe(text, num_merges):
    """Apprend `num_merges` fusions : à chaque tour, on fusionne la paire la plus fréquente."""
    tokens = list(text.encode("utf-8"))  # on part des 256 octets
    merges = {}
    for i in range(num_merges):
        pairs = Counter(zip(tokens, tokens[1:]))
        if not pairs:
            break
        best = max(pairs, key=pairs.get)
        merges[best] = 256 + i
        tokens = merge_pair(tokens, best, 256 + i)
    return merges


def bpe_encode(text, merges):
    tokens = list(text.encode("utf-8"))
    for pair, new_id in merges.items():  # dans l'ordre d'apprentissage
        tokens = merge_pair(tokens, pair, new_id)
    return tokens


def bpe_decode(ids, merges):
    vocab = {i: bytes([i]) for i in range(256)}
    for (a, b), new_id in merges.items():
        vocab[new_id] = vocab[a] + vocab[b]
    return b"".join(vocab[i] for i in ids).decode("utf-8", errors="replace")


def dpo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta=0.1):
    """Perte DPO. Les entrées sont les log-probabilités (somme sur les tokens) des deux réponses."""
    margin = (policy_chosen - ref_chosen) - (policy_rejected - ref_rejected)
    return -F.logsigmoid(beta * margin).mean()

# --------------------------------------------------------------------------
# Vérifications
# --------------------------------------------------------------------------
def _toy_batch(vocab, T=24, B=8, seed=0, step=1):
    """Séquences « n, n+step, n+2*step, ... » (modulo vocab) : une tâche minuscule."""
    g = torch.Generator().manual_seed(seed)
    start = torch.randint(0, vocab, (B, 1), generator=g)
    return (start + step * torch.arange(T + 1)) % vocab


def _train(model, data, steps, lr=3e-3):
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    model.train()
    first = last = None
    for _ in range(steps):
        logits, _ = model(data[:, :-1])
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), data[:, 1:].reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        first = loss.item() if first is None else first
        last = loss.item()
    return first, last


def check_causality():
    torch.manual_seed(0)
    m = TinyGPT(vocab_size=30).eval()
    a = torch.randint(0, 30, (1, 10))
    b = a.clone()
    b[0, 7] = (b[0, 7] + 1) % 30  # on change un token futur
    la, _ = m(a)
    lb, _ = m(b)
    assert torch.allclose(la[:, :7], lb[:, :7], atol=1e-6), "le futur ne doit pas influencer le passé"
    assert not torch.allclose(la[:, 7:], lb[:, 7:]), "mais il influence les positions suivantes"


def check_kv_cache():
    torch.manual_seed(0)
    m = TinyGPT(vocab_size=30).eval()
    prompt = torch.randint(0, 30, (1, 6))
    a = generate(m, prompt, 10, temperature=0, use_cache=True)
    b = generate(m, prompt, 10, temperature=0, use_cache=False)
    assert torch.equal(a, b), "le KV cache ne doit pas changer le résultat"


def check_training():
    torch.manual_seed(0)
    m = TinyGPT(vocab_size=30)
    first, last = _train(m, _toy_batch(30), steps=150)
    assert last < first * 0.3, (first, last)
    return first, last


def check_rope_relative():
    torch.manual_seed(0)
    q, k = torch.randn(1, 16), torch.randn(1, 16)
    d1 = (rope(q, torch.tensor([5])) * rope(k, torch.tensor([2]))).sum()    # écart 3
    d2 = (rope(q, torch.tensor([40])) * rope(k, torch.tensor([37]))).sum()  # écart 3
    d3 = (rope(q, torch.tensor([40])) * rope(k, torch.tensor([30]))).sum()  # écart 10
    assert torch.allclose(d1, d2, atol=1e-4), "le produit scalaire ne dépend que de l'écart"
    assert not torch.allclose(d1, d3, atol=1e-4)


def check_sampling():
    logits = torch.tensor([5.0, 4.0, 1.0, 0.0, -1.0])
    top1 = filter_logits(logits, top_k=1)
    assert torch.isfinite(top1).sum() == 1 and top1.argmax() == 0
    nucleus = filter_logits(logits, top_p=0.5)  # le meilleur token seul dépasse déjà 0,5
    assert torch.isfinite(nucleus).sum() == 1
    assert torch.isfinite(filter_logits(logits, top_p=0.99)).sum() >= 2
    cold = filter_logits(logits, temperature=0.1).softmax(-1)
    hot = filter_logits(logits, temperature=5.0).softmax(-1)
    assert cold.max() > logits.softmax(-1).max() > hot.max()


def check_lora():
    torch.manual_seed(0)
    base = TinyGPT(vocab_size=30)
    _train(base, _toy_batch(30, step=1), steps=150)  # « pré-entraînement » : la suite n, n+1, ...
    base.eval()
    x = torch.randint(0, 30, (2, 12))
    ref, _ = base(x)
    total_before, _ = count_params(base)

    base_snapshot = {k: v.clone() for k, v in base.state_dict().items()}
    model = apply_lora(base, targets=("qkv", "proj"), r=4, alpha=8)
    out, _ = model(x)
    assert torch.allclose(out, ref, atol=1e-6), "B = 0 : sortie identique au modèle de base"

    total, trainable = count_params(model)
    d = 64
    per_layer = lora_params(d, 3 * d, 4) + lora_params(d, d, 4)  # qkv + proj
    assert trainable == 2 * per_layer, (trainable, per_layer)  # 2 couches
    assert total == total_before + trainable

    # « Fine-tuning » : une nouvelle tâche, n, n+2, n+4, ... apprise avec les seuls paramètres LoRA.
    first, last = _train(model, _toy_batch(30, seed=1, step=2), steps=300)
    assert last < first * 0.5, (first, last)

    # Seuls A et B ont bougé : tout le reste est identique à l'état initial.
    for name, tensor in model.state_dict().items():
        clean = name.replace(".base", "")
        if name.endswith((".A", ".B")):
            continue
        assert torch.equal(tensor, base_snapshot[clean]), f"{name} a changé"

    # Fusion : même sortie, sans le chemin LoRA.
    model.eval()
    before, _ = model(x)
    for m in model.modules():
        if isinstance(m, LoRALinear):
            m.merge()
    merged, _ = model(x)
    assert torch.allclose(before, merged, atol=1e-5), "la fusion doit être exacte"
    for m in model.modules():
        if isinstance(m, LoRALinear):
            m.unmerge()
    again, _ = model(x)
    assert torch.allclose(before, again, atol=1e-5)
    return total, trainable, first, last


def check_memory_formulas():
    # Llama-2 7B : 32 couches, 32 têtes KV, dimension de tête 128, fp16.
    per_token = kv_cache_bytes_per_token(32, 32, 128, 2)
    assert per_token == 524_288  # 0,5 Mio par token
    assert per_token * 4096 == 2 * 1024**3  # 2 Gio pour 4096 tokens
    # Avec GQA (8 têtes KV) le cache est divisé par 4.
    assert kv_cache_bytes_per_token(32, 8, 128, 2) == per_token // 4
    # LoRA r=8 sur q et v d'un modèle 7B (d = 4096, 32 couches).
    assert 2 * 32 * lora_params(4096, 4096, 8) == 4_194_304



def check_bpe():
    text = "café ça va ? café ça va bien, café !" * 5
    merges = train_bpe(text, num_merges=20)
    ids = bpe_encode(text, merges)
    assert bpe_decode(ids, merges) == text, "encode puis decode doit redonner le texte"
    assert len(ids) < len(text.encode("utf-8")), "BPE compresse : moins de tokens que d'octets"
    assert bpe_decode(bpe_encode("café inconnu ☕", merges), merges) == "café inconnu ☕"  # jamais de « inconnu »
    return len(text.encode("utf-8")), len(ids)


def check_dpo():
    z = torch.zeros(4)
    same = dpo_loss(z, z, z, z)  # politique = référence : aucune préférence apprise
    assert torch.allclose(same, torch.tensor(math.log(2.0))), same
    better = dpo_loss(z + 2.0, z - 2.0, z, z)  # la politique préfère déjà la bonne réponse
    worse = dpo_loss(z - 2.0, z + 2.0, z, z)
    assert better < same < worse

if __name__ == "__main__":
    check_causality()
    print("causalité                     OK")
    check_kv_cache()
    print("KV cache = sans cache         OK")
    f, l = check_training()
    print(f"entraînement                  OK (perte {f:.2f} -> {l:.3f})")
    check_rope_relative()
    print("RoPE (position relative)      OK")
    check_sampling()
    print("température / top-k / top-p   OK")
    total, trainable, f, l = check_lora()
    print(f"LoRA                          OK ({trainable:,} paramètres entraînés sur {total:,}, "
          f"perte {f:.2f} -> {l:.2f})")
    check_memory_formulas()
    print("formules de mémoire           OK")
    n_bytes, n_tokens = check_bpe()
    print(f"BPE                           OK ({n_bytes} octets -> {n_tokens} tokens)")
    check_dpo()
    print("perte DPO (ln 2 au départ)    OK")
