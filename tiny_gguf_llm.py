#!/usr/bin/env python3
"""
tiny_gguf_llm.py
================

Ein winziges Transformer-Sprachmodell, KOMPLETT in NumPy - kein PyTorch,
kein Autograd. Jeder Forward- und Backward-Schritt ist von Hand geschrieben,
damit man genau sieht, was mit den Tensoren passiert.

Enthaelt:
  1. Einen GGUF-Reader/-Writer (echtes Binaerformat, siehe ggml/gguf-Spec)
  2. Alle NN-Bausteine (Linear, LayerNorm, GELU, Attention) mit manuellem
     forward()/backward() -> jede Funktion gibt einen "cache" zurueck, den
     die zugehoerige backward()-Funktion braucht (so wie man es von Hand
     auf Papier herleiten wuerde).
  3. Einen eigenen Adam-Optimizer (auch nur ein paar Zeilen NumPy).
  4. Einen Gradient-Check (numerische vs. analytische Ableitung), damit man
     dem handgeschriebenen Backward-Pass auch vertrauen kann.
  5. Eine Trainingsschleife auf Byte-Ebene (kein Tokenizer noetig).

Nutzung:
    python tiny_gguf_llm.py train --data mein_text.txt --steps 3000 --save model.gguf
    python tiny_gguf_llm.py inspect --gguf model.gguf
    python tiny_gguf_llm.py generate --gguf model.gguf --prompt "Der Hund"
    python tiny_gguf_llm.py gradcheck
"""

import argparse
import struct
import sys
import time

import numpy as np

RNG = np.random.default_rng(1234)

# =============================================================================
# 1. GGUF FORMAT  (siehe https://github.com/ggerganov/ggml/blob/master/docs/gguf.md)
# =============================================================================
#
# Datei-Layout:
#   uint32  magic            = 0x46554747  ("GGUF")
#   uint32  version           = 3
#   uint64  tensor_count
#   uint64  metadata_kv_count
#   -- metadata_kv_count x Key/Value Paare --
#   -- tensor_count x Tensor-Info (Name, Shape, Dtype, Offset) --
#   -- Padding bis zum naechsten "general.alignment"-Vielfachen --
#   -- rohe Tensor-Daten, jede aufs Alignment gerundet --

GGUF_MAGIC = 0x46554747
GGUF_VERSION = 3

# gguf metadata value types
T_UINT8, T_INT8, T_UINT16, T_INT16 = 0, 1, 2, 3
T_UINT32, T_INT32, T_FLOAT32, T_BOOL = 4, 5, 6, 7
T_STRING, T_ARRAY, T_UINT64, T_INT64, T_FLOAT64 = 8, 9, 10, 11, 12

# ggml tensor dtypes (nur die, die wir wirklich lesen/schreiben)
GGML_F32, GGML_F16 = 0, 1
GGML_TYPE_NAMES = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1",
    8: "Q8_0", 9: "Q8_1", 10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K",
    14: "Q6_K", 15: "Q8_K", 24: "I8", 25: "I16", 26: "I32", 27: "I64", 28: "F64",
}


class _Cursor:
    """Kleiner Hilfs-Reader, der sich seine Position im Byte-Buffer merkt."""

    def __init__(self, data: bytes):
        self.data = data
        self.off = 0

    def _unpack(self, fmt, size):
        v = struct.unpack_from(fmt, self.data, self.off)[0]
        self.off += size
        return v

    def u8(self):  return self._unpack("<B", 1)
    def i8(self):  return self._unpack("<b", 1)
    def u16(self): return self._unpack("<H", 2)
    def i16(self): return self._unpack("<h", 2)
    def u32(self): return self._unpack("<I", 4)
    def i32(self): return self._unpack("<i", 4)
    def u64(self): return self._unpack("<Q", 8)
    def i64(self): return self._unpack("<q", 8)
    def f32(self): return self._unpack("<f", 4)
    def f64(self): return self._unpack("<d", 8)
    def bool_(self): return self._unpack("<?", 1)

    def string(self):
        n = self.u64()
        s = self.data[self.off:self.off + n].decode("utf-8")
        self.off += n
        return s


_KV_READERS = {
    T_UINT8: _Cursor.u8, T_INT8: _Cursor.i8, T_UINT16: _Cursor.u16,
    T_INT16: _Cursor.i16, T_UINT32: _Cursor.u32, T_INT32: _Cursor.i32,
    T_FLOAT32: _Cursor.f32, T_BOOL: _Cursor.bool_, T_STRING: _Cursor.string,
    T_UINT64: _Cursor.u64, T_INT64: _Cursor.i64, T_FLOAT64: _Cursor.f64,
}


def _read_kv_value(cur: _Cursor, vtype: int):
    if vtype == T_ARRAY:
        atype = cur.u32()
        n = cur.u64()
        return [_read_kv_value(cur, atype) for _ in range(n)]
    return _KV_READERS[vtype](cur)


def gguf_read(path):
    """Liest eine .gguf-Datei komplett von Hand (kein llama.cpp/gguf-py noetig).

    Rueckgabe: (metadata: dict, tensors: dict[name -> np.ndarray|None],
                tensor_info: dict[name -> dict(shape, dtype_name, quantized)])
    """
    with open(path, "rb") as f:
        data = f.read()
    cur = _Cursor(data)

    magic = cur.u32()
    if magic != GGUF_MAGIC:
        raise ValueError(f"Keine gueltige GGUF-Datei (magic=0x{magic:08x})")
    version = cur.u32()
    n_tensors = cur.u64()
    n_kv = cur.u64()

    metadata = {}
    for _ in range(n_kv):
        key = cur.string()
        vtype = cur.u32()
        metadata[key] = _read_kv_value(cur, vtype)

    infos = []
    for _ in range(n_tensors):
        name = cur.string()
        n_dims = cur.u32()
        dims_gguf = [cur.u64() for _ in range(n_dims)]  # ne[0]=schnellste Dim (ggml-Konvention)
        shape = tuple(reversed(dims_gguf))               # -> numpy row-major shape
        ttype = cur.u32()
        offset = cur.u64()
        infos.append((name, shape, ttype, offset))

    alignment = metadata.get("general.alignment", 32)
    data_start = cur.off
    if data_start % alignment != 0:
        data_start += alignment - (data_start % alignment)

    tensors, tensor_info = {}, {}
    for name, shape, ttype, offset in infos:
        type_name = GGML_TYPE_NAMES.get(ttype, f"unknown({ttype})")
        n_elem = int(np.prod(shape)) if shape else 1
        abs_off = data_start + offset
        arr = None
        if ttype == GGML_F32:
            arr = np.frombuffer(data, dtype="<f4", count=n_elem, offset=abs_off).reshape(shape).copy()
        elif ttype == GGML_F16:
            arr = np.frombuffer(data, dtype="<f2", count=n_elem, offset=abs_off).astype(np.float32).reshape(shape)
        # sonst: quantisierter Typ -> wir dequantisieren hier bewusst NICHT
        # (Q4_K/Q6_K etc. haben eigene Block-Formate); wird als "quantized" markiert.
        tensors[name] = arr
        tensor_info[name] = dict(shape=shape, dtype=type_name, quantized=arr is None)

    metadata["_gguf_version"] = version
    return metadata, tensors, tensor_info


def _kv_bytes(key: str, value) -> bytes:
    out = struct.pack("<Q", len(key.encode())) + key.encode()

    def encode_scalar(v):
        if isinstance(v, bool):
            return struct.pack("<I", T_BOOL) + struct.pack("<?", v)
        if isinstance(v, int):
            return struct.pack("<I", T_INT32) + struct.pack("<i", v)
        if isinstance(v, float):
            return struct.pack("<I", T_FLOAT32) + struct.pack("<f", v)
        if isinstance(v, str):
            b = v.encode()
            return struct.pack("<I", T_STRING) + struct.pack("<Q", len(b)) + b
        raise TypeError(f"Nicht unterstuetzter Metadata-Typ: {type(v)}")

    if isinstance(value, list):
        # Array von Strings oder Zahlen -> einfachste Annahme: alle gleicher Typ
        if all(isinstance(v, str) for v in value):
            elem_type, elem_bytes = T_STRING, b"".join(
                struct.pack("<Q", len(v.encode())) + v.encode() for v in value)
        else:
            elem_type, elem_bytes = T_INT32, b"".join(struct.pack("<i", int(v)) for v in value)
        out += struct.pack("<I", T_ARRAY) + struct.pack("<I", elem_type) + struct.pack("<Q", len(value)) + elem_bytes
    else:
        out += encode_scalar(value)
    return out


def gguf_write(path, tensors: dict, metadata: dict, alignment: int = 32):
    """Schreibt Tensoren (dict[name -> float32 np.ndarray]) + Metadata als .gguf.

    Wir schreiben ausschliesslich F32 (GGML_F32) - fuer eigene from-scratch-
    Checkpoints reicht das, und man kann die Datei mit jedem GGUF-Reader
    (auch diesem hier) wieder oeffnen.
    """
    metadata = dict(metadata)
    metadata.setdefault("general.alignment", alignment)

    kv_blob = b"".join(_kv_bytes(k, v) for k, v in metadata.items())

    names = list(tensors.keys())
    tensor_data_blobs = []
    info_blob = b""
    running_offset = 0
    for name in names:
        arr = np.ascontiguousarray(tensors[name], dtype=np.float32)
        dims_gguf = list(reversed(arr.shape)) or [1]
        info_blob += struct.pack("<Q", len(name.encode())) + name.encode()
        info_blob += struct.pack("<I", len(dims_gguf))
        for d in dims_gguf:
            info_blob += struct.pack("<Q", d)
        info_blob += struct.pack("<I", GGML_F32)     # dtype
        info_blob += struct.pack("<Q", running_offset)  # offset in data section
        raw = arr.tobytes()
        pad = (-len(raw)) % alignment
        tensor_data_blobs.append(raw + b"\x00" * pad)
        running_offset += len(raw) + pad

    header = struct.pack("<I", GGUF_MAGIC) + struct.pack("<I", GGUF_VERSION)
    header += struct.pack("<Q", len(names)) + struct.pack("<Q", len(metadata))

    pre_data = header + kv_blob + info_blob
    pad = (-len(pre_data)) % alignment
    pre_data += b"\x00" * pad

    with open(path, "wb") as f:
        f.write(pre_data)
        for blob in tensor_data_blobs:
            f.write(blob)


# =============================================================================
# 2. NN-BAUSTEINE: von Hand geschriebener forward() + backward()
# =============================================================================
# Konvention: jede *_forward Funktion gibt (output, cache) zurueck.
#             jede *_backward Funktion bekommt (d_output, cache) und gibt
#             die Gradienten bzgl. aller Eingaben zurueck (in derselben
#             Reihenfolge wie die forward-Argumente).

def linear_forward(x, W, b):
    """y = x @ W + b .  x: (..., in), W: (in, out), b: (out,)"""
    y = x @ W + b
    return y, (x, W)


def linear_backward(dy, cache):
    x, W = cache
    x2d = x.reshape(-1, x.shape[-1])
    dy2d = dy.reshape(-1, dy.shape[-1])
    dW = x2d.T @ dy2d
    db = dy2d.sum(axis=0)
    dx = (dy2d @ W.T).reshape(x.shape)
    return dx, dW, db


def layernorm_forward(x, gamma, beta, eps=1e-5):
    mu = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    std = np.sqrt(var + eps)
    xhat = (x - mu) / std
    out = xhat * gamma + beta
    return out, (xhat, std, gamma)


def layernorm_backward(dout, cache):
    xhat, std, gamma = cache
    N = dout.shape[-1]
    reduce_axes = tuple(range(dout.ndim - 1))
    dgamma = (dout * xhat).sum(axis=reduce_axes)
    dbeta = dout.sum(axis=reduce_axes)
    dxhat = dout * gamma
    dx = (1.0 / (N * std)) * (
        N * dxhat - dxhat.sum(axis=-1, keepdims=True) - xhat * (dxhat * xhat).sum(axis=-1, keepdims=True)
    )
    return dx, dgamma, dbeta


def gelu_forward(x):
    """GELU, tanh-Approximation (wie GPT-2)."""
    c = np.sqrt(2.0 / np.pi)
    inner = c * (x + 0.044715 * x ** 3)
    t = np.tanh(inner)
    out = 0.5 * x * (1.0 + t)
    return out, (x, t, c)


def gelu_backward(dout, cache):
    x, t, c = cache
    dinner_dx = c * (1 + 3 * 0.044715 * x ** 2)
    dtanh = (1 - t ** 2) * dinner_dx
    dy_dx = 0.5 * (1 + t) + 0.5 * x * dtanh
    return dout * dy_dx


def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def causal_attention_forward(x, Wqkv, bqkv, Wproj, bproj, n_head):
    B, T, C = x.shape
    hs = C // n_head
    qkv, lin_cache = linear_forward(x, Wqkv, bqkv)          # (B,T,3C)
    q, k, v = np.split(qkv, 3, axis=-1)

    def split_heads(t):
        return t.reshape(B, T, n_head, hs).transpose(0, 2, 1, 3)  # (B,nh,T,hs)

    qh, kh, vh = split_heads(q), split_heads(k), split_heads(v)
    scores = (qh @ kh.transpose(0, 1, 3, 2)) / np.sqrt(hs)   # (B,nh,T,T)
    causal_mask = np.triu(np.ones((T, T), dtype=bool), k=1)
    scores = np.where(causal_mask, -1e10, scores)
    attn = softmax(scores, axis=-1)
    out_h = attn @ vh                                        # (B,nh,T,hs)
    out = out_h.transpose(0, 2, 1, 3).reshape(B, T, C)
    y, proj_cache = linear_forward(out, Wproj, bproj)
    cache = (qh, kh, vh, attn, lin_cache, proj_cache, n_head, B, T, C, hs, causal_mask)
    return y, cache


def causal_attention_backward(dy, cache):
    qh, kh, vh, attn, lin_cache, proj_cache, n_head, B, T, C, hs, causal_mask = cache
    dout, dWproj, dbproj = linear_backward(dy, proj_cache)
    dout_h = dout.reshape(B, T, n_head, hs).transpose(0, 2, 1, 3)

    dattn = dout_h @ vh.transpose(0, 1, 3, 2)
    dvh = attn.transpose(0, 1, 3, 2) @ dout_h

    # Softmax-Backward: dscores_i = attn_i * (dattn_i - sum_j(dattn_j*attn_j))
    dscores = attn * (dattn - (dattn * attn).sum(axis=-1, keepdims=True))
    dscores = np.where(causal_mask, 0.0, dscores)
    scale = 1.0 / np.sqrt(hs)
    dqh = (dscores @ kh) * scale
    dkh = (dscores.transpose(0, 1, 3, 2) @ qh) * scale

    def merge_heads(t):
        return t.transpose(0, 2, 1, 3).reshape(B, T, C)

    dq, dk, dv = merge_heads(dqh), merge_heads(dkh), merge_heads(dvh)
    dqkv = np.concatenate([dq, dk, dv], axis=-1)
    dx, dWqkv, dbqkv = linear_backward(dqkv, lin_cache)
    return dx, dWqkv, dbqkv, dWproj, dbproj


def softmax_cross_entropy(logits, targets):
    """logits: (B,T,V), targets: (B,T) int. Gibt (loss, dlogits) zurueck."""
    B, T, V = logits.shape
    probs = softmax(logits, axis=-1)
    idx_b, idx_t = np.meshgrid(np.arange(B), np.arange(T), indexing="ij")
    correct_logprobs = -np.log(probs[idx_b, idx_t, targets] + 1e-12)
    loss = correct_logprobs.mean()
    dlogits = probs.copy()
    dlogits[idx_b, idx_t, targets] -= 1.0
    dlogits /= (B * T)
    return loss, dlogits


# =============================================================================
# 3. DAS MODELL: Parameter-Init, Forward, Backward
# =============================================================================

def init_params(vocab_size, block_size, n_embd, n_head, n_layer):
    def randn(*shape, scale=0.02):
        return (RNG.standard_normal(shape) * scale).astype(np.float32)

    p = {
        "wte": randn(vocab_size, n_embd),
        "wpe": randn(block_size, n_embd),
        "ln_f.g": np.ones(n_embd, dtype=np.float32),
        "ln_f.b": np.zeros(n_embd, dtype=np.float32),
    }
    for i in range(n_layer):
        pre = f"h.{i}."
        p[pre + "ln1.g"] = np.ones(n_embd, dtype=np.float32)
        p[pre + "ln1.b"] = np.zeros(n_embd, dtype=np.float32)
        p[pre + "attn.qkv.w"] = randn(n_embd, 3 * n_embd)
        p[pre + "attn.qkv.b"] = np.zeros(3 * n_embd, dtype=np.float32)
        p[pre + "attn.proj.w"] = randn(n_embd, n_embd)
        p[pre + "attn.proj.b"] = np.zeros(n_embd, dtype=np.float32)
        p[pre + "ln2.g"] = np.ones(n_embd, dtype=np.float32)
        p[pre + "ln2.b"] = np.zeros(n_embd, dtype=np.float32)
        p[pre + "mlp.fc.w"] = randn(n_embd, 4 * n_embd)
        p[pre + "mlp.fc.b"] = np.zeros(4 * n_embd, dtype=np.float32)
        p[pre + "mlp.proj.w"] = randn(4 * n_embd, n_embd)
        p[pre + "mlp.proj.b"] = np.zeros(n_embd, dtype=np.float32)
    return p


def model_forward(params, idx, n_head, n_layer):
    """idx: (B,T) int array Token-IDs (hier: Byte-Werte 0..255).
    Rueckgabe: logits (B,T,V), cache (fuer backward)."""
    B, T = idx.shape
    tok_emb = params["wte"][idx]              # (B,T,C)
    pos_emb = params["wpe"][:T][None, :, :]   # (1,T,C)
    x = tok_emb + pos_emb

    layer_caches = []
    for i in range(n_layer):
        pre = f"h.{i}."
        ln1_out, ln1_c = layernorm_forward(x, params[pre + "ln1.g"], params[pre + "ln1.b"])
        attn_out, attn_c = causal_attention_forward(
            ln1_out, params[pre + "attn.qkv.w"], params[pre + "attn.qkv.b"],
            params[pre + "attn.proj.w"], params[pre + "attn.proj.b"], n_head)
        x1 = x + attn_out                      # residual 1

        ln2_out, ln2_c = layernorm_forward(x1, params[pre + "ln2.g"], params[pre + "ln2.b"])
        fc_out, fc_c = linear_forward(ln2_out, params[pre + "mlp.fc.w"], params[pre + "mlp.fc.b"])
        gelu_out, gelu_c = gelu_forward(fc_out)
        mlp_out, proj_c = linear_forward(gelu_out, params[pre + "mlp.proj.w"], params[pre + "mlp.proj.b"])
        x2 = x1 + mlp_out                      # residual 2

        layer_caches.append((ln1_c, attn_c, ln2_c, fc_c, gelu_c, proj_c, x, x1))
        x = x2

    ln_f_out, ln_f_c = layernorm_forward(x, params["ln_f.g"], params["ln_f.b"])
    logits = ln_f_out @ params["wte"].T        # weight tying: gleiche Matrix wie Embedding

    cache = dict(idx=idx, ln_f_c=ln_f_c, ln_f_out=ln_f_out, layer_caches=layer_caches,
                 tok_emb_shape=tok_emb.shape, T=T)
    return logits, cache


def model_backward(dlogits, params, cache, n_head, n_layer):
    grads = {k: np.zeros_like(v) for k, v in params.items()}
    ln_f_out = cache["ln_f_out"]

    # logits = ln_f_out @ wte.T  (weight tying -> wte bekommt Gradienten von 2 Stellen)
    grads["wte"] += dlogits.reshape(-1, dlogits.shape[-1]).T @ ln_f_out.reshape(-1, ln_f_out.shape[-1])
    dln_f_out = dlogits @ params["wte"]

    dx, dg, db = layernorm_backward(dln_f_out, cache["ln_f_c"])
    grads["ln_f.g"] += dg
    grads["ln_f.b"] += db

    for i in reversed(range(n_layer)):
        pre = f"h.{i}."
        ln1_c, attn_c, ln2_c, fc_c, gelu_c, proj_c, x_in, x1 = cache["layer_caches"][i]

        # x2 = x1 + mlp_out  -> Gradient teilt sich auf residual + mlp-Pfad
        dx2 = dx
        dmlp_out = dx2
        dgelu_out, dWproj, dbproj = linear_backward(dmlp_out, proj_c)
        dfc_out = gelu_backward(dgelu_out, gelu_c)
        dln2_out, dWfc, dbfc = linear_backward(dfc_out, fc_c)
        dx1_from_mlp, dg2, db2 = layernorm_backward(dln2_out, ln2_c)
        dx1 = dx2 + dx1_from_mlp  # residual: Gradient von x2 UND vom MLP-Zweig

        # x1 = x_in + attn_out
        dattn_out = dx1
        dln1_out, dWqkv, dbqkv, dWproj_attn, dbproj_attn = causal_attention_backward(dattn_out, attn_c)
        dx_in_from_attn, dg1, db1 = layernorm_backward(dln1_out, ln1_c)
        dx = dx1 + dx_in_from_attn  # residual: Gradient von x1 UND vom Attention-Zweig

        grads[pre + "mlp.proj.w"] += dWproj
        grads[pre + "mlp.proj.b"] += dbproj
        grads[pre + "mlp.fc.w"] += dWfc
        grads[pre + "mlp.fc.b"] += dbfc
        grads[pre + "ln2.g"] += dg2
        grads[pre + "ln2.b"] += db2
        grads[pre + "attn.qkv.w"] += dWqkv
        grads[pre + "attn.qkv.b"] += dbqkv
        grads[pre + "attn.proj.w"] += dWproj_attn
        grads[pre + "attn.proj.b"] += dbproj_attn
        grads[pre + "ln1.g"] += dg1
        grads[pre + "ln1.b"] += db1

    # x = tok_emb + pos_emb  -> Embedding-Gradienten per scatter-add
    idx = cache["idx"]
    T = cache["T"]
    np.add.at(grads["wte"], idx, dx)
    grads["wpe"][:T] += dx.sum(axis=0)

    return grads


# =============================================================================
# 4. EIGENER OPTIMIZER (Adam, von Hand)
# =============================================================================

class Adam:
    def __init__(self, params, lr=3e-3, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.0):
        self.lr, self.b1, self.b2, self.eps, self.wd = lr, beta1, beta2, eps, weight_decay
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, params, grads):
        self.t += 1
        for k in params:
            g = grads[k] + self.wd * params[k]
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * (g * g)
            mhat = self.m[k] / (1 - self.b1 ** self.t)
            vhat = self.v[k] / (1 - self.b2 ** self.t)
            params[k] -= self.lr * mhat / (np.sqrt(vhat) + self.eps)


# =============================================================================
# 5. GRADIENT-CHECK (numerisch vs. analytisch)
# =============================================================================

def gradient_check(vocab_size=20, block_size=8, n_embd=16, n_head=2, n_layer=2, n_check=6):
    print("Gradient-Check: numerische vs. analytische Ableitung\n")
    params = init_params(vocab_size, block_size, n_embd, n_head, n_layer)
    B, T = 2, block_size
    idx = RNG.integers(0, vocab_size, size=(B, T))
    targets = RNG.integers(0, vocab_size, size=(B, T))

    def loss_fn(p):
        logits, _ = model_forward(p, idx, n_head, n_layer)
        loss, _ = softmax_cross_entropy(logits, targets)
        return loss

    logits, cache = model_forward(params, idx, n_head, n_layer)
    loss, dlogits = softmax_cross_entropy(logits, targets)
    grads = model_backward(dlogits, params, cache, n_head, n_layer)

    eps = 1e-4
    keys = list(params.keys())
    RNG.shuffle(keys)
    max_rel_err = 0.0
    for k in keys[:n_check]:
        flat_idx = tuple(RNG.integers(0, s) for s in params[k].shape)
        orig = params[k][flat_idx]

        params[k][flat_idx] = orig + eps
        lp = loss_fn(params)
        params[k][flat_idx] = orig - eps
        lm = loss_fn(params)
        params[k][flat_idx] = orig

        num_grad = (lp - lm) / (2 * eps)
        ana_grad = grads[k][flat_idx]
        rel_err = abs(num_grad - ana_grad) / (abs(num_grad) + abs(ana_grad) + 1e-8)
        max_rel_err = max(max_rel_err, rel_err)
        print(f"  {k:20s} idx={flat_idx}  numerisch={num_grad: .6f}  analytisch={ana_grad: .6f}  rel_err={rel_err:.2e}")

    print(f"\nGroesster relativer Fehler: {max_rel_err:.2e}  ({'OK' if max_rel_err < 1e-2 else 'FEHLER!'})")
    return max_rel_err


# =============================================================================
# 6. DATEN (Byte-Level - kein Tokenizer noetig)
# =============================================================================

DEFAULT_TEXT = """Ein kleines Modell lernt Schritt fuer Schritt.
Jeder Tensor hat eine Form, jede Ableitung einen Ursprung.
NumPy reicht, um zu verstehen, was ein Transformer wirklich tut.
Vorwaerts rechnen, Fehler messen, rueckwaerts die Gradienten schicken.
Immer wieder, bis das Modell die Muster im Text erkennt.
""" * 40


def get_batch(byte_ids, block_size, batch_size):
    n = len(byte_ids) - block_size - 1
    starts = RNG.integers(0, n, size=batch_size)
    x = np.stack([byte_ids[s:s + block_size] for s in starts])
    y = np.stack([byte_ids[s + 1:s + block_size + 1] for s in starts])
    return x, y


# =============================================================================
# 7. TRAINING
# =============================================================================

def train(args):
    if args.data:
        with open(args.data, "r", encoding="utf-8") as f:
            text = f.read()
    else:
        text = DEFAULT_TEXT
        print("(kein --data angegeben, nutze eingebauten Beispieltext)\n")

    byte_ids = np.frombuffer(text.encode("utf-8"), dtype=np.uint8).astype(np.int64)
    vocab_size = 256  # Byte-Level: immer 256 moegliche Werte

    if args.load:
        print(f"Lade Startgewichte aus {args.load} ...")
        metadata, tensors, info = gguf_read(args.load)
        params = {k: v for k, v in tensors.items() if v is not None}
        n_embd = int(metadata["tinygpt.embedding_length"])
        n_head = int(metadata["tinygpt.attention.head_count"])
        n_layer = int(metadata["tinygpt.block_count"])
        block_size = int(metadata["tinygpt.context_length"])
        print(f"  Architektur: n_embd={n_embd} n_head={n_head} n_layer={n_layer} block_size={block_size}")
    else:
        n_embd, n_head, n_layer, block_size = args.n_embd, args.n_head, args.n_layer, args.block_size
        params = init_params(vocab_size, block_size, n_embd, n_head, n_layer)

    n_params = sum(v.size for v in params.values())
    print(f"Modell: {n_params:,} Parameter ueber {len(params)} Tensoren\n")
    print("Ein paar Tensor-Shapes zur Anschauung:")
    for k in list(params.keys())[:6]:
        print(f"  {k:20s} shape={params[k].shape}  dtype={params[k].dtype}")
    print("  ...\n")

    opt = Adam(params, lr=args.lr)
    t0 = time.time()
    for step in range(1, args.steps + 1):
        x, y = get_batch(byte_ids, block_size, args.batch_size)
        logits, cache = model_forward(params, x, n_head, n_layer)
        loss, dlogits = softmax_cross_entropy(logits, y)
        grads = model_backward(dlogits, params, cache, n_head, n_layer)
        opt.step(params, grads)

        if step % args.log_every == 0 or step == 1:
            dt = time.time() - t0
            print(f"step {step:5d}/{args.steps}  loss={loss:.4f}  ({dt:.1f}s)")

    if args.save:
        metadata = {
            "general.architecture": "tinygpt",
            "general.name": "tiny-gguf-llm",
            "tinygpt.context_length": block_size,
            "tinygpt.embedding_length": n_embd,
            "tinygpt.block_count": n_layer,
            "tinygpt.attention.head_count": n_head,
            "tinygpt.vocab_size": vocab_size,
        }
        gguf_write(args.save, params, metadata)
        print(f"\nGewichte gespeichert nach {args.save}")

    if args.prompt is not None:
        print("\nGeneriere Beispieltext:")
        print(generate(params, args.prompt, n_head, n_layer, block_size, n_tokens=200))


def generate(params, prompt, n_head, n_layer, block_size, n_tokens=200, temperature=0.8):
    ids = list(prompt.encode("utf-8"))
    for _ in range(n_tokens):
        ctx = np.array(ids[-block_size:])[None, :]
        logits, _ = model_forward(params, ctx, n_head, n_layer)
        last_logits = logits[0, -1] / temperature
        probs = softmax(last_logits)
        next_id = RNG.choice(len(probs), p=probs)
        ids.append(int(next_id))
    return bytes(ids).decode("utf-8", errors="replace")


def inspect(args):
    metadata, tensors, info = gguf_read(args.gguf)
    print(f"=== {args.gguf} ===\n")
    print("Metadata:")
    for k, v in metadata.items():
        if k != "_gguf_version":
            print(f"  {k}: {v}")
    print(f"\n{len(tensors)} Tensoren:")
    total = 0
    for name, meta in info.items():
        n_elem = int(np.prod(meta["shape"])) if meta["shape"] else 1
        total += n_elem
        flag = " (quantisiert, nicht dequantisiert)" if meta["quantized"] else ""
        print(f"  {name:20s} shape={meta['shape']!s:20s} dtype={meta['dtype']}{flag}")
        if not meta["quantized"]:
            arr = tensors[name]
            print(f"      min={arr.min(): .4f}  max={arr.max(): .4f}  mean={arr.mean(): .4f}  std={arr.std():.4f}")
    print(f"\nGesamt: {total:,} Werte")


def generate_cmd(args):
    metadata, tensors, info = gguf_read(args.gguf)
    params = {k: v for k, v in tensors.items() if v is not None}
    n_embd = int(metadata["tinygpt.embedding_length"])
    n_head = int(metadata["tinygpt.attention.head_count"])
    n_layer = int(metadata["tinygpt.block_count"])
    block_size = int(metadata["tinygpt.context_length"])
    print(generate(params, args.prompt, n_head, n_layer, block_size, n_tokens=args.n_tokens))


# =============================================================================
# 8. CLI
# =============================================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train", help="Modell from scratch (oder von --load) trainieren")
    t.add_argument("--data", type=str, default=None, help="Textdatei zum Trainieren (sonst eingebauter Beispieltext)")
    t.add_argument("--load", type=str, default=None, help=".gguf-Datei als Startgewichte laden")
    t.add_argument("--save", type=str, default="model.gguf", help="Wohin die trainierten Gewichte gespeichert werden")
    t.add_argument("--steps", type=int, default=1000)
    t.add_argument("--batch_size", type=int, default=32)
    t.add_argument("--block_size", type=int, default=64)
    t.add_argument("--n_embd", type=int, default=64)
    t.add_argument("--n_head", type=int, default=4)
    t.add_argument("--n_layer", type=int, default=2)
    t.add_argument("--lr", type=float, default=3e-3)
    t.add_argument("--log_every", type=int, default=100)
    t.add_argument("--prompt", type=str, default=None, help="Nach dem Training Beispieltext generieren")
    t.set_defaults(func=train)

    i = sub.add_parser("inspect", help="Tensoren einer .gguf-Datei anzeigen")
    i.add_argument("--gguf", type=str, required=True)
    i.set_defaults(func=inspect)

    g = sub.add_parser("generate", help="Text mit einem trainierten Modell generieren")
    g.add_argument("--gguf", type=str, required=True)
    g.add_argument("--prompt", type=str, default="Der ")
    g.add_argument("--n_tokens", type=int, default=300)
    g.set_defaults(func=generate_cmd)

    gc = sub.add_parser("gradcheck", help="Backward-Pass numerisch verifizieren")
    gc.set_defaults(func=lambda args: gradient_check())

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
