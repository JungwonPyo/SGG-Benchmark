import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def make_sine_position_encoding(
    in_feature_size, patch_size, d_model, temperature=10000, scale=2 * math.pi, device='cpu'
):
    h, w = in_feature_size[0] // (patch_size[0]), in_feature_size[1] // (patch_size[1])
    h = max(1, h)
    w = max(1, w)
    
    grid_y = torch.arange(1, h + 1, dtype=torch.float32, device=device)
    grid_x = torch.arange(1, w + 1, dtype=torch.float32, device=device)
    
    grid_y = grid_y / (h + 1e-6) * scale
    grid_x = grid_x / (w + 1e-6) * scale
    
    y_embed, x_embed = torch.meshgrid(grid_y, grid_x, indexing='ij')

    one_direction_feats = d_model // 2
    dim_t = torch.arange(one_direction_feats, dtype=torch.float32, device=device)
    dim_t = temperature ** (2 * (dim_t // 2) / one_direction_feats)

    pos_x = x_embed[:, :, None] / dim_t
    pos_y = y_embed[:, :, None] / dim_t
    
    pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
    pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
    
    pos = torch.cat((pos_y, pos_x), dim=2).flatten(0, 1)
    return pos[None]

class FourierPosEnc(nn.Module):
    """Fourier Feature Encoding for continuous spatial coordinates.

    Maps R^n → R^D via sin/cos at log-spaced frequencies, then projects to d_model.
    Much more expressive than a plain Linear for coordinates in [0,1] — the model
    doesn't need to learn to separate spatial frequencies from scratch.

    Example:
        box_pos_enc = FourierPosEnc(4, d_model)   # encodes [cx, cy, w, h]
        geo_enc     = FourierPosEnc(5, d_model)   # encodes [dx, dy, dist, area_ratio, ar]
    """
    def __init__(self, in_dim: int, d_model: int, num_freqs: int = 8):
        super().__init__()
        # Log-spaced: 1, 2, 4, …, 2^(F-1)  — covers both fine & coarse spatial structure
        freqs = 2.0 ** torch.arange(num_freqs).float()
        self.register_buffer("freqs", freqs)
        # in_dim coords × (sin + cos) × num_freqs → d_model
        self.proj = nn.Linear(in_dim * 2 * num_freqs, d_model, bias=False)
        nn.init.xavier_uniform_(self.proj.weight, gain=0.1)  # small init — additive residual

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [..., in_dim]  (any leading batch dims)
        angles = x.unsqueeze(-1) * self.freqs * math.pi   # [..., in_dim, F]
        features = torch.cat([angles.sin(), angles.cos()], dim=-1)  # [..., in_dim, 2F]
        return self.proj(features.flatten(-2))             # [..., d_model]


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization for stability and speed."""
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        orig_dtype = x.dtype
        x_float = x.to(torch.float32)
        norm = x_float.pow(2).mean(-1, keepdim=True)
        out = x_float * torch.rsqrt(norm + self.eps)
        return (self.weight * out).to(orig_dtype)

class FlashAttention(nn.Module):
    """
    Implements Scaled Dot Product Attention using PyTorch 2.0's optimized SDPA.
    This automatically uses Flash Attention if available on the GPU.
    """
    def __init__(self, embed_dim, num_heads, dropout=0.0, batch_first=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.batch_first = batch_first

    def forward(self, q, k, value, key_padding_mask=None, attn_mask=None, is_causal=False):
        """
        Args:
            q, k, value: (B, L, D) if batch_first=True
        """
        if not self.batch_first:
            q = q.transpose(0, 1)
            k = k.transpose(0, 1)
            value = value.transpose(0, 1)

        B, Lq, D = q.shape
        Lk = k.shape[1]
        
        # Project and split heads: (B, L, H, Dh) -> (B, H, L, Dh) for SDPA
        q = self.q_proj(q).view(B, Lq, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(k).view(B, Lk, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(B, Lk, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Convert key_padding_mask (B, Lk), True=padding → additive float (B, 1, 1, Lk).
        # SDPA broadcasts this over (B, H, Lq, Lk) automatically.
        if key_padding_mask is not None:
            pad_bias = torch.zeros_like(key_padding_mask, dtype=q.dtype)
            pad_bias = pad_bias.masked_fill(key_padding_mask, float('-inf'))
            pad_bias = pad_bias.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, Lk)
            attn_mask = pad_bias if attn_mask is None else attn_mask + pad_bias

        # SDPA requires attn_mask to match the query dtype when it is a float tensor.
        if attn_mask is not None and attn_mask.is_floating_point() and attn_mask.dtype != q.dtype:
            attn_mask = attn_mask.to(dtype=q.dtype)

        # Let PyTorch pick the optimal SDPA backend automatically (Flash, MemEfficient, or Math).
        # Avoid the deprecated sdp_kernel context manager which adds overhead and emits warnings.
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal,
        )
        
        # Reassemble
        out = out.transpose(1, 2).contiguous().view(B, Lq, D)
        out = self.out_proj(out)
        
        if not self.batch_first:
            out = out.transpose(0, 1)
            
        return out, None # Align with MultiheadAttention return signature

class TransformerEncoderLayer(nn.Module):
    """Optimized Transformer Encoder Layer with Flash Attention and RMSNorm."""
    def __init__(self, c1, cm=2048, num_heads=8, dropout=0.0, act=nn.GELU(), normalize_before=True):
        super().__init__()
        # Use our FlashAttention wrapper instead of nn.MultiheadAttention
        self.ma = FlashAttention(c1, num_heads, dropout=dropout, batch_first=True)
        
        self.fc1 = nn.Linear(c1, cm)
        self.fc2 = nn.Linear(cm, c1)
        
        # Updated to RMSNorm for speed/stability
        self.norm1 = RMSNorm(c1)
        self.norm2 = RMSNorm(c1)
        
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.act = act
        self.normalize_before = normalize_before

    def forward(self, src, pos=None, key_padding_mask=None, attn_mask=None):
        if self.normalize_before:
            src2 = self.norm1(src)
            q = k = src2 if pos is None else src2 + pos
            # FlashAttention returns (output, weights), we only need output
            attn_out, _ = self.ma(q, k, value=src2,
                                  key_padding_mask=key_padding_mask,
                                  attn_mask=attn_mask)
            src = src + self.dropout1(attn_out)

            src2 = self.norm2(src)
            src = src + self.dropout2(self.fc2(self.dropout(self.act(self.fc1(src2)))))
            return src

        q = k = src if pos is None else src + pos
        attn_out, _ = self.ma(q, k, value=src,
                              key_padding_mask=key_padding_mask,
                              attn_mask=attn_mask)
        src = src + self.dropout1(attn_out)
        src = self.norm1(src)
        src = src + self.dropout2(self.fc2(self.dropout(self.act(self.fc1(src)))))
        return self.norm2(src)

class AIFI(TransformerEncoderLayer):
    """All-Levels Feature Interaction (image-level global context)."""
    def __init__(self, c1, cm=2048, num_heads=8, dropout=0, act=nn.GELU(), normalize_before=True):
        super().__init__(c1, cm, num_heads, dropout, act, normalize_before)

    def forward(self, x):
        b, c, h, w = x.shape
        # Cache the sincos PE to avoid recomputing it (and copying CPU→GPU) every forward.
        # The cache is invalidated automatically when spatial dimensions or device changes.
        cache_key = (w, h, c, x.device.type, x.device.index if x.device.type == 'cuda' else 0)
        if not hasattr(self, '_pe_cache_key') or self._pe_cache_key != cache_key:
            self._pe_cache_key = cache_key
            self._pe_cached = self.build_2d_sincos_position_embedding(w, h, c).to(x.device)
        pos_embed = self._pe_cached
        # Flatten [B, C, H, W] to [B, HxW, C] -> batch_first=True expected by FlashAttention
        x_flat = x.flatten(2).permute(0, 2, 1)
        out = super().forward(x_flat, pos=pos_embed)
        return out.permute(0, 2, 1).view(b, c, h, w).contiguous()

    @staticmethod
    def build_2d_sincos_position_embedding(w, h, embed_dim=256, temp=10000.0):
        grid_w = torch.arange(w, dtype=torch.float32)
        grid_h = torch.arange(h, dtype=torch.float32)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing="ij")
        pos_dim = embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1.0 / (temp**omega)
        out_w = grid_w.flatten()[..., None] @ omega[None]
        out_h = grid_h.flatten()[..., None] @ omega[None]
        return torch.cat([torch.sin(out_w), torch.cos(out_w), torch.sin(out_h), torch.cos(out_h)], 1)[None]

class MLP(nn.Module):
    """Efficient YOLO Multi-layer Perceptron."""
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, act=nn.GELU):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim, *h], [*h, output_dim]))
        self.act = act()

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = self.act(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x