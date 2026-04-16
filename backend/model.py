import torch
import torch.nn as nn
import math

# =========================
# Patch Embedding
# =========================
class PatchEmbedding(nn.Module):
    def __init__(self, img_size=256, patch_size=16, in_chans=3, embed_dim=512):
        super().__init__()
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )
        
        # Positional embedding MUST match training size (256 patches)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.n_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        x = self.proj(x)              # (B, D, H/ps, W/ps)
        x = x.flatten(2).transpose(1, 2)  # (B, N, D)
        x = x + self.pos_embed        # Add positional embeddings
        return x


# =========================
# Transformer Encoder Block
# =========================
class TransformerEncoder(nn.Module):
    def __init__(self, dim=512, heads=8, mlp_dim=2048, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=heads,
            dropout=dropout,
            batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, dim)
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + self.drop(attn_out))
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.drop(ffn_out))
        return x


# =========================
# Main ViT Model
# =========================
class ViTCustom(nn.Module):
    def __init__(
        self,
        num_classes,
        img_size=256,       # MUST MATCH training
        patch_size=16,
        dim=512,
        depth=4,
        heads=8,
        mlp_dim=2048
    ):
        super().__init__()

        self.patch_embed = PatchEmbedding(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=dim
        )

        self.encoder = nn.Sequential(
            *[TransformerEncoder(dim, heads, mlp_dim) for _ in range(depth)]
        )

        self.head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes),
        )

    def forward(self, x):
        x = self.patch_embed(x)  # (B, 256, D)
        x = self.encoder(x)      # (B, 256, D)
        x = x.mean(dim=1)        # Global average pooling
        return self.head(x)


# =========================
# Build model (used by backend)
# =========================
def build_model(num_classes):
    return ViTCustom(
        num_classes=num_classes,
        img_size=256,         # FIXED
        patch_size=16,
        dim=512,
        depth=4,
        heads=8,
        mlp_dim=2048
    )
