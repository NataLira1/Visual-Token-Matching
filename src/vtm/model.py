from __future__ import annotations

import re
from typing import Iterable, Sequence

import torch
from torch import nn
import torch.nn.functional as F


def _safe_key(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]", "_", name)


class TinyCNNBackbone(nn.Module):
    """Backbone mínimo para testes offline; replica a interface multi-nível."""

    def __init__(self, embed_dim: int, patch_size: int) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_prefix_tokens = 0
        self.patch_embed = nn.Conv2d(3, embed_dim, patch_size, stride=patch_size)
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(embed_dim, embed_dim, 3, padding=1),
                    nn.GELU(),
                    nn.Conv2d(embed_dim, embed_dim, 3, padding=1),
                )
                for _ in range(4)
            ]
        )

    def forward_levels(self, images: torch.Tensor, indices: Sequence[int]) -> list[torch.Tensor]:
        x = self.patch_embed(images)
        levels = []
        for index, block in enumerate(self.blocks):
            x = x + block(x)
            if index in indices:
                levels.append(x.flatten(2).transpose(1, 2))
        return levels


class ImageEncoder(nn.Module):
    def __init__(
        self,
        backbone_name: str,
        image_size: int,
        feature_blocks: Sequence[int],
        pretrained: bool = True,
        embed_dim: int | None = None,
        patch_size: int = 16,
    ) -> None:
        super().__init__()
        self.feature_blocks = tuple(feature_blocks)
        self.image_size = image_size
        if backbone_name == "tiny_cnn":
            if embed_dim is None:
                raise ValueError("embed_dim é obrigatório para tiny_cnn.")
            self.backbone = TinyCNNBackbone(embed_dim, patch_size)
            self.embed_dim = embed_dim
            self.patch_size = patch_size
            self._is_timm = False
        else:
            try:
                import timm
            except ImportError as exc:
                raise RuntimeError("Instale timm==1.0.26 para usar o encoder ViT.") from exc
            self.backbone = timm.create_model(
                backbone_name,
                pretrained=pretrained,
                img_size=image_size,
                num_classes=0,
            )
            self.embed_dim = int(self.backbone.embed_dim)
            patch = self.backbone.patch_embed.patch_size
            self.patch_size = int(patch[0] if isinstance(patch, tuple) else patch)
            self._is_timm = True
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False

    @property
    def grid_size(self) -> int:
        return self.image_size // self.patch_size

    def train(self, mode: bool = True) -> "ImageEncoder":
        super().train(mode)
        self.backbone.eval()
        return self

    def _forward_timm(self, images: torch.Tensor) -> list[torch.Tensor]:
        model = self.backbone
        x = model.patch_embed(images)
        x = model._pos_embed(x)
        if hasattr(model, "patch_drop"):
            x = model.patch_drop(x)
        if hasattr(model, "norm_pre"):
            x = model.norm_pre(x)
        levels = []
        for index, block in enumerate(model.blocks):
            x = block(x)
            if index in self.feature_blocks:
                prefix = int(getattr(model, "num_prefix_tokens", 1))
                levels.append(x[:, prefix:])
        if len(levels) != len(self.feature_blocks):
            raise RuntimeError(
                f"Foram solicitados {len(self.feature_blocks)} níveis, mas o "
                f"backbone produziu {len(levels)}."
            )
        return levels

    def forward(
        self,
        images: torch.Tensor,
        task_bias: torch.Tensor | None = None,
    ) -> list[torch.Tensor]:
        with torch.no_grad():
            if self._is_timm:
                levels = self._forward_timm(images)
            else:
                levels = self.backbone.forward_levels(images, self.feature_blocks)
        if task_bias is not None:
            if task_bias.shape != (len(levels), self.embed_dim):
                raise ValueError(
                    f"task_bias deve ter shape {(len(levels), self.embed_dim)}, "
                    f"recebido {tuple(task_bias.shape)}."
                )
            levels = [level + task_bias[index][None, None] for index, level in enumerate(levels)]
        return levels


class LabelEncoder(nn.Module):
    def __init__(
        self,
        image_size: int,
        patch_size: int,
        embed_dim: int,
        depth: int,
        num_heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.grid_size = image_size // patch_size
        self.patch_embed = nn.Conv2d(1, embed_dim, patch_size, stride=patch_size)
        self.position = nn.Parameter(torch.zeros(1, self.grid_size**2, embed_dim))
        self.blocks = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=embed_dim,
                    nhead=num_heads,
                    dim_feedforward=embed_dim * 4,
                    dropout=dropout,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(depth)
            ]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(depth)])
        nn.init.trunc_normal_(self.position, std=0.02)

    def forward(self, masks: torch.Tensor) -> list[torch.Tensor]:
        x = self.patch_embed(masks).flatten(2).transpose(1, 2)
        x = x + self.position
        levels = []
        for block, norm in zip(self.blocks, self.norms):
            x = block(x)
            levels.append(norm(x))
        return levels


class MatchingLevel(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.q_norm = nn.LayerNorm(embed_dim)
        self.k_norm = nn.LayerNorm(embed_dim)
        self.v_norm = nn.LayerNorm(embed_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.output_norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        query: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        return_attention: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        output, weights = self.attention(
            self.q_norm(query),
            self.k_norm(keys),
            self.v_norm(values),
            need_weights=return_attention,
            average_attn_weights=False,
        )
        return self.output_norm(output + query), weights


class TokenMatcher(nn.Module):
    def __init__(self, levels: int, embed_dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.levels = nn.ModuleList(
            [MatchingLevel(embed_dim, num_heads, dropout) for _ in range(levels)]
        )

    def forward(
        self,
        query_levels: Sequence[torch.Tensor],
        support_image_levels: Sequence[torch.Tensor],
        support_label_levels: Sequence[torch.Tensor],
        return_attention: bool = False,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor | None]]:
        matched, attention = [], []
        for module, query, keys, values in zip(
            self.levels, query_levels, support_image_levels, support_label_levels
        ):
            output, weights = module(query, keys, values, return_attention)
            matched.append(output)
            attention.append(weights)
        return matched, attention


class UpsampleBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(8 if channels >= 8 else 1, channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        return x + self.block(x)


class LabelDecoder(nn.Module):
    def __init__(
        self,
        levels: int,
        embed_dim: int,
        width: int,
        grid_size: int,
        image_size: int,
    ) -> None:
        super().__init__()
        self.grid_size = grid_size
        self.image_size = image_size
        self.projections = nn.ModuleList([nn.Linear(embed_dim, width) for _ in range(levels)])
        self.fusion = nn.Sequential(
            nn.Conv2d(width * levels, width, 1),
            nn.GroupNorm(8 if width >= 8 else 1, width),
            nn.GELU(),
        )
        self.upsample = nn.ModuleList([UpsampleBlock(width) for _ in range(4)])
        self.head = nn.Conv2d(width, 1, 1)

    def forward(self, levels: Sequence[torch.Tensor]) -> torch.Tensor:
        maps = []
        for projection, tokens in zip(self.projections, levels):
            projected = projection(tokens)
            batch, count, channels = projected.shape
            if count != self.grid_size**2:
                raise ValueError(
                    f"Esperados {self.grid_size**2} tokens; recebidos {count}."
                )
            maps.append(
                projected.transpose(1, 2).reshape(
                    batch, channels, self.grid_size, self.grid_size
                )
            )
        x = self.fusion(torch.cat(maps, dim=1))
        for block in self.upsample:
            if x.shape[-1] >= self.image_size:
                break
            x = block(x)
        if x.shape[-2:] != (self.image_size, self.image_size):
            x = F.interpolate(
                x,
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            )
        return self.head(x)


class VTM(nn.Module):
    def __init__(
        self,
        image_size: int,
        train_tasks: Iterable[str],
        backbone: str = "vit_tiny_patch16_224",
        pretrained: bool = True,
        feature_blocks: Sequence[int] = (2, 5, 8, 11),
        label_depth: int = 4,
        num_heads: int = 4,
        decoder_width: int = 96,
        dropout: float = 0.0,
        embed_dim: int | None = None,
        patch_size: int = 16,
    ) -> None:
        super().__init__()
        self.image_encoder = ImageEncoder(
            backbone,
            image_size,
            feature_blocks,
            pretrained,
            embed_dim,
            patch_size,
        )
        self.num_levels = len(feature_blocks)
        if label_depth != self.num_levels:
            raise ValueError("label_depth deve ser igual ao número de feature_blocks.")
        dim = self.image_encoder.embed_dim
        self.label_encoder = LabelEncoder(
            image_size,
            self.image_encoder.patch_size,
            dim,
            label_depth,
            num_heads,
            dropout,
        )
        self.matcher = TokenMatcher(self.num_levels, dim, num_heads, dropout)
        self.decoder = LabelDecoder(
            self.num_levels,
            dim,
            decoder_width,
            self.image_encoder.grid_size,
            image_size,
        )
        self.task_biases = nn.ParameterDict(
            {
                _safe_key(task): nn.Parameter(torch.zeros(self.num_levels, dim))
                for task in train_tasks
            }
        )

    @property
    def bias_shape(self) -> tuple[int, int]:
        return self.num_levels, self.image_encoder.embed_dim

    def bias_for(self, task: str) -> torch.Tensor:
        key = _safe_key(task)
        if key not in self.task_biases:
            raise KeyError(f"Task bias desconhecido: {task}")
        return self.task_biases[key]

    @staticmethod
    def _flatten_support(levels: Sequence[torch.Tensor], batch: int, shots: int) -> list[torch.Tensor]:
        return [level.reshape(batch, shots * level.shape[1], level.shape[2]) for level in levels]

    def forward(
        self,
        query_images: torch.Tensor,
        support_images: torch.Tensor,
        support_masks: torch.Tensor,
        task: str | None = None,
        task_bias: torch.Tensor | None = None,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor | None]]:
        if (task is None) == (task_bias is None):
            raise ValueError("Informe exatamente um entre task e task_bias.")
        bias = self.bias_for(task) if task is not None else task_bias
        batch, shots = support_images.shape[:2]
        if query_images.shape[0] != batch or support_masks.shape[:2] != (batch, shots):
            raise ValueError("Batch de query, imagens support e máscaras support incompatíveis.")

        query_levels = self.image_encoder(query_images, bias)
        flat_images = support_images.flatten(0, 1)
        flat_masks = support_masks.flatten(0, 1)
        support_image_levels = self.image_encoder(flat_images, bias)
        support_label_levels = self.label_encoder(flat_masks)
        support_image_levels = self._flatten_support(support_image_levels, batch, shots)
        support_label_levels = self._flatten_support(support_label_levels, batch, shots)
        matched, attention = self.matcher(
            query_levels,
            support_image_levels,
            support_label_levels,
            return_attention,
        )
        logits = self.decoder(matched)
        return (logits, attention) if return_attention else logits


class FrozenFeatureBaseline(nn.Module):
    """Decoder pequeno ajustado diretamente sobre features congeladas."""

    def __init__(self, image_encoder: ImageEncoder, decoder_width: int = 96) -> None:
        super().__init__()
        self.image_encoder = image_encoder
        dim = image_encoder.embed_dim
        self.decoder = nn.Sequential(
            nn.Conv2d(dim, decoder_width, 1),
            nn.GELU(),
            nn.Conv2d(decoder_width, decoder_width, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(decoder_width, 1, 1),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        tokens = self.image_encoder(images)[-1]
        batch, _, channels = tokens.shape
        grid = self.image_encoder.grid_size
        features = tokens.transpose(1, 2).reshape(batch, channels, grid, grid)
        logits = self.decoder(features)
        return F.interpolate(
            logits,
            size=(self.image_encoder.image_size, self.image_encoder.image_size),
            mode="bilinear",
            align_corners=False,
        )
