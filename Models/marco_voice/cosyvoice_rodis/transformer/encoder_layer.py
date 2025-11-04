# 

# Copyright (c) 2021 Mobvoi Inc (Binbin Zhang, Di Wu)
#               2022 Xingchen Song (sxc19@mails.tsinghua.edu.cn)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Modified from ESPnet(https://github.com/espnet/espnet)
"""Encoder self-attention layer definition."""

from typing import Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F
class TransformerEncoderLayer(nn.Module):
    """Encoder layer module.

    Args:
        size (int): Input dimension.
        self_attn (torch.nn.Module): Self-attention module instance.
            `MultiHeadedAttention` or `RelPositionMultiHeadedAttention`
            instance can be used as the argument.
        feed_forward (torch.nn.Module): Feed-forward module instance.
            `PositionwiseFeedForward`, instance can be used as the argument.
        dropout_rate (float): Dropout rate.
        normalize_before (bool):
            True: use layer_norm before each sub-block.
            False: to use layer_norm after each sub-block.
    """

    def __init__(
        self,
        size: int,
        self_attn: torch.nn.Module,
        feed_forward: torch.nn.Module,
        dropout_rate: float,
        normalize_before: bool = True,
    ):
        """Construct an EncoderLayer object."""
        super().__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.norm1 = nn.LayerNorm(size, eps=1e-5)
        self.norm2 = nn.LayerNorm(size, eps=1e-5)
        self.dropout = nn.Dropout(dropout_rate)
        self.size = size
        self.normalize_before = normalize_before

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        pos_emb: torch.Tensor,
        mask_pad: torch.Tensor = torch.ones((0, 0, 0), dtype=torch.bool),
        att_cache: torch.Tensor = torch.zeros((0, 0, 0, 0)),
        cnn_cache: torch.Tensor = torch.zeros((0, 0, 0, 0)),
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute encoded features.

        Args:
            x (torch.Tensor): (#batch, time, size)
            mask (torch.Tensor): Mask tensor for the input (#batch, time，time),
                (0, 0, 0) means fake mask.
            pos_emb (torch.Tensor): just for interface compatibility
                to ConformerEncoderLayer
            mask_pad (torch.Tensor): does not used in transformer layer,
                just for unified api with conformer.
            att_cache (torch.Tensor): Cache tensor of the KEY & VALUE
                (#batch=1, head, cache_t1, d_k * 2), head * d_k == size.
            cnn_cache (torch.Tensor): Convolution cache in conformer layer
                (#batch=1, size, cache_t2), not used here, it's for interface
                compatibility to ConformerEncoderLayer.
        Returns:
            torch.Tensor: Output tensor (#batch, time, size).
            torch.Tensor: Mask tensor (#batch, time, time).
            torch.Tensor: att_cache tensor,
                (#batch=1, head, cache_t1 + time, d_k * 2).
            torch.Tensor: cnn_cahce tensor (#batch=1, size, cache_t2).

        """
        residual = x
        if self.normalize_before:
            x = self.norm1(x)
        x_att, new_att_cache = self.self_attn(x, x, x, mask, pos_emb=pos_emb, cache=att_cache)
        x = residual + self.dropout(x_att)
        if not self.normalize_before:
            x = self.norm1(x)

        residual = x
        if self.normalize_before:
            x = self.norm2(x)
        x = residual + self.dropout(self.feed_forward(x))
        if not self.normalize_before:
            x = self.norm2(x)

        fake_cnn_cache = torch.zeros((0, 0, 0), dtype=x.dtype, device=x.device)
        return x, mask, new_att_cache, fake_cnn_cache

class ConformerEncoderLayer(nn.Module):
    """Encoder layer module.
    Args:
        size (int): Input dimension.
        self_attn (torch.nn.Module): Self-attention module instance.
            `MultiHeadedAttention` or `RelPositionMultiHeadedAttention`
            instance can be used as the argument.
        feed_forward (torch.nn.Module): Feed-forward module instance.
            `PositionwiseFeedForward` instance can be used as the argument.
        feed_forward_macaron (torch.nn.Module): Additional feed-forward module
             instance.
            `PositionwiseFeedForward` instance can be used as the argument.
        conv_module (torch.nn.Module): Convolution module instance.
            `ConvlutionModule` instance can be used as the argument.
        dropout_rate (float): Dropout rate.
        normalize_before (bool):
            True: use layer_norm before each sub-block.
            False: use layer_norm after each sub-block.
    """

    def __init__(
        self,
        size: int,
        self_attn: torch.nn.Module,
        feed_forward: Optional[nn.Module] = None,
        feed_forward_macaron: Optional[nn.Module] = None,
        conv_module: Optional[nn.Module] = None,
        dropout_rate: float = 0.1,
        normalize_before: bool = True,
    ):
        """Construct an EncoderLayer object."""
        super().__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.feed_forward_macaron = feed_forward_macaron
        self.conv_module = conv_module
        self.norm_ff = nn.LayerNorm(size, eps=1e-5)  # for the FNN module
        self.norm_mha = nn.LayerNorm(size, eps=1e-5)  # for the MHA module
        if feed_forward_macaron is not None:
            self.norm_ff_macaron = nn.LayerNorm(size, eps=1e-5)
            self.ff_scale = 0.5
        else:
            self.ff_scale = 1.0
        if self.conv_module is not None:
            self.norm_conv = nn.LayerNorm(size, eps=1e-5)  # for the CNN module
            self.norm_final = nn.LayerNorm(
                size, eps=1e-5)  # for the final output of the block
        self.dropout = nn.Dropout(dropout_rate)
        self.size = size
        self.normalize_before = normalize_before

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        pos_emb: torch.Tensor,
        mask_pad: torch.Tensor = torch.ones((0, 0, 0), dtype=torch.bool),
        att_cache: torch.Tensor = torch.zeros((0, 0, 0, 0)),
        cnn_cache: torch.Tensor = torch.zeros((0, 0, 0, 0)),
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute encoded features.

        Args:
            x (torch.Tensor): (#batch, time, size)
            mask (torch.Tensor): Mask tensor for the input (#batch, time，time),
                (0, 0, 0) means fake mask.
            pos_emb (torch.Tensor): positional encoding, must not be None
                for ConformerEncoderLayer.
            mask_pad (torch.Tensor): batch padding mask used for conv module.
                (#batch, 1，time), (0, 0, 0) means fake mask.
            att_cache (torch.Tensor): Cache tensor of the KEY & VALUE
                (#batch=1, head, cache_t1, d_k * 2), head * d_k == size.
            cnn_cache (torch.Tensor): Convolution cache in conformer layer
                (#batch=1, size, cache_t2)
        Returns:
            torch.Tensor: Output tensor (#batch, time, size).
            torch.Tensor: Mask tensor (#batch, time, time).
            torch.Tensor: att_cache tensor,
                (#batch=1, head, cache_t1 + time, d_k * 2).
            torch.Tensor: cnn_cahce tensor (#batch, size, cache_t2).
        """

        # whether to use macaron style
        if self.feed_forward_macaron is not None:
            residual = x
            if self.normalize_before:
                x = self.norm_ff_macaron(x)
            x = residual + self.ff_scale * self.dropout(
                self.feed_forward_macaron(x))
            if not self.normalize_before:
                x = self.norm_ff_macaron(x)

        # multi-headed self-attention module
        residual = x
        if self.normalize_before:
            x = self.norm_mha(x)
        x_att, new_att_cache = self.self_attn(x, x, x, mask, pos_emb,
                                              att_cache)
        x = residual + self.dropout(x_att)
        if not self.normalize_before:
            x = self.norm_mha(x)

        # convolution module
        # Fake new cnn cache here, and then change it in conv_module
        new_cnn_cache = torch.zeros((0, 0, 0), dtype=x.dtype, device=x.device)
        if self.conv_module is not None:
            residual = x
            if self.normalize_before:
                x = self.norm_conv(x)
            x, new_cnn_cache = self.conv_module(x, mask_pad, cnn_cache)
            x = residual + self.dropout(x)

            if not self.normalize_before:
                x = self.norm_conv(x)

        # feed forward module
        residual = x
        if self.normalize_before:
            x = self.norm_ff(x)

        x = residual + self.ff_scale * self.dropout(self.feed_forward(x))
        if not self.normalize_before:
            x = self.norm_ff(x)

        if self.conv_module is not None:
            x = self.norm_final(x)

        return x, mask, new_att_cache, new_cnn_cache

class StyleMultiHeadAttention(nn.Module):
    def __init__(self, embed_dim=512, num_heads=8, dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        # 投影层
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

        # 输出层 + 残差连接
        # self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        # self.layer_norm = nn.LayerNorm(embed_dim)  # 用于残差连接后的归一化

    def forward(self, style, text, key_padding_mask=None):
        """
        输入:
            style: [batch_size, 1, embed_dim]   (如 [71, 1, 512])
            text: [batch_size, seq_len, embed_dim]  (如 [71, 240, 512])
            key_padding_mask: [batch_size, 1, seq_len]  (如 [71, 1, 240])

        输出:
            output: [batch_size, seq_len, embed_dim]  (与输入text同维度)
            attn_weights: [batch_size, num_heads, seq_len]
        """
        batch_size, seq_len, _ = text.shape

        # 扩展style的序列长度以匹配text
        style_expanded = style.expand(-1, seq_len, -1)  # [71, 240, 512]

        # 投影Q,K,V
        q = self.q_proj(style_expanded)  # [71, 240, 512]
        k = self.k_proj(text)            # [71, 240, 512]
        v = self.v_proj(text)            # [71, 240, 512]

        # 分割多头
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # [71, 8, 240, 64]
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)   # [71, 8, 240, 64]
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)   # [71, 8, 240, 64]

        # 计算注意力
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [71, 8, 240, 240]

        # 应用mask
        if key_padding_mask is not None:
            mask = key_padding_mask # .unsqueeze(1)  # [71, 1, 1, 240]
            mask = mask.expand(-1, self.num_heads, -1, -1)
            mask = mask.expand(-1, -1, attn_scores.size(2), -1)
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # attentoin
        attended = torch.matmul(attn_weights, v)  # [71, 8, 240, 64]
        attended = attended.transpose(1, 2).reshape(batch_size, seq_len, -1)  # [71, 240, 512]

        # 投影输出 + 残差连接
        # output = self.out_proj(attended)
        # output = self.layer_norm(output + text)  # 残差连接 + LayerNorm

        return attended # , attn_weights

class StyleConformerEncoderLayer(nn.Module):
    """Enhanced Encoder layer module with style attention.

    Args:
        size (int): Input dimension.
        self_attn (torch.nn.Module): Self-attention module instance.
        style_attn (torch.nn.Module): Style attention module instance.
        feed_forward (torch.nn.Module): Feed-forward module instance.
        feed_forward_macaron (torch.nn.Module): Additional feed-forward module instance.
        conv_module (torch.nn.Module): Convolution module instance.
        dropout_rate (float): Dropout rate.
        normalize_before (bool): Whether to use layer_norm before each sub-block.
        style_dim (int): Dimension of style embeddings.
    """

    def __init__(
        self,
        size: int,
        self_attn: torch.nn.Module,
        # style_attn: Optional[nn.Module] = None,
        feed_forward: Optional[nn.Module] = None,
        feed_forward_macaron: Optional[nn.Module] = None,
        conv_module: Optional[nn.Module] = None,
        dropout_rate: float = 0.1,
        normalize_before: bool = True,
        # style_dim: int = 512,
    ):
        super().__init__()
        self.self_attn = self_attn
        # self.style_attn = style_attn
        self.feed_forward = feed_forward
        self.feed_forward_macaron = feed_forward_macaron
        self.conv_module = conv_module
        self.dropout = nn.Dropout(dropout_rate)
        self.size = size
        self.normalize_before = normalize_before

        # Normalization layers
        self.style_attn = StyleMultiHeadAttention()
        self.norm_ff = nn.LayerNorm(size, eps=1e-5)
        self.norm_mha = nn.LayerNorm(size, eps=1e-5)
        if self.style_attn is not None:
            self.norm_style = nn.LayerNorm(size, eps=1e-5)
            # self.style_proj = nn.Linear(style_dim, size)

        if feed_forward_macaron is not None:
            self.norm_ff_macaron = nn.LayerNorm(size, eps=1e-5)
            self.ff_scale = 0.5
        else:
            self.ff_scale = 1.0

        if conv_module is not None:
            self.norm_conv = nn.LayerNorm(size, eps=1e-5)
            self.norm_final = nn.LayerNorm(size, eps=1e-5)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        pos_emb: torch.Tensor,
        style_embed: Optional[torch.Tensor] = None,
        mask_pad: torch.Tensor = torch.ones((0, 0, 0), dtype=torch.bool),
        att_cache: torch.Tensor = torch.zeros((0, 0, 0, 0)),
        cnn_cache: torch.Tensor = torch.zeros((0, 0, 0, 0)),
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute encoded features with style modeling.

        Args:
            x (torch.Tensor): Input tensor (#batch, time, size)
            mask (torch.Tensor): Mask tensor for the input (#batch, time, time)
            pos_emb (torch.Tensor): Positional encoding
            style_embed (torch.Tensor): Style embedding (#batch, style_dim)
            mask_pad (torch.Tensor): Batch padding mask for conv module
            att_cache (torch.Tensor): Attention cache
            cnn_cache (torch.Tensor): Convolution cache

        Returns:
            torch.Tensor: Output tensor
            torch.Tensor: Mask tensor
            torch.Tensor: Updated attention cache
            torch.Tensor: Updated convolution cache
        """
        # Macaron-style feed forward (pre-normalization)
        if self.feed_forward_macaron is not None:#不进
            residual = x
            if self.normalize_before:
                x = self.norm_ff_macaron(x)
            x = residual + self.ff_scale * self.dropout(
                self.feed_forward_macaron(x))
            if not self.normalize_before:
                x = self.norm_ff_macaron(x)

        # Multi-headed self-attention module
        residual = x
        if self.normalize_before:
            x = self.norm_mha(x)
        x_att, new_att_cache = self.self_attn(x, x, x, mask, pos_emb, att_cache)
        x = residual + self.dropout(x_att)
        if not self.normalize_before:
            x = self.norm_mha(x)

        # Style attention module (new)
        if self.style_attn is not None and style_embed is not None:
            residual = x
            if self.normalize_before:
                x = self.norm_style(x)

            # Project style embedding to match attention dimensions
            # style_projected = self.style_proj(style_embed.transpose(1, 0)).unsqueeze(1)  # (batch, 1, size) (66x512 and 64x512)

            # Style attention: attend to content features based on style
            # Using style as query, content as key and value
            x_style = self.style_attn(
                style_embed.expand(-1, x.size(1), -1),  # Expand style to sequence length torch.Size([67, 1, 512]) mask [71, 1, 240]
                x,  # Content as key
                # x,  # Content as value
                mask.unsqueeze(1)   # Adjust mask dimensions
            )
            # Combine style-attended features with original features
            x = residual + self.dropout(x_style.squeeze(1))
            if not self.normalize_before:
                x = self.norm_style(x)

        # Convolution module
        new_cnn_cache = torch.zeros((0, 0, 0), dtype=x.dtype, device=x.device)
        if self.conv_module is not None:
            residual = x
            if self.normalize_before:
                x = self.norm_conv(x)
            x, new_cnn_cache = self.conv_module(x, mask_pad, cnn_cache)
            x = residual + self.dropout(x)
            if not self.normalize_before:
                x = self.norm_conv(x)

        # Feed forward module
        residual = x
        if self.normalize_before:
            x = self.norm_ff(x)
        x = residual + self.ff_scale * self.dropout(self.feed_forward(x))
        if not self.normalize_before:
            x = self.norm_ff(x)

        if self.conv_module is not None:
            x = self.norm_final(x)

        return x, mask, new_att_cache, new_cnn_cache