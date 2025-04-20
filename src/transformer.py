"""Contains transofmer backbone model.

"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """
    Encode positions with sine–cosine alternation like in Vaswani et al, 2017.

    Parameters
    ----------
    std_vect_dim : int
        Token (embedding) dimension.
    max_len : int
        Maximum sequence length you expect. Make this as large as
        the longest sequence you will see; excess positions are ignored.

    """
    def __init__(
        self, 
        std_vect_dim: int, 
        max_len: int
    ):
        super().__init__()

        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, std_vect_dim, 2).float() \
            * (-math.log(10000.0) / std_vect_dim)
        )
        pe = torch.zeros(max_len, std_vect_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Register as buffer so it moves with .to(device) but not as parameter
        self.register_buffer("pe", pe.unsqueeze(0)) # (1, max_len, std_vect_dim)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Shape (batch, seq_len, std_vect_dim).

        Returns
        -------
        torch.Tensor
            Input plus position encodings, same shape.

        """
        return x + self.pe[:, : x.size(1), :]


class TransformerBackbone(nn.Module):
    """
    Transformer encoder for fixed‑width (std_vect_dim-D) token sequences.

    """
    def __init__(
        self,
        std_vect_dim: int,
        n_heads: int,
        n_layers: int,
        dim_feedforward: int,
        dropout: float,
        max_seq_len: int,
        layer_norm_eps: float,
        activation: str
    ):
        super().__init__()
        self.pos_enc = PositionalEncoding(std_vect_dim, max_seq_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=std_vect_dim,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True,  # (batch, seq, feature)
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.final_norm = nn.LayerNorm(std_vect_dim, eps=layer_norm_eps)

    def forward(
        self,
        tokens: torch.Tensor,
        src_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        tokens : torch.Tensor
            Float tensor of shape (batch, seq_len, 1024).
        src_key_padding_mask : torch.BoolTensor, optional
            Shape (batch, seq_len); True for PAD positions to ignore.

        Returns
        -------
        torch.Tensor
            Encoded sequence, same shape as input.

        """
        x = self.pos_enc(tokens)
        x = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        return self.final_norm(x)

