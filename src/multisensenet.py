"""Contains all components of MultiSenseNet model.

"""
import math
import hashlib
import torch
import torch.nn as nn
import torch.nn.functional as F

def make_model(
    cfg: "congifuration.ExperimentConfiguration",
    taskname: str
): 
    """
    Make Multisensenet model.

    Parameters
    ----------
    cfg : congifuration.ExperimentConfiguration
        Class object that bundles configurations for experiment.
    taskname : str
        
    """
    # initialize model
    model = TransformerBackbone(
        std_vect_dim=cfg.std_vect_dim,
        n_heads=cfg.n_heads,
        n_layers=cfg.n_layers,
        dim_feedforward=cfg.dim_feedforward,
        dropout=cfg.dropout,
        max_seq_len=cfg.max_seq_len,
        layer_norm_eps=cfg.layer_norm_eps,
        activation=cfg.activation
    ).to(cfg.torch_device)


    return model 




def _make_delim(
    tag: str, 
    std_vect_dim: int
) -> torch.Tensor:
    """
    Deterministically generate a delimiter vector of length `std_vect_dim`
    from a textual tag.  The SHA‑256 hash gives a reproducible 32‑byte seed,
    which is reduced to 32 bits and used to seed an independent RNG.

    Parameters
    ----------
    tag : str
        Tag that indicates which delimiter to generator. Used by RNG.
    std_vect_dim : int
        Dimension of canonical data format vectors expected by Transformer 
        backbone model. Used by RNG.
    
    Returns:
    ----------
    vec : torch.Tensor
        Unique and deterministically reproducible delimiter token for requested
        tag and std_vect_dim. 

    """
    # 1. hash “tag:dim” → 32‑byte hex, keep first 8 hex chars → 32‑bit int
    seed = int(hashlib.sha256(f"{tag}:{std_vect_dim}".encode()).hexdigest()[:8], 16)
    gen = torch.Generator()
    gen.manual_seed(seed)                       # reproducible, tag‑specific
    vec = torch.randn(std_vect_dim, generator=gen)   # N(0, 1), same scale as normal tokens
    vec = F.normalize(vec, p=2, dim=0)          # ‖vec‖₂ = 1  (optional but good)

    return vec


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

        # --- dedicated delimiter tokens ------------------------------------
        # They live on the same vector space as ordinary tokens,
        # but are *buffers* (fixed), not trainable parameters.
        self.register_buffer("task_description_delimtoken",
            _make_delim("TASK_DESC", std_vect_dim))
        self.register_buffer("data_point_delimtoken",
            _make_delim("DATA_POINT", std_vect_dim))
        self.register_buffer("modality_description_delimtoken",
            _make_delim("MODALITY_DESC", std_vect_dim))
        self.register_buffer("numeric_modality_delimtoken",
            _make_delim("NUMERIC_MODALITY", std_vect_dim))


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Shape (batch, seq_len, std_vect_dim).

        Returns
        ----------
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
        ----------
        torch.Tensor
            Encoded sequence, same shape as input.

        """
        # Create canonical format using delimiter tokens. All of shape (D,).
        delim_task = self.pos_enc.task_description_delimtoken
        delim_datapoint = self.pos_enc.data_point_delimtoken
        delim_modality = self.pos_enc.modality_description_delimtoken
        delim_numeric = self.pos_enc.numeric_modality_delimtoken

        # add positional encoding.
        x = self.pos_enc(tokens)

        # do forward pass.
        x = self.encoder(x, src_key_padding_mask=src_key_padding_mask)

        return self.final_norm(x)