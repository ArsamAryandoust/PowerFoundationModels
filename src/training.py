"""Contains training, validation and testing procedure.

"""
import torch 

def train_model(
    cfg: "congifuration.ExperimentConfiguration",
    model: "transformer.TransformerBackbone",
    taskdata_dict: tuple,
    save: bool
):
    """ 
    Train backbone network and encoder parts.

    Parameters
    ----------
    cfg : congifuration.ExperimentConfiguration
        Class object that bundles configurations for experiment.
    model : transformer.TransformerBackbone
        Backbone transformer model.
    taskdata_dict : dict
        Task data loaded as requested by user.
    save : bool 
        Whether or not to save updates to model weights.

    """
    # Placeholder tests for model
    BATCH = 2
    SEQ_LEN = 128

    dummy_input = torch.randn(
        BATCH, SEQ_LEN, cfg.std_vect_dim, device=cfg.torch_device
    )
    padding_mask = torch.zeros(
        BATCH, SEQ_LEN, dtype=torch.bool, device=cfg.torch_device
    )

    with torch.no_grad():
        output = model(dummy_input, src_key_padding_mask=padding_mask)

    print("Output shape:", output.shape)  # (2, 128, 1024)

    



