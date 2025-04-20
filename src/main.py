"""Main entry point to program for running experiments associated with study.


Example usage:
--------------
$ python main.py

"""
import configuration
import transformer 

PATH_CONFIG = '../config.yml'


if __name__ == "__main__":

    # parse configurations into dictionary
    cfg = configuration.ExperimentConfiguration(PATH_CONFIG)
    
    BATCH = 2
    SEQ_LEN = 128
    std_vect_dim = cfg.std_vect_dim
    device = cfg.torch_device

    model = transformer.TransformerBackbone().to(device)

    dummy_input = torch.randn(BATCH, SEQ_LEN, std_vect_dim, device=device)
    padding_mask = torch.zeros(BATCH, SEQ_LEN, dtype=torch.bool, device=device)

    with torch.no_grad():
        output = model(dummy_input, src_key_padding_mask=padding_mask)

    print("Output shape:", output.shape)  # (2, 128, 1024)
