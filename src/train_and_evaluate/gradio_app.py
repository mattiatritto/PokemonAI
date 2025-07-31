import os
import re
import gradio as gr
import torch
from torchvision.transforms import ToPILImage

from model.t2i_generator import TextToImageModel
from train_and_evaluate.dataset import PokemonDataset



# Configuration for the inference
EMBEDDING_DIM = 256
N_HEADS_ENOCDER = 1
NUM_TRANSFORMER_ENCODER_LAYERS = 1
DIM_FEEDFORWARD_ENCODER = 512
DROPOUT_ENCODER = 0
NOISE_ENCODER = False

N_HEADS_ATTENTION = 1
DROPOUT_ATTENTION = 0
DECODER_DIM = 256

NOISE_DIM = 256
DROPOUT_DECODER = 0

CONTEXT_DIM = 256
OUTPUT_SIZE = 215
CONFIG_NAME = "ft_lr_0.0001_wd_1e-06_nhe_1_dfn_512_ntel_1"
CHECKPOINT_DIR = "checkpoints"
# CHECKPOINT_DIR = "best_models"



def get_latest_checkpoint(checkpoint_dir=f"../../results/trained_models/{CONFIG_NAME}/{CHECKPOINT_DIR}/"):

    """
    Finds the latest checkpoint file in the given directory matching pattern 'best_model_epoch_<num>.pt'.
    
    Args:
        checkpoint_dir (str): Directory to scan for checkpoint files.
    
    Returns:
        str: Full path to the latest checkpoint file.
    
    Raises:
        FileNotFoundError: If no checkpoint file is found.
    """

    pattern = re.compile(r"model_epoch_(\d+)\.pt")
    max_epoch = -1
    latest_ckpt = None

    for fname in os.listdir(checkpoint_dir):
        match = pattern.match(fname)
        if match:
            epoch = int(match.group(1))
            if epoch > max_epoch:
                max_epoch = epoch
                latest_ckpt = fname

    if latest_ckpt is None:
        raise FileNotFoundError(f"No checkpoint files found in {checkpoint_dir}")

    return os.path.join(checkpoint_dir, latest_ckpt)



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = TextToImageModel(
    embedding_dim=EMBEDDING_DIM,
    n_heads_encoder=N_HEADS_ENOCDER,
    num_transformer_encoder_layers=NUM_TRANSFORMER_ENCODER_LAYERS,
    dim_feedforward_encoder=DIM_FEEDFORWARD_ENCODER,
    dropout_encoder=DROPOUT_ENCODER,
    noise_encoder=NOISE_ENCODER,

    n_heads_attention=N_HEADS_ATTENTION,
    dropout_attention=DROPOUT_ATTENTION,
    decoder_dim=DECODER_DIM,

    noise_dim=NOISE_DIM,
    dropout_decoder=DROPOUT_DECODER
)


# Load the latest checkpoint found in the config directory specified
last_checkpoint = get_latest_checkpoint(f"../../results/trained_models/{CONFIG_NAME}/{CHECKPOINT_DIR}/")
checkpoint = torch.load(last_checkpoint, map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)
model.eval()



def from_tensor_to_image(image):
    """
    Converts a normalized tensor image to a PIL Image.

    Args:
        image_tensor (torch.Tensor): Image tensor with values in range [-1, 1].

    Returns:
        PIL.Image.Image: Converted PIL Image.
    """
        
    image = ((image.cpu() + 1) / 2).clamp(0, 1)
    return ToPILImage()(image.squeeze(0))



def generate_pokemon(description):

    """
    Generates a Pokémon sprite image based on a textual description.

    Args:
        description (str): Text description of the Pokémon.

    Returns:
        PIL.Image.Image: Generated Pokémon sprite image.
    """
        
    model.eval()
    with torch.no_grad():
        dataset = PokemonDataset("../../data/test.csv", "../../data/test_sprites")
        token_tensor, attention_mask = dataset.tokenize_text(description, dataset.tokenizer)
        token_tensor = token_tensor.unsqueeze(0).to(device)         # Shape: [1, seq_len]
        attention_mask = attention_mask.unsqueeze(0).to(device)     # Shape: [1, seq_len]
        generated_image = model(token_tensor, attention_mask)  # [1, C, H, W]
        img_pil = from_tensor_to_image(generated_image)

        return img_pil



if __name__ == "__main__":
    demo = gr.Interface(fn=generate_pokemon, inputs=gr.Textbox(label="Pokémon Description"), outputs=gr.Image(type="pil", label="Generated Pokémon Sprite"), title="Pokémon Text-to-Image Generator", description="Enter a Pokémon description like 'A small, fiery dragon with wings' and generate a sprite.")
    demo.launch(server_name="0.0.0.0", server_port=7860)