import os
import itertools
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from torchvision import transforms
from model.t2i_generator import TextToImageModel
from dataset import PokemonDataset
from evaluation import evaluate
import open_clip
from torchvision.transforms.functional import to_pil_image
import pandas as pd


# Hyperparameters
NUM_EPOCHS = 100
BATCH_SIZE = 10
RANDOM_SEED = 42
IMAGES_VAL = 10
PERCEPTUAL_LOSS_WEIGHT = 0

EMBEDDING_DIM = 256
DROPOUT_ENCODER = 0.3
NOISE_ENCODER = False

N_HEADS_ATTENTION = 1
DROPOUT_ATTENTION = 0.3
DECODER_DIM = 256

NOISE_DIM = 256
DROPOUT_DECODER = 0.3


CONTEXT_DIM = 256
OUTPUT_SIZE = 215


GRID_PARAMETERS = {
"learning_rate": [1e-4, 1e-5],
"weight_decay": [1e-6, 1e-5],
"n_heads_encoder": [2, 4],
"dim_feedforward_encoder": [512, 1024],
"num_transformers_encoder_layers": [2, 3]
}


"""
GRID_PARAMETERS = {
"learning_rate": [1e-4],
"weight_decay": [1e-6],
"n_heads_encoder": [1],
"dim_feedforward_encoder": [512],
"num_transformers_encoder_layers": [1]
}
"""




class CLIPLoss(nn.Module):

    """
    CLIP-based loss function using cosine similarity between image and text features.

    Attributes:
        model: The pre-trained CLIP model.
        preprocess: The image preprocessing pipeline compatible with the CLIP model.
        device: Device where tensors will be allocated.
    """
        
    def __init__(self, device):
        super(CLIPLoss, self).__init__()
        self.model, _, self.preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
        self.model = self.model.to(device).eval()
        self.device = device



    def forward(self, images, texts):
        """
        Forward pass to compute CLIP loss.

        Args:
            images (torch.Tensor): Batch of images as tensors in range [-1, 1], shape (B, C, H, W).
            texts (List[str]): Corresponding text descriptions for the images.

        Returns:
            torch.Tensor: Scalar CLIP loss value (1 - cosine similarity).
        """
                
        # Normalize image from [-1, 1] to [0, 1]
        images = (images + 1) / 2

        # Convert tensors to PIL images and preprocess
        pil_images = [to_pil_image(image.cpu()) for image in images]
        processed_images = torch.stack([self.preprocess(img).to(self.device) for img in pil_images])

        # Tokenize text
        text_tokens = open_clip.tokenize(texts).to(self.device)

        # Encode with CLIP
        image_features = self.model.encode_image(processed_images)
        text_features = self.model.encode_text(text_tokens)

        # Normalize features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # Compute cosine similarity
        similarity = torch.sum(image_features * text_features, dim=-1)

        # CLIP loss = 1 - cosine similarity
        return 1 - similarity.mean()


    
def save_image_comparison(images_pred, images_gt, path_prefix, max_images=IMAGES_VAL, pdf_filename="image_comparisons.pdf"):

    """
    Save a side-by-side comparison of predicted and ground-truth images into a multi-page PDF.

    Args:
        images_pred (List[Tensor]): List of predicted image tensors.
        images_gt (List[Tensor]): List of ground-truth image tensors.
        path_prefix (str): Directory path prefix where the PDF will be saved.
        max_images (int): Maximum number of images to save.
        pdf_filename (str): Filename for the output PDF.
    """
        
    to_pil = transforms.ToPILImage()
    os.makedirs(path_prefix, exist_ok=True)
    pdf_path = os.path.join(path_prefix, pdf_filename)
    
    with PdfPages(pdf_path) as pdf:
        for i in range(min(len(images_pred), max_images)):
            fig, axes = plt.subplots(2, 1, figsize=(4, 6))

            pred = from_tensor_to_image(images_pred[i])
            gt = from_tensor_to_image(images_gt[i])

            axes[0].imshow(to_pil(gt))
            axes[0].set_title(f"GT Image #{i}")
            axes[0].axis("off")

            axes[1].imshow(to_pil(pred))
            axes[1].set_title(f"Predicted Image #{i}")
            axes[1].axis("off")

            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

    print(f"[Success]: PDF saved at: {pdf_path}")
    


def from_tensor_to_image(image):
    """
    Normalize an image tensor from [-1, 1] to [0, 1] for visualization.

    Args:
        image (torch.Tensor): Image tensor.

    Returns:
        torch.Tensor: Normalized image tensor clamped between 0 and 1.
    """
        
    image = ((image.cpu() + 1) / 2).clamp(0, 1)
    return image



def train(model, train_loader, val_loader, device, num_epochs, save_dir, plot_dir, learning_rate, weight_decay):

    """
    Train the Text-to-Image model.

    Args:
        model (nn.Module): The model to train.
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
        device (torch.device): Device to use for computation.
        num_epochs (int): Number of epochs to train.
        save_dir (str): Directory to save checkpoints.
        plot_dir (str): Directory to save image comparisons and plots.
        learning_rate (float): Learning rate for optimizer.
        weight_decay (float): Weight decay (L2 regularization) for optimizer.

    Returns:
        float: Best validation loss achieved during training.
    """
        
    criterion_l1 = nn.L1Loss()
    # clip_loss_fn = CLIPLoss(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    checkpoints_dir = os.path.join(save_dir, "checkpoints")
    best_models_dir = os.path.join(save_dir, "best_models")
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(best_models_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    best_val_loss = float('inf')
    train_losses, val_losses = [], []

    for epoch in tqdm(range(1, num_epochs + 1), desc="Training Epochs", unit="epoch"):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            input_ids, attention_mask, real_images, text_descriptions = batch
            input_ids, attention_mask, real_images = input_ids.to(device), attention_mask.to(device), real_images.to(device)

            optimizer.zero_grad()
            generated_images = model(input_ids, attention_mask)

            # clip_loss = clip_loss_fn(generated_images, text_descriptions)
            l1_loss = criterion_l1(generated_images, real_images)

            loss = l1_loss

            loss.backward()
            optimizer.step()
            train_loss += loss.item() * input_ids.size(0)

        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                input_ids, attention_mask, real_images, text_descriptions = batch
                input_ids, attention_mask, real_images = input_ids.to(device), attention_mask.to(device), real_images.to(device)

                generated_images = model(input_ids, attention_mask)
                l1_loss = criterion_l1(generated_images, real_images)
                # clip_loss = clip_loss_fn(generated_images, text_descriptions)
                loss = l1_loss
                val_loss += loss.item() * input_ids.size(0)

        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)

        print(f"[Epoch {epoch}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        checkpoint_path = os.path.join(checkpoints_dir, f"model_epoch_{epoch}.pt")
        # torch.save({
        #     'epoch': epoch,
        #     'model_state_dict': model.state_dict(),
        #     'optimizer_state_dict': optimizer.state_dict(),
        #     'val_loss': val_loss
        # }, checkpoint_path)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_checkpoint_path = os.path.join(best_models_dir, f"best_model_epoch_{epoch}.pt")
            # torch.save({
            #     'epoch': epoch,
            #     'model_state_dict': model.state_dict(),
            #     'optimizer_state_dict': optimizer.state_dict(),
            #     'val_loss': val_loss
            # }, best_checkpoint_path)

        # Image visualization every 10 epochs
        if epoch % 10 == 0:
            def collect_images(loader, split):
                model.eval()
                all_preds = []
                all_gts = []
                with torch.no_grad():
                    for batch in loader:
                        input_ids, attention_mask, real_images, _ = batch
                        input_ids, attention_mask, real_images = input_ids.to(device), attention_mask.to(device), real_images.to(device)
                        generated_images = model(input_ids, attention_mask)
                        all_preds.extend(generated_images.cpu())
                        all_gts.extend(real_images.cpu())
                        if len(all_preds) >= IMAGES_VAL:
                            break
                save_image_comparison(all_preds, all_gts, os.path.join(plot_dir, f"epoch_{epoch}", split))

            collect_images(train_loader, "train")
            collect_images(val_loader, "val")

    # Plot loss curves
    plt.figure()
    plt.plot(range(1, num_epochs + 1), train_losses, label="Train Loss")
    plt.plot(range(1, num_epochs + 1), val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Learning Curves")
    plt.legend()
    plt.grid(True)
    plot_path = os.path.join(plot_dir, "learning_curves.png")
    plt.savefig(plot_path)
    plt.close()

    return best_val_loss



def run_grid_search(save_dir_base="../../results"):
    """
    Run a grid search over specified hyperparameters and train models accordingly.
    Save results (best validation loss, test loss, CLIP score) to a CSV file.

    Args:
        save_dir_base (str): Base directory to save all grid search outputs and results.

    Returns:
        list: A list of dictionaries containing the results for each parameter combination.
    """
    # Initialize datasets and dataloaders
    train_dataset = PokemonDataset(csv_path="../../data/train.csv", sprite_dir="../../data/train_sprites")
    val_dataset = PokemonDataset(csv_path="../../data/val.csv", sprite_dir="../../data/val_sprites")
    test_dataset = PokemonDataset(csv_path="../../data/test.csv", sprite_dir="../../data/test_sprites")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    # Initialize results storage
    results = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Grid search over hyperparameter combinations
    param_combinations = list(itertools.product(*GRID_PARAMETERS.values()))
    for lr, wd, n_heads_encoder, dim_feedforward_encoder, num_transformers_encoder_layers in tqdm(param_combinations, desc="Grid Search", unit="config"):
        config_name = f"ft_lr_{lr}_wd_{wd}_nhe_{n_heads_encoder}_dfn_{dim_feedforward_encoder}_ntel_{num_transformers_encoder_layers}"
        print(f"\n[Training] Running config: {config_name}")

        # Initialize model
        model = TextToImageModel(
            embedding_dim=EMBEDDING_DIM,
            n_heads_encoder=n_heads_encoder,
            num_transformer_encoder_layers=num_transformers_encoder_layers,
            dim_feedforward_encoder=dim_feedforward_encoder,
            dropout_encoder=DROPOUT_ENCODER,
            noise_encoder=NOISE_ENCODER,
            n_heads_attention=N_HEADS_ATTENTION,
            dropout_attention=DROPOUT_ATTENTION,
            decoder_dim=DECODER_DIM,
            noise_dim=NOISE_DIM,
            dropout_decoder=DROPOUT_DECODER
        )
        model.to(device)

        # Set up directories
        save_dir = os.path.join(save_dir_base, "trained_models", config_name)
        plot_dir = os.path.join(save_dir_base, "plots", config_name)

        # Train the model
        best_val_loss = train(model, train_loader, val_loader, device, NUM_EPOCHS, save_dir, plot_dir, lr, wd)

        # Evaluate on test set
        test_loss, clip_score_avg = evaluate(model, test_loader, device)

        # Collect images for test set
        def collect_images(loader, split):
            model.eval()
            all_preds = []
            all_gts = []
            with torch.no_grad():
                for batch in loader:
                    input_ids, attention_mask, real_images, _ = batch
                    input_ids, attention_mask, real_images = input_ids.to(device), attention_mask.to(device), real_images.to(device)
                    generated_images = model(input_ids, attention_mask)
                    all_preds.extend(generated_images.cpu())
                    all_gts.extend(real_images.cpu())
            save_image_comparison(all_preds, all_gts, os.path.join(plot_dir, "epoch", split), max_images=len(test_loader.dataset))

        collect_images(test_loader, "test")

        # Store results for this configuration
        result = {
            'config_name': config_name,
            'learning_rate': lr,
            'weight_decay': wd,
            'best_val_loss': best_val_loss,
            'test_loss': test_loss,
            'clip_score_avg': clip_score_avg,
        }
        results.append(result)
        print(f"[Finished] Config: {config_name} | Best Val Loss: {best_val_loss:.4f} | "
              f"Test Loss: {test_loss:.4f} | CLIP Score: {clip_score_avg:.4f}")

    # Save results to CSV
    os.makedirs(save_dir_base, exist_ok=True)
    csv_path = os.path.join(save_dir_base, "grid_search_results.csv")
    results_df = pd.DataFrame(results)
    results_df.to_csv(csv_path, index=False)
    print(f"[Success] Grid search results saved to: {csv_path}")

    return results

if __name__ == "__main__":
    run_grid_search()

