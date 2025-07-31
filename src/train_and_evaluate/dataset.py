import os
import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer



class PokemonDataset(Dataset):
    """
    PokemonDataset: A PyTorch Dataset class for Pokémon text-to-image generation.

    This dataset class:
    - Loads Pokémon metadata and sprite images.
    - Constructs enriched text descriptions.
    - Applies text tokenization using a local BERT tokenizer.
    - Preprocesses image data for model input.

    Assumptions:
    - Images are RGBA and must be converted to RGB with a white background.
    - Images are resized to 215x215 and normalized to [-1, 1].
    """



    def __init__(self, csv_path, sprite_dir):
        """
        Args:
            csv_path (str): Path to the CSV file containing Pokémon metadata.
            sprite_dir (str): Path to the directory containing Pokémon sprite images.
        """
        self.data = self.load_data(csv_path, sprite_dir)
        self.tokenizer = AutoTokenizer.from_pretrained("../model/bert-mini-local")
        self.vocab_size = self.tokenizer.vocab_size



    def __len__(self):
        """
        Returns:
            int: Number of samples in the dataset.
        """
        return len(self.data)



    def __getitem__(self, idx):
        """
        Retrieves a single sample from the dataset.

        Args:
            idx (int): Index of the sample.

        Returns:
            input_ids (Tensor): Tokenized input IDs.
            attention_mask (Tensor): Attention mask for input.
            image (Tensor): Preprocessed image tensor.
            description (str): Original enriched description text.
        """
        entry = self.data[idx]
        input_ids, attention_mask = self.tokenize_text(entry["description"], tokenizer=self.tokenizer)
        image = self.preprocess_image(entry["image_path"])
        return input_ids, attention_mask, image, entry["description"]
    


    def load_data(self, csv_path, sprite_dir):
        """
        Loads and preprocesses metadata and image paths from the CSV file.

        Args:
            csv_path (str): Path to metadata CSV.
            sprite_dir (str): Path to sprite images.

        Returns:
            List[Dict]: List of dictionaries with image path and enriched descriptions.

        Raises:
            AssertionError: If an expected image file does not exist.
        """
        df = pd.read_csv(csv_path, sep=';')
        data = []

        for _, row in df.iterrows():
            raw_id = str(row["national_number"])
            id_str = raw_id.split("_")[0]
            description = str(row["description"]).strip()
            primary_type = str(row["primary_type"]).capitalize()
            secondary_type = str(row["secondary_type"]).capitalize() if pd.notna(row["secondary_type"]) else None
            classification = str(row["classification"]).strip()
            sprite_filename = f"{id_str.zfill(3)}.png"
            sprite_path = os.path.join(sprite_dir, sprite_filename)
            assert os.path.exists(sprite_path), f"[Error]: Missing sprite for #{id_str}"
            
            # Create enriched description
            enriched_description = f"{description} This Pokémon is classified as a {classification}. It is of {primary_type} type"
            if secondary_type:
                enriched_description += f" and {secondary_type} type"
            # enriched_description = description
            
            data.append({"number": id_str, "description": enriched_description, "image_path": sprite_path})

        return data
    


    def tokenize_text(self, text, tokenizer):
        """
        Tokenizes input text using a pretrained tokenizer.

        Args:
            text (str): Input text description.
            tokenizer (transformers.PreTrainedTokenizer): BERT-compatible tokenizer.

        Returns:
            Tuple[Tensor, Tensor]: Tokenized input IDs and attention mask.
        """
        encoding = tokenizer(text, padding='max_length', truncation=True, max_length=256, return_tensors="pt")
        input_ids = encoding["input_ids"]
        attention_mask = encoding["attention_mask"]
        return input_ids.squeeze(0), attention_mask.squeeze(0)
    


    def preprocess_image(self, image_path):
        """
        Loads and normalizes an image from file.

        - Converts RGBA to RGB using a white background.
        - Resizes to (215, 215).
        - Normalizes to [-1, 1].
        - Converts to PyTorch tensor (C, H, W).

        Args:
            image_path (str): Path to the image file.

        Returns:
            Tensor: Normalized image tensor of shape (3, 215, 215).

        Raises:
            AssertionError: If the final image tensor shape is incorrect.
        """
        image = Image.open(image_path).convert("RGBA")
        background = Image.new("RGB", image.size, (255, 255, 255))
        background.paste(image, mask=image.split()[3])
        image = background

        if image.size != (215, 215):
            image = image.resize((215, 215), Image.LANCZOS)
        
        img_array = np.array(image)
        img_array = img_array / 255.0
        img_array = (img_array - 0.5) / 0.5
        img_tensor = torch.FloatTensor(img_array).permute(2, 0, 1)
        assert img_tensor.shape == (3, 215, 215), f"[Error]: Image tensor shape mismatch: {img_tensor.shape}"
        return img_tensor