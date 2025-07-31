import os
import shutil
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split



# Parameters to change
RANDOM_SEED = 42
NUM_AUGMENTS = 0
AUGMENT = False



def augment_image_and_save(original_path, dest_dir, base_name, num_augments):
    """
    Generates augmented images by rotating the original image.

    Args:
        original_path (str): Path to the original image.
        dest_dir (str): Destination directory to save augmented images.
        base_name (str): Base name for the new image files.
        num_augments (int): Number of rotated versions to create.
    """
        
    image = Image.open(original_path)

    for i in range(1, num_augments + 1):
        angle = 360 * i / (num_augments + 1)
        rotated = image.rotate(angle, expand=True)
        
        background = Image.new("RGB", rotated.size, (255, 255, 255))
        background.paste(rotated, mask=rotated.split()[3])
        
        aug_name = f"{base_name}_aug{i}.png"
        aug_path = os.path.join(dest_dir, aug_name)
        background.save(aug_path)



def split_dataset_and_images(csv_path="../../data/original_dataset/pokemon_data.csv", sprite_dir="../../data/original_dataset/sprites", train_csv_path="../../data/train.csv", val_csv_path="../../data/val.csv", test_csv_path="../../data/test.csv", train_sprite_dir="../../data/train_sprites", val_sprite_dir="../../data/val_sprites", test_sprite_dir="../../data/test_sprites", val_ratio=0.1, test_ratio=0.1, random_seed=RANDOM_SEED, augment=AUGMENT, augment_copies=NUM_AUGMENTS):

    """
    Splits the Pokémon dataset and copies corresponding sprite images to train, val, and test folders.

    Optionally applies rotation-based augmentation to images.

    Args:
        csv_path (str): Path to the CSV containing Pokémon metadata.
        sprite_dir (str): Directory containing original sprite images.
        train_csv_path (str): Output CSV file path for the training set.
        val_csv_path (str): Output CSV file path for the validation set.
        test_csv_path (str): Output CSV file path for the test set.
        train_sprite_dir (str): Directory to save training sprite images.
        val_sprite_dir (str): Directory to save validation sprite images.
        test_sprite_dir (str): Directory to save test sprite images.
        val_ratio (float): Fraction of the dataset to use as validation.
        test_ratio (float): Fraction of the dataset to use as test.
        random_seed (int): Seed for reproducible random splitting.
        augment (bool): Whether to apply augmentation to the training images.
        augment_copies (int): Number of augmented images to generate per sample.

    Raises:
        ValueError: If no valid images are found for any split.

    Output:
        Saves train/val/test CSV files and corresponding images in their directories.
    """

    df = pd.read_csv(csv_path, sep=';')

    train_val_df, test_df = train_test_split(df, test_size=test_ratio, random_state=random_seed, shuffle=True)
    val_adjusted_ratio = val_ratio / (1.0 - test_ratio)
    train_df, val_df = train_test_split(train_val_df, test_size=val_adjusted_ratio, random_state=random_seed, shuffle=True)

    os.makedirs(train_sprite_dir, exist_ok=True)
    os.makedirs(val_sprite_dir, exist_ok=True)
    os.makedirs(test_sprite_dir, exist_ok=True)
    
    def copy_and_augment(df_split, split_dir, sprite_dir, augment=True, augment_copies=NUM_AUGMENTS):
        """
        Copies and optionally augments a dataset split.

        Args:
            df_split (DataFrame): Subset of Pokémon metadata.
            split_dir (str): Directory to save the split's images.
            sprite_dir (str): Path to source sprite images.
            augment (bool): Whether to apply rotation-based augmentation.
            augment_copies (int): Number of augmentations per image.

        Returns:
            DataFrame: Updated metadata including augmented image entries.
        """
                
        augmented_rows = []
        os.makedirs(split_dir, exist_ok=True)

        for _, row in df_split.iterrows():
            number = str(row["national_number"])
            base_id = number.split('_')[0]
            # Try both padded and non-padded filenames
            padded_id = f"{int(base_id):03}"
            img_name = f"{number}.png" if '_' in number else f"{padded_id}.png"
            src_path = os.path.join(sprite_dir, img_name)
            # Fallback to non-padded filename
            if not os.path.exists(src_path):
                img_name = f"{base_id}.png"
                src_path = os.path.join(sprite_dir, img_name)
            dst_path = os.path.join(split_dir, img_name)

            if not os.path.exists(src_path):
                print(f"[Warning]: Image {src_path} does not exist. Skipping Pokémon #{number}.")
                continue

            shutil.copy(src_path, dst_path)
            original_row = row.to_dict()
            augmented_rows.append(original_row)

            if augment:
                augment_image_and_save(src_path, split_dir, padded_id, num_augments=augment_copies)
                for i in range(1, augment_copies + 1):
                    aug_row = row.to_dict()
                    aug_row["national_number"] = f"{padded_id}_aug{i}"
                    augmented_rows.append(aug_row)

        augmented_df = pd.DataFrame(augmented_rows)
        if augmented_df.empty:
            raise ValueError(f"[Error]: No valid images found in {sprite_dir}. Check sprite_dir or df_split.")
        return augmented_df
                

    train_df_aug = copy_and_augment(train_df, train_sprite_dir, sprite_dir=sprite_dir)
    val_df_aug = copy_and_augment(val_df, val_sprite_dir, sprite_dir=sprite_dir)
    test_df_aug = copy_and_augment(test_df, test_sprite_dir, sprite_dir=sprite_dir)

    train_df_aug.to_csv(train_csv_path, index=False, sep=';')
    val_df_aug.to_csv(val_csv_path, index=False, sep=';')
    test_df_aug.to_csv(test_csv_path, index=False, sep=';')

    print(f"[Success]: Split complete:")
    print(f"- {len(train_df_aug)} train samples -> {train_sprite_dir}")
    print(f"- {len(val_df_aug)} val samples   -> {val_sprite_dir}")
    print(f"- {len(test_df_aug)} test samples  -> {test_sprite_dir}")



if __name__ == "__main__":
    split_dataset_and_images()