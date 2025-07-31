import numpy as np
import  torchvision.transforms as transforms
import clip

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.inception import inception_v3




def evaluate(model, test_loader, device):
    """
    Evaluate the text-to-image model on a test dataset.

    Computes:
    - L1 loss between generated and real images.
    - Average CLIP similarity score between generated images and text prompts.
    - Inception score of generated images.

    Args:
        model (torch.nn.Module): The trained text-to-image generation model.
        test_loader (DataLoader): DataLoader for test dataset yielding
            (input_ids, attention_mask, real_images, text_prompts).
        device (torch.device): Device to perform computations on (CPU/GPU).

    Returns:
        tuple: (average_test_loss, average_clip_score, inception_score)
    """

    criterion = nn.L1Loss()
    model.eval()
    test_loss = 0.0
    clip_score_total = 0.0
    all_preds = []

    clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
    clip_model.eval()


    with torch.no_grad():
        for batch in test_loader:
            input_ids, attention_mask, real_images, text_prompts = batch
            input_ids, attention_mask, real_images = input_ids.to(device), attention_mask.to(device), real_images.to(device)
            generated_images = model(input_ids, attention_mask)
            loss = criterion(generated_images, real_images)
            test_loss += loss.item() * input_ids.size(0)

            for i in range(len(generated_images)):
                image = generated_images[i]
                image_clip = clip_preprocess(transforms.ToPILImage()(image.cpu())).unsqueeze(0).to(device)
                raw_text = text_prompts[i]
                text_tokens = clip.tokenize([raw_text]).to(device)
                img_feat = clip_model.encode_image(image_clip)
                txt_feat = clip_model.encode_text(text_tokens)
                img_feat /= img_feat.norm(dim=-1, keepdim=True)
                txt_feat /= txt_feat.norm(dim=-1, keepdim=True)
                clip_score = (img_feat @ txt_feat.T).item()
                clip_score_total += clip_score

    test_loss /= len(test_loader.dataset)
    clip_score_avg = clip_score_total / len(test_loader.dataset)
    print(f"Test Loss: {test_loss:.4f} | CLIP Score: {clip_score_avg:.4f}")
    return test_loss, clip_score_avg