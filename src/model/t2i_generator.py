import torch.nn as nn
from model.encoder import TextEncoder
from model.attention import AttentionBlock
from model.decoder import ImageGenerator



class TextToImageModel(nn.Module):
    
    """
    TextToImageModel generates an image from a textual input using a transformer-based text encoder,
    attention mechanism, and an image decoder.

    This model combines:
    - A BERT-based text encoder for extracting meaningful language features.
    - A cross-attention mechanism to summarize text context into a fixed-size vector.
    - An image generator that conditions on both the text context and the full encoder output
      through multiple attention-guided decoding layers.

    Args:
        embedding_dim (int): Dimensionality of token embeddings and encoder outputs.
        n_heads_encoder (int): Number of self-attention heads in the text encoder.
        num_transformer_encoder_layers (int): Number of transformer layers in the text encoder.
        dim_feedforward_encoder (int): Dimension of the feedforward network in the transformer encoder.
        dropout_encoder (float): Dropout rate used in the encoder.
        noise_encoder (bool): Whether to add noise to the text encoding (for robustness or diversity).
        n_heads_attention (int): Number of heads in the attention mechanism used in both encoder summary and decoder.
        dropout_attention (float): Dropout rate in attention blocks.
        decoder_dim (int): Dimensionality of the context vector passed to the decoder.
        noise_dim (int): Size of the random noise vector concatenated with text context in the decoder.
        dropout_decoder (float): Dropout rate used within the image generator.

    Forward Inputs:
        input_ids (Tensor): Tokenized input text, shape [batch_size, seq_len].
        attention_mask (Tensor, optional): Binary mask indicating which tokens are valid (1) or padding (0), shape [batch_size, seq_len].

    Returns:
        Tensor: Generated image of shape [batch_size, 3, 215, 215], with pixel values in range [-1, 1].
    """
    
    def __init__(self, embedding_dim, n_heads_encoder, num_transformer_encoder_layers, 
                 dim_feedforward_encoder, dropout_encoder, noise_encoder,
                 n_heads_attention, dropout_attention, decoder_dim, 
                 noise_dim, dropout_decoder):
        
        super(TextToImageModel, self).__init__()
        self.noise_encoder = noise_encoder
        self.text_encoder = TextEncoder(embedding_dim=embedding_dim, n_heads=n_heads_encoder, num_layers=num_transformer_encoder_layers, dim_feedforward=dim_feedforward_encoder, dropout=dropout_encoder, bert_ckpt_path="../model/bert-mini-local")
        self.attention = AttentionBlock(encoder_dim=embedding_dim, decoder_dim=decoder_dim, num_heads=n_heads_attention, dropout=dropout_attention)
        self.decoder = ImageGenerator(text_context_dim=decoder_dim, noise_dim=noise_dim, image_size=215, dropout=dropout_decoder, encoder_dim=embedding_dim, num_heads_attention=n_heads_attention, dropout_attention=dropout_attention)



    def forward(self, input_ids, attention_mask=None):
        # Encode text input
        encoder_outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask, add_noise=self.noise_encoder)

        # Compute initial context vector
        context_vector, attn_weights = self.attention(encoder_outputs, attention_mask)

        # Pass encoder outputs and attention mask to decoder for iterative attention
        generated_image = self.decoder(context_vector, encoder_outputs, attention_mask)
        return generated_image