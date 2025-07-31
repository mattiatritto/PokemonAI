import torch.nn as nn
import torch.nn.functional as F



class AttentionBlock(nn.Module):
    """A multi-head attention block to summarize encoder outputs into a context vector.

    Processes encoder outputs using multi-head scaled dot-product attention
    to produce a context vector for downstream tasks, such as image generation in a text-to-image model.

    Args:
        encoder_dim (int): Dimensionality of the input encoder outputs.
        decoder_dim (int): Dimensionality of the output context vector.
        num_heads (int): Number of attention heads.
        dropout (float, optional): Dropout rate for attention.
    """

    def __init__(self, encoder_dim, decoder_dim, num_heads, dropout):
        super(AttentionBlock, self).__init__()
        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim
        self.num_heads = num_heads

        # Ensure decoder_dim is divisible by num_heads, because each head receives a piece of the input
        assert decoder_dim % num_heads == 0, f"[Error]: Decoder dimension ({decoder_dim}) must be divisible by num_heads ({num_heads})"

        # Multi-head attention layer
        self.attn = nn.MultiheadAttention(embed_dim=decoder_dim, num_heads=num_heads, dropout=dropout, batch_first=True)

        # Linear layer to project mean-pooled encoder output to query
        self.query_proj = nn.Linear(encoder_dim, decoder_dim)

        # Linear layer to project context vector if needed
        self.out_proj = nn.Linear(encoder_dim, decoder_dim) if encoder_dim != decoder_dim else nn.Identity()



    def forward(self, encoder_outputs, attention_mask):
        """Computes multi-head attention over encoder outputs to produce a context vector.

        Args:
            encoder_outputs (torch.Tensor): Encoder embeddings, shape [batch_size, seq_len, encoder_dim].
            attention_mask (torch.Tensor): Mask for valid tokens (1) and padding (0), shape [batch_size, seq_len].

        Returns:
            tuple: (context_vector, attn_weights)
                - context_vector: Weighted context vector, shape [batch_size, decoder_dim].
                - attn_weights: Attention weights, shape [batch_size, num_heads, 1, seq_len].
        """

        # encoder_outputs has shape 10x256x256
        # attention_mask has shape 10x256



        # Mean-pool encoder outputs
        masked_encoder_outputs = encoder_outputs * attention_mask.unsqueeze(-1) # unsqueeze adds a new dimension at the end
        # masked_encoder_outputs has shape 10x256x256



        sum_masked = masked_encoder_outputs.sum(dim=1)
        # sum_maksed has shape 10x256
        num_tokens = attention_mask.sum(dim=1).unsqueeze(-1)
        # num_tokens has shape 10x1
        mean_encoder_output = sum_masked / (num_tokens + 1e-9) # Avoid divison by zero
        # mean_encoder_output has shape 10x256



        # Project to decoder_dim
        query = self.query_proj(mean_encoder_output).unsqueeze(1)  # [batch_size, 1, decoder_dim]
        # query has shape 10x1x256





        # Multi-head attention
        key_padding_mask = (attention_mask == 0)  # Masks padding tokens
        # key_padding_mask has shape 10x256

        attn_output, attn_weights = self.attn(query=query, key=encoder_outputs, value=encoder_outputs, key_padding_mask=key_padding_mask, need_weights=True)
        # attn_output has shape 10x1x256
        # attn_weights has shape 10x1x256


        context_vector = attn_output.squeeze(1)
        context_vector = self.out_proj(context_vector)
        # context_vector has shape 10x256 

        return context_vector, attn_weights