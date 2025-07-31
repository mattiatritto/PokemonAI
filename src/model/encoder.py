import torch
import torch.nn as nn
from transformers import BertModel



class TextEncoder(nn.Module):

    """A PyTorch module that encodes text inputs using a pre-trained BERT model followed by a transformer encoder.

    This module processes tokenized text inputs (e.g., Pokémon descriptions) to produce contextual embeddings,
    which can be used in a text-to-image generation pipeline. It combines a pre-trained BERT model for initial
    text encoding with a custom transformer encoder for further refinement.

    Args:
        embedding_dim (int): The dimensionality of the output embeddings. Must match the BERT model's hidden size.
        max_text_length (int): The maximum length of input text sequences (in tokens).
        n_heads (int): Number of attention heads in the transformer encoder.
        num_layers (int): Number of transformer encoder layers.
        dim_feedforward (int): Dimension of the feedforward network in the transformer encoder.
        dropout (float): Dropout rate for the transformer encoder.
        bert_ckpt_path (str): Path or identifier (e.g., 'bert-base-uncased') for the pre-trained BERT model.

    Raises:
        AssertionError: If the BERT model's hidden size does not match `embedding_dim`.
    """
        
    def __init__(self, embedding_dim, n_heads, num_layers, dim_feedforward, dropout, bert_ckpt_path):
        super(TextEncoder, self).__init__()

        self.bert_mini = BertModel.from_pretrained(bert_ckpt_path)
        bert_embedding_dim = self.bert_mini.config.hidden_size
        self.noise_std = 0.05
        assert bert_embedding_dim == embedding_dim, f"[Error]: BERT-mini hidden size ({bert_embedding_dim}) does not match embedding_dim ({embedding_dim})"
        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=n_heads, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)



    def forward(self, input_ids, attention_mask, add_noise=False):

        """Processes tokenized text inputs through BERT and a transformer encoder to produce embeddings.

        Args:
            input_ids (torch.Tensor): Token IDs for input text, shape [batch_size, seq_len].
            attention_mask (torch.Tensor): Attention mask indicating valid tokens (1) and padding (0),
                shape [batch_size, seq_len].

        Returns:
            torch.Tensor: Refined embeddings, shape [batch_size, seq_len, embedding_dim].
        """

        # input_ids has shape 10x256
        # attention_mask has shape 10x256



        outputs = self.bert_mini(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        # last_hidden_state has shape 10x256x256



        # Optionally add Gaussian noise to BERT embeddings
        if add_noise and self.training:  # Only add noise during training
            noise = torch.randn_like(last_hidden_state, device=last_hidden_state.device) * self.noise_std
            last_hidden_state = last_hidden_state + noise

        transformer_mask = (attention_mask == 0)
        # tranformer_mask has shape 10x256

        transformer_output = self.transformer(last_hidden_state, src_key_padding_mask=transformer_mask)
        # transformer_output has shape 10x256x256
        
        return transformer_output