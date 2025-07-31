import torch
import torch.nn as nn
import torch.nn.functional as F
from model.attention import AttentionBlock



class ImageGenerator(nn.Module):
    """
    ImageGenerator generates images conditioned on text context and visual encoder outputs,
    using a combination of transposed convolutions and attention-modulated decoding layers.

    Args:
        text_context_dim (int): Dimension of the input text context vector.
        noise_dim (int): Dimension of the random noise vector to introduce stochasticity.
        image_size (int): Target output image size (unused directly but can be used for checks).
        dropout (float): Dropout probability for decoder layers.
        encoder_dim (int): Dimensionality of the encoder outputs used for attention.
        num_heads_attention (int): Number of attention heads in the AttentionBlock.
        dropout_attention (float): Dropout rate within the attention mechanism.

    Architecture:
        - Concatenates text context and noise vector.
        - Projects to initial spatial feature map of shape [1024, 8, 8].
        - Five decoding stages using ConvTranspose2d to upsample to [256, 256].
        - Each stage includes:
            - Convolutional upsampling
            - BatchNorm, ReLU, Dropout
            - Attention block using encoder outputs
            - Channel-wise projection of context to match feature maps
        - Final stage outputs a [3, 215, 215] image using a ConvTranspose2d layer with Tanh activation.

    Forward Inputs:
        text_context (Tensor): Shape [batch_size, text_context_dim], representing embedded text features.
        encoder_outputs (Tensor): Shape [batch_size, seq_len, encoder_dim].
        attention_mask (Tensor): Shape [batch_size, seq_len], with 1s for valid positions and 0s for masked ones.

    Returns:
        Tensor: Generated image of shape [batch_size, 3, 215, 215], with values in range [-1, 1].
    """
        
    def __init__(self, text_context_dim, noise_dim, image_size, dropout, encoder_dim, num_heads_attention, dropout_attention):
        super(ImageGenerator, self).__init__()
        
        self.text_context_dim = text_context_dim
        self.noise_dim = noise_dim
        self.image_size = image_size
        self.num_channels = 3
        self.input_dim = text_context_dim + noise_dim
        self.init_size = 8
        self.init_channels = 1024
        self.dropout = dropout
        self.encoder_dim = encoder_dim
        self.decoder_dim = 256

        # Initial linear layer
        self.fc_init = nn.Linear(self.input_dim, self.init_channels * self.init_size * self.init_size)
        self.fc_dropout = nn.Dropout(p=self.dropout)



        # Define decoder layers with attention at each step
        self.decoder_layers = nn.ModuleList([
            # Layer 1: 8x8 -> 16x16
            nn.ModuleDict({
                'conv': nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1, bias=False),
                'bn': nn.BatchNorm2d(512),
                'relu': nn.ReLU(True),
                'dropout': nn.Dropout2d(p=self.dropout),
                'attention': AttentionBlock(encoder_dim=encoder_dim, decoder_dim=self.decoder_dim, num_heads=num_heads_attention, dropout=dropout_attention),
                'proj': nn.Linear(self.decoder_dim, 512)  # Project context to match conv output channels
            }),
            # Layer 2: 16x16 -> 32x32
            nn.ModuleDict({
                'conv': nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
                'bn': nn.BatchNorm2d(256),
                'relu': nn.ReLU(True),
                'dropout': nn.Dropout2d(p=self.dropout),
                'attention': AttentionBlock(encoder_dim=encoder_dim, decoder_dim=self.decoder_dim, num_heads=num_heads_attention, dropout=dropout_attention),
                'proj': nn.Linear(self.decoder_dim, 256)
            }),
            # Layer 3: 32x32 -> 64x64
            nn.ModuleDict({
                'conv': nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
                'bn': nn.BatchNorm2d(128),
                'relu': nn.ReLU(True),
                'dropout': nn.Dropout2d(p=self.dropout),
                'attention': AttentionBlock(encoder_dim=encoder_dim, decoder_dim=self.decoder_dim, num_heads=num_heads_attention, dropout=dropout_attention),
                'proj': nn.Linear(self.decoder_dim, 128)
            }),
            # Layer 4: 64x64 -> 128x128
            nn.ModuleDict({
                'conv': nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
                'bn': nn.BatchNorm2d(64),
                'relu': nn.ReLU(True),
                'dropout': nn.Dropout2d(p=self.dropout),
                'attention': AttentionBlock(encoder_dim=encoder_dim, decoder_dim=self.decoder_dim, num_heads=num_heads_attention, dropout=dropout_attention),
                'proj': nn.Linear(self.decoder_dim, 64)
            }),
            # Layer 5: 128x128 -> 256x256
            nn.ModuleDict({
                'conv': nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, bias=False),
                'bn': nn.BatchNorm2d(32),
                'relu': nn.ReLU(True),
                'attention': AttentionBlock(encoder_dim=encoder_dim, decoder_dim=self.decoder_dim, num_heads=num_heads_attention, dropout=dropout_attention),
                'proj': nn.Linear(self.decoder_dim, 32)
            }),
            # Final layer: Adjust to 215x215
            nn.ModuleDict({
                'conv': nn.ConvTranspose2d(32, self.num_channels, kernel_size=4, stride=1, padding=22, bias=False),
                'tanh': nn.Tanh()
            })
        ])
        self.apply(self._init_weights)



    def _init_weights(self, module):
        if isinstance(module, (nn.ConvTranspose2d, nn.Linear)):
            nn.init.normal_(module.weight.data, 0.0, 0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias.data, 0)
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.constant_(module.weight.data, 1.0)
            nn.init.constant_(module.bias.data, 0)



    def forward(self, text_context, encoder_outputs, attention_mask):

        # text_context has shape 10x256



        batch_size = text_context.shape[0]
        noise = torch.randn(batch_size, self.noise_dim, device=text_context.device)
        # noise has shape 10x256


        x = torch.cat((text_context, noise), dim=1)
        # x has shape 10x512

        x = self.fc_init(x)
        # x has shape 10x65536


        x = self.fc_dropout(x)
        x = x.view(batch_size, self.init_channels, self.init_size, self.init_size)
        # x has shape 10x1024x8x8


        # Process through decoder layers
        for i, layer in enumerate(self.decoder_layers[:-1]):  # Exclude final layer
            # Apply convolution, batch norm, ReLU, and dropout
            x = layer['conv'](x)
            # x has shape 10x512x16x16
            # x has shape 10x256x32x32
            # x has shape 10x128x64x64
            # x has shape 10x64x32x32
            # x has shape 10x32x256x156

            x = layer['bn'](x)
            x = layer['relu'](x)
            if i != 4:
                x = layer['dropout'](x)

            # Prepare feature map for attention
            batch_size, channels, height, width = x.shape
            
            # Apply attention
            context_vector, _ = layer['attention'](encoder_outputs, attention_mask)
            # context_vector has shape 10x256
            
            # Project context to match channels and reshape to [batch_size, channels, 1, 1]
            context_vector = layer['proj'](context_vector).unsqueeze(-1).unsqueeze(-1)
            # 10x512x1x1
            # 10x256x1x1
            # 10x128x1x1
            # 10x64x1x1
            # 10x32x1x1

            x = x + context_vector.expand(-1, -1, height, width)  # Additive modulation with context vectors
            # x has shape 10x512x16x16
            # x has shape 10x256x32x32
            # x has shape 10x128x64x64
            # x has shape 10x64x128x128
            # x has shape 10x32x256x256


        # Final layer
        x = self.decoder_layers[-1]['conv'](x)
        # x has shape 10x3x215x215
        generated_image = self.decoder_layers[-1]['tanh'](x)
        
        return generated_image