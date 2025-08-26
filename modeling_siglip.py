from typing import Optional, Tuple
import torch 
import torch.nn as nn 
class SiglipVisionConfig:
    def __init__(
        self,
        hidden_size = 768,
        intermediate_size = 3072,
        num_hidden_layers = 12,
        num_attention_heads = 12,
        num_channels = 3,
        image_size = 224, # default is the smallest paligemma modedl
        Patch_size = 16,
        layer_norm_ep = 1e-6,
        attention_dropout = 0.0, # maybe will use it in the future
        num_image_tokens: int = None,   # how many image embeddings for each image
        **kwargs
    ):
        super.__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.image_size = image_size
        self.patch_size = Patch_size
        self.attention_dropout = attention_dropout
        self.layer_norm_eps = layer_norm_ep
        self.num_image_tokens = num_image_tokens
    




# this is a single layer pipeline for the transformer shown in 2_transformer_arch
class SUglipEncoderLayer(nn.Module):
    def __init__(self, config:SiglipVisionConfig):
        super.__init__()
        self.embed_dim = config.hidden_size
        self.self_atn = SiglipAttention(config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps = config.layer_norm_eps)
        self.mlp = SiglipMLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps = config.layer_norm_eps)
        
        
    def forward(
        self,
        hidden_states: torch.Tensor) -> torch.Tensor:
        
        residual = hidden_states   # [B, Num_patches, Embed_dim]
        
        hidden_states = self.layer_norm1(residual)                     # [B, num_patches, Embed_dim]
        hidden_states, _ = self.self_atn(hidden_states=hidden_states)  # [B, num_patches, Embed_dim]
        hidden_states = residual + hidden_states                       # [B, num_patches, Embed_dim]
        
        residual = hidden_states
        
        hidden_states = self.layer_norm2(residual)                     # [B, num_patches, Embed_dim]
        hidden_states = self.mlp(hidden_states)                        # [B, num_patches, Embed_dim]
        hidden_states = residual + hidden_states                       # [B, num_patches, Embed_dim]
        
        return hidden_states
        



class SiglipVisionEmbeddings(nn.Module):
    def __init__(self,config: SiglipVisionConfig):
        super.__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size
        
        # basically we want to extract first level info using convolution on these 16 patches without overlap so the kernal is so big.
        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=config.hidden_size,
            kernel_size=config.patch_size,
            stride = config.patch_size,
            padding = "valid",  # no padding
        )
        
        self.num_patches = (self.image_size//self.patch_size)**2
        self.num_positions = self.num_patches
        self.position_embedding = nn.Embedding(self.num_positions,self.embed_dim)
        
        # this below registers the postiions of each extracted path feature (0,1,2,3,4...)
        self.register_buffer(
            "position_ids",
            torch.arange(self.num_patches).expand((1,-1)),
            persistent=False,
        )
        
        
    def forward(self, pixel_values:torch.FloatTensor) -> torch.Tensor:
        B,C,H,W = pixel_values.shape

        # Basically feature extraction by a normal convolution,  [B,C,H,W] -> [B, embed_dim, H_new, W_new]
        patch_embeds = self.patch_embedding(pixel_values)        
        
        # Flattening it for 1D embeddings,  [B, embed_dim, H_new, W_new] -> [B, embed_dim, num_patches]  (num_patches = H_new * W_new)
        embeddings = patch_embeds.flatten(2)

        # concat with positional info
        embeddings = embeddings + self.position_embedding(self.position_ids)
        
        return embeddings
    
    
    
    
    
# the main vision transformer pipeline 
class SiglipVisionTransformer(nn.Module):
    def __init__(self, config = SiglipVisionConfig):
        super.__init__()
        self.config = config
        embed_dim = config.hidden_size
        
        # define embedding extractino, transformer forward pass and final layer norm 
        self.embeddings = SiglipVisionEmbeddings(config)
        self.encoder = SiglipEncoder(config)
        self.post_layernorm = nn.LayerNorm(embed_dim, eps = config.layer_norm_eps)
        
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # [B,C,H,W] -> [B,Num_patches,Embed_dim]
        
        hidden_states = self.embeddings(pixel_values)   # basically extract patches and embed them
        last_hidden_state = self.encoder(input_embeds = hidden_states)  # this is going to be the main list of transformer layers
        last_hidden_state = self.post_layernorm(last_hidden_state)
        
        return last_hidden_state
        
    
# pass to the main vision transformer
class SiglipVisionModel(nn.Module):
    def __init__(self,config: SiglipVisionConfig):
        super.__init__()
        self.config = config
        self.vision_model = SiglipVisionTransformer(config)
        
    def forward(self,pixel_values) -> Tuple:
        # [B,C,H,W] -> [B,num_patches, embed_dim]
        return self.vision_model(pixel_values=pixel_values)