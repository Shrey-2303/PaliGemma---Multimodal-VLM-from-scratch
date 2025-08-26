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
    
# the main vision transformer 
class SiglipVisionTransformer(nn.Module):
    def __init__(self, config = SiglipVisionConfig):
        super.__init__()
        self.config = config
        embed_dim = config.hidden_size
        
        self.embeddings = SiglipVisionEmbeddings(config)
        self.encoder = SiglipEncoder(config)
        self.post_layernorm = nn.LayerNorm(embed_dim, eps = config.layer_norm_eps)
        
    
    
# pass to the main vision transformer
class SiglipVisionModel(nn.Module):
    def __init__(self,config: SiglipVisionConfig):
        super.__init__()
        self.config = config
        self.vision_model = SiglipVisionTransformer(config)
        
    def forward(self,pixel_values) -> Tuple:
        # [B,C,H,W] -> [B,num_patches, embed_dim]
        return self.vision_model(pixel_values=pixel_values)