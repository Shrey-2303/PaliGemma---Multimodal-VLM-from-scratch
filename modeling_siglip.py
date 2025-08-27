from typing import Optional, Tuple
import torch 
import torch.nn as nn 


'''
The main config class for the entire model
'''
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
    


'''
This entire thing is basically to introduce non linearity and contextualize te embedding in higher dim and shrinking to original
'''
class SiglipMLP(nn.Module):
    def __init__(self,config):
        super.__init__()
        self.config = config 
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)
        
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        
        hidden_states = self.fc1(hidden_states)
        hidden_states = nn.functional.gelu(hidden_states, approximate="tanh")
        hidden_states = self.fc2(hidden_states)
        
        return hidden_states
        




class SiglipAttention(nn.Module):
    """ Multi head attention from original Attention is all you need paper"""
    
    def __init__(self,config):
        super.__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim** (-0.5)   ### 1/sqrt(head_dim)
        self.dropout = config.attention_dropout
        
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)
        
    def forward(
        self,
        hidden_states: torch.Tensor,) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        
        B, seq_len, _ = hidden_states.size() # [B, num_patches, Embed_dim]
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        # [B, num_heafs, num_patches, head_dim] * [B, Num_heads, head_dim, Num_patches]
        query_states = query_states.view(B,seq_len, self.num_heads, self.head_dim).transpose(1,2)
        key_states = key_states.view(B,seq_len, self.num_heads, self.head_dim).transpose(1,2)
        value_states = value_states.view(B,seq_len, self.num_heads, self.head_dim).transpose(1,2)
        
        # calculating the atttention using the formula Q*K.t / sqrt(d_k)
        attn_weights = (torch.matmul(query_states,key_states.transpose(2,3)) * self.scale)
        
        if attn_weights.size() != (B, self.num_heads, seq_len, seq_len):
            raise ValueError(
                f"Attention weights should be of size {(B, self.num_heads, seq_len, seq_len)}, but is "
                f" {attn_weights.size()}"
            )
            
        # now apply softmax for probs
        attn_weights = nn.functional.softmax(attn_weights, dim = -1, dtype=torch.float32).to(query_states.dtype)
        
        # drop out but paper didnt use it
        attn_weights = nn.functional.dropout(attn_weights, p = self.dropout, training = self.training)
        
        # multiply atttn with value states -> [B, num_heads,num_patches, head_dim]
        attn_output = torch.matmul(attn_weights, value_states)
        
        if attn_output.size() != (B, self.num_heads, seq_len, self.head_dim):
            raise ValueError(
                f"attn_output should be of size {(B, self.num_heads, seq_len, self.head_dim)} instead it is"
                f" {attn_output.size()}"
            )

        # [B, num_heads, num_patches, head_dim] -> [B, num_patches, num_heads, head_dim]
        attn_output = attn_output.transpose(1,2).contiguous()
        
        # collapse and contatinate the heads
        attn_output = attn_output.reshape(B, seq_len,self.embed_dim)
        
        # final projection so each head has some info from other heads
        attn_output = self.out_proj(attn_output)
        
        return attn_output,attn_weights
        



'''
This is a single layer pipeline for the transformer shown in 2_transformer_arch
'''
class SiglipEncoderLayer(nn.Module):
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
        


class SiglipEncoder(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        self.config = config
        self.layers = nn.ModuleList(
            [SiglipEncoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        
    def forward(
        self,
        inputs_embeds: torch.Tensor) -> torch.Tensor:
        
        hidden_sttaes = inputs_embeds
        for encoder_layer in self.layers:
            # [B, num_patches, embed_dim] -> [B, num_patches, embed_dim]
            hidden_states = encoder_layer(hidden_states)
            
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