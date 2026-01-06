from typing import Optional, Tuple, List
import math
import torch
import torch.nn as nn
from siglip import SiglipVisionConfig, SiglipVisionModel


class GemmaConfig:
  def __init__(
    self,
    vocab_size,
    hidden_size, # embedding size of each vector
    intermediate_size,
    num_hidden_layers,
    num_attention_heads, # num heads for query (grouped query attention)
    num_key_value_heads, # num heads for key/value
    head_dim=256,
    max_position_embedding=8192,
    rms_norm_eps=1e-6,
    rope_theta=10000.0,
    attention_bias=False,
    attention_dropout=0.0,
    pad_token_id=None,
    **kwargs
  ):
    pass


class PaliGemmaConfig:
  def __init__(
    self,
    vision_config=None,
    text_config=None,
    ignore_index=-100,
    image_token_index=256000,  # <image> placeholder token index
    vocab_size=257152,
    projection_dim=2048,  # final dimension image features should be resized to (projection layer)
    hidden_size=2048,  # embedding size of language model
    pad_token_id=None,
    **kwargs,
  ):
    super().__init__()
    self.ignore_index = ignore_index
    self.image_token_index = image_token_index
    self.vocab_size = vocab_size
    self.projection_dim = projection_dim
    self.hidden_size = hidden_size
    self.vision_config = vision_config
    self.is_encoder_decoder = False
    self.pad_token_id = pad_token_id

    self.vision_config = SiglipVisionConfig(**vision_config)
    self.text_config = text_config

    self.text_config = GemmaConfig(**text_config, pad_token_id=pad_token_id)
    self.vocab_size = self.text_config.vocab_size

    self.text_config.num_image_tokens = (self.vision_config.image_size // self.vision_config.patch_size) ** 2
    self.vision_config.projection_dim = projection_dim


class PaliGemmaForConditionalGeneration(nn.Module):
  def __init__(self, config: PaliGemmaConfig):
    super().__init__()
    self.config = config
    self.vision_tower = SiglipVisionModel(config.vision_config)
    self.multi_modal_projector = PaliGemmaMultiModalProjector(config)  # Linear layer after SigLIP, before decoder
    self.vocab_size = config.vocab_size

    language_model = GemmaForCausalLM(config.text_config)
    self.language_model = language_model

    self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else -1
  
  def tie_weights(self):
    self.language_model.tie_weights()
  
  def forward(
    self,
    input_ids: torch.LongTensor = None,
    pixel_values: torch.FloatTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    kv_cache: Optional[KVCache] = None,
  ) -> Tuple:
    assert torch.all(attention_mask == 1), "The input cannot be padded"

    # 1. Extract the input embeddings
    # shape: (batch_size, seq_len, hidden_size)
    inputs_embeds = self.language_model.get_input_embeddings()(input_ids)

    # 2. Merge text and images
    # [batch_size, channels, height, width] -> [batch_size, num_patches, embed_dim]
    selected_image_feature = self.vision_tower(pixel_values.to(inputs_embeds.dtype))
    # [batch_size, num_patches, embed_dim] -> [batch_size, num_patches, hidden_size]
    # resize image embeddings into same size as text embeddings using multi-modal projector
    image_features = self.multi_modal_projector(selected_image_feature)

    # Merge embeddings of the image and text tokens
    inputs_embeds, attention_mask, position_ids = self._merge_input_ids_with_image_features(image_features, input_embeds, input_ids, attention_mask, kv_cache)

    outputs = self.language_model(
      attention_mask=attention_mask,
      position_ids=position_ids,
      inputs_embeds=inputs_embeds,
      kv_cache=kv_cache,
    )

    return outputs
