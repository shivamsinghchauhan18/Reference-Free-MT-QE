import torch
import torch.nn as nn
from transformers import AutoModel, PreTrainedModel, AutoConfig

class MetricX25Evaluator(nn.Module):
    """
    MetricX-25 Model adapting Gemma 3 for Quality Estimation (Encoder-Only).
    """
    def __init__(self, model_name: str = "google/gemma-3-12b-it", freeze_encoder: bool = False, use_lora: bool = True):
        super().__init__()
        
        # Load backbone
        # We use AutoModel which typically loads the base model without head
        # We need to handle the "is_decoder=False" part. 
        # Gemma is a causal LM, so it defaults to causal attention.
        # Transformers library doesn't always support simply flipping is_decoder=False for CausalLMs 
        # without some hacking or specific config changes if the architecture supports it.
        # MetricX documentation suggests they "initialize as encoder". 
        # For practical implementation with HF Transformers, we often just remove the attention mask's causal triangle
        # or use a specific configuration if available. 
        # Let's try loading it and manipulating the config.
        
        self.config = AutoConfig.from_pretrained(model_name)
        # Force non-causal attention if possible in config, though specific implementation depends on model code.
        # Gemma2/3 typically uses SDPA. We will handle attention mask in forward pass if needed
        # or rely on the fact that if we provide a full attention mask of 1s, it might still do causal.
        # However, for true encoder behavior, we need bidirectional attention. 
        # Standard HF Causal models check 'is_decoder'.
        
        # NOTE: For this implementation, we will load the model and assume we can pass 
        # attention_mask that allows bidirectional flow if we don't pass causal=True implicitly.
        # But for Causal architectures, HF usually forces causal masking.
        # A robust way is to wrap the underlying model implementation or use a helper that strips the mask.
        
        from transformers import BitsAndBytesConfig
        
        # QLoRA Configuration for Memory Efficiency
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        
        self.backbone = AutoModel.from_pretrained(
            model_name, 
            trust_remote_code=True,
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        
        if freeze_encoder and not use_lora:
            for param in self.backbone.parameters():
                param.requires_grad = False
                
        # Regression Head
        # Gemma hidden size
        # FIX: Handle Gemma 3 nested configuration
        if hasattr(self.config, "hidden_size"):
            hidden_size = self.config.hidden_size
        elif hasattr(self.config, "text_config") and hasattr(self.config.text_config, "hidden_size"):
            # Gemma 3 stores text params inside text_config
            hidden_size = self.config.text_config.hidden_size
        else:
            # Fallback or error if neither exists
            raise AttributeError(f"Could not find 'hidden_size' in config for {model_name}")
        
        self.regression_head = nn.Sequential(
            nn.Linear(hidden_size, 1024),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 1)
        )
        
    def forward(self, input_ids, attention_mask=None):
        # We need to ensure bidirectional attention.
        # If the backbone is a CausalLM, it might automatically apply causal masking.
        # We can try to rely on the model accepting 'attention_mask' 
        # and checking if it disables causal masking when a custom mask is provided.
        # Many modern HF models (Gemma, Llama) use 'is_causal' in forward.
        
        # Ideally, we'd pass use_cache=False to save memory
        outputs = self.backbone(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            use_cache=False,
            output_hidden_states=True
        )
        
        # Last hidden state: [batch, seq, hidden]
        last_hidden_state = outputs.last_hidden_state
        
        # Mean Pooling (ignoring padding)
        if attention_mask is not None:
            # Expand mask: [batch, seq] -> [batch, seq, hidden]
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
            sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            pooled_output = sum_embeddings / sum_mask
        else:
            pooled_output = last_hidden_state.mean(dim=1)
            
        # Regression
        score = self.regression_head(pooled_output)
        
        return score.squeeze(-1) # [batch]
