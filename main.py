import json
from typing import Dict, Literal, Optional
from dataclasses import dataclass

# Precision to bytes mapping
PRECISION_BYTES = {
    "float32": 4,
    "bfloat16": 2,
    "float16": 2,
    "int8": 1,
    "int4": 0.5
}

PrecisionType = Literal["float32", "bfloat16", "float16", "int8", "int4"]


@dataclass
class MoEConfig:
    """Configuration for MoE model parameters"""
    V: int  # vocab_size
    h: int  # hidden_size
    l: int  # num_layers
    a: int  # num_attention_heads
    N: int  # num_experts
    f_mult: float  # expert multiplier
    s: int  # sequence_length - for kv-cache calculation
    top_k: int  # number of experts activated per token
    
    @classmethod
    def from_dict(cls, config: Dict) -> 'MoEConfig':
        """Create config from dictionary"""
        # Remove non-MoEConfig fields
        config_params = {k: v for k, v in config.items() 
                        if k in ['V', 'h', 'l', 'a', 'N', 'f_mult', 's', 'top_k']}
        return cls(**config_params)
    
    @classmethod
    def get_default_config(cls) -> Dict:
        """Return default configuration as dictionary"""
        return {
            "V": 32000,
            "h": 4096,
            "l": 32,
            "a": 32,
            "N": 8,
            "f_mult": 1.25,
            "s": 2048,
            "top_k": 2
        }


class MoEMemoryCalculator:
    """Calculate memory requirements for MoE models"""
    
    def __init__(self, config: MoEConfig, precision: PrecisionType):
        self.config = config
        self.precision = precision
        self.bytes_per_param = PRECISION_BYTES[precision]
    
    # ============ MEMORY CALCULATIONS ============
    
    def calculate_embedding_weights(self) -> int:
        """
        Embedding Weights (B) = 2 * k * V * h
        Factor of 2 accounts for input and output embedding matrices
        """
        k = self.bytes_per_param
        V = self.config.V
        h = self.config.h
        return 2 * k * V * h
    
    def calculate_ln_weights(self) -> int:
        """
        LN Weights (B) = 4 * k * h
        """
        k = self.bytes_per_param
        h = self.config.h
        return 4 * k * h
    
    def calculate_attention_weights(self) -> int:
        """
        Attention Weights (B) = 4 * k * h * (h + 1)
        4 weight matrices: query, key, value, output, each h x h
        """
        k = self.bytes_per_param
        h = self.config.h
        return 4 * k * h * (h + 1)
    
    def calculate_router_weights(self) -> int:
        """
        Router Weights (B) = k * N * (h + 1)
        Weight matrix of size h x N with learnable router weights
        """
        k = self.bytes_per_param
        N = self.config.N
        h = self.config.h
        return k * N * (h + 1)
    
    def calculate_moe_layer_weights(self) -> int:
        """
        MoE Layer Weights (B) = k * N * h * (3 * f_mult * h + 2 * f_mult + 1)
        Each expert uses SwiGLU with three linear transformations
        """
        k = self.bytes_per_param
        N = self.config.N
        h = self.config.h
        f_mult = self.config.f_mult
        return k * N * h * (3 * f_mult * h + 2 * f_mult + 1)
    
    def calculate_decoder_weights(self) -> int:
        """
        Decoder Weights (B) = LN + Attention + Router + MoE Layer
        Combines all components with layer norms already included
        """
        ln = self.calculate_ln_weights()
        attention = self.calculate_attention_weights()
        router = self.calculate_router_weights()
        moe_layer = self.calculate_moe_layer_weights()
        
        return ln + attention + router + moe_layer
    
    def calculate_model_weights(self) -> int:
        """
        Model Weights (B) = Embedding + l * Decoder
        Total weights across all layers
        """
        embedding = self.calculate_embedding_weights()
        decoder = self.calculate_decoder_weights()
        l = self.config.l
        
        return embedding + l * decoder
    
    def calculate_kv_cache(self) -> int:
        """
        KV-Cache (B) = 2 * k * l * s * h
        Cache for keys (k) and values (v) across layers l,
        sequence length s, with h/a dimension per attention head
        """
        k = self.bytes_per_param
        l = self.config.l
        s = self.config.s
        h = self.config.h
        
        return 2 * k * l * s * h
    
    # ============ FLOPS CALCULATIONS ============
    
    def calculate_embedding_flops(self) -> int:
        """
        Embedding Compute (FLOPs) = 4 * s * V * h
        """
        s = self.config.s
        V = self.config.V
        h = self.config.h
        return 4 * s * V * h
    
    def calculate_ln_flops(self) -> int:
        """
        LN Compute (FLOPs) = 14 * s * h
        """
        s = self.config.s
        h = self.config.h
        return 14 * s * h
    
    def calculate_attention_flops(self) -> int:
        """
        Attention Compute (FLOPs) = s * (8 * h^2 + 4 * s * h + 3 * s * a)
        """
        s = self.config.s
        h = self.config.h
        a = self.config.a
        return s * (8 * h**2 + 4 * s * h + 3 * s * a)
    
    def calculate_rope_flops(self) -> int:
        """
        RoPE Compute (FLOPs) = 0.75 * h
        """
        h = self.config.h
        return 0.75 * h
    
    def calculate_router_flops(self) -> int:
        """
        Router Compute (FLOPs) = s * N * (2 * h + 3)
        """
        s = self.config.s
        N = self.config.N
        h = self.config.h
        return s * N * (2 * h + 3)
    
    def calculate_moe_layer_flops(self) -> int:
        """
        MoE Layer Compute (FLOPs) = 2 * top_k * s * f_mult * h * (4 * h + 3)
        """
        top_k = self.config.top_k
        s = self.config.s
        f_mult = self.config.f_mult
        h = self.config.h
        return 2 * top_k * s * f_mult * h * (4 * h + 3)
    
    def calculate_linear_layer_flops(self) -> int:
        """
        Linear Layer Compute (FLOPs) = 2 * s * V * h
        """
        s = self.config.s
        V = self.config.V
        h = self.config.h
        return 2 * s * V * h
    
    def calculate_decoder_flops(self) -> int:
        """
        Decoder Compute (FLOPs) = LN + Attention + RoPE + Router + MoE Layer
        """
        ln = self.calculate_ln_flops()
        attention = self.calculate_attention_flops()
        rope = self.calculate_rope_flops()
        router = self.calculate_router_flops()
        moe_layer = self.calculate_moe_layer_flops()
        
        return ln + attention + rope + router + moe_layer
    
    def calculate_prefill_flops(self) -> int:
        """
        Prefill (FLOPs) = Embedding + l * Decoder
        """
        embedding = self.calculate_embedding_flops()
        decoder = self.calculate_decoder_flops()
        l = self.config.l
        
        return embedding + l * decoder
    
    # Decode FLOPs calculations (with s=1)
    
    def calculate_attention_flops_decode(self) -> int:
        """
        Attention Compute w/ KV-Cache (FLOPs) = 8 * h^2 + 4 * s * h + 3 * s * a
        Note: s here is the context length (cached tokens)
        """
        s = self.config.s  # context length for decode
        h = self.config.h
        a = self.config.a
        return 8 * h**2 + 4 * s * h + 3 * s * a
    
    def calculate_decoder_flops_decode(self) -> int:
        """
        Decoder Compute w/ KV-Cache = LN + Attention w/ KV-Cache + RoPE + Router + MoE Layer
        All components use s=1 except attention which uses cached context
        """
        # LN, RoPE, Router, MoE all use s=1
        original_s = self.config.s
        self.config.s = 1
        
        ln = self.calculate_ln_flops()
        rope = self.calculate_rope_flops()
        router = self.calculate_router_flops()
        moe_layer = self.calculate_moe_layer_flops()
        
        # Restore original s
        self.config.s = original_s
        
        # Attention uses cached context
        attention = self.calculate_attention_flops_decode()
        
        return ln + attention + rope + router + moe_layer
    
    def calculate_decode_flops(self) -> int:
        """
        Decode (FLOPs) = Embedding_{s=1} + l * Decoder w/ KV-Cache_{s=1}
        """
        original_s = self.config.s
        
        # Embedding uses s=1
        self.config.s = 1
        embedding = self.calculate_embedding_flops()
        self.config.s = original_s
        
        # Decoder uses KV-cache version
        decoder = self.calculate_decoder_flops_decode()
        l = self.config.l
        
        return embedding + l * decoder
    
    def calculate_total(self) -> Dict[str, float]:
        """
        Calculate total memory requirements and FLOPs
        Returns memory in bytes, GB, FLOPs, and breakdown
        """
        weights_bytes = self.calculate_model_weights()
        kv_cache_bytes = self.calculate_kv_cache()
        total_bytes = weights_bytes + kv_cache_bytes
        
        prefill_flops = self.calculate_prefill_flops()
        decode_flops = self.calculate_decode_flops()
        
        # Convert to GB
        bytes_to_gb = 1024 ** 3
        
        return {
            "weights_gb": weights_bytes / bytes_to_gb,
            "kv_cache_gb": kv_cache_bytes / bytes_to_gb,
            "total_gb": total_bytes / bytes_to_gb,
            "weights_bytes": weights_bytes,
            "kv_cache_bytes": kv_cache_bytes,
            "total_bytes": total_bytes,
            "prefill_flops": prefill_flops,
            "decode_flops": decode_flops,
            "precision": self.precision
        }


def load_config_from_json(config_path: str = "moe_config.json") -> tuple[MoEConfig, str]:
    """
    Load configuration from JSON file
    
    Args:
        config_path: Path to JSON config file
    
    Returns:
        Tuple of (MoEConfig object, precision string)
    """
    try:
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        
        precision = config_data.get("precision", "bfloat16")
        config = MoEConfig.from_dict(config_data)
        return config, precision
        
    except FileNotFoundError:
        print(f"Config file '{config_path}' not found. Using default configuration.")
        return MoEConfig.from_dict(MoEConfig.get_default_config()), "bfloat16"
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON file: {e}")
        print("Using default configuration.")
        return MoEConfig.from_dict(MoEConfig.get_default_config()), "bfloat16"


def list_available_configs(config_path: str = "moe_config.json"):
    """List all available configurations in the JSON file"""
    try:
        with open(config_path, 'r') as f:
            configs = json.load(f)
        
        if not isinstance(configs, list):
            print("Config file must contain an array of configurations.")
            return
        
        print("\nAvailable Configurations:")
        print("=" * 50)
        for i, config in enumerate(configs):
            config_name = config.get("name", f"Config {i}")
            print(f"  {i}: {config_name}")
        print("=" * 50)
        
    except FileNotFoundError:
        print(f"Config file '{config_path}' not found.")
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON file: {e}")


def calculate_moe_metrics(config: MoEConfig, precision: PrecisionType, 
                         config_name: Optional[str] = None) -> str:
    """
    Main function to calculate memory requirements and FLOPs
    
    Args:
        config: MoEConfig object
        precision: Precision type for weights
        config_name: Optional name of the configuration for display
    
    Returns:
        Formatted string with memory requirements and FLOPs
    """
    calculator = MoEMemoryCalculator(config, precision)
    metrics = calculator.calculate_total()
    
    # Convert FLOPs to TFLOPs
    prefill_tflops = metrics['prefill_flops'] / 1e12
    decode_tflops = metrics['decode_flops'] / 1e12
    
    config_header = f" ({config_name})" if config_name else ""
    
    result = f"""
Memory Requirements for MoE Model{config_header}
{'=' * 50}
Configuration:
  Vocab Size (V): {config.V:,}
  Hidden Size (h): {config.h:,}
  Num Layers (l): {config.l}
  Attention Heads (a): {config.a}
  Num Experts (N): {config.N}
  Expert Multiplier (f_mult): {config.f_mult}
  Sequence Length (s): {config.s:,}
  Top-K Experts: {config.top_k}
  Precision: {precision}

Memory Breakdown:
  Model Weights: {metrics['weights_gb']:.2f} GB
  KV-Cache: {metrics['kv_cache_gb']:.2f} GB
{'=' * 50}
TOTAL MEMORY NEEDED: {metrics['total_gb']:.2f} GB

FLOPs Requirements:
  Prefill FLOPs: {prefill_tflops:.2f} TFLOPs
  Decode FLOPs (per token): {decode_tflops:.6f} TFLOPs
{'=' * 50}
"""
    return result


def main():
    """Example usage with JSON config file"""
    print("MoE Memory & FLOPs Calculator")
    print("=" * 50)
    
    # List available configurations
    list_available_configs()
    
    # Get config index from user
    config_input = input("\nEnter config number (or press Enter for 0): ").strip()
    config_index = 0
    if config_input:
        try:
            config_index = int(config_input)
        except ValueError:
            print("Invalid input. Using config 0.")
            config_index = 0
    
    # Load configuration
    config, precision = load_config_from_json(config_index=config_index)
    
    # Calculate and display results
    config_name = f"Config {config_index}"
    print(calculate_moe_metrics(config, precision, config_name))
    
    # Option to override precision
    print("\nAvailable precisions: float32, bfloat16, float16, int8, int4")
    override = input(f"Override precision (currently {precision}, press Enter to keep): ").strip().lower()
    
    if override and override in PRECISION_BYTES:
        precision = override
        print(calculate_moe_metrics(config, precision, config_name))


if __name__ == "__main__":
    main()
