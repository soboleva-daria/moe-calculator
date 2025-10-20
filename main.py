import json
from typing import Dict, Literal, Optional
from dataclasses import dataclass
try:
    from transformers import AutoConfig as _HF_AutoConfig  # type: ignore
    _HAVE_HF = True
except Exception:
    _HF_AutoConfig = None
    _HAVE_HF = False

# Precision to bytes mapping
PRECISION_BYTES = {
    "float32": 4,
    "bfloat16": 2,
    "float16": 2,
    "int8": 1,
    "int4": 0.5
}

PrecisionType = Literal["float32", "bfloat16", "float16", "int8", "int4"]


# --- NEW (tiny): GPU presets for roofline (defaults pick H100-SXM => B*≈1989/3.35≈590) ---
_GPU_PRESETS = {
    "H100-SXM":      {"peak_tflops_bf16": 1979.0, "hbm_gbps": 3350.0},
    "A100-80GB":     {"peak_tflops_bf16": 312.0,  "hbm_gbps": 2039.0},
    "Generic-800TF": {"peak_tflops_bf16": 800.0,  "hbm_gbps": 350.0},
}
_DEFAULT_GPU = "H100-SXM"
_DEFAULT_ETA_GEMM = 0.40  # usable GEMM fraction for matmul (kept simple)


@dataclass
class MoEConfig:
    """Configuration for MoE model parameters"""
    V: int              # vocab_size
    h: int              # hidden_size
    l: int              # num_layers
    a: int              # num_attention_heads
    N: int              # num_experts
    f_mult: float       # expert multiplier
    s: int              # sequence_length - for kv-cache calculation
    top_k: int          # number of experts activated per token
    hf_model_name: Optional[str] = None   # if provided, override fields from HF config
    a_kv: Optional[int] = None            # optional KV heads (for GQA)
    gpu_name: Optional[str] = None
    gpu_peak_tflops_bf16: Optional[float] = None   # e.g. 1979.0
    gpu_hbm_gbps: Optional[float] = None           # e.g. 3350.0
    
    @classmethod
    def from_dict(cls, config: Dict) -> 'MoEConfig':
        allowed = ['V','h','l','a','N','f_mult','s','top_k','hf_model_name','a_kv','gpu_name','gpu_peak_tflops_bf16','gpu_hbm_gbps']
        base = cls.get_default_config()
        base.update({k: v for k, v in config.items() if k in allowed})
        return cls(**base)


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
            "top_k": 2,
            "hf_model_name": None,
            "a_kv": None,
            "gpu_name": "H100-SXM",
            "gpu_peak_tflops_bf16": None,
            "gpu_hbm_gbps": None
        }

    def resolve_hf_overrides(self) -> 'MoEConfig':
        """
        If hf_model_name is set and transformers is available, override
        config fields from the HF AutoConfig. Otherwise return self unchanged.
        """
        if not self.hf_model_name:
            return self
        if not _HAVE_HF:
            print("[HF override] transformers not available; using provided/default config.")
            return self

        try:
            cfg = _HF_AutoConfig.from_pretrained(self.hf_model_name, trust_remote_code=True)
        except Exception as e:
            print(f"[HF override] failed to load '{self.hf_model_name}': {e}\nUsing provided/default config.")
            return self

        # Keep current values as fallbacks; only override when discoverable
        V = int(getattr(cfg, "vocab_size", self.V))
        h = int(getattr(cfg, "hidden_size", getattr(cfg, "n_embed", self.h)))
        l = int(getattr(cfg, "num_hidden_layers", getattr(cfg, "num_layers", self.l)))
        a = int(getattr(cfg, "num_attention_heads", self.a))

        # MoE detection: number of experts (if present and >1); else dense (N=1)
        N_found = None
        for nm in [
            "num_experts", "n_experts", "moe_num_experts", "router_num_experts",
            "experts", "num_moe_experts"
        ]:
            if hasattr(cfg, nm):
                try:
                    cand = int(getattr(cfg, nm))
                    if cand > 0:
                        N_found = cand
                        break
                except Exception:
                    pass
        N = N_found if N_found is not None else 1  # assume dense if not present

        # top_k (many names)
        top_k = self.top_k
        for nm in ["num_experts_per_tok", "top_k", "moe_topk", "routing_top_k", "router_top_k"]:

            if hasattr(cfg, nm):
                try:
                    cand = int(getattr(cfg, nm))
                    if 1 <= cand <= 64:  # sanity bound
                        top_k = cand
                        break
                except Exception:
                    pass
        if N <= 1:
            top_k = 1

        # FFN multiplier via intermediate size
        try:
            d_ff = int(getattr(cfg, "intermediate_size",
                            getattr(cfg, "ffn_hidden_size",
                                    getattr(cfg, "mlp_hidden_dim", int(self.f_mult * h)))))
            f_mult = float(d_ff) / float(h)
        except Exception:
            f_mult = self.f_mult

        # GQA detection via KV heads
        a_kv = self.a_kv
        for nm in ["num_key_value_heads", "kv_num_heads", "num_kv_heads"]:
            if hasattr(cfg, nm):
                try:
                    cand = int(getattr(cfg, nm))
                    if cand > 0:
                        a_kv = cand
                        break
                except Exception:
                    pass

        # If anything actually changed, print a one-liner so users see it
        changed = (V != self.V or h != self.h or l != self.l or a != self.a or
                N != self.N or top_k != self.top_k or f_mult != self.f_mult or
                a_kv != self.a_kv)
        if changed:
            print(f"[HF override] loaded {self.hf_model_name} → "
                f"h={h}, L={l}, a={a}, N={N}, top_k={top_k}, f_mult≈{f_mult:.3f}, a_kv={a_kv or 'n/a'}")
        else:
            print(f"[HF override] loaded {self.hf_model_name} but no recognized fields changed.")

        return MoEConfig(
            V=V, h=h, l=l, a=a, N=N, f_mult=f_mult, s=self.s, top_k=top_k,
            hf_model_name=self.hf_model_name, a_kv=a_kv,
            gpu_name=self.gpu_name,
            gpu_peak_tflops_bf16=self.gpu_peak_tflops_bf16,
            gpu_hbm_gbps=self.gpu_hbm_gbps,
        )


class MoEMemoryCalculator:
    """Calculate memory requirements and FLOPs for MoE/Dense (GQA-aware)"""

    def __init__(self, config: MoEConfig, precision: PrecisionType):
        self.config = config.resolve_hf_overrides()
        self.precision = precision
        self.bytes_per_param = PRECISION_BYTES[precision]
        self._gqa_ratio = (self.config.a_kv / self.config.a) if (self.config.a_kv and self.config.a) else 1.0

    # ============ MEMORY CALCULATIONS ============
    
    def calculate_embedding_weights(self) -> float:
        """
        Embedding Weights (B) = 2 * k * V * h
        Factor of 2 accounts for input and output embedding matrices
        """
        k = self.bytes_per_param
        V = self.config.V
        h = self.config.h
        return 2 * k * V * h
    
    def calculate_ln_weights(self) -> float:
        """
        LN Weights (B) = 4 * k * h
        """
        k = self.bytes_per_param
        h = self.config.h
        return 4 * k * h
    
    def calculate_attention_weights(self) -> float:
        """
        Attention Weights (B) with GQA:
        Q/O: 2 * k * h^2; K/V: 2 * r * k * h^2  => total = (2 + 2r) * k * h^2
        (Falls back to r=1 for non-GQA models)
        """
        k = self.bytes_per_param
        h = self.config.h
        r = self._gqa_ratio
        return (2 + 2 * r) * k * (h ** 2)

    def calculate_router_weights(self) -> float:
        """
        Router Weights (B) = k * N * h
        Weight matrix of size N x h with learnable router weights
        """
        k = self.bytes_per_param
        N = self.config.N
        h = self.config.h
        return k * N * h
    
    def calculate_moe_layer_weights(self) -> float:
        """
        MoE Layer Weights (B) = 3 * k * N * f_mult * h^2
        Each expert uses SwiGLU with three linear transformations
        """
        k = self.bytes_per_param
        N = self.config.N
        h = self.config.h
        f_mult = self.config.f_mult
        return 3 * k * N * f_mult * h ** 2
    
    def calculate_decoder_weights(self) -> float:
        """
        Decoder Weights (B) = LN + Attention + Router + MoE Layer
        Combines all components with layer norms already included
        """
        ln = self.calculate_ln_weights()
        attention = self.calculate_attention_weights()
        router = self.calculate_router_weights()
        moe_layer = self.calculate_moe_layer_weights()
        
        return ln + attention + router + moe_layer
    
    def calculate_model_weights(self) -> float:
        """
        Model Weights (B) = Embedding + l * Decoder
        Total weights across all layers
        """
        embedding = self.calculate_embedding_weights()
        decoder = self.calculate_decoder_weights()
        l = self.config.l
        
        return embedding + l * decoder
    
    def calculate_kv_cache(self) -> float:
        """
        KV-Cache (B) with GQA:
        = 2 * k * l * s * (r * h)
        Cache for keys (k) and values (v) across layers l,
        sequence length s; K/V width is r*h with GQA (defaults to h if r=1)
        """
        k = self.bytes_per_param
        l = self.config.l
        s = self.config.s
        h = self.config.h
        r = self._gqa_ratio
        return 2 * k * l * s * r * h
    
    # ============ FLOPS CALCULATIONS ============
    
    def calculate_ln_flops(self) -> float:
        """
        LN Compute (FLOPs) = 14 * s * h
        """
        s = self.config.s
        h = self.config.h
        return 14 * s * h
    
    def calculate_attention_flops(self) -> float:
        """
        Attention Compute (FLOPs) = s * (8 * h^2 + 4 * s * h + 3 * s * a)
        """
        s = self.config.s
        h = self.config.h
        a = self.config.a
        r = self._gqa_ratio
        return s * (4 * (1 + r) * h**2 + 4 * s * h + 3 * s * a)
    
    def calculate_rope_flops(self) -> float:
        """
        RoPE Compute (FLOPs) = 0.75 * s * h
        """
        s = self.config.s
        h = self.config.h
        return 0.75 * s * h
    
    def calculate_router_flops(self) -> float:
        """
        Router Compute (FLOPs) = s * N * (2 * h + 3)
        """
        s = self.config.s
        N = self.config.N
        h = self.config.h
        return s * N * (2 * h + 3)
    
    def calculate_moe_layer_flops(self) -> float:
        """
        MoE Layer Compute (FLOPs) = 6 * top_k * s * f_mult * h * (h + 1)
        """
        top_k = self.config.top_k
        s = self.config.s
        f_mult = self.config.f_mult
        h = self.config.h
        return 6 * top_k * s * f_mult * h * (h + 1)
    
    def calculate_unembedding_flops(self) -> float:
        """
        Unembedding Compute (FLOPs) = 2 * s * V * h
        """
        s = self.config.s
        V = self.config.V
        h = self.config.h
        return 2 * s * V * h
    
    def calculate_decoder_flops(self) -> float:
        """
        Decoder Compute (FLOPs) = LN + Attention + RoPE + Router + MoE Layer
        """
        ln = self.calculate_ln_flops()
        attention = self.calculate_attention_flops()
        rope = self.calculate_rope_flops()
        router = self.calculate_router_flops()
        moe_layer = self.calculate_moe_layer_flops()
        
        return ln + attention + rope + router + moe_layer
    
    def calculate_prefill_flops(self) -> float:
        """
        Prefill (FLOPs) = l * Decoder + Unembedding
        """
        decoder = self.calculate_decoder_flops()
        unembedding = self.calculate_unembedding_flops()
        l = self.config.l
        
        return l * decoder + unembedding
    
    # Decode FLOPs calculations (with s=1)
    
    def calculate_attention_flops_decode(self) -> float:
        """
        Attention Compute w/ KV-Cache (FLOPs) = 4 * (1 + r) * h^2 + 4 * s * h + 3 * s * a
        (GQA-aware; falls back to r=1)
        Note: s here is the context length (cached tokens)
        """
        s = self.config.s  # context length for decode
        h = self.config.h
        a = self.config.a
        r = self._gqa_ratio
        return 4 * (1 + r) * h** 2 + 4 * s * h + 3 * s * a

    def calculate_decoder_flops_decode(self) -> float:
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
    
    def calculate_decode_flops(self) -> float:
        """
        Decode (FLOPs) = l * Decoder w/ KV-Cache_{s=1} + Unembedding_{s=1}
        """
        # Decoder uses KV-cache version
        decoder = self.calculate_decoder_flops_decode()

        original_s = self.config.s
        self.config.s = 1
        unembedding = self.calculate_unembedding_flops()
        self.config.s = original_s
        l = self.config.l
        
        return l * decoder + unembedding
    
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

# --- NEW (tiny): matmul-only roofline helpers (no signature changes elsewhere) ---

def _auto_gqa_ratio(cfg: MoEConfig) -> float:
    return (cfg.a_kv / cfg.a) if (cfg.a_kv and cfg.a) else 1.0

def _gpu_params_from_config(cfg: MoEConfig) -> Dict[str, object]:
    name = (cfg.gpu_name or _DEFAULT_GPU)
    gpu = _GPU_PRESETS.get(name, _GPU_PRESETS[_DEFAULT_GPU]).copy()
    if cfg.gpu_peak_tflops_bf16 is not None:
        gpu["peak_tflops_bf16"] = float(cfg.gpu_peak_tflops_bf16)
    if cfg.gpu_hbm_gbps is not None:
        gpu["hbm_gbps"] = float(cfg.gpu_hbm_gbps)
    gpu["name"] = name
    return gpu


def _roofline_optimal_batches(cfg: MoEConfig, precision: PrecisionType,
                              gpu_name: str = _DEFAULT_GPU,
                              eta_gemm: float = _DEFAULT_ETA_GEMM,
                              ffn_gate_factor: float = 2.0) -> Dict[str, float]:
    """
    Compute hardware B* (dense) or (B_sat, B*) for MoE using matmul-only bytes/flops.
    """
    
    gpu = _gpu_params_from_config(cfg)
    F_peak = gpu["peak_tflops_bf16"] * 1e12
    BW = gpu["hbm_gbps"] * 1e9
    F_use = eta_gemm * F_peak


    beta = PRECISION_BYTES[precision]
    r = _auto_gqa_ratio(cfg)

    # per-layer matmul 'weights' in elements
    A = (2.0 + 2.0 * r) * cfg.h * cfg.h                   # Q,K,V,O projections (GQA-aware)
    d_ff = cfg.f_mult * cfg.h
    U = (1.0 + ffn_gate_factor) * cfg.h * d_ff            # SwiGLU => 3*h*d_ff (gate=2)

    F_use = eta_gemm * F_peak

    if cfg.N and cfg.N > 1:
        # MoE
        top_k = int(cfg.top_k if cfg.top_k else 1)
        N = int(cfg.N)
        B_sat = N / float(top_k)
        # saturated crossover
        B_star = (beta / 2.0) * (F_use / BW) * (A + N * U) / (A + top_k * U)
        return {"kind": "moe", "B_sat": B_sat, "B_star": B_star}
    else:
        # Dense
        B_star = (beta / 2.0) * (F_use / BW)
        return {"kind": "dense", "B_star": B_star}

def _plot_roofline(cfg: MoEConfig, precision: PrecisionType,
                   gpu_name: str = _DEFAULT_GPU,
                   eta_gemm: float = _DEFAULT_ETA_GEMM,
                   ffn_gate_factor: float = 2.0) -> None:
    try:
        import math
        import matplotlib.pyplot as plt
    except Exception:
        print("[roofline] matplotlib not available; skipping plot.")
        return

    gpu = _gpu_params_from_config(cfg)
    F_peak = gpu["peak_tflops_bf16"] * 1e12
    BW = gpu["hbm_gbps"] * 1e9
    F_use = eta_gemm * F_peak

    beta = PRECISION_BYTES[precision]
    r = _auto_gqa_ratio(cfg)

    # Matmul elements per layer
    A = (2.0 + 2.0 * r) * cfg.h * cfg.h
    d_ff = cfg.f_mult * cfg.h
    U = (1.0 + ffn_gate_factor) * cfg.h * d_ff

    F_use = eta_gemm * F_peak  # keep consistent with above
    is_moe = (cfg.N and cfg.N > 1)
    top_k = cfg.top_k if cfg.top_k else 1
    N = cfg.N if cfg.N else 1

    def E_active(B: int) -> float:
        return min(N, top_k * B) if is_moe else 0.0

    def flops_per_step(B: int) -> float:
        if is_moe:
            return 2.0 * B * cfg.l * (A + top_k * U)
        else:
            return 2.0 * B * cfg.l * (A + U)

    def bytes_per_step(B: int) -> float:
        if is_moe:
            return beta * cfg.l * (A + U * E_active(B))
        else:
            return beta * cfg.l * (A + U)

    def t_step(B: int) -> float:
        t_compute = flops_per_step(B) / F_use
        t_mem = bytes_per_step(B) / BW
        return max(t_compute, t_mem)

    # Sweep
    Bmin, Bmax, n = 1, 8192, 80
    xs = sorted(set(max(Bmin, int(round(10 ** (math.log10(Bmin) + i * (math.log10(Bmax) - math.log10(Bmin)) / (n - 1))))) for i in range(n)))
    lats_ms = [1e3 * t_step(B) for B in xs]
    thrus = [B / t_step(B) for B in xs]
    lat_min, lat_max = min(lats_ms), max(lats_ms)
    thr_min, thr_max = min(thrus), max(thrus)

    # Compute regimes
    opt = _roofline_optimal_batches(cfg, precision, gpu_name, eta_gemm, ffn_gate_factor)
    fig, ax1 = plt.subplots(figsize=(10.5, 6.0))
    ax1.set_xscale("log")
    ax1.set_xlabel("Batch size (B)")
    ax1.set_ylabel("Latency per step [ms]")
    ax1.plot(xs, lats_ms, color="tab:blue", linewidth=2, label="Latency")
    ax2 = ax1.twinx()
    ax2.set_ylabel("Throughput [tokens/s]")
    ax2.plot(xs, thrus, color="tab:red", linewidth=2, label="Throughput")

    # Align axes so traces coincide at B=1, keep throughput top at 2×max
    lat1, thr1 = lats_ms[0], thrus[0]
    ax1.set_ylim(lat_min, lat_max)
    T_max_target = 2.0 * thr_max
    # match relative position of B=1 across axes
    pos = (lat1 - lat_min) / max(lat_max - lat_min, 1e-12)
    T_min_align = thr1 - pos * (T_max_target - thr_min)
    ax2.set_ylim(T_min_align, T_max_target)

    # Shade regimes
    if opt["kind"] == "moe":
        B_sat = opt["B_sat"]
        B_star = opt["B_star"]
        ax1.axvspan(Bmin, B_sat, facecolor="#eef3ff", alpha=0.7, zorder=0, label="B ≤ B_sat")
        ax1.axvspan(B_sat, B_star, facecolor="#fff7e6", alpha=0.6, zorder=0, label="B_sat < B ≤ B*")
        ax1.axvspan(B_star, Bmax, facecolor="#e8f7ef", alpha=0.6, zorder=0, label="B > B*")
        ax1.axvline(B_sat, color="gray", linestyle=":", linewidth=1.5)
        ax1.axvline(B_star, color="black", linestyle="--", linewidth=1.8)
        ax1.text(B_sat, lat_max * 0.92, f"B_sat ≈ {B_sat:.0f}", rotation=90, va="top", ha="left", color="gray")
        ax1.text(B_star, lat_max * 0.92, f"B* ≈ {B_star:.0f}", rotation=90, va="top", ha="left", color="black")
        title = f"MoE roofline (matmul-only) | h={cfg.h}, d_ff≈{int(d_ff)}, L={cfg.l}, top_k={top_k}, N={N}"
    else:
        B_star = opt["B_star"]
        ax1.axvspan(Bmin, B_star, facecolor="#fff7e6", alpha=0.6, zorder=0, label="B ≤ B*")
        ax1.axvspan(B_star, Bmax, facecolor="#e8f7ef", alpha=0.6, zorder=0, label="B > B*")
        ax1.axvline(B_star, color="black", linestyle="--", linewidth=1.8)
        ax1.text(B_star, lat_max * 0.92, f"B* ≈ {B_star:.0f}", rotation=90, va="top", ha="left", color="black")
        title = f"Dense roofline (matmul-only) | h={cfg.h}, d_ff≈{int(d_ff)}, L={cfg.l}"

    ax1.set_title(
        f"{title} | {precision}, η={eta_gemm:.2f}, "
        f"peak={gpu['peak_tflops_bf16']:.0f} TF/s, HBM={gpu['hbm_gbps']:.0f} GB/s, GPU={gpu['name']}"
    )
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")
    fig.tight_layout()
    plt.show()


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
    config = calculator.config  # (may include HF overrides)
    config_header = f" ({config_name})" if config_name else ""

    # --- NEW (tiny): optimal batch information (matmul-only, hardware) ---
    opt = _roofline_optimal_batches(config, precision)
    if opt["kind"] == "moe":
        opt_line = f"  Optimal B (MoE): B_sat ≈ {opt['B_sat']:.1f},  B* ≈ {opt['B_star']:.1f}"
    else:
        opt_line = f"  Optimal B (Dense): B* ≈ {opt['B_star']:.1f}"
    gpu = _gpu_params_from_config(config)
    gpu_line = f"  GPU: {gpu['name']} | peak={gpu['peak_tflops_bf16']:.0f} TF/s, HBM={gpu['hbm_gbps']:.0f} GB/s"

    # Safe GQA print
    r_str = f"{(config.a_kv/config.a):.3f}" if (config.a_kv and config.a) else "n/a"
    
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
  HF Model: {config.hf_model_name or 'None'}
  KV Heads (a_kv): {config.a_kv or 'None'}  (GQA r = {r_str})

Memory Breakdown:
  Model Weights: {metrics['weights_gb']:.2f} GB
  KV-Cache: {metrics['kv_cache_gb']:.2f} GB
{'=' * 50}
TOTAL MEMORY NEEDED: {metrics['total_gb']:.2f} GB

FLOPs Requirements:
  Prefill FLOPs: {prefill_tflops:.2f} TFLOPs
  Decode FLOPs (per token): {decode_tflops:.6f} TFLOPs

Optimal Batch (roofline, matmul-only):
{opt_line}
{gpu_line}
{'=' * 50}
"""
    return result


def main():
    """Example usage with JSON config file"""
    print("MoE Memory & FLOPs Calculator")
    print("=" * 50)
    
    # Load configuration
    config, precision = load_config_from_json()
    config.hf_model_name = "Qwen/Qwen3-30B-A3B"
    config = config.resolve_hf_overrides()
    # Calculate and display results (only once, with final precision)
    print(calculate_moe_metrics(config, precision, "moe_config.json"))

    _plot_roofline(config, precision)


if __name__ == "__main__":
    main()
