from typing import Sequence
from ..libllaisys import (
    LIB_LLAISYS,
    DeviceType,
    DataType,
    LlaisysQwen2Meta,
    LlaisysQwen2Weights,
    llaisysQwen2Model_t,
    llaisysDeviceType_t,
)
from ..tensor import Tensor
from ctypes import c_int64, c_size_t, c_int, c_char, c_void_p, POINTER, addressof

from pathlib import Path
import safetensors
from safetensors import safe_open
import json
import numpy as np
import torch


class Qwen2:

    def __init__(self, model_path, device: DeviceType = DeviceType.CPU):

        model_path = Path(model_path)

        # Load config
        with open(model_path / "config.json", "r") as f:
            config = json.load(f)

        # Check device availability before allocating anything
        api = LIB_LLAISYS.llaisysGetRuntimeAPI(llaisysDeviceType_t(device.value))
        ndev = api.contents.get_device_count()
        if ndev == 0:
            raise RuntimeError(
                f"No devices available for device type '{device.name}'. "
                "Make sure the library is compiled with the correct backend "
                "and the hardware is accessible."
            )

        self._device = device

        # Create meta
        meta = LlaisysQwen2Meta()
        meta.dtype = DataType.BF16.value
        meta.nlayer = config["num_hidden_layers"]
        meta.hs = config["hidden_size"]
        meta.nh = config["num_attention_heads"]
        meta.nkvh = config["num_key_value_heads"]
        meta.dh = config["hidden_size"] // config["num_attention_heads"]
        meta.di = config["intermediate_size"]
        meta.maxseq = config.get("max_position_embeddings", 4096)
        meta.voc = config["vocab_size"]
        meta.epsilon = config["rms_norm_eps"]
        meta.theta = config.get("rope_theta", 10000.0)
        meta.end_token = config.get("eos_token_id", 151643)
        
        # Create model
        device_id = c_int(0)
        self._model = LIB_LLAISYS.llaisysQwen2ModelCreate(
            POINTER(LlaisysQwen2Meta)(meta),
            device.value,
            POINTER(c_int)(device_id),
            1
        )
        
        # Get weights pointer
        self._weights = LIB_LLAISYS.llaisysQwen2ModelWeights(self._model)
        
        # Load weights from safetensors
        for file in sorted(model_path.glob("*.safetensors")):
            print(f"Loading weights from {file.name}...")
            with safe_open(file, framework="pt", device="cpu") as f:
                for name in f.keys():
                    print(f"  Loading {name}... ", end="", flush=True)
                    tensor_data = f.get_tensor(name)
                    self._load_weight(name, tensor_data)
                    print("OK")
        print("All weights loaded successfully!")
        
        self._meta = meta
    
    def _load_weight(self, name: str, data):
        """Load a single weight tensor"""
        # Convert to numpy array and keep alive during load
        if isinstance(data, torch.Tensor):
            print(f"shape={data.shape}, dtype={data.dtype}")
            if data.dtype == torch.bfloat16:
                # For bfloat16, view as uint16 first, then convert to numpy
                data_np = data.cpu().view(torch.uint16).numpy()
            else:
                # For other types, convert to numpy
                data_np = data.cpu().numpy()
        elif hasattr(data, 'ctypes'):
            # Already numpy array
            data_np = data
        else:
            raise TypeError(f"Unsupported data type: {type(data)}")
        
        # Ensure contiguous memory layout
        if not data_np.flags['C_CONTIGUOUS']:
            data_np = np.ascontiguousarray(data_np)
        
        print(f"numpy shape={data_np.shape}, dtype={data_np.dtype}, contiguous={data_np.flags['C_CONTIGUOUS']}")
        data_ptr = c_void_p(data_np.ctypes.data)
        print(f"data_ptr={data_ptr}")
        
        weights = self._weights.contents
        print(f"weights={weights}")
        
        # Parse weight name
        if name == "model.embed_tokens.weight":
            tensor = Tensor(tensor=weights.in_embed)
            print(f"tensor object created, calling load...")
            tensor.load(data_ptr)
        elif name == "lm_head.weight":
            print(f"Accessing weights.out_embed...")
            tensor = Tensor(tensor=weights.out_embed)
            print(f"tensor object created, calling load...")
            tensor.load(data_ptr)
        elif name == "model.norm.weight":
            tensor = Tensor(tensor=weights.out_norm_w)
            tensor.load(data_ptr)
        elif "model.layers." in name:
            # Parse layer index
            parts = name.split(".")
            layer_idx = int(parts[2])
            
            if "input_layernorm.weight" in name:
                tensor = Tensor(tensor=weights.attn_norm_w[layer_idx])
                tensor.load(data_ptr)
            elif "self_attn.q_proj.weight" in name:
                tensor = Tensor(tensor=weights.attn_q_w[layer_idx])
                tensor.load(data_ptr)
            elif "self_attn.q_proj.bias" in name:
                tensor = Tensor(tensor=weights.attn_q_b[layer_idx])
                tensor.load(data_ptr)
            elif "self_attn.k_proj.weight" in name:
                tensor = Tensor(tensor=weights.attn_k_w[layer_idx])
                tensor.load(data_ptr)
            elif "self_attn.k_proj.bias" in name:
                tensor = Tensor(tensor=weights.attn_k_b[layer_idx])
                tensor.load(data_ptr)
            elif "self_attn.v_proj.weight" in name:
                tensor = Tensor(tensor=weights.attn_v_w[layer_idx])
                tensor.load(data_ptr)
            elif "self_attn.v_proj.bias" in name:
                tensor = Tensor(tensor=weights.attn_v_b[layer_idx])
                tensor.load(data_ptr)
            elif "self_attn.o_proj.weight" in name:
                tensor = Tensor(tensor=weights.attn_o_w[layer_idx])
                tensor.load(data_ptr)
            elif "post_attention_layernorm.weight" in name:
                tensor = Tensor(tensor=weights.mlp_norm_w[layer_idx])
                tensor.load(data_ptr)
            elif "mlp.gate_proj.weight" in name:
                tensor = Tensor(tensor=weights.mlp_gate_w[layer_idx])
                tensor.load(data_ptr)
            elif "mlp.up_proj.weight" in name:
                tensor = Tensor(tensor=weights.mlp_up_w[layer_idx])
                tensor.load(data_ptr)
            elif "mlp.down_proj.weight" in name:
                tensor = Tensor(tensor=weights.mlp_down_w[layer_idx])
                tensor.load(data_ptr)
            else:
                print(f"WARNING: Unmatched weight name: {name}")


    def generate(
        self,
        inputs: Sequence[int],
        max_new_tokens: int = None,
        top_k: int = 1,
        top_p: float = 0.8,
        temperature: float = 0.8,
    ):
        # Explicitly activate the correct device context before any C++ call.
        # This is necessary because the thread-local Context is shared across
        # calls and may have been switched by other code (e.g., HuggingFace).
        LIB_LLAISYS.llaisysSetContextRuntime(llaisysDeviceType_t(self._device.value), c_int(0))

        # Convert inputs to token array
        input_tokens = (c_int64 * len(inputs))(*inputs)

        # First forward pass with prompt
        next_token = LIB_LLAISYS.llaisysQwen2ModelInfer(
            self._model,
            input_tokens,
            len(inputs)
        )

        # Generate tokens
        generated = list(inputs) + [next_token]

        for _ in range(max_new_tokens - 1):
            if next_token == self._meta.end_token:
                break

            # Forward pass with single token
            token_array = (c_int64 * 1)(next_token)
            next_token = LIB_LLAISYS.llaisysQwen2ModelInfer(
                self._model,
                token_array,
                1
            )
            generated.append(next_token)

        return generated
    
    def __del__(self):
        if hasattr(self, "_model") and self._model is not None:
            LIB_LLAISYS.llaisysQwen2ModelDestroy(self._model)
            self._model = None
