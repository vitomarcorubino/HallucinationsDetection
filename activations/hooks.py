import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Union
from functools import partial


class ActivationHookContext:
    """
    Context manager to capture intermediate activations from specific model layers.
    Optimized for systems with limited RAM (e.g., 8GB) and compatible with
    structured layer maps (Hidden, MLP, Attention).
    """

    def __init__(self, model: nn.Module, layer_map: Dict[str, List[str]],
                 move_to_cpu: bool = True, last_token_only: bool = True):
        self.model = model
        self.layer_map = layer_map
        self.move_to_cpu = move_to_cpu
        self.last_token_only = last_token_only
        self.handles = []

        # Nested structure to store tensors: { 'activation_type': { layer_index: tensor } }
        # Example: { 'hidden': { 0: tensor, 1: tensor }, 'mlp': { 0: tensor ... } }
        self.captured_activations: Dict[str, Dict[int, torch.Tensor]] = {
            act_type: {} for act_type in layer_map.keys()
        }

    def __enter__(self):
        """
        Registers forward hooks on the model modules specified in the layer_map.
        Triggered when entering the 'with' block.
        """
        for act_type, module_paths in self.layer_map.items():
            for layer_idx, path in enumerate(module_paths):
                try:
                    # USE GET_SUBMODULE: This is faster than iterating through all named_modules()
                    # because it directly accesses the target component via its string path.
                    module = self.model.get_submodule(path)

                    # PARTIAL APPLICATION: We use partial to "bake" the act_type and
                    # layer_idx into the hook function, allowing the same generic
                    # _hook_fn to handle multiple different layers uniquely.
                    hook_fn = partial(self._hook_fn, act_type = act_type, layer_idx = layer_idx)
                    handle = module.register_forward_hook(hook_fn)
                    self.handles.append(handle)
                except AttributeError:
                    print(f"⚠️ Warning: Layer path '{path}' not found in the model.")
        return self

    def _hook_fn(self, module, inputs, outputs, act_type, layer_idx):
        """
        The actual hook executed during the model's forward pass.
        Captures the output tensor of a specific layer.
        """
        # TUPLE HANDLING: Many Transformer layers (especially Attention blocks) return a tuple
        # (e.g., hidden_states, attention_weights).
        # We only want the primary tensor (hidden_states) located at index 0.
        tensor = outputs[0] if isinstance(outputs, tuple) else outputs

        # LAST TOKEN EXTRACTION for generation and probing tasks.
        # Most models output a 3D tensor: [batch, sequence_length, hidden_dimension].
        # We slice it to keep only the final token index (-1) to represent
        # the model's "decision point" or summary state
        if self.last_token_only and len(tensor.shape) == 3:
            # Resulting shape: [batch, hidden_dimension]
            tensor = tensor[:, -1, :]

        # MEMORY OPTIMIZATION: Move the tensor to CPU immediately.
        # This keeps the GPU (VRAM) or primary system RAM available for the
        # heavy LLM weights rather than storing intermediate activation data.
        if self.move_to_cpu:
            tensor = tensor.detach().cpu()
        else:
            tensor = tensor.detach()

        # Store the processed tensor in our dictionary
        self.captured_activations[act_type][layer_idx] = tensor

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Ensures all hooks are removed from the model when exiting the 'with' block.
        This prevents memory leaks and ensures subsequent runs start with a clean model.
        """
        for handle in self.handles:
            handle.remove()
        self.handles.clear()