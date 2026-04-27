from typing import Dict, List

def get_layer_map(model_id: str, layers: List[int]) -> Dict[str, List[str]]:
    """
    Returns the correct layer paths based on the model family.
    """
    model_id_lower = model_id.lower()

    # --- GEMMA Family ---
    if "gemma" in model_id_lower:
        return {
            "hidden": [f"model.language_model.layers.{i}" for i in layers],
            "mlp": [f"model.language_model.layers.{i}.mlp" for i in layers],
            "attn": [f"model.language_model.layers.{i}.self_attn.o_proj" for i in layers]
        }

    # --- LLAMA Family ---
    elif "llama" in model_id_lower:
        return {
            "hidden": [f"model.layers.{i}" for i in layers],
            "mlp": [f"model.layers.{i}.mlp" for i in layers],
            "attn": [f"model.layers.{i}.self_attn.o_proj" for i in layers]
        }

    # --- MISTRAL Family ---
    elif "mistral" in model_id_lower:
        return {
            "hidden": [f"model.layers.{i}" for i in layers],
            "mlp": [f"model.layers.{i}.mlp" for i in layers],
            "attn": [f"model.layers.{i}.self_attn.o_proj" for i in layers]
        }

    else:
        raise ValueError(f"Model architecture for '{model_id}' not supported yet. "
                         f"Please add it to activations/layer_configs.py")