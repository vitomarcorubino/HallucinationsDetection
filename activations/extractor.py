import torch
from tqdm import tqdm
from typing import List, Dict, Tuple
from .cache_utils import save_activation_tensor
from .hooks import ActivationHookContext


class ActivationExtractor:
    """
    Handles the extraction of internal model states (activations)
    from a given dataset using PyTorch forward hooks.
    """
    def __init__(self, model, tokenizer, config):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config

    def extract_from_dataset(self, samples: List[Tuple[str, int]], dataset_name: str,
                             label_name: str, layer_map: Dict[str, List[str]]):
        """
        Iterates through samples to extract activations at the last token position.

        Args:
            samples: List of tuples containing (full_text, instance_id).
            dataset_name: Name of the current dataset (e.g., 'nq_swap' or 'covidqa').
            label_name: Group name for storage ('hallucinated' vs 'not_hallucinated').
            layer_map: Dictionary mapping activation types to specific model module paths.
        """
        # Set model to evaluation mode to disable dropout
        self.model.eval()

        print(f"🚀 Starting extraction for {label_name} group...")

        for text, instance_id in tqdm(samples):
            # TOKENIZATION: Convert text to input tensors
            # Truncation and max_length are prevent Out-Of-Memory errors on systems with limited RAM (e.g., 8GB)
            inputs = self.tokenizer(
                text,
                return_tensors = "pt",
                truncation = True,
                max_length = self.config.MAX_SEQ_LENGTH
            ).to(self.config.DEVICE)

            # HOOK MANAGEMENT: Use the context manager to safely attach/detach hooks.
            # This ensures that even if an error occurs, hooks are removed from the model.
            with ActivationHookContext(self.model, layer_map) as hook:
                # FORWARD PASS: Run the model without calculating gradients
                # torch.no_grad()  reduces memory consumption during inference
                with torch.no_grad():
                    self.model(**inputs)

                # DATA RETRIEVAL: Access the activations from the hook context.
                # The hook already isolated the last token and moved it to the CPU.
                for act_type, layers in hook.captured_activations.items():
                    for layer_idx, tensor in layers.items():
                        # STORAGE: Save individual tensors to disk (.pt files)
                        # .squeeze(0) removes the batch dimension [1, hidden_dim] -> [hidden_dim]
                        # as we process samples one by one.
                        save_activation_tensor(
                            tensor.squeeze(0),
                            layer_idx,
                            instance_id,
                            act_type,
                            dataset_name,
                            label_name,
                            self.config
                        )

            # MEMORY CLEANUP: Clear intermediate tensors to maintain system stability
            if torch.cuda.is_available():
                torch.cuda.empty_cache()