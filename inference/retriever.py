# inference/retriever.py
import torch
from sentence_transformers import SentenceTransformer
import config


class GemmaRetriever:
    def __init__(self, model_id: str = "google/embeddinggemma-300m"):
        print(f"📡 Initializing Retriever: {model_id}...")
        # Use the config specified device (in my case a CPU)
        self.model = SentenceTransformer(model_id, device = str(config.DEVICE))


    def get_top_k(self, query: str, documents, k: int = 2):
        """
        Finds the top-k documents for the given query.
        Handles list, numpy arrays, or pandas series for 'documents'.
        """
        # Fix the ambiguous truth value check
        if documents is None or len(documents) == 0:
            return ""

        # Standardize documents into a list of strings if they aren't already
        # (HuggingFace datasets often return numpy arrays of objects)
        doc_list = list(documents) if not isinstance(documents, list) else documents

        # Encoding
        query_emb = self.model.encode(query, convert_to_tensor = True)
        doc_embs = self.model.encode(doc_list, convert_to_tensor = True)

        # Similarity
        similarities = self.model.similarity(query_emb, doc_embs)[0]

        # Selection
        top_k = min(k, len(doc_list))
        indices = torch.topk(similarities, k = top_k).indices

        selected_docs = [doc_list[i] for i in indices]
        return "\n\n".join(selected_docs)