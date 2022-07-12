from sklearn.metrics.pairwise import cosine_similarity
import clip
import numpy as np
import torch

def encode_texts(texts_list, model):
    """turn list of strings into an array of clip tokens"""

    X_tokens = clip.tokenize(texts_list, truncate = True)

    text_embs = model.encode_text(X_tokens).float()

    text_embs = text_embs / text_embs.norm(dim=-1, keepdim=True)

    return text_embs.detach().numpy()


def get_modifier_influence(modifier, base_prompts, model):
    """Encode text to clip embeddings
    calculate the cosine similarity matrix
    slice matrix with index to get respective prompts inverted similarity
    """

    full_prompts = [f'{str(base_prompt)}, {modifier}' for base_prompt in base_prompts]

    full_emb = encode_texts(full_prompts, model)
    base_emb = encode_texts(base_prompts, model)

    sim_matrix = cosine_similarity(full_emb, base_emb)

    diffs = [max(0, (1 - sim[i])) for i, sim in enumerate(sim_matrix)]

    return dict(zip(base_prompts, diffs))
