from typing import List, Union

from fastapi import FastAPI
from pydantic import BaseModel
import clip
import numpy as np
import torch
import pandas as pd

from artist_studies.influence import get_modifier_influence

app = FastAPI()

device = "cuda" if torch.cuda.is_available() else "cpu"

model, _preprocess = clip.load("ViT-B/32", device=device, jit=False)

class ModifierRequest(BaseModel):
    modifier_text : str
    base_prompts: List[str] = []

@app.get("/")
async def index():
    return {'Hello': 'World'}

@app.get("/modifier_influence/")
async def modifier_influence(modifier_text):

    prompts = pd.read_csv('raw_data/base_prompts.txt', delimiter = "\n", header = None)

    base_prompts = [f'{str(base_prompt[0])}' for base_prompt in prompts.values]

    influence_dict = get_modifier_influence(modifier_text, base_prompts, model)

    return float(np.mean(list(influence_dict.values())))

@app.post("/modifier_influence_custom/")
async def modifier_influence_custom(modifier: ModifierRequest):

    return get_modifier_influence(modifier.modifier_text, modifier.base_prompts, model)
