import numpy as np
import torch
import open_clip
import os

def similarity_filter(sample, clip_model, thresh=0.24, device='cpu'):
    video_embeddings = np.load(sample['.npy'])
    video_embeddings = np.mean(video_embeddings, axis=0)
    video_embeddings /= np.linalg.norm(video_embeddings, axis=-1, keepdims=True)
    with open(sample['.txt'], 'r') as f:
        txt = str(f.read())

    with torch.no_grad():
         toks = open_clip.tokenize(txt).to(device)
         txt_embeddings = clip_model.encode_text(toks).detach().cpu().numpy().squeeze(0)
    txt_embeddings /= np.linalg.norm(txt_embeddings, axis=-1, keepdims=True)

    return video_embeddings @ txt_embeddings.T > thresh
