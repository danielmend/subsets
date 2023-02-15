import numpy as np
import torch
import open_clip
import os

def similarity_filter_CLIP_embeddings(tempdir, clip_model, thresh=0.24, device='cuda:0'):
    embs = sorted([f for f in os.listdir(tempdir) if f.endswith(".npy")])
    keys = [x.split(".npy")[0] for x in embs]

    video_embeddings = []
    txt_embeddings = []
    for emb, key in zip(embs, keys):
        embeddings = np.load(tempdir + '/' + emb)
        with open(tempdir + '/' + key + '.txt', 'rb') as f:
            txt = str(f.read())
        with torch.no_grad():
            toks = open_clip.tokenize(txt).to(device)
            txt_embed = clip_model.encode_text(toks).detach().cpu().numpy().squeeze(0)
        video_embeddings.append(np.mean(embeddings, axis=0))
        txt_embeddings.append(txt_embed)
    video_embeddings = np.array(video_embeddings)
    video_embeddings /= np.linalg.norm(video_embeddings, axis=-1, keepdims=True)

    txt_embeddings = np.array(txt_embeddings)
    txt_embeddings /= np.linalg.norm(txt_embeddings, axis=-1, keepdims=True)
    similarities = np.diagonal(video_embeddings @ txt_embeddings.T)
    keys_to_keep = np.asarray(keys)[similarities > thresh].tolist()
    return keys_to_keep
