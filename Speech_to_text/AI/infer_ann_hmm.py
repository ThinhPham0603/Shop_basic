import os
import base64
import json
import pickle
import librosa
import numpy as np
import torch
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import hashes

# ---------- Load models -----------------
MODEL_DIR = "models"
id2sent = np.load(os.path.join(MODEL_DIR, "id2sentence.npy"), allow_pickle=True).item()
pca, hmm_model = pickle.load(open(os.path.join(MODEL_DIR, "pca_hmm.pkl"), "rb"))

# ANN phải khớp với lúc train
from train_step2 import ANN, fft_feat, DEVICE, EMB_DIM

# --- SET INPUT_DIM và OUTPUT_DIM đúng ---
INPUT_DIM = 257        # Theo config model cũ
OUTPUT_DIM = 2204      # Số lượng câu khi train model ban đầu

# Khởi tạo ANN đúng output_dim
ann = ANN(INPUT_DIM, OUTPUT_DIM).to(DEVICE)
ann.load_state_dict(torch.load(os.path.join(MODEL_DIR, "ann_fft.pt"), map_location=DEVICE))
ann.eval()

# ---------- RSA tiện ích ---------------
def rsa_wrap(txt: str):
    prv = rsa.generate_private_key(65537, 2048)
    pub = prv.public_key()
    ct = pub.encrypt(txt.encode(),
            padding.OAEP(mgf=padding.MGF1(hashes.SHA256()),
                         algorithm=hashes.SHA256(), label=None))
    pt = prv.decrypt(ct, padding.OAEP(
            mgf=padding.MGF1(hashes.SHA256()),
            algorithm=hashes.SHA256(), label=None))
    return base64.b64encode(ct).decode(), pt.decode()

# ---------- Hàm suy diễn ----------------
def speech2text_rsa(wav_path: str):
    fft = torch.tensor(fft_feat(wav_path))[None].to(DEVICE)   # (1, seq_len, D)
    with torch.no_grad():
        emb = ann(fft)[1].cpu().numpy()                       # (1, EMB_DIM)
    emb_pca = pca.transform(emb)                              # (1, 50)
    idx = hmm_model.predict(emb_pca)[0]                       # dự đoán state id
    text = str(id2sent.get(idx, "Unknown"))                   # lấy câu từ id
    enc, dec = rsa_wrap(text)
    return {"plain": text, "encrypted_base64": enc, "decrypted": dec}

# ---------- Test local ------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav", type=str, required=True, help="Path to WAV file")
    args = parser.parse_args()

    result = speech2text_rsa(args.wav)
    print(json.dumps(result, ensure_ascii=False, indent=2))
