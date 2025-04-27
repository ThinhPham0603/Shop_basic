import os, json, pickle, librosa, numpy as np, pandas as pd, torch, seaborn as sns, matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, top_k_accuracy_score, confusion_matrix
from hmmlearn import hmm

# ---------- Hyper ------------------------
CSV, WAV_D = "data/processed_vi.csv", "data/wav"
SR, N_FFT, HOP, MAX_F = 16000, 512, 256, 200
EMB_DIM, BATCH, EPOCH = 64, 32, 10        # 10 epoch là đủ khi train=val
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs("models", exist_ok=True)

# ---------- helper -----------------------
def fft_feat(path: str):
    y, _ = librosa.load(path, sr=SR)
    S, _ = librosa.magphase(librosa.stft(y, n_fft=N_FFT, hop_length=HOP))
    feat = librosa.amplitude_to_db(S, ref=np.max).T
    if feat.shape[0] < MAX_F:
        feat = np.vstack([feat, np.zeros((MAX_F - feat.shape[0], feat.shape[1]))])
    else:
        feat = feat[:MAX_F]
    return feat.astype(np.float32)

# ---------- Dataset ----------------------
class FFTset(Dataset):
    def __init__(self, rows):
        self.rows = rows[["wav", "label"]].values
    def __len__(self): return len(self.rows)
    def __getitem__(self, idx):
        wav, lab = self.rows[idx]
        return torch.tensor(fft_feat(os.path.join(WAV_D, wav))), torch.tensor(lab)

def collate(b):
    xs, ys = zip(*b)
    return torch.stack(xs), torch.tensor(ys)

# ---------- ANN --------------------------
class ANN(nn.Module):
    def __init__(self, in_dim, n_cls):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, 128, batch_first=True)
        self.emb = nn.Linear(128, EMB_DIM)
        self.cls = nn.Linear(EMB_DIM, n_cls)
    def forward(self, x):
        h, _ = self.lstm(x)
        z = torch.relu(self.emb(h[:, -1]))
        return self.cls(z), z

# -------------- main ---------------------
def main():
    # 1. chuẩn bị dữ liệu
    df = pd.read_csv(CSV).sample(frac=1, random_state=42)
    df["label"] = np.arange(len(df))                 # 1 nhãn / câu
    np.save("models/id2sentence.npy", df["sentence"].values)

    fft_dim = fft_feat(os.path.join(WAV_D, df.iloc[0].wav)).shape[1]
    ann = ANN(fft_dim, len(df)).to(DEVICE)
    opt = torch.optim.Adam(ann.parameters(), 1e-3)
    loss_fn = nn.CrossEntropyLoss()

    # train & "val" trên toàn bộ dữ liệu (vì mọi nhãn xuất hiện đúng 1 lần)
    ds = FFTset(df)
    dl = DataLoader(ds, BATCH, shuffle=True, collate_fn=collate, num_workers=0)

    for ep in range(EPOCH):
        ann.train()
        for x, y in dl:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logit, _ = ann(x)
            loss = loss_fn(logit, y)
            opt.zero_grad(); loss.backward(); opt.step()

        # --- metric trên chính tập train (demo) ----
        ann.eval(); y_true, y_pred, y_prob = [], [], []
        with torch.no_grad():
            for x, y in dl:
                logit, _ = ann(x.to(DEVICE))
                prob = logit.softmax(-1).cpu()
                y_true.extend(y.tolist())
                y_pred.extend(prob.argmax(-1).tolist())
                y_prob.append(prob.numpy())
        y_prob = np.vstack(y_prob)
        acc = accuracy_score(y_true, y_pred)
        acc5 = top_k_accuracy_score(y_true, y_prob, k=5,
                                    labels=np.arange(len(df)))
        print(f"Ep {ep:02d} | train_acc {acc:.3f} | top5 {acc5:.3f}")

    torch.save(ann.state_dict(), "models/ann_fft.pt")

    # 2. trích embedding toàn bộ ➜ PCA ➜ HMM
    ann.eval(); embs = []
    with torch.no_grad():
        for x,_ in DataLoader(ds, BATCH, collate_fn=collate):
            embs.append(ann(x.to(DEVICE))[1].cpu().numpy())
    embs = np.vstack(embs)

    pca = PCA(50).fit(embs)
    gmmhmm = hmm.GaussianHMM(6, "diag", n_iter=1000).fit(pca.transform(embs))
    pickle.dump([pca, gmmhmm], open("models/pca_hmm.pkl", "wb"))

    # 3. save metrics & (tuỳ chọn) conf-matrix
    json.dump({"train_acc": float(acc), "train_acc5": float(acc5)},
              open("models/metrics.json", "w"), ensure_ascii=False)

    print("✅ ANN, PCA, HMM & metrics saved to /models")

if __name__ == "__main__":
    import multiprocessing as mp; mp.freeze_support()
    main()
