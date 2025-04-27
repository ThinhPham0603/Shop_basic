import os
import csv
import subprocess
import re
import pandas as pd
from pathlib import Path

# Cấu hình
FFMPEG = r"C:\Users\PC\Downloads\ffmpeg-7.1.1-essentials_build\ffmpeg-7.1.1-essentials_build\bin\ffmpeg.exe"
SRC_DIR = Path("AI/data/cv-corpus-21.0-delta-2025-03-14-vi")
CLIPS_DIR = Path(r"C:\Users\PC\Downloads\Speech_to_text\Speech_to_text\AI\data\cv-corpus-21.0-delta-2025-03-14-vi\cv-corpus-21.0-delta-2025-03-14\vi\clips")
OUT_DIR = Path("data/wav")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Hàm clean câu
def clean_sentence(t: str):
    t = t.lower()
    t = re.sub(r"[^a-z0-9à-ỹ\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()

# Chuẩn bị đọc file
tsv_path = Path(r"C:\Users\PC\Downloads\Speech_to_text\Speech_to_text\AI\data\cv-corpus-21.0-delta-2025-03-14-vi\cv-corpus-21.0-delta-2025-03-14\vi\validated.tsv")
rows = []
bad = 0

# Đọc validated.tsv và xử lý
with open(tsv_path, encoding="utf-8") as f:
    reader = csv.DictReader(f, delimiter="\t")
    for r in reader:
        mp3_file = CLIPS_DIR / r["path"]  # nối path cực sạch

        if not mp3_file.exists():
            continue  # clip không tồn tại

        wav_name = Path(r["path"]).with_suffix(".wav").name
        wav_file = OUT_DIR / wav_name

        cmd = [
            FFMPEG, "-loglevel", "error", "-y",
            "-i", str(mp3_file),
            "-ar", "16000", "-ac", "1",
            str(wav_file)
        ]

        try:
            subprocess.run(cmd, timeout=6, check=True)

            if wav_file.stat().st_size < 2048:
                bad += 1
                wav_file.unlink(missing_ok=True)
                continue

            sentence = clean_sentence(r["sentence"])
            if sentence:  # chỉ lưu nếu còn text
                rows.append((wav_name, sentence))

        except subprocess.SubprocessError:
            bad += 1
            continue

# Ghi ra processed_vi.csv
df = pd.DataFrame(rows, columns=["wav", "sentence"])
df.to_csv("data/processed_vi.csv", index=False, encoding="utf-8")

print(f"Hoàn tất: {len(df)} file hợp lệ — bỏ qua {bad} file lỗi.")
