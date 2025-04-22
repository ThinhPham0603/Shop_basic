import os
import numpy as np
import pandas as pd
import pickle
import librosa
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from hmmlearn import hmm
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical

# ====== Trích xuất đặc trưng FFT ======
DATA_PATH = "cv_processed_vi_wav"
train_df = pd.read_csv("vi/train.tsv", sep="\t")

def extract_fft_features(file_path, sample_rate=16000, n_fft=512, hop_length=256):
    y, sr = librosa.load(file_path, sr=sample_rate)
    D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    S, _ = librosa.magphase(D)
    log_S = librosa.amplitude_to_db(S, ref=np.max)
    return log_S.T

X, y = [], []

print(" Đang trích xuất đặc trưng từ file .wav...")
for index, row in train_df.iterrows():
    wav_file = os.path.join(DATA_PATH, row["path"].replace(".mp3", ".wav"))
    if os.path.exists(wav_file):
        X.append(extract_fft_features(wav_file))
        y.append(" ".join(row["sentence"].lower().split()))  

with open("train_fft_features_vi.pkl", "wb") as f:
    pickle.dump((X, y), f)

print("Đã lưu đặc trưng vào 'train_fft_features_vi.pkl'.")

# ======  Huấn luyện mô hình ANN + HMM ======
# Load lại dữ liệu
with open("train_fft_features_vi.pkl", "rb") as f:
    X, y = pickle.load(f)

max_frames = 200
feature_dim = X[0].shape[1]

X_padded = np.array([  
    np.pad(x[:max_frames], ((0, max(0, max_frames - x.shape[0])), (0, 0)), mode="constant")
    for x in X
])

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)  
num_classes = len(np.unique(y_encoded))
y_one_hot = to_categorical(y_encoded, num_classes=num_classes)

# Lưu LabelEncoder
with open("label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)
print("Đã lưu label_encoder.pkl.")

X_train, X_val, y_train, y_val = train_test_split(X_padded, y_one_hot, test_size=0.1, random_state=42)

# Hàm xây dựng mô hình ANN
def build_ann_model(input_shape, output_dim):
    inputs = tf.keras.Input(shape=input_shape)
    x = layers.LSTM(128, return_sequences=False)(inputs)
    x = layers.Dense(64, activation="relu")(x)
    outputs = layers.Dense(output_dim, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model

ann_model = build_ann_model((max_frames, feature_dim), num_classes)
ann_model.summary()
print(" Bắt đầu huấn luyện ANN...")
ann_model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, batch_size=32)

ann_model.save("speech_to_text_ann_vi.keras")
print("Đã lưu mô hình ANN vào 'speech_to_text_ann_vi.keras'.")

feature_extractor = tf.keras.Model(inputs=ann_model.input, outputs=ann_model.layers[-2].output)
X_features = feature_extractor.predict(X_padded)

pca = PCA(n_components=50)
X_features_reduced = pca.fit_transform(X_features)
with open("pca_model.pkl", "wb") as f:
    pickle.dump(pca, f)
print("Đã lưu PCA model vào 'pca_model.pkl'.")
# Train HMM
print("Đang huấn luyện HMM...")
hmm_model = hmm.GaussianHMM(n_components=5, covariance_type="diag", n_iter=1000)
hmm_model.fit(X_features_reduced)

with open("hmm_model_vi.pkl", "wb") as f:
    pickle.dump(hmm_model, f)
print("Đã lưu mô hình HMM vào 'hmm_model_vi.pkl'.")
print("HOÀN TẤT: Đã huấn luyện và lưu mô hình ANN + HMM + LabelEncoder.")

# import os
# import numpy as np
# import pandas as pd
# import pickle
# import librosa
# from sklearn.preprocessing import LabelEncoder
# from sklearn.model_selection import train_test_split
# from hmmlearn import hmm
# import tensorflow as tf
# from tensorflow.keras import layers
# from tensorflow.keras.utils import to_categorical


# DATA_PATH = "cv_processed_vi_wav"
# train_df = pd.read_csv("vi/train.tsv", sep="\t")

# def extract_fft_features(file_path, sample_rate=16000, n_fft=512, hop_length=256):
#     y, sr = librosa.load(file_path, sr=sample_rate)
#     D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
#     S, _ = librosa.magphase(D)
#     log_S = librosa.amplitude_to_db(S, ref=np.max)
#     return log_S.T  

# X, y = [], []

# print("Đang trích xuất đặc trưng từ file .wav...")
# for index, row in train_df.iterrows():
#     wav_file = os.path.join(DATA_PATH, row["path"].replace(".mp3", ".wav"))
#     if os.path.exists(wav_file):
#         X.append(extract_fft_features(wav_file))
#         y.append(row["sentence"].split()[0])

# with open("train_fft_features_vi.pkl", "wb") as f:
#     pickle.dump((X, y), f)

# print(" Đã lưu đặc trưng vào 'train_fft_features_vi.pkl'.")


# with open("train_fft_features_vi.pkl", "rb") as f:
#     X, y = pickle.load(f)

# max_frames = 200
# feature_dim = X[0].shape[1]

# X_padded = np.array([
#     np.pad(x[:max_frames], ((0, max(0, max_frames - x.shape[0])), (0, 0)), mode="constant")
#     for x in X
# ])

# label_encoder = LabelEncoder()
# y_encoded = label_encoder.fit_transform(y)
# num_classes = len(np.unique(y_encoded))
# y_one_hot = to_categorical(y_encoded, num_classes=num_classes)


# with open("label_encoder.pkl", "wb") as f:
#     pickle.dump(label_encoder, f)

# X_train, X_val, y_train, y_val = train_test_split(X_padded, y_one_hot, test_size=0.1, random_state=42)


# def build_ann_model(input_shape, output_dim):
#     inputs = tf.keras.Input(shape=input_shape)
#     x = layers.LSTM(128, return_sequences=True)(inputs)
#     x = layers.LSTM(128, return_sequences=False)(x)
#     outputs = layers.Dense(output_dim, activation="softmax")(x)

#     model = tf.keras.Model(inputs, outputs)
#     model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
#     return model

# ann_model = build_ann_model((max_frames, feature_dim), num_classes)
# print("Bắt đầu huấn luyện ANN...")
# ann_model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100, batch_size=32)

# ann_model.save("speech_to_text_ann_vi.keras")
# print("Đã lưu mô hình ANN vào 'speech_to_text_ann_vi.keras'.")


# feature_extractor = tf.keras.Model(inputs=ann_model.input, outputs=ann_model.layers[-2].output)
# X_features = feature_extractor.predict(X_padded)

# X_features = X_features.reshape(X_features.shape[0], -1)

# print("Đang huấn luyện HMM...")
# hmm_model = hmm.GaussianHMM(n_components=10, covariance_type="diag", n_iter=1000)
# hmm_model.fit(X_features)

# with open("hmm_model_vi.pkl", "wb") as f:
#     pickle.dump(hmm_model, f)

# print("HOÀN TẤT: Đã huấn luyện và lưu mô hình ANN + HMM + LabelEncoder.")
