# import os
# os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
# import sys
# import numpy as np
# import librosa
# import tensorflow as tf
# from datetime import datetime, timezone, timedelta
# from pydub import AudioSegment
# from hmmlearn import hmm
# import pickle
# from sklearn.decomposition import PCA
# sys.stdout.reconfigure(encoding='utf-8')

# # Chuyển đổi WebM sang WAV
# def convert_webm_to_wav(webm_path, wav_path):
#     try:
#         audio = AudioSegment.from_file(webm_path, format="webm")
#         audio = audio.set_frame_rate(16000).set_channels(1)
#         audio.export(wav_path, format="wav", codec="pcm_s16le")
#         os.remove(webm_path)
#     except Exception as e:
#         print(f"Lỗi chuyển đổi: {e}")

# # Trích xuất đặc trưng MFCC
# def extract_mfcc(audio_path, sample_rate=16000, n_mfcc=13, max_frames=200):
#     y, sr = librosa.load(audio_path, sr=sample_rate)
#     mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
#     num_frames = mfcc.shape[1]
    
#     if num_frames < max_frames:
#         padding = np.zeros((n_mfcc, max_frames - num_frames))
#         mfcc = np.hstack((mfcc, padding))
#     elif num_frames > max_frames:
#         mfcc = mfcc[:, :max_frames]
    
#     return mfcc.T  


# def recognize_speech(ann_model, hmm_model, audio_path):
#     mfcc = extract_mfcc(audio_path)
#     mfcc = np.expand_dims(mfcc, axis=0)  
    
#     feature_extractor = tf.keras.Model(inputs=ann_model.input, outputs=ann_model.layers[-2].output)
#     features = feature_extractor.predict(mfcc, verbose=0) 
#     predicted_state = hmm_model.predict(features)

#     return decode_output(predicted_state)


# # Giải mã đầu ra HMM thành văn bản
# def decode_output(predictions):
#     char_map = "abcdefghijklmnopqrstuvwxyz '"
#     print("Raw predictions:", predictions)
#     decoded_text = "".join(char_map[p] if p < len(char_map) else '' for p in predictions)
#     return decoded_text

# if __name__ == "__main__":
#     if len(sys.argv) < 2:
#         print("Vui lòng cung cấp đường dẫn file .webm")
#         sys.exit(1)
    
#     webm_path = sys.argv[1]
#     uploads_dir = "uploads"
#     if not os.path.exists(uploads_dir):
#         os.makedirs(uploads_dir)
    
#     vn_time = datetime.now(timezone.utc) + timedelta(hours=7)
#     wav_filename = vn_time.strftime("%Y-%m-%d_%Hh-%M") + ".wav"
#     wav_path = os.path.join(uploads_dir, wav_filename)
    
#     convert_webm_to_wav(webm_path, wav_path)

#     model_path = "D:/REACT/Test/econmerce-backend/src/scripts/speech_to_text_ann.keras"
#     hmm_path = "D:/REACT/Test/econmerce-backend/src/scripts/hmm_model.pkl"
   
#     if not os.path.exists(model_path) or not os.path.exists(hmm_path):
#         print("Không tìm thấy mô hình ANN hoặc HMM, vui lòng huấn luyện trước.")
#         sys.exit(1)
    
#     ann_model = tf.keras.models.load_model(model_path)
#     with open(hmm_path, "rb") as f:
#         hmm_model = pickle.load(f)
#     print("HMM Model components:", hmm_model.n_components)
#     print("HMM Model feature size:", hmm_model.startprob_.shape)

#     text = recognize_speech(ann_model, hmm_model, wav_path)
#     print(f"Văn bản nhận diện: {text}")




# import os
# os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
# import sys
# import numpy as np
# import librosa
# import tensorflow as tf
# from datetime import datetime, timezone, timedelta
# from pydub import AudioSegment
# from hmmlearn import hmm
# import pickle
# from sklearn.decomposition import PCA
# sys.stdout.reconfigure(encoding='utf-8')
# import re
# # Chuyển đổi WebM sang WAV
# def convert_webm_to_wav(webm_path, wav_path):
#     try:
#         audio = AudioSegment.from_file(webm_path, format="webm")
#         audio = audio.set_frame_rate(16000).set_channels(1)
#         audio.export(wav_path, format="wav", codec="pcm_s16le")
#         os.remove(webm_path)
#     except Exception as e:
#         print(f"Lỗi chuyển đổi: {e}")

# # Trích xuất đặc trưng MFCC
# def extract_mfcc(audio_path, sample_rate=16000, n_mfcc=13, max_frames=200):
#     y, sr = librosa.load(audio_path, sr=sample_rate)
#     mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
#     num_frames = mfcc.shape[1]
    
#     if num_frames < max_frames:
#         padding = np.zeros((n_mfcc, max_frames - num_frames))
#         mfcc = np.hstack((mfcc, padding))
#     elif num_frames > max_frames:
#         mfcc = mfcc[:, :max_frames]
    
#     return mfcc.T  


# def recognize_speech(ann_model, hmm_model, audio_path):
#     mfcc = extract_mfcc(audio_path)
#     mfcc = np.expand_dims(mfcc, axis=0)  

#     feature_extractor = tf.keras.Model(inputs=ann_model.input, outputs=ann_model.layers[-2].output)
#     features = feature_extractor.predict(mfcc, verbose=0)
#     features = features.reshape(features.shape[1], -1)  
#     pca = PCA(n_components=50)
#     features_reduced = pca.fit_transform(features)
#     predicted_state = hmm_model.predict(features_reduced)

#     return decode_output(predicted_state)

# # Giải mã đầu ra HMM thành văn bản
# # def decode_output(predictions):
# #     char_map = "abcdefghijklmnopqrstuvwxyz '"
# #     decoded_text = "".join(char_map[p] if p < len(char_map) else '' for p in predictions)
# #     return decoded_text


# def clean_text(text):
#     text = re.sub(r"\s+", " ", text)  
#     text = re.sub(r"([a-z])\1{2,}", r"\1", text)  
#     return text.strip()

# def decode_output(predictions):
#     char_map = "abcdefghijklmnopqrstuvwxyz '"
#     decoded_text = "".join(char_map[p] if 0 <= p < len(char_map) else '' for p in predictions)
    
#     return clean_text(decoded_text)
 
# if __name__ == "__main__":
#     if len(sys.argv) < 2:
#         print("Vui lòng cung cấp đường dẫn file .webm")
#         sys.exit(1)
    
#     webm_path = sys.argv[1]
#     uploads_dir = "uploads"
#     if not os.path.exists(uploads_dir):
#         os.makedirs(uploads_dir)
    
#     vn_time = datetime.now(timezone.utc) + timedelta(hours=7)
#     wav_filename = vn_time.strftime("%Y-%m-%d_%Hh-%M") + ".wav"
#     wav_path = os.path.join(uploads_dir, wav_filename)
    
#     convert_webm_to_wav(webm_path, wav_path)

#     model_path = "D:/REACT/Test/econmerce-backend/src/scripts/speech_to_text_ann.keras"
#     hmm_path = "D:/REACT/Test/econmerce-backend/src/scripts/hmm_model.pkl"
   
#     if not os.path.exists(model_path) or not os.path.exists(hmm_path):
#         print("Không tìm thấy mô hình ANN hoặc HMM, vui lòng huấn luyện trước.")
#         sys.exit(1)
    
#     ann_model = tf.keras.models.load_model(model_path)
#     with open(hmm_path, "rb") as f:
#         hmm_model = pickle.load(f)

#     text = recognize_speech(ann_model, hmm_model, wav_path)
#     print(f"Văn bản nhận diện: {text}")



import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import sys
import numpy as np
import librosa
import tensorflow as tf
from datetime import datetime, timezone, timedelta
from pydub import AudioSegment
from hmmlearn import hmm
import pickle
from sklearn.decomposition import PCA
import re

sys.stdout.reconfigure(encoding='utf-8')

# Chuyển đổi WebM sang WAV
def convert_webm_to_wav(webm_path, wav_path):
    try:
        audio = AudioSegment.from_file(webm_path, format="webm")
        audio = audio.set_frame_rate(16000).set_channels(1)
        audio.export(wav_path, format="wav", codec="pcm_s16le")
        os.remove(webm_path)
    except Exception as e:
        print(f"Lỗi chuyển đổi: {e}")

# Trích xuất đặc trưng FFT
def extract_fft_features(audio_path, sample_rate=16000, n_fft=512, hop_length=256, max_frames=200):
    y, sr = librosa.load(audio_path, sr=sample_rate)
    D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    S, _ = librosa.magphase(D)
    log_S = librosa.amplitude_to_db(S, ref=np.max)
    fft_features = log_S.T

    # Cắt hoặc padding về max_frames
    if fft_features.shape[0] < max_frames:
        padding = np.zeros((max_frames - fft_features.shape[0], fft_features.shape[1]))
        fft_features = np.vstack([fft_features, padding])
    else:
        fft_features = fft_features[:max_frames]

    return fft_features

# Làm sạch văn bản đầu ra
def clean_text(text):
    text = re.sub(r"\s+", " ", text)  
    text = re.sub(r"([a-z])\1{2,}", r"\1", text)  
    return text.strip()

# Nhận diện giọng nói
def recognize_speech(ann_model, hmm_model, pca_model, label_encoder, audio_path):
    try:
        fft = extract_fft_features(audio_path)
        fft = np.expand_dims(fft, axis=0)

        feature_extractor = tf.keras.Model(inputs=ann_model.input, outputs=ann_model.layers[-2].output)
        features = feature_extractor.predict(fft, verbose=0)
        features = features.reshape(features.shape[0], -1)

        features_reduced = pca_model.transform(features)
        predicted_state = hmm_model.predict(features_reduced)
        predicted_text = label_encoder.inverse_transform(predicted_state)
        return clean_text(" ".join(predicted_text))
    
    except Exception as e:
        print("Lỗi trong recognize_speech:", e)
        return ""

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Vui lòng cung cấp đường dẫn file .webm")
        sys.exit(1)

    webm_path = sys.argv[1]
    uploads_dir = "uploads"
    if not os.path.exists(uploads_dir):
        os.makedirs(uploads_dir)

    vn_time = datetime.now(timezone.utc) + timedelta(hours=7)
    wav_filename = vn_time.strftime("%Y-%m-%d_%Hh-%M") + ".wav"
    wav_path = os.path.join(uploads_dir, wav_filename)

    convert_webm_to_wav(webm_path, wav_path)

    
    model_path = "D:/REACT/Test/econmerce-backend/src/scripts/speech_to_text_ann_vi.keras"
    hmm_path = "D:/REACT/Test/econmerce-backend/src/scripts/hmm_model_vi.pkl"
    pca_path = "D:/REACT/Test/econmerce-backend/src/scripts/pca_model.pkl"
    label_encoder_path = "D:/REACT/Test/econmerce-backend/src/scripts/label_encoder.pkl"

    if not all(os.path.exists(p) for p in [model_path, hmm_path, pca_path, label_encoder_path]):
        print("Không tìm thấy đầy đủ mô hình đã huấn luyện (ANN, HMM, PCA, LabelEncoder)")
        sys.exit(1)

    ann_model = tf.keras.models.load_model(model_path)

    with open(hmm_path, "rb") as f:
        hmm_model = pickle.load(f)

    with open(pca_path, "rb") as f:
        pca_model = pickle.load(f)

    with open(label_encoder_path, "rb") as f:
        label_encoder = pickle.load(f)
    text = recognize_speech(ann_model, hmm_model, pca_model, label_encoder, wav_path)

    print(f"Văn bản nhận diện: {text}")

