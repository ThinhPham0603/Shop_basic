import sys
import os
from pydub import AudioSegment
from datetime import datetime, timezone, timedelta
import speech_recognition as sr
import azure.cognitiveservices.speech as speechsdk
sys.stdout.reconfigure(encoding='utf-8')


def convert_webm_to_wav(webm_path, wav_path):
    try:
        audio = AudioSegment.from_file(webm_path, format="webm")
        audio = audio.set_frame_rate(16000).set_channels(1)
        audio.export(wav_path, format="wav", codec="pcm_s16le")
        os.remove(webm_path)
    except Exception as e:
        print(f"Lỗi khi chuyển đổi hoặc xóa file: {e}")


def recognize_speech_from_google(audio_path):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio_data = recognizer.record(source)
    try:
        return recognizer.recognize_google(audio_data, language="vi-VN")
    except sr.UnknownValueError:
        return "Không thể nhận diện giọng nói"
    except sr.RequestError as e:
        return f"Lỗi kết nối: {e}"

def recognize_speech_from_azure(audio_path, bing_key):
    speech_config = speechsdk.SpeechConfig(subscription=bing_key, region="eastasia")
    audio_config = speechsdk.AudioConfig(filename=audio_path)
    speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)
    result = speech_recognizer.recognize_once()
    if result.reason == speechsdk.ResultReason.RecognizedSpeech:
        return result.text
    elif result.reason == speechsdk.ResultReason.NoMatch:
        return "Không thể nhận diện giọng nói"
    else:
        return f"Lỗi nhận diện: {result.cancellation_details.reason}"

def recognize_speech_from_sphinx(audio_path):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio_data = recognizer.record(source)
    try:
        return recognizer.recognize_sphinx(audio_data)
    except sr.UnknownValueError:
        return "Không thể nhận diện giọng nói"
    except sr.RequestError as e:
        return f"Lỗi kết nối: {e}"

def recognize_speech_from_wit(audio_path, wit_key):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio_data = recognizer.record(source)
    try:
        return recognizer.recognize_wit(audio_data, key=wit_key)
    except sr.UnknownValueError:
        return "Không thể nhận diện giọng nói"
    except sr.RequestError as e:
        return f"Lỗi kết nối: {e}"

def recognize_speech_from_houndify(audio_path, client_id, client_key):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio_data = recognizer.record(source)
    try:
        text, confidence = recognizer.recognize_houndify(audio_data, client_id=client_id, client_key=client_key)
        return text
    except sr.UnknownValueError:
        return "Không thể nhận diện giọng nói"
    except sr.RequestError as e:
        return f"Lỗi kết nối: {e}"

def recognize_speech_from_whisper(audio_path):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio_data = recognizer.record(source)
    try:
        return recognizer.recognize_whisper(audio_data, language="vi", model="medium")
    except sr.UnknownValueError:
        return "Không thể nhận diện giọng nói"
    except sr.RequestError as e:
        return f"Lỗi kết nối: {e}"

def recognize_speech_from_file(audio_path, tts_service, **kwargs):
    if tts_service == "google":
        return recognize_speech_from_google(audio_path)
    elif tts_service == "azure":
        bing_key = kwargs.get('bing_key')
        if not bing_key:
            return "Azure API key missing"
        return recognize_speech_from_azure(audio_path, bing_key)
    elif tts_service == "sphinx":
        return recognize_speech_from_sphinx(audio_path)
    elif tts_service == "whisper":
        return recognize_speech_from_whisper(audio_path)
    elif tts_service == "wit":
        wit_key = kwargs.get('wit_key')
        if not wit_key:
            return "Wit.ai key missing"
        return recognize_speech_from_wit(audio_path, wit_key)
    elif tts_service == "houndify":
        client_id = kwargs.get('client_id')
        client_key = kwargs.get('client_key')
        if not client_id or not client_key:
            return "Houndify client credentials missing"
        return recognize_speech_from_houndify(audio_path, client_id, client_key)
    else:
        return "Service not supported"

if __name__ == "__main__":
    # Đọc tham số từ dòng lệnh
    webm_path = sys.argv[1]
    tts_service = sys.argv[2]  
    
    wit_key = bing_key = client_id = client_key = None
    if tts_service == "wit":
        wit_key = "UDCGR3GW3D44RAQLBKZ77TRZUZG7DBHQ" 
    elif tts_service == "azure":
        bing_key = "CmnTE9u8Ye3zwpsqphhBKXKYZZPCCXaaDuoX8IkyBCGpLcgGNm0vJQQJ99BCAC3pKaRXJ3w3AAAYACOGqh9z"
    elif tts_service == "houndify":
        client_id =  "qK6HnwXKm2QQ2gaRnJbG4g=="
        client_key =  "XRLzd-VGYITHRQhMoYIrLcTh19mwxmuuj3IsZJIhc3zb0keCBAbvc1jF8UfOJdPuwJGdDKgUenF_yQwXyOYDDQ=="
    
    uploads_dir = "D:\\REACT\\Test\\econmerce-backend\\src\\uploads"
    if not os.path.exists(uploads_dir):
        os.makedirs(uploads_dir)
    vn_time = datetime.now(timezone.utc) + timedelta(hours=7)
    wav_filename = vn_time.strftime("%Y-%m-%d_%Hh-%M") + ".wav"
    wav_path = os.path.join(uploads_dir, wav_filename)
    convert_webm_to_wav(webm_path, wav_path)
    transcription = recognize_speech_from_file(wav_path, tts_service, wit_key=wit_key, bing_key=bing_key, client_id=client_id, client_key=client_key)
    print(transcription)







