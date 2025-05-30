import streamlit as st
import sounddevice as sd
import whisper
import librosa
import numpy as np

def record_audio(duration= 8, samplerate=44100):
    audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype=np.int16)
    st.write('recording')
    sd.wait()
    st.write('recording complete')
    return audio, samplerate

def preprocess_audio(audio_input, original_sr = 44100, target_sr = 16000):
    audio_input = audio_input.squeeze()
    audio_input = audio_input.astype(np.float32) / 32768.0
    audio_input = librosa.resample(audio_input, orig_sr=original_sr, target_sr=target_sr)
    audio_input = whisper.pad_or_trim(audio_input)
    return audio_input