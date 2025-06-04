import librosa
import numpy as np
import os
from config import FMA_LARGE_PATH,SAMPLE_RATE,DURATION,NUM_SEGMENTS,N_MELS,HOP_LENGTH





def audio_path(track_id):
    """Genera il percorso del file audio"""
    tid_str = '{:06d}'.format(track_id)
    return os.path.join(FMA_LARGE_PATH, tid_str[:3], tid_str + '.mp3')

def extract_features(file_path):
    """Estrae features mel-spectrogram"""
    try:
        audio, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION)
        samples_per_segment = int(DURATION * SAMPLE_RATE / NUM_SEGMENTS)
        features = []
        
        for i in range(NUM_SEGMENTS):
            start = samples_per_segment * i
            end = start + samples_per_segment
            
            if end <= len(audio):
                segment = audio[start:end]
                mel_spec = librosa.feature.melspectrogram(
                    y=segment, sr=sr, n_mels=N_MELS, hop_length=HOP_LENGTH)
                log_mel_spec = librosa.power_to_db(mel_spec)
                log_mel_spec = (log_mel_spec - np.mean(log_mel_spec)) / np.std(log_mel_spec)
                features.append(log_mel_spec)
        
        return np.array(features)
    except Exception as e:
        print(f"Errore processing {file_path}: {e}")
        return None
