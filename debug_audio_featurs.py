#!/usr/bin/env python3
"""
Script per debuggare le features audio estratte
Verifica se le features catturano caratteristiche distintive del genere
"""

import librosa
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path

def analyze_audio_features(file_path, sample_rate=22050, duration=30):
    """Analizza features audio dettagliate"""
    print(f"🎵 Analisi features: {Path(file_path).name}")
    
    # Carica audio
    audio, sr = librosa.load(file_path, sr=sample_rate, duration=duration)
    
    # Features basilari
    print(f"\n📊 Caratteristiche basilari:")
    print(f"   Durata: {len(audio)/sr:.1f}s")
    print(f"   RMS Energy: {np.mean(librosa.feature.rms(y=audio)):.4f}")
    print(f"   Zero Crossing Rate: {np.mean(librosa.feature.zero_crossing_rate(audio)):.4f}")
    
    # Spectral features
    spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
    spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)[0]
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)[0]
    
    print(f"\n🌈 Spectral features:")
    print(f"   Spectral Centroid: {np.mean(spectral_centroids):.1f} Hz")
    print(f"   Spectral Rolloff: {np.mean(spectral_rolloff):.1f} Hz")
    print(f"   Spectral Bandwidth: {np.mean(spectral_bandwidth):.1f} Hz")
    
    # Tempo e ritmo
    tempo, beats = librosa.beat.beat_track(y=audio, sr=sr)
    print(f"\n🥁 Tempo e ritmo:")
    print(f"   Tempo stimato: {tempo:.1f} BPM")
    print(f"   Beats rilevati: {len(beats)}")
    
    # Chroma (tonalità)
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
    dominant_chroma = np.argmax(np.mean(chroma, axis=1))
    chroma_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    
    print(f"\n🎼 Analisi tonale:")
    print(f"   Tonalità dominante: {chroma_names[dominant_chroma]}")
    print(f"   Varianza tonale: {np.var(np.mean(chroma, axis=1)):.4f}")
    
    # MFCC features (come nel modello)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    print(f"\n🔢 MFCC features:")
    print(f"   MFCC mean: {np.mean(mfccs, axis=1)[:5]}")
    print(f"   MFCC std: {np.std(mfccs, axis=1)[:5]}")
    
    # Mel-spectrogram (come usato dal modello)
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128, hop_length=512)
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    
    print(f"\n🌈 Mel-spectrogram:")
    print(f"   Shape: {log_mel_spec.shape}")
    print(f"   Range: {log_mel_spec.min():.1f} to {log_mel_spec.max():.1f} dB")
    print(f"   Mean energy: {np.mean(log_mel_spec):.1f} dB")
    
    return {
        'rms_energy': np.mean(librosa.feature.rms(y=audio)),
        'zcr': np.mean(librosa.feature.zero_crossing_rate(audio)),
        'spectral_centroid': np.mean(spectral_centroids),
        'spectral_rolloff': np.mean(spectral_rolloff),
        'spectral_bandwidth': np.mean(spectral_bandwidth),
        'tempo': tempo,
        'dominant_chroma': dominant_chroma,
        'chroma_variance': np.var(np.mean(chroma, axis=1)),
        'mel_spec_shape': log_mel_spec.shape,
        'mel_spec_mean': np.mean(log_mel_spec)
    }

def compare_genre_characteristics():
    """Stampa caratteristiche tipiche per genere"""
    print(f"\n📋 CARATTERISTICHE TIPICHE PER GENERE:")
    print(f"=" * 50)
    
    print(f"🤘 METAL/ROCK:")
    print(f"   - Alto RMS Energy (>0.1)")
    print(f"   - Alto Spectral Centroid (>2000 Hz)")
    print(f"   - Tempo veloce (120-180 BPM)")
    print(f"   - Alta varianza tonale")
    
    print(f"\n🎷 SOUL/R&B:")
    print(f"   - RMS Energy moderato (0.05-0.1)")
    print(f"   - Spectral Centroid medio (1000-2000 Hz)")
    print(f"   - Tempo medio (80-120 BPM)")
    print(f"   - Patterns ritmici complessi")
    
    print(f"\n🎸 POP:")
    print(f"   - RMS Energy bilanciato")
    print(f"   - Spectral features moderate")
    print(f"   - Tempo dance (100-130 BPM)")
    
    print(f"\n🎵 CLASSICAL:")
    print(f"   - Dinamiche variabili")
    print(f"   - Range spettrale ampio")
    print(f"   - Tempo variabile")
    print(f"   - Alta complessità tonale")

def main():
    parser = argparse.ArgumentParser(description='Debug Audio Features')
    parser.add_argument('--file', required=True, help='File audio da analizzare')
    
    args = parser.parse_args()
    
    print("🔍 DEBUG AUDIO FEATURES")
    print("=" * 40)
    
    if not Path(args.file).exists():
        print(f"❌ File non trovato: {args.file}")
        return
    
    # Analizza features
    features = analyze_audio_features(args.file)
    
    # Mostra caratteristiche tipiche
    compare_genre_characteristics()
    
    # Suggerimenti diagnostici
    print(f"\n💡 VALUTAZIONE:")
    print(f"=" * 30)
    
    if features['rms_energy'] > 0.1:
        print(f"✅ Alta energia - coerente con rock/metal")
    else:
        print(f"⚠️ Bassa energia - potrebbe confondere il modello")
    
    if features['spectral_centroid'] > 2000:
        print(f"✅ Alto spectral centroid - coerente con metal")
    else:
        print(f"⚠️ Basso spectral centroid - potrebbe sembrare soul/jazz")
    
    if features['tempo'] > 120:
        print(f"✅ Tempo veloce - coerente con metal")
    else:
        print(f"⚠️ Tempo lento - potrebbe sembrare soul/ballad")

if __name__ == "__main__":
    main()