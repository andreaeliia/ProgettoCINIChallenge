#!/usr/bin/env python3
"""
Script Test Singolo MP3 - Classificazione Generi Musicali
Testa il modello bilanciato su un singolo file MP3
VERSIONE CORRETTA con layer personalizzate
"""

import os
import sys
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.utils import custom_object_scope
import json
import argparse
from pathlib import Path

# Importa configurazione se disponibile
try:
    import config
    SAMPLE_RATE = config.SAMPLE_RATE
    DURATION = config.DURATION
    N_MELS = config.N_MELS
    HOP_LENGTH = config.HOP_LENGTH
    NUM_SEGMENTS = config.NUM_SEGMENTS
except ImportError:
    print("⚠️ config.py non trovato, uso configurazione default")
    SAMPLE_RATE = 22050
    DURATION = 30
    N_MELS = 128
    HOP_LENGTH = 512
    NUM_SEGMENTS = 10

# ==================== LAYER PERSONALIZZATE ====================
class PANNsGlobalPooling(layers.Layer):
    """Layer PANNs per Global Pooling personalizzato"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def call(self, inputs):
        # inputs shape: (batch, height, width, channels)
        
        # Media lungo l'asse del tempo (width)
        x = tf.reduce_mean(inputs, axis=2)  # (batch, height, channels)
        
        # Max + Mean pooling lungo l'asse delle frequenze (height)
        x1 = tf.reduce_max(x, axis=1)  # (batch, channels)
        x2 = tf.reduce_mean(x, axis=1)  # (batch, channels) 
        x = x1 + x2  # Combinazione PANNs style
        
        return x
    
    def get_config(self):
        return super().get_config()

class PANNsConvBlock(layers.Layer):
    """Blocco convoluzionale PANNs style"""
    def __init__(self, out_channels, **kwargs):
        super().__init__(**kwargs)
        self.out_channels = out_channels
        
        self.conv1 = layers.Conv2D(out_channels, (3, 3), padding='same', use_bias=False)
        self.conv2 = layers.Conv2D(out_channels, (3, 3), padding='same', use_bias=False)
        self.bn1 = layers.BatchNormalization()
        self.bn2 = layers.BatchNormalization()
    
    def call(self, inputs, pool_size=(2, 2), pool_type='avg', training=None):
        x = inputs
        x = tf.nn.relu(self.bn1(self.conv1(x), training=training))
        x = tf.nn.relu(self.bn2(self.conv2(x), training=training))
        
        if pool_size != (1, 1):  # Solo se dobbiamo fare pooling
            if pool_type == 'avg':
                x = layers.AveragePooling2D(pool_size)(x)
            elif pool_type == 'max':
                x = layers.MaxPooling2D(pool_size)(x)
            elif pool_type == 'avg+max':
                x1 = layers.AveragePooling2D(pool_size)(x)
                x2 = layers.MaxPooling2D(pool_size)(x)
                x = x1 + x2
        
        return x
    
    def get_config(self):
        config = super().get_config()
        config.update({'out_channels': self.out_channels})
        return config

# ==================== CLASSE MP3 TESTER CORRETTA ====================
class MP3Tester:
    """Classe per testare file MP3 con il modello addestrato"""
    
    def __init__(self, model_path: str, config_path: str = None):
        self.model_path = model_path
        self.config_path = config_path
        self.model = None
        self.mappings = None
        
        self.load_model_and_config()
    
    def load_model_and_config(self):
        """Carica modello e configurazione generi con layer personalizzate"""
        print("🔄 Caricamento modello e configurazione...")
        
        # Carica modello con custom object scope
        try:
            # Registra le layer personalizzate
            custom_objects = {
                'PANNsConvBlock': PANNsConvBlock,
                'PANNsGlobalPooling': PANNsGlobalPooling
            }
            
            with custom_object_scope(custom_objects):
                self.model = tf.keras.models.load_model(self.model_path, compile=False)
            
            print(f"   ✅ Modello caricato: {self.model_path}")
            
            # Stampa architettura per verifica
            print(f"   📐 Input shape: {self.model.input_shape}")
            print(f"   📊 Parametri: {self.model.count_params():,}")
            
        except Exception as e:
            print(f"   ❌ Errore caricamento modello: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
        
        # Carica mappings generi
        if self.config_path and os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)
                self.mappings = config_data.get('mappings', {})
                print(f"   ✅ Configurazione caricata: {self.config_path}")
                
                # Stampa info mappings
                if self.mappings:
                    print(f"   🎵 Generi top: {len(self.mappings.get('top_genres', []))}")
                    print(f"   🎯 Generi medium: {len(self.mappings.get('medium_genres', []))}")
                    print(f"   🎨 Generi all: {len(self.mappings.get('all_genres', []))}")
                
            except Exception as e:
                print(f"   ⚠️ Errore caricamento config: {e}")
                self._create_default_mappings()
        else:
            print("   ⚠️ File configurazione non trovato, uso mappings default")
            self._create_default_mappings()
    
    def _create_default_mappings(self):
        """Crea mappings default se non disponibili"""
        # Mappings reali basati sui 16 generi del tuo dataset (ordinati per frequenza)
        real_genres = [
            'Experimental', 'Rock', 'Electronic', 'Hip-Hop', 'Folk', 'Pop',
            'Instrumental', 'Classical', 'International', 'Spoken', 'Jazz',
            'Old-Time / Historic', 'Blues', 'Soul-RnB', 'Country', 'Easy Listening'
        ]
        
        self.mappings = {
            'top_genres': real_genres,  # 16 generi reali
            'medium_genres': real_genres,  # Stessi generi per medium level
            'all_genres': real_genres   # Stessi generi per multi-label
        }
        print("   📝 Usati mappings reali dei 16 generi FMA")
    
    def extract_audio_features(self, file_path: str, use_full_song: bool = True) -> np.ndarray:
        """Estrae features audio dal file MP3 - opzione per canzone completa"""
        try:
            print(f"🎵 Estrazione features da: {os.path.basename(file_path)}")
            
            # Carica audio completo (senza limite di durata)
            if use_full_song:
                audio, sr = librosa.load(file_path, sr=SAMPLE_RATE)
                full_duration = len(audio) / sr
                print(f"   📊 Audio completo: {len(audio)} campioni, {sr}Hz, {full_duration:.1f}s")
                
                # Dividi in chunks di 30 secondi
                chunk_samples = int(DURATION * SAMPLE_RATE)
                num_chunks = int(np.ceil(len(audio) / chunk_samples))
                
                print(f"   🔄 Processando {num_chunks} chunks di {DURATION}s...")
                
                all_features = []
                
                for chunk_idx in range(num_chunks):
                    start_sample = chunk_idx * chunk_samples
                    end_sample = min(start_sample + chunk_samples, len(audio))
                    
                    # Estrai chunk
                    chunk_audio = audio[start_sample:end_sample]
                    
                    # Se chunk troppo corto, padding con zero
                    if len(chunk_audio) < chunk_samples:
                        padding = chunk_samples - len(chunk_audio)
                        chunk_audio = np.pad(chunk_audio, (0, padding), mode='constant')
                    
                    # Processa chunk come nel training (segmentazione interna)
                    chunk_features = self._process_audio_chunk(chunk_audio, sr, chunk_idx + 1)
                    if chunk_features is not None:
                        all_features.extend(chunk_features)
                
                if not all_features:
                    print(f"   ❌ Nessuna feature estratta da {num_chunks} chunks")
                    return None
                
                # Converti in array
                features_array = np.array(all_features, dtype=np.float32)
                if len(features_array.shape) == 3:
                    features_array = np.expand_dims(features_array, axis=-1)
                
                print(f"   ✅ Features estratte: {features_array.shape} da {num_chunks} chunks")
                return features_array
            
            else:
                # Modalità originale (solo 30s)
                audio, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION)
                print(f"   📊 Audio (30s): {len(audio)} campioni, {sr}Hz, {len(audio)/sr:.1f}s")
                
                chunk_features = self._process_audio_chunk(audio, sr, 1)
                if chunk_features is None:
                    return None
                
                features_array = np.array(chunk_features, dtype=np.float32)
                if len(features_array.shape) == 3:
                    features_array = np.expand_dims(features_array, axis=-1)
                
                print(f"   ✅ Features estratte: {features_array.shape}")
                return features_array
            
        except Exception as e:
            print(f"   ❌ Errore estrazione features: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _process_audio_chunk(self, audio_chunk: np.ndarray, sr: int, chunk_num: int) -> list:
        """Processa un singolo chunk di 30 secondi come nel training"""
        try:
            chunk_duration = len(audio_chunk) / sr
            print(f"      Chunk {chunk_num}: {chunk_duration:.1f}s")
            
            # Segmentazione interna (come nel training)
            samples_per_segment = int(len(audio_chunk) / NUM_SEGMENTS)
            features = []
            
            for i in range(NUM_SEGMENTS):
                start = samples_per_segment * i
                end = start + samples_per_segment
                
                if end <= len(audio_chunk):
                    segment = audio_chunk[start:end]
                    
                    # Estrai mel-spectrogram
                    mel_spec = librosa.feature.melspectrogram(
                        y=segment,
                        sr=sr,
                        n_mels=N_MELS,
                        hop_length=HOP_LENGTH
                    )
                    
                    # Log scale
                    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
                    
                    # Normalizzazione
                    log_mel_spec = (log_mel_spec - np.mean(log_mel_spec)) / (np.std(log_mel_spec) + 1e-8)
                    
                    features.append(log_mel_spec)
            
            print(f"         → {len(features)} segmenti estratti")
            return features
            
        except Exception as e:
            print(f"      ❌ Errore processing chunk {chunk_num}: {e}")
            return None
    
    def predict_genres(self, features: np.ndarray) -> dict:
        """Esegue predizione generi musicali"""
        try:
            print("🔄 Esecuzione predizione...")
            
            # Predizione
            predictions = self.model.predict(features, verbose=0)
            
            # Gestisci diversi formati di output
            if isinstance(predictions, list) and len(predictions) == 3:
                top_pred, medium_pred, all_pred = predictions
            elif isinstance(predictions, list) and len(predictions) == 1:
                # Solo un output
                top_pred = predictions[0]
                medium_pred = predictions[0]
                all_pred = predictions[0]
            else:
                # Output singolo
                top_pred = predictions
                medium_pred = predictions
                all_pred = predictions
            
            print(f"   📊 Predetto su {len(features)} segmenti")
            print(f"   🎯 Shape predizioni: top={top_pred.shape}, medium={medium_pred.shape}, all={all_pred.shape}")
            
            # Aggregazione risultati
            results = self._aggregate_predictions(top_pred, medium_pred, all_pred)
            
            return results
            
        except Exception as e:
            print(f"   ❌ Errore predizione: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _aggregate_predictions(self, top_pred: np.ndarray, medium_pred: np.ndarray, all_pred: np.ndarray) -> dict:
        """Aggrega predizioni da multipli segmenti (da canzone completa)"""
        
        num_segments = len(top_pred)
        print(f"   🔄 Aggregazione da {num_segments} segmenti totali...")
        
        # Media delle probabilità tra tutti i segmenti
        top_avg = np.mean(top_pred, axis=0)
        medium_avg = np.mean(medium_pred, axis=0)
        all_avg = np.mean(all_pred, axis=0)
        
        # Trova generi predetti
        top_idx = np.argmax(top_avg)
        medium_idx = np.argmax(medium_avg)
        
        # Top-K per multi-label
        all_top_indices = np.argsort(all_avg)[::-1][:10]
        
        # Calcola stabilità migliorata per canzone completa
        top_predictions_per_segment = [np.argmax(pred) for pred in top_pred]
        top_genre_consistency = len(set(top_predictions_per_segment)) / len(top_predictions_per_segment)
        
        # Statistiche temporali per canzone completa
        chunk_size = 10  # NUM_SEGMENTS per chunk di 30s
        num_chunks = num_segments // chunk_size if num_segments >= chunk_size else 1
        
        chunk_predictions = []
        if num_chunks > 1:
            for i in range(num_chunks):
                start_idx = i * chunk_size
                end_idx = min(start_idx + chunk_size, num_segments)
                chunk_preds = top_pred[start_idx:end_idx]
                chunk_avg = np.mean(chunk_preds, axis=0)
                chunk_predictions.append(np.argmax(chunk_avg))
        
        # Crea risultati strutturati
        results = {
            'top_genre': {
                'name': self.mappings['top_genres'][top_idx] if top_idx < len(self.mappings['top_genres']) else f'Unknown_{top_idx}',
                'confidence': float(top_avg[top_idx]),
                'all_probabilities': {
                    self.mappings['top_genres'][i] if i < len(self.mappings['top_genres']) else f'Unknown_{i}': float(prob)
                    for i, prob in enumerate(top_avg)
                }
            },
            'medium_genre': {
                'name': self.mappings['medium_genres'][medium_idx] if medium_idx < len(self.mappings['medium_genres']) else f'Unknown_{medium_idx}',
                'confidence': float(medium_avg[medium_idx])
            },
            'all_genres': [
                {
                    'name': self.mappings['all_genres'][idx] if idx < len(self.mappings['all_genres']) else f'Unknown_{idx}',
                    'confidence': float(all_avg[idx])
                }
                for idx in all_top_indices if all_avg[idx] > 0.1  # Soglia minima
            ],
            'segments_analyzed': num_segments,
            'chunks_analyzed': num_chunks,
            'consistency': {
                'top_genre_stability': float(np.std(top_predictions_per_segment)),
                'avg_confidence': float(np.mean(np.max(top_pred, axis=1))),
                'temporal_consistency': 1.0 - top_genre_consistency,  # 1.0 = perfetta consistenza
                'chunk_agreement': float(len(set(chunk_predictions)) / len(chunk_predictions)) if chunk_predictions else 1.0
            },
            'temporal_analysis': {
                'segment_predictions': top_predictions_per_segment,
                'chunk_predictions': chunk_predictions,
                'confidence_over_time': [float(np.max(pred)) for pred in top_pred]
            }
        }
        
        return results
    
    def print_results(self, results: dict, file_path: str):
        """Stampa risultati in formato leggibile con analisi temporale"""
        print(f"\n🎯 RISULTATI CLASSIFICAZIONE: {os.path.basename(file_path)}")
        print("=" * 60)
        
        # Genere principale
        top = results['top_genre']
        print(f"🏆 GENERE PRINCIPALE:")
        print(f"   {top['name']} (confidenza: {top['confidence']:.3f})")
        
        # Top 3 generi principali
        top_probs = sorted(top['all_probabilities'].items(), key=lambda x: x[1], reverse=True)[:3]
        print(f"\n🥇 TOP 3 GENERI PRINCIPALI:")
        for i, (genre, prob) in enumerate(top_probs, 1):
            print(f"   {i}. {genre}: {prob:.3f}")
        
        # Genere specifico
        medium = results['medium_genre']
        print(f"\n🎯 SOTTOGENERE:")
        print(f"   {medium['name']} (confidenza: {medium['confidence']:.3f})")
        
        # All genres (multi-label)
        if results['all_genres']:
            print(f"\n🎵 GENERI MULTIPLI (Top 5):")
            for i, genre_info in enumerate(results['all_genres'][:5], 1):
                print(f"   {i}. {genre_info['name']}: {genre_info['confidence']:.3f}")
        
        # Statistiche predizione migliorata
        consistency = results['consistency']
        chunks_analyzed = results.get('chunks_analyzed', 1)
        
        print(f"\n📊 QUALITÀ PREDIZIONE:")
        print(f"   Segmenti analizzati: {results['segments_analyzed']}")
        print(f"   Chunks di 30s analizzati: {chunks_analyzed}")
        print(f"   Durata totale analizzata: ~{chunks_analyzed * 30}s")
        
        # Analisi consistenza temporale
        temporal_consistency = consistency.get('temporal_consistency', 0)
        if temporal_consistency > 0.8:
            print(f"   Consistenza temporale: ✅ Molto alta ({temporal_consistency:.3f})")
        elif temporal_consistency > 0.6:
            print(f"   Consistenza temporale: 👍 Buona ({temporal_consistency:.3f})")
        else:
            print(f"   Consistenza temporale: ⚠️ Variabile ({temporal_consistency:.3f})")
        
        print(f"   Confidenza media: {consistency['avg_confidence']:.3f}")
        
        # Analisi temporale dettagliata se disponibile
        if 'temporal_analysis' in results and chunks_analyzed > 1:
            temporal = results['temporal_analysis']
            chunk_preds = temporal.get('chunk_predictions', [])
            
            if chunk_preds:
                print(f"\n🕒 EVOLUZIONE TEMPORALE:")
                chunk_genres = [self.mappings['top_genres'][pred] if pred < len(self.mappings['top_genres']) else f'Unknown_{pred}' 
                              for pred in chunk_preds]
                
                for i, genre in enumerate(chunk_genres):
                    time_start = i * 30
                    time_end = (i + 1) * 30
                    print(f"   {time_start:3d}s-{time_end:3d}s: {genre}")
                
                # Analisi cambiamenti
                genre_changes = len(set(chunk_genres))
                if genre_changes == 1:
                    print(f"   📈 Genere stabile per tutta la canzone")
                elif genre_changes <= len(chunk_genres) // 2:
                    print(f"   📊 Genere relativamente stabile ({genre_changes} cambiamenti)")
                else:
                    print(f"   ⚠️ Genere molto variabile ({genre_changes} cambiamenti)")
        
        # Raccomandazioni migliorate
        avg_conf = consistency['avg_confidence']
        temp_cons = temporal_consistency
        
        if avg_conf > 0.7 and temp_cons > 0.7:
            print(f"\n   🎉 Predizione molto affidabile su tutta la canzone!")
        elif avg_conf > 0.5 and temp_cons > 0.5:
            print(f"\n   👍 Predizione affidabile")
        elif temp_cons < 0.3:
            print(f"\n   ⚠️ Canzone eterogenea - potrebbe contenere più generi o sezioni diverse")
        else:
            print(f"\n   ⚠️ Predizione incerta - il brano potrebbe essere difficile da classificare")
    
    def test_file(self, file_path: str, use_full_song: bool = True) -> dict:
        """Testa un singolo file MP3"""
        mode_text = "CANZONE COMPLETA" if use_full_song else "PRIMI 30 SECONDI"
        print(f"🎧 TEST FILE MP3 ({mode_text}): {file_path}")
        print("=" * 60)
        
        # Verifica file
        if not os.path.exists(file_path):
            print(f"❌ File non trovato: {file_path}")
            return None
        
        if not file_path.lower().endswith(('.mp3', '.wav', '.flac', '.m4a')):
            print(f"⚠️ Formato file non supportato: {file_path}")
            print("   Formati supportati: .mp3, .wav, .flac, .m4a")
        
        # Estrazione features
        features = self.extract_audio_features(file_path, use_full_song=use_full_song)
        if features is None:
            return None
        
        # Predizione
        results = self.predict_genres(features)
        if results is None:
            return None
        
        # Stampa risultati
        self.print_results(results, file_path)
        
        return results
    
    def batch_test(self, folder_path: str, extensions: list = ['.mp3', '.wav', '.flac'], use_full_song: bool = True) -> dict:
        """Testa tutti i file audio in una cartella"""
        mode_text = "CANZONI COMPLETE" if use_full_song else "PRIMI 30 SECONDI"
        print(f"📁 BATCH TEST ({mode_text}): {folder_path}")
        print("=" * 60)
        
        folder = Path(folder_path)
        if not folder.exists():
            print(f"❌ Cartella non trovata: {folder_path}")
            return {}
        
        # Trova file audio
        audio_files = []
        for ext in extensions:
            audio_files.extend(list(folder.glob(f'*{ext}')))
            audio_files.extend(list(folder.glob(f'*{ext.upper()}')))
        
        if not audio_files:
            print(f"❌ Nessun file audio trovato in: {folder_path}")
            return {}
        
        print(f"📊 Trovati {len(audio_files)} file audio")
        
        results = {}
        for i, audio_file in enumerate(audio_files, 1):
            print(f"\n[{i}/{len(audio_files)}] {audio_file.name}")
            result = self.test_file(str(audio_file), use_full_song=use_full_song)
            if result:
                results[audio_file.name] = result
        
        # Statistiche batch
        self._print_batch_stats(results)
        
        return results
    
    def _print_batch_stats(self, results: dict):
        """Stampa statistiche del batch test"""
        if not results:
            return
        
        print(f"\n📈 STATISTICHE BATCH")
        print("=" * 40)
        
        # Conta generi predetti
        top_genres = [r['top_genre']['name'] for r in results.values()]
        genre_counts = {}
        for genre in top_genres:
            genre_counts[genre] = genre_counts.get(genre, 0) + 1
        
        print(f"📊 Distribuzione generi predetti:")
        for genre, count in sorted(genre_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"   {genre}: {count} file")
        
        # Confidenza media
        avg_confidence = np.mean([r['top_genre']['confidence'] for r in results.values()])
        print(f"\n🎯 Confidenza media: {avg_confidence:.3f}")

def main():
    """Funzione principale"""
    parser = argparse.ArgumentParser(description='Test Modello su File MP3 Singoli')
    parser.add_argument('--model', required=True, help='Path al modello .h5')
    parser.add_argument('--config', help='Path al file configurazione JSON')
    parser.add_argument('--file', help='File MP3 singolo da testare')
    parser.add_argument('--folder', help='Cartella con file MP3 per batch test')
    parser.add_argument('--save', help='File output per salvare risultati JSON')
    parser.add_argument('--full-song', action='store_true', default=True,
                        help='Analizza canzone completa (default: True)')
    parser.add_argument('--30s-only', dest='thirty_s_only', action='store_true', 
                        help='Analizza solo primi 30 secondi')
    
    args = parser.parse_args()
    
    # Determina modalità di analisi
    use_full_song = args.full_song and not args.thirty_s_only
    
    mode_text = "CANZONE COMPLETA" if use_full_song else "PRIMI 30 SECONDI"
    print("🎵 TEST CLASSIFICAZIONE GENERI MUSICALI")
    print(f"📊 Modalità: {mode_text}")
    print("=" * 50)
    
    # Verifica modello
    if not os.path.exists(args.model):
        print(f"❌ Modello non trovato: {args.model}")
        return
    
    # Inizializza tester
    tester = MP3Tester(args.model, args.config)
    
    # Esegui test
    results = None
    
    if args.file:
        # Test singolo file
        results = tester.test_file(args.file, use_full_song=use_full_song)
        
    elif args.folder:
        # Batch test
        results = tester.batch_test(args.folder, use_full_song=use_full_song)
        
    else:
        print("❌ Specifica --file per singolo test o --folder per batch test")
        return
    
    # Salva risultati se richiesto
    if args.save and results:
        try:
            # Aggiungi info modalità ai risultati
            if isinstance(results, dict) and 'top_genre' in results:
                # Singolo file
                results['analysis_mode'] = {
                    'full_song': use_full_song,
                    'mode_description': mode_text
                }
            elif isinstance(results, dict):
                # Batch test
                for filename, result in results.items():
                    if isinstance(result, dict):
                        result['analysis_mode'] = {
                            'full_song': use_full_song,
                            'mode_description': mode_text
                        }
            
            with open(args.save, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False, default=str)
            print(f"\n💾 Risultati salvati in: {args.save}")
        except Exception as e:
            print(f"\n❌ Errore salvataggio: {e}")
    
    print(f"\n✅ Test completato!")
    if use_full_song:
        print("💡 Hai testato la canzone completa - risultati più accurati!")
    else:
        print("💡 Hai testato solo 30s - per risultati migliori usa --full-song")

if __name__ == "__main__":
    main()