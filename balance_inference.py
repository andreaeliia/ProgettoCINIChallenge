#!/usr/bin/env python3
"""
Sistema di Inferenza per Classificazione Generi Musicali Bilanciati
Supporta predizione su singoli file MP3 con analisi completa
"""

import os
import numpy as np
import librosa
import tensorflow as tf
import json
import argparse
from typing import Dict, List, Tuple, Any
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ==================== CONFIGURAZIONE ====================
class InferenceConfig:
    # Audio settings (devono corrispondere al training)
    SAMPLE_RATE = 22050
    DURATION = 30.0
    N_MELS = 128
    HOP_LENGTH = 512
    N_FFT = 2048
    NUM_SEGMENTS = 10
    
    # Inference settings
    CONFIDENCE_THRESHOLD = 0.1
    TOP_K_RESULTS = 5
    SEGMENT_OVERLAP = 0.5  # Per predizioni più robuste
    
    # Output settings
    SAVE_ANALYSIS = True
    GENERATE_PLOTS = True

config = InferenceConfig()

# ==================== ESTRATTORE FEATURES PER INFERENZA ====================
class InferenceFeatureExtractor:
    """
    Estrattore features per inferenza con segmentazione avanzata
    """
    def __init__(self):
        self.sample_rate = config.SAMPLE_RATE
        self.duration = config.DURATION
        self.n_mels = config.N_MELS
        self.hop_length = config.HOP_LENGTH
        self.n_fft = config.N_FFT
        self.num_segments = config.NUM_SEGMENTS
    
    def extract_features_robust(self, file_path: str, overlap_ratio: float = 0.5) -> np.ndarray:
        """
        Estrae features con sovrapposizione per predizioni più robuste
        """
        try:
            print(f"🎵 Caricamento audio: {os.path.basename(file_path)}")
            
            # Carica audio
            audio, sr = librosa.load(file_path, sr=self.sample_rate)
            total_duration = len(audio) / sr
            
            print(f"   📊 Durata: {total_duration:.1f}s, Sample rate: {sr}Hz")
            
            # Calcola parametri segmentazione
            segment_samples = int(self.duration * self.sample_rate / self.num_segments)
            hop_samples = int(segment_samples * (1 - overlap_ratio))
            
            features = []
            segment_timestamps = []
            
            # Estrai segmenti con sovrapposizione
            for start in range(0, len(audio) - segment_samples + 1, hop_samples):
                end = start + segment_samples
                segment = audio[start:end]
                timestamp = start / sr
                
                # Estrai mel-spectrogram
                mel_spec = librosa.feature.melspectrogram(
                    y=segment,
                    sr=sr,
                    n_mels=self.n_mels,
                    hop_length=self.hop_length,
                    n_fft=self.n_fft
                )
                
                # Converti in log scale
                log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
                
                # Normalizzazione robusta
                mean = np.mean(log_mel_spec)
                std = np.std(log_mel_spec)
                if std > 0:
                    log_mel_spec = (log_mel_spec - mean) / std
                else:
                    log_mel_spec = log_mel_spec - mean
                
                features.append(log_mel_spec)
                segment_timestamps.append(timestamp)
            
            # Se audio troppo corto, crea almeno un segmento
            if len(features) == 0:
                print("   ⚠️ Audio troppo corto, padding...")
                padded_audio = np.pad(audio, (0, segment_samples - len(audio)), 'constant')
                
                mel_spec = librosa.feature.melspectrogram(
                    y=padded_audio[:segment_samples],
                    sr=sr,
                    n_mels=self.n_mels,
                    hop_length=self.hop_length,
                    n_fft=self.n_fft
                )
                
                log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
                mean = np.mean(log_mel_spec)
                std = np.std(log_mel_spec)
                if std > 0:
                    log_mel_spec = (log_mel_spec - mean) / std
                
                features.append(log_mel_spec)
                segment_timestamps.append(0.0)
            
            features_array = np.array(features, dtype=np.float32)
            print(f"   ✅ Estratti {len(features)} segmenti: {features_array.shape}")
            
            return features_array, segment_timestamps
            
        except Exception as e:
            print(f"❌ Errore nell'estrazione features: {e}")
            return None, None

# ==================== ANALIZZATORE PREDIZIONI ====================
class PredictionAnalyzer:
    """
    Analizza e aggrega predizioni da più segmenti
    """
    def __init__(self, mappings: Dict, class_weights: Dict = None):
        self.mappings = mappings
        self.class_weights = class_weights
    
    def analyze_hierarchical_predictions(self, predictions: List[np.ndarray], 
                                       timestamps: List[float]) -> Dict[str, Any]:
        """
        Analizza predizioni gerarchiche con multiple strategie di aggregazione
        """
        top_pred, medium_pred, all_pred = predictions
        
        analysis = {
            'segment_analysis': self._analyze_segments(top_pred, medium_pred, all_pred, timestamps),
            'aggregated_results': self._aggregate_predictions(top_pred, medium_pred, all_pred),
            'confidence_analysis': self._analyze_confidence(top_pred, medium_pred, all_pred),
            'temporal_analysis': self._analyze_temporal_patterns(top_pred, medium_pred, timestamps)
        }
        
        return analysis
    
    def _analyze_segments(self, top_pred: np.ndarray, medium_pred: np.ndarray, 
                         all_pred: np.ndarray, timestamps: List[float]) -> List[Dict]:
        """Analizza predizioni per ogni singolo segmento"""
        segment_results = []
        
        for i, (timestamp, top_probs, med_probs, all_probs) in enumerate(
            zip(timestamps, top_pred, medium_pred, all_pred)):
            
            # Top genre per questo segmento
            top_idx = np.argmax(top_probs)
            top_genre = self.mappings['top_genres'][top_idx]
            top_confidence = top_probs[top_idx]
            
            # Medium genre per questo segmento
            med_idx = np.argmax(med_probs)
            med_genre = self.mappings['medium_genres'][med_idx]
            med_confidence = med_probs[med_idx]
            
            # Top-K all genres per questo segmento
            all_top_indices = np.argsort(all_probs)[::-1][:config.TOP_K_RESULTS]
            all_genres = [(self.mappings['all_genres'][idx], all_probs[idx]) 
                         for idx in all_top_indices if all_probs[idx] > config.CONFIDENCE_THRESHOLD]
            
            segment_results.append({
                'segment_id': i,
                'timestamp': timestamp,
                'top_genre': {'name': top_genre, 'confidence': float(top_confidence)},
                'medium_genre': {'name': med_genre, 'confidence': float(med_confidence)},
                'all_genres': [{'name': name, 'confidence': float(conf)} for name, conf in all_genres],
                'entropy': {
                    'top': float(self._calculate_entropy(top_probs)),
                    'medium': float(self._calculate_entropy(med_probs))
                }
            })
        
        return segment_results
    
    def _aggregate_predictions(self, top_pred: np.ndarray, medium_pred: np.ndarray, 
                             all_pred: np.ndarray) -> Dict[str, Any]:
        """Aggrega predizioni con multiple strategie"""
        
        # === TOP LEVEL AGGREGATION ===
        
        # Voting (most frequent)
        top_votes = [np.argmax(pred) for pred in top_pred]
        top_vote_counts = Counter(top_votes)
        most_voted_idx = top_vote_counts.most_common(1)[0][0]
        most_voted_genre = self.mappings['top_genres'][most_voted_idx]
        
        # Average confidence
        top_avg_probs = np.mean(top_pred, axis=0)
        top_avg_idx = np.argmax(top_avg_probs)
        top_avg_genre = self.mappings['top_genres'][top_avg_idx]
        
        # Weighted average (se class weights disponibili)
        if self.class_weights and 'top' in self.class_weights:
            weights = np.array([self.class_weights['top'].get(genre, 1.0) 
                              for genre in self.mappings['top_genres']])
            top_weighted_probs = np.average(top_pred, axis=0, weights=weights)
            top_weighted_idx = np.argmax(top_weighted_probs)
            top_weighted_genre = self.mappings['top_genres'][top_weighted_idx]
        else:
            top_weighted_genre = top_avg_genre
            top_weighted_probs = top_avg_probs
        
        # === MEDIUM LEVEL AGGREGATION ===
        
        medium_votes = [np.argmax(pred) for pred in medium_pred]
        medium_vote_counts = Counter(medium_votes)
        med_most_voted_idx = medium_vote_counts.most_common(1)[0][0]
        med_most_voted_genre = self.mappings['medium_genres'][med_most_voted_idx]
        
        medium_avg_probs = np.mean(medium_pred, axis=0)
        med_avg_idx = np.argmax(medium_avg_probs)
        med_avg_genre = self.mappings['medium_genres'][med_avg_idx]
        
        # === ALL GENRES AGGREGATION ===
        
        all_avg_probs = np.mean(all_pred, axis=0)
        all_top_indices = np.argsort(all_avg_probs)[::-1][:10]  # Top 10
        all_top_genres = [(self.mappings['all_genres'][idx], all_avg_probs[idx]) 
                         for idx in all_top_indices]
        
        return {
            'top_genre': {
                'voting': {'name': most_voted_genre, 'votes': top_vote_counts[most_voted_idx], 
                          'total_segments': len(top_pred)},
                'average_confidence': {'name': top_avg_genre, 'confidence': float(top_avg_probs[top_avg_idx])},
                'weighted_average': {'name': top_weighted_genre, 'confidence': float(top_weighted_probs[top_weighted_idx])}
            },
            'medium_genre': {
                'voting': {'name': med_most_voted_genre, 'votes': medium_vote_counts[med_most_voted_idx]},
                'average_confidence': {'name': med_avg_genre, 'confidence': float(medium_avg_probs[med_avg_idx])}
            },
            'all_genres': {
                'top_genres': [{'name': name, 'confidence': float(conf)} for name, conf in all_top_genres],
                'threshold_genres': [{'name': name, 'confidence': float(conf)} 
                                   for name, conf in all_top_genres 
                                   if conf > config.CONFIDENCE_THRESHOLD]
            }
        }
    
    def _analyze_confidence(self, top_pred: np.ndarray, medium_pred: np.ndarray, 
                          all_pred: np.ndarray) -> Dict[str, Any]:
        """Analizza distribuzioni di confidenza"""
        
        # Calcola entropie per ogni segmento
        top_entropies = [self._calculate_entropy(pred) for pred in top_pred]
        med_entropies = [self._calculate_entropy(pred) for pred in medium_pred]
        
        # Calcola max confidence per ogni segmento
        top_max_confs = [np.max(pred) for pred in top_pred]
        med_max_confs = [np.max(pred) for pred in medium_pred]
        
        return {
            'entropy_stats': {
                'top': {
                    'mean': float(np.mean(top_entropies)),
                    'std': float(np.std(top_entropies)),
                    'min': float(np.min(top_entropies)),
                    'max': float(np.max(top_entropies))
                },
                'medium': {
                    'mean': float(np.mean(med_entropies)),
                    'std': float(np.std(med_entropies)),
                    'min': float(np.min(med_entropies)),
                    'max': float(np.max(med_entropies))
                }
            },
            'confidence_stats': {
                'top': {
                    'mean': float(np.mean(top_max_confs)),
                    'std': float(np.std(top_max_confs)),
                    'min': float(np.min(top_max_confs)),
                    'max': float(np.max(top_max_confs))
                },
                'medium': {
                    'mean': float(np.mean(med_max_confs)),
                    'std': float(np.std(med_max_confs)),
                    'min': float(np.min(med_max_confs)),
                    'max': float(np.max(med_max_confs))
                }
            },
            'prediction_quality': {
                'top_consistent': np.std(top_max_confs) < 0.1,
                'medium_consistent': np.std(med_max_confs) < 0.1,
                'high_entropy_warning': np.mean(top_entropies) > 2.0
            }
        }
    
    def _analyze_temporal_patterns(self, top_pred: np.ndarray, medium_pred: np.ndarray, 
                                 timestamps: List[float]) -> Dict[str, Any]:
        """Analizza pattern temporali nelle predizioni"""
        
        # Analizza cambiamenti di predizione nel tempo
        top_changes = 0
        med_changes = 0
        
        prev_top = np.argmax(top_pred[0])
        prev_med = np.argmax(medium_pred[0])
        
        for i in range(1, len(top_pred)):
            curr_top = np.argmax(top_pred[i])
            curr_med = np.argmax(medium_pred[i])
            
            if curr_top != prev_top:
                top_changes += 1
            if curr_med != prev_med:
                med_changes += 1
                
            prev_top = curr_top
            prev_med = curr_med
        
        # Calcola stabilità
        top_stability = 1.0 - (top_changes / (len(top_pred) - 1))
        med_stability = 1.0 - (med_changes / (len(medium_pred) - 1))
        
        return {
            'stability': {
                'top_genre': float(top_stability),
                'medium_genre': float(med_stability)
            },
            'changes': {
                'top_genre': top_changes,
                'medium_genre': med_changes
            },
            'duration_analysis': {
                'total_duration': timestamps[-1] + (timestamps[1] - timestamps[0]) if len(timestamps) > 1 else 0,
                'segments_analyzed': len(timestamps)
            }
        }
    
    def _calculate_entropy(self, probabilities: np.ndarray) -> float:
        """Calcola entropia di una distribuzione di probabilità"""
        # Evita log(0)
        probs = np.clip(probabilities, 1e-8, 1.0)
        return -np.sum(probs * np.log(probs))

# ==================== SISTEMA DI INFERENZA COMPLETO ====================
class MusicGenreInference:
    """
    Sistema completo di inferenza per classificazione generi musicali
    """
    def __init__(self, model_path: str, config_path: str):
        self.model_path = model_path
        self.config_path = config_path
        self.model = None
        self.config_data = None
        self.feature_extractor = InferenceFeatureExtractor()
        self.analyzer = None
        
        self.load_model_and_config()
    
    def load_model_and_config(self):
        """Carica modello e configurazione"""
        print("🔄 Caricamento modello e configurazione...")
        
        try:
            # Carica modello
            self.model = tf.keras.models.load_model(
                self.model_path,
                custom_objects={
                    'FocalLoss': lambda **kwargs: tf.keras.losses.CategoricalCrossentropy(),
                    'WeightedBinaryCrossEntropy': lambda **kwargs: tf.keras.losses.BinaryCrossentropy()
                },
                compile=False
            )
            print("   ✅ Modello caricato")
            
            # Carica configurazione
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self.config_data = json.load(f)
            print("   ✅ Configurazione caricata")
            
            # Inizializza analyzer
            self.analyzer = PredictionAnalyzer(
                self.config_data['mappings'],
                self.config_data.get('class_weights')
            )
            
            print(f"   📊 Generi disponibili:")
            print(f"      Top: {len(self.config_data['mappings']['top_genres'])}")
            print(f"      Medium: {len(self.config_data['mappings']['medium_genres'])}")
            print(f"      All: {len(self.config_data['mappings']['all_genres'])}")
            
        except Exception as e:
            print(f"❌ Errore nel caricamento: {e}")
            raise e
    
    def predict_single_file(self, audio_path: str, overlap_ratio: float = 0.5) -> Dict[str, Any]:
        """
        Predice generi per un singolo file audio
        """
        print(f"\n🎵 ANALISI: {os.path.basename(audio_path)}")
        print("=" * 60)
        
        # Estrai features
        features, timestamps = self.feature_extractor.extract_features_robust(
            audio_path, overlap_ratio
        )
        
        if features is None:
            return {'error': 'Impossibile estrarre features dal file audio'}
        
        # Aggiungi dimensione canale se necessaria
        if len(features.shape) == 3:
            features = np.expand_dims(features, axis=-1)
        
        print(f"🔄 Esecuzione predizione su {len(features)} segmenti...")
        
        # Esegui predizione
        try:
            predictions = self.model.predict(features, verbose=0)
            print("   ✅ Predizione completata")
        except Exception as e:
            print(f"   ❌ Errore nella predizione: {e}")
            return {'error': f'Errore nella predizione: {e}'}
        
        # Analizza risultati
        analysis = self.analyzer.analyze_hierarchical_predictions(predictions, timestamps)
        
        # Aggiungi metadati
        analysis['metadata'] = {
            'file_path': audio_path,
            'file_name': os.path.basename(audio_path),
            'segments_analyzed': len(features),
            'overlap_ratio': overlap_ratio,
            'total_duration': timestamps[-1] + (timestamps[1] - timestamps[0]) if len(timestamps) > 1 else 0
        }
        
        # Stampa risultati principali
        self._print_results(analysis)
        
        return analysis
    
    def _print_results(self, analysis: Dict[str, Any]):
        """Stampa risultati principali in formato leggibile"""
        print("\n🎯 RISULTATI PRINCIPALI:")
        print("-" * 40)
        
        agg = analysis['aggregated_results']
        
        # Top Genre
        top_voting = agg['top_genre']['voting']
        top_confidence = agg['top_genre']['average_confidence']
        
        print(f"🏆 GENERE PRINCIPALE:")
        print(f"   Voting: {top_voting['name']} ({top_voting['votes']}/{top_voting['total_segments']} segmenti)")
        print(f"   Confidence: {top_confidence['name']} ({top_confidence['confidence']:.3f})")
        
        # Medium Genre
        med_voting = agg['medium_genre']['voting']
        med_confidence = agg['medium_genre']['average_confidence']
        
        print(f"\n🎯 GENERE SPECIFICO:")
        print(f"   Voting: {med_voting['name']} ({med_voting['votes']} segmenti)")
        print(f"   Confidence: {med_confidence['name']} ({med_confidence['confidence']:.3f})")
        
        # All Genres Top 5
        print(f"\n🎵 TOP 5 GENERI (Multi-label):")
        for i, genre in enumerate(agg['all_genres']['top_genres'][:5], 1):
            print(f"   {i}. {genre['name']}: {genre['confidence']:.3f}")
        
        # Qualità predizione
        quality = analysis['confidence_analysis']['prediction_quality']
        print(f"\n📊 QUALITÀ PREDIZIONE:")
        print(f"   Consistenza Top: {'✅' if quality['top_consistent'] else '⚠️'}")
        print(f"   Consistenza Medium: {'✅' if quality['medium_consistent'] else '⚠️'}")
        if quality['high_entropy_warning']:
            print(f"   ⚠️ Alta incertezza rilevata")
        
        # Stabilità temporale
        stability = analysis['temporal_analysis']['stability']
        print(f"\n⏱️ STABILITÀ TEMPORALE:")
        print(f"   Top Genre: {stability['top_genre']:.2%}")
        print(f"   Medium Genre: {stability['medium_genre']:.2%}")
    
    def batch_predict(self, audio_folder: str, file_extensions: List[str] = ['.mp3', '.wav', '.flac']) -> Dict[str, Any]:
        """
        Predice generi per tutti i file in una cartella
        """
        print(f"📁 BATCH PREDICTION: {audio_folder}")
        print("=" * 60)
        
        # Trova file audio
        audio_files = []
        for ext in file_extensions:
            audio_files.extend(Path(audio_folder).glob(f'*{ext}'))
            audio_files.extend(Path(audio_folder).glob(f'*{ext.upper()}'))
        
        print(f"📊 Trovati {len(audio_files)} file audio")
        
        results = {}
        errors = []
        
        for i, audio_file in enumerate(audio_files, 1):
            print(f"\n[{i}/{len(audio_files)}] Processando: {audio_file.name}")
            
            try:
                result = self.predict_single_file(str(audio_file))
                if 'error' not in result:
                    results[audio_file.name] = result
                else:
                    errors.append({'file': audio_file.name, 'error': result['error']})
            except Exception as e:
                errors.append({'file': audio_file.name, 'error': str(e)})
        
        # Statistiche batch
        batch_stats = self._analyze_batch_results(results)
        
        return {
            'results': results,
            'errors': errors,
            'statistics': batch_stats,
            'total_files': len(audio_files),
            'successful': len(results),
            'failed': len(errors)
        }
    
    def _analyze_batch_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analizza risultati batch per statistiche"""
        if not results:
            return {}
        
        # Raccogli generi predetti
        top_genres = []
        medium_genres = []
        
        for result in results.values():
            agg = result['aggregated_results']
            top_genres.append(agg['top_genre']['voting']['name'])
            medium_genres.append(agg['medium_genre']['voting']['name'])
        
        # Conta distribuzioni
        top_counts = Counter(top_genres)
        medium_counts = Counter(medium_genres)
        
        return {
            'top_genre_distribution': dict(top_counts),
            'medium_genre_distribution': dict(medium_counts),
            'most_common_top': top_counts.most_common(5),
            'most_common_medium': medium_counts.most_common(5),
            'unique_top_genres': len(top_counts),
            'unique_medium_genres': len(medium_counts)
        }
    
    def save_results(self, results: Dict[str, Any], output_path: str):
        """Salva risultati in formato JSON"""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False, default=str)
            print(f"💾 Risultati salvati in: {output_path}")
        except Exception as e:
            print(f"❌ Errore nel salvataggio: {e}")

# ==================== MAIN INTERFACE ====================
def main():
    """Interfaccia principale per inferenza"""
    parser = argparse.ArgumentParser(description='Inferenza Classificazione Generi Musicali Bilanciata')
    parser.add_argument('--model', required=True, help='Path al modello .h5')
    parser.add_argument('--config', required=True, help='Path al file di configurazione JSON')
    parser.add_argument('--file', help='Singolo file audio da analizzare')
    parser.add_argument('--folder', help='Cartella con file audio per batch prediction')
    parser.add_argument('--overlap', type=float, default=0.5, help='Overlap ratio tra segmenti (default: 0.5)')
    parser.add_argument('--output', help='File di output per risultati JSON')
    parser.add_argument('--threshold', type=float, default=0.1, help='Soglia di confidenza (default: 0.1)')
    
    args = parser.parse_args()
    
    # Aggiorna configurazione
    config.CONFIDENCE_THRESHOLD = args.threshold
    
    print("🎧 SISTEMA DI INFERENZA GENERI MUSICALI BILANCIATI")
    print("=" * 60)
    
    # Verifica file
    if not os.path.exists(args.model):
        print(f"❌ Modello non trovato: {args.model}")
        return
    
    if not os.path.exists(args.config):
        print(f"❌ Configurazione non trovata: {args.config}")
        return
    
    # Inizializza sistema
    try:
        inference_system = MusicGenreInference(args.model, args.config)
    except Exception as e:
        print(f"❌ Errore nell'inizializzazione: {e}")
        return
    
    # Esegui predizione
    results = None
    
    if args.file:
        # Single file prediction
        if not os.path.exists(args.file):
            print(f"❌ File audio non trovato: {args.file}")
            return
        
        results = inference_system.predict_single_file(args.file, args.overlap)
        
    elif args.folder:
        # Batch prediction
        if not os.path.exists(args.folder):
            print(f"❌ Cartella non trovata: {args.folder}")
            return
        
        results = inference_system.batch_predict(args.folder)
        
        # Stampa statistiche batch
        if results and 'statistics' in results:
            print(f"\n📊 STATISTICHE BATCH:")
            print(f"   File processati: {results['successful']}/{results['total_files']}")
            print(f"   Errori: {results['failed']}")
            
            if results['statistics']:
                print(f"   Generi top più comuni:")
                for genre, count in results['statistics']['most_common_top'][:3]:
                    print(f"      {genre}: {count} file")
    
    else:
        print("❌ Specifica --file per singolo file o --folder per batch prediction")
        return
    
    # Salva risultati
    if results and args.output:
        inference_system.save_results(results, args.output)
    
    print("\n✅ Inferenza completata!")

if __name__ == "__main__":
    main()