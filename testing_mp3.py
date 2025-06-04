#!/usr/bin/env python3
"""
MP3 Genre Classification Test Script - DIAGNOSTIC VERSION
Added debugging and fixed metric calculation issues
"""

import os
import numpy as np
import librosa
import tensorflow as tf
import json
import argparse
from typing import List, Tuple, Dict, Any
from collections import Counter
import matplotlib.pyplot as plt
from scipy import stats

def extract_features_fixed(audio_path: str, segment_duration: float = 30.0, 
                          overlap_ratio: float = 0.5,
                          target_shape: Tuple[int, int] = (128, 130)) -> np.ndarray:
    """
    Enhanced audio feature extraction with overlapping segments for full song analysis
    """
    try:
        # Load audio
        y, sr = librosa.load(audio_path, sr=22050)
        total_duration = len(y) / sr
        print(f"📊 Audio loaded: {len(y)} samples, {sr}Hz, {total_duration:.2f}s")
        
        segments = []
        segment_samples = int(segment_duration * sr)
        hop_samples = int(segment_samples * (1 - overlap_ratio))
        
        print(f"🎵 Creating segments: {segment_duration}s each, {overlap_ratio*100:.0f}% overlap")
        
        # Create overlapping segments
        for start in range(0, len(y) - segment_samples + 1, hop_samples):
            end = start + segment_samples
            segment = y[start:end]
            
            # Extract mel spectrogram with exact parameters from training
            mel_spec = librosa.feature.melspectrogram(
                y=segment,
                sr=sr,
                n_mels=target_shape[0],  # 128 mel bins
                n_fft=2048,
                hop_length=512,
                win_length=2048
            )
            
            # Convert to log scale
            log_mel = librosa.power_to_db(mel_spec, ref=np.max)
            
            # Ensure correct time dimension (width)
            if log_mel.shape[1] > target_shape[1]:
                log_mel = log_mel[:, :target_shape[1]]
            elif log_mel.shape[1] < target_shape[1]:
                pad_width = target_shape[1] - log_mel.shape[1]
                log_mel = np.pad(log_mel, ((0, 0), (0, pad_width)), 'constant')
            
            # FIXED: Better normalization
            log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-8)
            
            # Add channel dimension: (128, 130, 1)
            segment_features = log_mel[..., np.newaxis]
            segments.append(segment_features)
            
            # DEBUG: Print statistics for first segment
            if len(segments) == 1:
                print(f"🔍 First segment stats - Min: {segment_features.min():.3f}, "
                      f"Max: {segment_features.max():.3f}, Mean: {segment_features.mean():.3f}")
        
        # Handle short songs
        if len(segments) == 0:
            print(f"⚠️ Song shorter than {segment_duration}s, padding...")
            segment = np.pad(y, (0, max(0, segment_samples - len(y))), 'constant')
            
            mel_spec = librosa.feature.melspectrogram(
                y=segment,
                sr=sr,
                n_mels=target_shape[0],
                n_fft=2048,
                hop_length=512,
                win_length=2048
            )
            
            log_mel = librosa.power_to_db(mel_spec, ref=np.max)
            
            if log_mel.shape[1] > target_shape[1]:
                log_mel = log_mel[:, :target_shape[1]]
            elif log_mel.shape[1] < target_shape[1]:
                pad_width = target_shape[1] - log_mel.shape[1]
                log_mel = np.pad(log_mel, ((0, 0), (0, pad_width)), 'constant')
            
            log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-8)
            segment_features = log_mel[..., np.newaxis]
            segments.append(segment_features)
        
        result = np.array(segments, dtype=np.float32)
        print(f"🎵 Extracted {len(segments)} segments with shape: {result.shape}")
        
        # DEBUG: Check for identical segments
        if len(segments) > 1:
            segment_similarities = []
            for i in range(1, len(segments)):
                similarity = np.corrcoef(segments[0].flatten(), segments[i].flatten())[0,1]
                segment_similarities.append(similarity)
            avg_similarity = np.mean(segment_similarities)
            print(f"🔍 Average segment similarity: {avg_similarity:.3f} (1.0 = identical)")
            if avg_similarity > 0.99:
                print("⚠️ WARNING: Segments are too similar - might indicate processing error")
        
        return result
        
    except Exception as e:
        print(f"❌ Error in feature extraction: {str(e)}")
        return np.array([])

def create_hierarchical_model(input_shape: Tuple[int, int, int] = (128, 130, 1), 
                             num_top: int = 10, 
                             num_medium: int = 25, 
                             num_all: int = 163) -> tf.keras.Model:
    """Create the exact HierarchicalGenreModel architecture"""
    inputs = tf.keras.layers.Input(shape=input_shape, name='audio_input', dtype='float32')
    
    # First conv block
    x = tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu')(inputs)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.2)(x)

    # Second conv block
    x = tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    # Global pooling and shared dense layer
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    shared = tf.keras.layers.Dense(512, activation='relu')(x)

    # Output branches
    top_output = tf.keras.layers.Dense(num_top, activation='softmax', name='top_genres', dtype='float32')(shared)
    med_output = tf.keras.layers.Dense(num_medium, activation='softmax', name='medium_genres', dtype='float32')(shared)
    all_output = tf.keras.layers.Dense(num_all, activation='sigmoid', name='all_genres', dtype='float32')(shared)

    model = tf.keras.Model(inputs, [top_output, med_output, all_output])
    return model

def load_model_alternative(model_path: str, mappings: Dict = None) -> tf.keras.Model:
    """Alternative model loading method that recreates the exact architecture"""
    try:
        print("🔧 Creating model with exact HierarchicalGenreModel architecture...")
        
        # Get dimensions from mappings if available
        num_top = len(mappings.get('top_genres', [])) if mappings else 10
        num_medium = len(mappings.get('medium_genres', [])) if mappings else 25
        num_all = len(mappings.get('all_genres', [])) if mappings else 163
        
        print(f"📊 Model dimensions: Top={num_top}, Medium={num_medium}, All={num_all}")
        
        # Create the exact model architecture
        model = create_hierarchical_model(
            input_shape=(128, 130, 1),
            num_top=num_top,
            num_medium=num_medium,
            num_all=num_all
        )
        
        # Try to load weights
        try:
            model.load_weights(model_path)
            print("✅ Weights loaded successfully")
            return model
        except Exception as e:
            print(f"⚠️ Direct weight loading failed: {e}")
            try:
                # Try loading weights by name, skipping mismatches
                model.load_weights(model_path, by_name=True, skip_mismatch=True)
                print("✅ Weights loaded by name (some layers may be skipped)")
                return model
            except Exception as e2:
                print(f"❌ Weight loading failed: {e2}")
                raise Exception("Could not load model weights")
    
    except Exception as e:
        print(f"❌ Alternative loading failed: {str(e)}")
        raise e

def load_model_and_mappings(model_path: str, mappings_path: str) -> Tuple[Any, Dict]:
    """Load the trained model and genre mappings"""
    try:
        print("🔄 Loading model and mappings...")
        
        # Load mappings first
        with open(mappings_path, 'r') as f:
            mappings = json.load(f)
        
        print(f"✅ Mappings loaded: {len(mappings.get('top_genres', []))} top genres, "
              f"{len(mappings.get('medium_genres', []))} medium genres, "
              f"{len(mappings.get('all_genres', []))} all genres")
        
        # Try standard loading first
        try:
            model = tf.keras.models.load_model(model_path)
            print("✅ Model loaded successfully")
        except Exception as e:
            print(f"⚠️ Standard load failed: {e}")
            try:
                model = tf.keras.models.load_model(model_path, compile=False)
                print("✅ Model loaded without compilation")
            except Exception as e2:
                print(f"⚠️ Non-compiled load failed: {e2}")
                model = load_model_alternative(model_path, mappings)
        
        return model, mappings
        
    except Exception as e:
        print(f"❌ Error loading model/mappings: {str(e)}")
        return None, None

def predict_genre(model: Any, features: np.ndarray, mappings: Dict) -> Dict[str, Any]:
    """Make predictions using the loaded model - FIXED VERSION"""
    try:
        print(f"🔄 Making predictions on {len(features)} segments...")
        
        # DEBUG: Check input statistics
        print(f"🔍 Input features stats - Min: {features.min():.3f}, Max: {features.max():.3f}, "
              f"Mean: {features.mean():.3f}, Std: {features.std():.3f}")
        
        predictions = model.predict(features, verbose=0)
        
        # Handle hierarchical outputs: [top_genres, medium_genres, all_genres]
        top_pred, medium_pred, all_pred = predictions
        
        # DEBUG: Print raw prediction statistics
        print(f"🔍 Top predictions stats - Min: {top_pred.min():.3f}, Max: {top_pred.max():.3f}")
        print(f"🔍 Medium predictions stats - Min: {medium_pred.min():.3f}, Max: {medium_pred.max():.3f}")  
        print(f"🔍 All predictions stats - Min: {all_pred.min():.3f}, Max: {all_pred.max():.3f}")
        
        # FIXED: Check for flat distributions (sign of untrained model)
        for i, pred in enumerate(top_pred[:3]):  # Check first 3 segments
            entropy = -np.sum(pred * np.log(pred + 1e-8))
            max_entropy = np.log(len(pred))  # Maximum possible entropy
            normalized_entropy = entropy / max_entropy
            print(f"🔍 Segment {i+1} entropy: {normalized_entropy:.3f} (1.0 = uniform distribution)")
            if normalized_entropy > 0.95:
                print(f"⚠️ WARNING: Segment {i+1} has very flat distribution - model might be untrained")
        
        # Get predictions for each segment
        top_idx = np.argmax(top_pred, axis=1)
        medium_idx = np.argmax(medium_pred, axis=1)
        
        # FIXED: Get actual max probabilities, not indices
        top_max_probs = np.max(top_pred, axis=1)
        medium_max_probs = np.max(medium_pred, axis=1)
        
        # For multi-label (all_genres), use threshold
        all_threshold = 0.5
        all_pred_binary = (all_pred > all_threshold).astype(int)
        
        # FIXED: Add debug info for predictions
        print(f"🔍 Top genre predictions:")
        for i in range(min(5, len(top_idx))):  # Show first 5
            genre_name = mappings.get('top_genres', [])[top_idx[i]] if top_idx[i] < len(mappings.get('top_genres', [])) else f'Unknown_{top_idx[i]}'
            print(f"   Segment {i+1}: {genre_name} (prob: {top_max_probs[i]:.3f}, idx: {top_idx[i]})")
        
        results = {
            'top_genre': {
                'indices': top_idx.tolist(),
                'probabilities': top_max_probs.tolist(),  # FIXED: Use actual max probabilities
                'all_probabilities': top_pred.tolist(),
                'names': [mappings.get('top_genres', [])[i] if i < len(mappings.get('top_genres', [])) else f'Unknown_{i}' for i in top_idx]
            },
            'medium_genre': {
                'indices': medium_idx.tolist(),
                'probabilities': medium_max_probs.tolist(),  # FIXED: Use actual max probabilities
                'all_probabilities': medium_pred.tolist(),
                'names': [mappings.get('medium_genres', [])[i] if i < len(mappings.get('medium_genres', [])) else f'Unknown_{i}' for i in medium_idx]
            },
            'all_genres': {
                'predictions': all_pred_binary.tolist(),
                'probabilities': all_pred.tolist(),
                'names': mappings.get('all_genres', [])
            }
        }
        
        return results
        
    except Exception as e:
        print(f"❌ Prediction error: {str(e)}")
        return {}

def analyze_full_song(results: Dict[str, Any], num_segments: int, audio_path: str) -> Dict[str, Any]:
    """Advanced analysis of full song using multiple aggregation methods - FIXED VERSION"""
    if not results:
        print("❌ No results to analyze")
        return {}
    
    print(f"\n📊 ANALISI COMPLETA CANZONE: {os.path.basename(audio_path)}")
    print("=" * 80)
    
    analysis = {}
    
    # TOP LEVEL GENRES - FIXED ANALYSIS
    print("🎵 TOP LEVEL GENRES:")
    top_genres = results['top_genre']['names']
    top_probs = results['top_genre']['probabilities']
    
    # DEBUG: Check for suspicious identical values
    unique_probs = set([round(p, 6) for p in top_probs])
    if len(unique_probs) == 1:
        print(f"⚠️ WARNING: All probabilities are identical ({list(unique_probs)[0]:.6f}) - likely model issue")
    
    # Voting method (most frequent)
    genre_votes = Counter(top_genres)
    most_voted = genre_votes.most_common(1)[0]
    
    # Confidence averaging method
    genre_confidences = {}
    for genre, prob in zip(top_genres, top_probs):
        if genre not in genre_confidences:
            genre_confidences[genre] = []
        genre_confidences[genre].append(prob)
    
    avg_confidences = {genre: np.mean(probs) for genre, probs in genre_confidences.items()}
    most_confident = max(avg_confidences, key=avg_confidences.get)
    
    # FIXED: Weighted voting (vote count * average confidence)
    weighted_scores = {}
    for genre, probs in genre_confidences.items():
        vote_weight = len(probs) / num_segments  # Normalize by total segments
        confidence_weight = np.mean(probs)
        weighted_scores[genre] = vote_weight * confidence_weight
    
    most_weighted = max(weighted_scores, key=weighted_scores.get)
    
    print(f"   📊 Analisi {num_segments} segmenti:")
    for i, (genre, prob) in enumerate(zip(top_genres, top_probs)):
        print(f"      Segmento {i+1}: {genre} ({prob:.4f})")  # More decimal places
    
    print(f"\n   🗳️  VOTING: {most_voted[0]} ({most_voted[1]}/{num_segments} segmenti)")
    print(f"   🎯 CONFIDENCE: {most_confident} (avg: {avg_confidences[most_confident]:.4f})")
    print(f"   ⚖️  WEIGHTED: {most_weighted} (score: {weighted_scores[most_weighted]:.4f})")
    
    # FIXED: Add distribution analysis
    print(f"   📈 DISTRIBUZIONE:")
    for genre, count in genre_votes.most_common():
        percentage = (count / num_segments) * 100
        avg_conf = avg_confidences[genre]
        print(f"      {genre}: {count} segmenti ({percentage:.1f}%), conf avg: {avg_conf:.4f}")
    
    analysis['top_genre'] = {
        'voting_winner': most_voted[0],
        'voting_count': most_voted[1],
        'confidence_winner': most_confident,
        'confidence_score': avg_confidences[most_confident],
        'weighted_winner': most_weighted,
        'weighted_score': weighted_scores[most_weighted],
        'all_votes': dict(genre_votes),
        'all_confidences': avg_confidences,
        'distribution_warning': len(unique_probs) == 1
    }
    
    # MEDIUM LEVEL GENRES - SAME FIXES
    print(f"\n🎵 MEDIUM LEVEL GENRES:")
    medium_genres = results['medium_genre']['names']
    medium_probs = results['medium_genre']['probabilities']
    
    medium_votes = Counter(medium_genres)
    medium_most_voted = medium_votes.most_common(1)[0]
    
    medium_confidences = {}
    for genre, prob in zip(medium_genres, medium_probs):
        if genre not in medium_confidences:
            medium_confidences[genre] = []
        medium_confidences[genre].append(prob)
    
    medium_avg_confidences = {genre: np.mean(probs) for genre, probs in medium_confidences.items()}
    medium_most_confident = max(medium_avg_confidences, key=medium_avg_confidences.get)
    
    print(f"   🗳️  VOTING: {medium_most_voted[0]} ({medium_most_voted[1]}/{num_segments} segmenti)")
    print(f"   🎯 CONFIDENCE: {medium_most_confident} (avg: {medium_avg_confidences[medium_most_confident]:.4f})")
    
    analysis['medium_genre'] = {
        'voting_winner': medium_most_voted[0],
        'voting_count': medium_most_voted[1],
        'confidence_winner': medium_most_confident,
        'confidence_score': medium_avg_confidences[medium_most_confident],
        'all_votes': dict(medium_votes),
        'all_confidences': medium_avg_confidences
    }
    
    # ALL GENRES (Multi-label analysis) - FIXED
    print(f"\n🎵 ALL GENRES (Multi-label):")
    all_probs = np.array(results['all_genres']['probabilities'])
    all_names = results['all_genres']['names']
    
    # Average probabilities across segments
    avg_probs = np.mean(all_probs, axis=0)
    
    # FIXED: Check for suspicious uniform distribution
    prob_std = np.std(avg_probs)
    prob_range = np.max(avg_probs) - np.min(avg_probs)
    print(f"🔍 Multi-label stats - Std: {prob_std:.4f}, Range: {prob_range:.4f}")
    if prob_std < 0.01:
        print("⚠️ WARNING: Very low variance in multi-label predictions - possible model issue")
    
    # Get top genres by average probability
    top_indices = np.argsort(avg_probs)[::-1][:10]  # Top 10
    
    print("   🏆 Top 10 generi per probabilità media:")
    for i, idx in enumerate(top_indices):
        if idx < len(all_names):
            print(f"      {i+1}. {all_names[idx]}: {avg_probs[idx]:.4f}")
    
    analysis['all_genres'] = {
        'top_10': [(all_names[idx], float(avg_probs[idx])) for idx in top_indices if idx < len(all_names)],
        'average_probabilities': avg_probs.tolist(),
        'stats': {
            'std': float(prob_std),
            'range': float(prob_range),
            'uniform_warning': prob_std < 0.01
        }
    }
    
    # CONSENSUS ANALYSIS
    print(f"\n🎯 CONSENSUS FINALE:")
    print(f"   🥇 Top Genre (Weighted): {most_weighted}")
    print(f"   🥈 Medium Genre (Voting): {medium_most_voted[0]}")
    print(f"   🥉 Best Multi-label: {all_names[top_indices[0]] if top_indices[0] < len(all_names) else 'Unknown'}")
    
    # FIXED: Add model health warning
    if analysis['top_genre']['distribution_warning'] or analysis['all_genres']['stats']['uniform_warning']:
        print(f"\n⚠️  ATTENZIONE: Il modello potrebbe non essere addestrato correttamente!")
        print(f"   - Distribuzioni troppo uniformi o identiche")
        print(f"   - Probabilità sospettosamente simili")
    
    analysis['consensus'] = {
        'final_top_genre': most_weighted,
        'final_medium_genre': medium_most_voted[0],
        'final_best_multilabel': all_names[top_indices[0]] if top_indices[0] < len(all_names) else 'Unknown',
        'confidence_score': weighted_scores[most_weighted],
        'model_health_warning': analysis['top_genre']['distribution_warning'] or analysis['all_genres']['stats']['uniform_warning']
    }
    
    return analysis

def save_analysis(analysis: Dict, audio_path: str) -> None:
    """Save detailed analysis to JSON file"""
    output_file = f"{os.path.splitext(audio_path)[0]}_genre_analysis.json"
    
    analysis_output = {
        'file': os.path.basename(audio_path),
        'analysis_timestamp': str(tf.timestamp()),
        'results': analysis
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(analysis_output, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 Analisi salvata in: {output_file}")

def main():
    """Main testing function"""
    parser = argparse.ArgumentParser(description='Full Song Genre Classification Analysis - Debug Version')
    parser.add_argument('--file', required=True, help='Path to audio file')
    parser.add_argument('--model', required=True, help='Path to trained model')
    parser.add_argument('--mappings', required=True, help='Path to genre mappings JSON')
    parser.add_argument('--overlap', type=float, default=0.5, help='Overlap ratio between segments (default: 0.5)')
    parser.add_argument('--save', action='store_true', help='Save detailed analysis to JSON')
    
    args = parser.parse_args()
    
    print("🎧 ANALISI COMPLETA GENERE MUSICALE - DEBUG VERSION")
    print("=" * 60)
    
    # Check if files exist
    if not os.path.exists(args.file):
        print(f"❌ File audio non trovato: {args.file}")
        return
    
    if not os.path.exists(args.model):
        print(f"❌ File modello non trovato: {args.model}")
        return
    
    if not os.path.exists(args.mappings):
        print(f"❌ File mappings non trovato: {args.mappings}")
        return
    
    # Load model and mappings
    model, mappings = load_model_and_mappings(args.model, args.mappings)
    if model is None:
        return
    
    print(f"\n🎵 Elaborazione: {args.file}")
    print("=" * 50)
    
    # Extract features with overlapping segments
    features = extract_features_fixed(args.file, overlap_ratio=args.overlap)
    
    if len(features) == 0:
        print("❌ Nessuna feature estratta")
        return
    
    print(f"✅ Features estratte con successo: {features.shape}")
    
    # Make predictions
    results = predict_genre(model, features, mappings)
    
    # Perform full song analysis
    analysis = analyze_full_song(results, len(features), args.file)
    
    # Save analysis if requested
    if args.save and analysis:
        save_analysis(analysis, args.file)
    
    print("\n✅ Analisi completata!")
    
    # FINAL DIAGNOSIS
    if analysis and analysis['consensus']['model_health_warning']:
        print("\n🔍 DIAGNOSI FINALE:")
        print("   Il problema sembra essere nel MODELLO, non nel codice.")
        print("   Possibili cause:")
        print("   - Modello non addestrato o pesi non caricati correttamente")
        print("   - Architettura del modello non corrispondente ai pesi")
        print("   - Normalizzazione dei dati diversa dal training")

if __name__ == "__main__":
    main()