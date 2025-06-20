#!/usr/bin/env python3
"""
FMA Dataset Inspector - Verifica struttura del dataset per generi musicali
"""

import pandas as pd
import numpy as np
import os
from collections import Counter
import json

def inspect_fma_dataset(metadata_path):
    """
    Ispeziona il dataset FMA per comprendere la struttura dei generi
    """
    print("🔍 ISPEZIONE DATASET FMA LARGE")
    print("=" * 60)
    
    # Carica tracks.csv con header multi-livello
    tracks_path = os.path.join(metadata_path, 'tracks.csv')
    tracks = pd.read_csv(tracks_path, index_col=0, header=[0, 1])
    
    print(f"📊 Dataset caricato: {len(tracks)} tracce totali")
    print(f"📊 Colonne principali: {len(tracks.columns)}")
    
    # Esamina struttura colonne
    print("\n🗂️ STRUTTURA COLONNE:")
    level_0_cols = tracks.columns.get_level_values(0).unique()
    for l0 in level_0_cols:
        l1_cols = tracks.columns.get_level_values(1)[tracks.columns.get_level_values(0) == l0]
        print(f"  {l0}:")
        for l1 in l1_cols:
            print(f"    - {l1}")
    
    # Focus sui generi
    print("\n🎵 ANALISI GENERI:")
    
    # Genere top-level
    if ('track', 'genre_top') in tracks.columns:
        genre_top = tracks[('track', 'genre_top')].dropna()
        top_genres = genre_top.value_counts()
        print(f"\n📈 GENERI TOP-LEVEL ({len(top_genres)} generi):")
        print(f"   Totale tracce con genere top: {len(genre_top)}")
        print(f"   Distribuzione top 10:")
        for genre, count in top_genres.head(10).items():
            print(f"     {genre}: {count} tracce ({count/len(genre_top)*100:.1f}%)")
        
        # Calcola sbilanciamento
        max_count = top_genres.max()
        min_count = top_genres.min()
        imbalance_ratio = max_count / min_count
        print(f"   📊 Sbilanciamento: {imbalance_ratio:.1f}:1 (max/min)")
    
    # Generi multipli
    if ('track', 'genres') in tracks.columns:
        genres_all = tracks[('track', 'genres')].dropna()
        print(f"\n📈 GENERI MULTIPLI:")
        print(f"   Tracce con lista generi: {len(genres_all)}")
        print(f"   Esempio generi multipli: {genres_all.iloc[0] if len(genres_all) > 0 else 'N/A'}")
    
    # Subset disponibili
    if ('set', 'subset') in tracks.columns:
        subsets = tracks[('set', 'subset')].value_counts()
        print(f"\n📦 SUBSET DISPONIBILI:")
        for subset, count in subsets.items():
            print(f"   {subset}: {count} tracce")
    
    # Carica genres.csv per gerarchia
    genres_path = os.path.join(metadata_path, 'genres.csv')
    if os.path.exists(genres_path):
        genres_df = pd.read_csv(genres_path, index_col=0)
        print(f"\n🏗️ GERARCHIA GENERI (da genres.csv):")
        print(f"   Totale generi nel database: {len(genres_df)}")
        print(f"   Colonne: {list(genres_df.columns)}")
        
        # Analizza gerarchia
        root_genres = genres_df[genres_df['parent'] == 0]
        print(f"   Generi radice (top-level): {len(root_genres)}")
        print(f"   Lista generi radice: {list(root_genres['title'])}")
        
        # Livelli di profondità
        max_depth = calculate_genre_depth(genres_df)
        print(f"   Profondità massima gerarchia: {max_depth}")
    
    return tracks, genres_df if 'genres_df' in locals() else None

def calculate_genre_depth(genres_df):
    """Calcola la profondità massima della gerarchia dei generi"""
    def get_depth(genre_id, visited=None):
        if visited is None:
            visited = set()
        if genre_id in visited or genre_id == 0:
            return 0
        visited.add(genre_id)
        parent_id = genres_df.loc[genre_id, 'parent'] if genre_id in genres_df.index else 0
        return 1 + get_depth(parent_id, visited)
    
    max_depth = 0
    for genre_id in genres_df.index:
        depth = get_depth(genre_id)
        max_depth = max(max_depth, depth)
    return max_depth

def analyze_genre_balance(tracks, genres_df=None):
    """
    Analizza lo sbilanciamento dei generi e suggerisce strategie di bilanciamento
    """
    print("\n⚖️ ANALISI SBILANCIAMENTO GENERI")
    print("=" * 50)
    
    # Analizza subset large
    large_subset = tracks[tracks[('set', 'subset')] == 'large']
    genre_top = large_subset[('track', 'genre_top')].dropna()
    
    genre_counts = genre_top.value_counts()
    
    print(f"📊 Distribuzione generi nel subset 'large':")
    print(f"   Totale tracce: {len(genre_top)}")
    print(f"   Generi unici: {len(genre_counts)}")
    
    # Statistiche distribuzione
    mean_count = genre_counts.mean()
    std_count = genre_counts.std()
    median_count = genre_counts.median()
    
    print(f"\n📈 Statistiche distribuzione:")
    print(f"   Media: {mean_count:.1f} tracce per genere")
    print(f"   Mediana: {median_count:.1f} tracce per genere")
    print(f"   Deviazione standard: {std_count:.1f}")
    print(f"   Coefficiente di variazione: {std_count/mean_count:.2f}")
    
    # Categorie di sbilanciamento
    q25, q75 = genre_counts.quantile([0.25, 0.75])
    
    underrepresented = genre_counts[genre_counts < q25]
    balanced = genre_counts[(genre_counts >= q25) & (genre_counts <= q75)]
    overrepresented = genre_counts[genre_counts > q75]
    
    print(f"\n🏷️ Categorie bilanciamento:")
    print(f"   Sottorapresentati (<Q1): {len(underrepresented)} generi")
    print(f"   Bilanciati (Q1-Q3): {len(balanced)} generi")
    print(f"   Sovrarapresentati (>Q3): {len(overrepresented)} generi")
    
    # Suggerimenti di bilanciamento
    print(f"\n💡 STRATEGIE DI BILANCIAMENTO CONSIGLIATE:")
    
    # Class weights
    class_weights = compute_class_weights(genre_counts)
    print(f"   1. Class Weights:")
    print(f"      - Peso minimo: {min(class_weights.values()):.3f}")
    print(f"      - Peso massimo: {max(class_weights.values()):.3f}")
    print(f"      - Rapporto max/min: {max(class_weights.values())/min(class_weights.values()):.1f}")
    
    # Sampling strategies
    max_samples = int(mean_count + std_count)
    min_samples = max(100, int(mean_count - std_count))
    
    print(f"   2. Resampling Strategy:")
    print(f"      - Max samples per classe: {max_samples}")
    print(f"      - Min samples per classe: {min_samples}")
    print(f"      - Oversample generi con < {min_samples} tracce")
    print(f"      - Undersample generi con > {max_samples} tracce")
    
    return {
        'genre_counts': genre_counts,
        'class_weights': class_weights,
        'resampling_thresholds': {'max': max_samples, 'min': min_samples},
        'statistics': {
            'mean': mean_count,
            'std': std_count,
            'median': median_count,
            'cv': std_count/mean_count
        }
    }

def compute_class_weights(genre_counts):
    """Calcola i pesi delle classi per bilanciamento"""
    total_samples = genre_counts.sum()
    num_classes = len(genre_counts)
    
    # Balanced class weights: n_samples / (n_classes * n_samples_for_class)
    class_weights = {}
    for genre, count in genre_counts.items():
        weight = total_samples / (num_classes * count)
        class_weights[genre] = weight
    
    return class_weights


def convert_numpy_types(obj):
    """Converte tipi numpy in tipi Python per JSON"""
    import numpy as np
    
    if isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj
def generate_balanced_config(analysis_results, output_file="balanced_config.json"):
    """Genera configurazione per bilanciamento"""
    config = {
        'dataset_info': {
            'total_genres': len(analysis_results['genre_counts']),
            'total_samples': int(analysis_results['genre_counts'].sum()),
            'mean_samples_per_genre': float(analysis_results['statistics']['mean']),
            'coefficient_of_variation': float(analysis_results['statistics']['cv'])
        },
        'class_weights': {k: float(v) for k, v in analysis_results['class_weights'].items()},
        'resampling_strategy': {
            'max_samples_per_class': analysis_results['resampling_thresholds']['max'],
            'min_samples_per_class': analysis_results['resampling_thresholds']['min'],
            'oversample_threshold': analysis_results['resampling_thresholds']['min'],
            'undersample_threshold': analysis_results['resampling_thresholds']['max']
        },
        'training_recommendations': {
            'use_class_weights': True,
            'use_weighted_sampling': True,
            'focal_loss_recommended': analysis_results['statistics']['cv'] > 1.0,
            'batch_balancing_recommended': True
        }
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        config_safe = convert_numpy_types(config)
        json.dump(config_safe, f, indent=2, ensure_ascii=False)
        
    
    print(f"📄 Configurazione bilanciamento salvata in: {output_file}")
    return config

if __name__ == "__main__":
    # Configura path del metadata
    METADATA_PATH = "modello3/fma_metadata"  # Modifica questo path
    
    if not os.path.exists(METADATA_PATH):
        print(f"❌ Path metadata non trovato: {METADATA_PATH}")
        print("💡 Modifica la variabile METADATA_PATH nel codice")
        exit(1)
    
    try:
        # Ispeziona dataset
        tracks, genres_df = inspect_fma_dataset(METADATA_PATH)
        
        # Analizza bilanciamento
        analysis = analyze_genre_balance(tracks, genres_df)
        
        # Genera configurazione
        config = generate_balanced_config(analysis)
        
        print("\n✅ Ispezione completata!")
        print("💡 Usa il file 'balanced_config.json' per configurare il training bilanciato")
        
    except Exception as e:
        print(f"❌ Errore durante l'ispezione: {e}")
        import traceback
        traceback.print_exc()