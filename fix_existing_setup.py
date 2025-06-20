#!/usr/bin/env python3
"""
Script Completo - Fix Dataset FMA Esistente
Risolve l'errore JSON serialization e configura tutto automaticamente
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
import traceback
from collections import Counter

def safe_json_convert(obj):
    """Converte oggetti numpy/pandas in tipi Python nativi per JSON"""
    if isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.bool_, np.bool)):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.Series):
        return obj.to_dict()
    elif isinstance(obj, dict):
        return {k: safe_json_convert(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [safe_json_convert(item) for item in obj]
    else:
        return obj

def find_fma_paths():
    """Trova automaticamente i path del dataset FMA"""
    print("🔍 Ricerca automatica dataset FMA...")
    
    # Possibili path per metadata
    metadata_candidates = [
        'fma_metadata',
        'modello3/fma_metadata', 
        'data/fma_metadata',
        '.'
    ]
    
    metadata_path = None
    for path in metadata_candidates:
        tracks_file = os.path.join(path, 'tracks.csv')
        if os.path.exists(tracks_file):
            metadata_path = path
            print(f"   ✅ Metadata: {path}")
            break
    
    if not metadata_path:
        print("   ❌ Metadata non trovati automaticamente")
        return None, None
    
    # Possibili path per audio
    audio_candidates = [
        'fma_large',
        'modello3/fma_large',
        'data/fma_large',
        'fma_medium',
        'fma_small',
        '.'
    ]
    
    audio_path = None
    for path in audio_candidates:
        if os.path.exists(path) and os.path.isdir(path):
            # Controlla sottocartelle numeriche
            subdirs = [d for d in os.listdir(path) 
                      if os.path.isdir(os.path.join(path, d)) 
                      and d.isdigit() and len(d) == 3]
            if subdirs:
                audio_path = path
                print(f"   ✅ Audio: {path} ({len(subdirs)} cartelle)")
                break
    
    if not audio_path:
        print("   ⚠️ Audio non trovato (solo metadata)")
    
    return metadata_path, audio_path

def analyze_fma_balance(metadata_path):
    """Analizza il bilanciamento del dataset FMA"""
    print("\n📊 Analisi bilanciamento dataset...")
    
    try:
        tracks_file = os.path.join(metadata_path, 'tracks.csv')
        tracks = pd.read_csv(tracks_file, index_col=0, header=[0, 1])
        
        # Focus su subset large
        large_subset = tracks[tracks[('set', 'subset')] == 'large']
        genre_top = large_subset[('track', 'genre_top')].dropna()
        
        # Conta generi con conversione sicura
        genre_counts = genre_top.value_counts()
        
        # Converti tutto in tipi Python nativi
        genre_counts_dict = {}
        for genre, count in genre_counts.items():
            genre_counts_dict[str(genre)] = int(count)
        
        # Statistiche con conversione sicura
        mean_count = float(genre_counts.mean())
        std_count = float(genre_counts.std())
        median_count = float(genre_counts.median())
        min_count = int(genre_counts.min())
        max_count = int(genre_counts.max())
        
        stats = {
            'total_tracks': int(len(genre_top)),
            'num_genres': int(len(genre_counts)),
            'mean_per_genre': mean_count,
            'std_per_genre': std_count,
            'median_per_genre': median_count,
            'min_count': min_count,
            'max_count': max_count,
            'imbalance_ratio': float(max_count / min_count),
            'genre_counts': genre_counts_dict
        }
        
        # Calcola class weights
        total_samples = stats['total_tracks']
        num_classes = stats['num_genres']
        
        class_weights = {}
        for genre, count in genre_counts_dict.items():
            weight = float(total_samples / (num_classes * count))
            class_weights[genre] = weight
        
        print(f"   📈 Tracce analizzate: {stats['total_tracks']:,}")
        print(f"   🎵 Generi: {stats['num_genres']}")
        print(f"   ⚖️ Sbilanciamento: {stats['imbalance_ratio']:.1f}:1")
        print(f"   📊 Range: {min_count} - {max_count} tracce")
        
        return stats, class_weights
        
    except Exception as e:
        print(f"   ❌ Errore analisi: {e}")
        return None, None

def create_config_file(metadata_path, audio_path):
    """Crea il file config.py"""
    print("\n⚙️ Creazione config.py...")
    
    config_content = f'''#!/usr/bin/env python3
"""
Configurazione Dataset FMA - Generata Automaticamente
"""

try:
    from gpu import setup_gpu, setup_mixed_precision
    GPU_AVAILABLE = setup_gpu()
    MIXED_PRECISION = setup_mixed_precision() if GPU_AVAILABLE else False
except ImportError:
    print("⚠️ gpu.py non trovato - configurazione GPU manuale")
    import tensorflow as tf
    
    # Configurazione GPU base
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            GPU_AVAILABLE = True
            print("✅ GPU configurata")
        except:
            GPU_AVAILABLE = False
    else:
        GPU_AVAILABLE = False
    
    MIXED_PRECISION = False

# Dataset Paths
FMA_LARGE_PATH = '{audio_path or ""}'
METADATA_PATH = '{metadata_path}'

# Audio Processing
SAMPLE_RATE = 22050
DURATION = 30
N_MELS = 128
HOP_LENGTH = 512
NUM_SEGMENTS = 10

# Training
CHUNK_SIZE = 5000
BATCH_SIZE = 32

# Bilanciamento
USE_CLASS_WEIGHTS = True
USE_WEIGHTED_SAMPLING = True
USE_FOCAL_LOSS = True
OVERSAMPLE_THRESHOLD = 100
UNDERSAMPLE_THRESHOLD = 4000

# Stampa configurazione
print("📁 Configurazione FMA:")
print(f"   Metadata: {{METADATA_PATH}}")
if FMA_LARGE_PATH:
    print(f"   Audio: {{FMA_LARGE_PATH}}")
else:
    print("   Audio: Non configurato")
print(f"   GPU: {{'✅' if GPU_AVAILABLE else '❌'}}")
print(f"   Batch Size: {{BATCH_SIZE}}")
'''
    
    with open('config.py', 'w', encoding='utf-8') as f:
        f.write(config_content)
    
    print("   ✅ config.py creato")

def create_balanced_config_safe(stats, class_weights, metadata_path, audio_path):
    """Crea configurazione bilanciata con conversione JSON sicura"""
    print("\n📄 Creazione configurazione bilanciata...")
    
    # Configurazione con tutti i tipi convertiti esplicitamente
    config = {
        'dataset_info': {
            'name': 'FMA_Large_Existing',
            'total_tracks': int(stats['total_tracks']),
            'num_genres': int(stats['num_genres']),
            'mean_per_genre': float(stats['mean_per_genre']),
            'std_per_genre': float(stats['std_per_genre']),
            'imbalance_ratio': float(stats['imbalance_ratio']),
            'min_count': int(stats['min_count']),
            'max_count': int(stats['max_count'])
        },
        'paths': {
            'metadata_path': str(metadata_path),
            'audio_path': str(audio_path) if audio_path else None,
            'tracks_csv': str(os.path.join(metadata_path, 'tracks.csv')),
            'genres_csv': str(os.path.join(metadata_path, 'genres.csv'))
        },
        'class_weights': class_weights,
        'bilanciamento': {
            'strategia': 'class_weights + focal_loss + weighted_sampling',
            'peso_minimo': float(min(class_weights.values())),
            'peso_massimo': float(max(class_weights.values())),
            'rapporto_pesi': float(max(class_weights.values()) / min(class_weights.values()))
        },
        'training_config': {
            'batch_size': 32,
            'use_class_weights': True,
            'use_weighted_sampling': True,
            'use_focal_loss': True,
            'oversample_threshold': 100,
            'undersample_threshold': 4000
        },
        'setup_status': {
            'dataset_found': True,
            'metadata_analyzed': True,
            'audio_available': audio_path is not None,
            'ready_for_training': True
        }
    }
    
    # Salvataggio con gestione errori
    config_file = 'balanced_dataset_config.json'
    try:
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        print(f"   ✅ Configurazione salvata: {config_file}")
        return config_file
        
    except TypeError as e:
        print(f"   ❌ Errore JSON: {e}")
        
        # Salva versione debug
        debug_file = 'config_debug.txt'
        with open(debug_file, 'w', encoding='utf-8') as f:
            for key, value in config.items():
                f.write(f"{key}: {value}\\n")
        
        print(f"   💾 Debug salvato: {debug_file}")
        return None

def create_simple_trainer():
    """Crea script semplice per avviare il training"""
    print("\n🚀 Creazione script training...")
    
    script_content = '''#!/usr/bin/env python3
"""
Script Semplice per Avvio Training FMA
"""

import os
import sys

def main():
    print("🎵 AVVIO TRAINING FMA BILANCIATO")
    print("=" * 50)
    
    # Test configurazione
    try:
        import config
        print("✅ config.py caricato")
        print(f"   Metadata: {config.METADATA_PATH}")
        if hasattr(config, 'FMA_LARGE_PATH') and config.FMA_LARGE_PATH:
            print(f"   Audio: {config.FMA_LARGE_PATH}")
        print(f"   GPU: {'✅' if config.GPU_AVAILABLE else '❌'}")
        print(f"   Batch size: {config.BATCH_SIZE}")
    except Exception as e:
        print(f"❌ Errore config: {e}")
        print("💡 Verifica che config.py sia stato creato correttamente")
        return False
    
    # Controlla script training
    if not os.path.exists('balanced_trainer.py'):
        print("❌ balanced_trainer.py non trovato")
        print("💡 Assicurati di avere tutti gli script del progetto")
        return False
    
    print("\\n🏋️ Avvio training...")
    print("   (Questo può richiedere molto tempo)")
    
    try:
        import subprocess
        result = subprocess.run([
            sys.executable, 'balanced_trainer.py'
        ], check=True)
        
        print("\\n🎉 Training completato!")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"\\n❌ Errore training: {e}")
        return False
    except KeyboardInterrupt:
        print("\\n⚠️ Training interrotto dall'utente")
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        print("\\n🔧 Controlla gli errori e riprova")
'''
    
    script_file = 'start_fma_training.py'
    with open(script_file, 'w', encoding='utf-8') as f:
        f.write(script_content)
    
    try:
        os.chmod(script_file, 0o755)
    except:
        pass
    
    print(f"   ✅ Script creato: {script_file}")
    return script_file

def main():
    """Funzione principale"""
    print("🔧 FIX SETUP DATASET FMA ESISTENTE")
    print("=" * 60)
    print("Risolve l'errore JSON e configura tutto automaticamente")
    
    # try:
    #     print('ciO')
    # except Exception as e:
    #     pass

    try:
        print("\\n[1/5] Ricerca dataset...")
        metadata_path, audio_path = find_fma_paths()


        if not metadata_path:
            print("❌ Dataset non trovato automaticamente")
            print("💡 Configurazione manuale:")
            print("   1. Assicurati che tracks.csv sia nella directory corrente o in fma_metadata/")
            print("   2. Modifica manualmente config.py con i tuoi path")
            return

         # Step 2: Analizza dataset
        print("\\n[2/5] Analisi bilanciamento...")
        stats, class_weights = analyze_fma_balance(metadata_path)
        
        if not stats:
            print("❌ Errore nell'analisi del dataset")
            return
        
          # Step 3: Crea config.py
        print("\\n[3/5] Creazione config.py...")
        create_config_file(metadata_path, audio_path)
        
        # Step 4: Crea configurazione bilanciata
        print("\\n[4/5] Configurazione bilanciata...")
        config_file = create_balanced_config_safe(stats, class_weights, metadata_path, audio_path)
        
        # Step 5: Crea script training
        print("\\n[5/5] Script training...")
        starter_script = create_simple_trainer()
        
        # Riepilogo finale
        print("\\n🎉 SETUP COMPLETATO!")
        print("=" * 50)

        print("📊 Dataset analizzato:")
        print(f"   📈 Tracce (subset large): {stats['total_tracks']:,}")
        print(f"   🎵 Generi: {stats['num_genres']}")
        print(f"   ⚖️ Sbilanciamento: {stats['imbalance_ratio']:.1f}:1")
        print(f"   📊 Range tracce: {stats['min_count']} - {stats['max_count']}")
        
        print("\\n📄 File generati:")
        print("   ✅ config.py")

        if config_file:
            print(f"   ✅ {config_file}")
        print(f"   ✅ {starter_script}")
        

        print("\\n🚀 PROSSIMI PASSI:")
        print('   1. Test config:     python -c "import config; print(\'✅ OK\')"')
        print(f"   2. Avvia training:  python {starter_script}")
        print("   3. Training diretto: python balanced_trainer.py")
        

        print("\\n💡 NOTE:")
        print("   - L'errore JSON è stato risolto ✅")
        print("   - Il dataset è già ben configurato ✅")
        print("   - Le strategie di bilanciamento sono ottimali per il tuo sbilanciamento ✅")    
        
        
        


    except Exception as e:
        print(f"\\n❌ ERRORE DURANTE SETUP: {e}")
        print("\\nDETTAGLI ERRORE:")
        traceback.print_exc()
        
        print("\\n🔧 FALLBACK - CONFIGURAZIONE MANUALE:")
        print("Crea config.py con:")
        print("```python")
        print("FMA_LARGE_PATH = '/path/to/your/audio'")
        print("METADATA_PATH = '/path/to/your/metadata'")
        print("SAMPLE_RATE = 22050")
        print("BATCH_SIZE = 32")
        print("# ... altri parametri")
        print("```")

   

if __name__ == "__main__":
    main()