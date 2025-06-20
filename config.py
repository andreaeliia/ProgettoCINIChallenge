#!/usr/bin/env python3
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
FMA_LARGE_PATH = 'modello3/fma_large'
METADATA_PATH = 'fma_metadata'

# Audio Processing
SAMPLE_RATE = 22050
DURATION = 30
N_MELS = 128
HOP_LENGTH = 512
NUM_SEGMENTS = 10

# Training
CHUNK_SIZE = 5000
BATCH_SIZE = 4

# Bilanciamento
USE_CLASS_WEIGHTS = True
USE_WEIGHTED_SAMPLING = True
USE_FOCAL_LOSS = True
OVERSAMPLE_THRESHOLD = 100
UNDERSAMPLE_THRESHOLD = 4000

# Stampa configurazione
print("📁 Configurazione FMA:")
print(f"   Metadata: {METADATA_PATH}")
if FMA_LARGE_PATH:
    print(f"   Audio: {FMA_LARGE_PATH}")
else:
    print("   Audio: Non configurato")
print(f"   GPU: {'✅' if GPU_AVAILABLE else '❌'}")
print(f"   Batch Size: {BATCH_SIZE}")
