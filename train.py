#!/usr/bin/env python3
"""
Main script con configurazione GPU CORRETTA
"""

# ⚠️ IMPORTANTE: Configurazione GPU DEVE essere PRIMA di ogni import TensorFlow
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Riduce log TensorFlow

import tensorflow as tf
from tensorflow.keras import mixed_precision

# CONFIGURAZIONE GPU - PRIMA DI TUTTO!
def setup_gpu_first():
    """Configurazione GPU che DEVE essere eseguita per prima"""
    print("🔧 Configurazione GPU in corso...")
    
    # Lista GPU fisiche
    gpus = tf.config.list_physical_devices('GPU')
    
    if not gpus:
        print("⚠️ Nessuna GPU trovata - utilizzo CPU") 
        return False, False
    
    gpu_available = False
    mixed_precision_enabled = False
    
    try:
        # CRITICO: Configurare PRIMA di qualsiasi operazione TF
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            
        print(f"✅ GPU configurata: {len(gpus)} dispositivi")
        gpu_available = True
        
        # Test rapido GPU
        with tf.device('/GPU:0'):
            test = tf.constant([1.0, 2.0, 3.0])
            result = tf.reduce_sum(test)
            print(f"✅ Test GPU OK: {result.numpy()}")
            
    except RuntimeError as e:
        print(f"❌ Errore GPU: {e}")
        if "memory growth" in str(e).lower():
            print("ℹ️ Memoria già configurata (riavvio script)")
            gpu_available = True  # Probabilmente OK comunque
        else:
            return False, False
    
    # Mixed Precision solo se GPU OK
    if gpu_available:
        try:
            policy = mixed_precision.Policy('mixed_float16')
            mixed_precision.set_global_policy(policy)
            mixed_precision_enabled = True
            print("✅ Mixed Precision abilitata")
        except Exception as e:
            print(f"⚠️ Mixed Precision fallita: {e}")
    
    return gpu_available, mixed_precision_enabled

# ESEGUI CONFIGURAZIONE GPU SUBITO
GPU_AVAILABLE, MIXED_PRECISION = setup_gpu_first()

# ORA importa il resto (dopo configurazione GPU!)
import json
import numpy as np
import warnings

# I tuoi import locali
from model import HierarchicalGenreModel
from processor import HierarchicalGenreProcessor
from preprocessing import prepare_hierarchical_dataset, create_tf_dataset_from_chunks
from utils.audio import extract_features
from loader import DataLoader

warnings.filterwarnings("ignore")

def predict_hierarchical(model, file_path, mappings):
    """Predizione con device corretto"""
    device = '/GPU:0' if GPU_AVAILABLE else '/CPU:0'
    print(f"🎧 Predizione su {device}...")

    features = extract_features(file_path)
    if features is None:
        return None

    features = features[..., np.newaxis].astype(np.float32)

    with tf.device(device):
        predictions = model.predict(features, batch_size=len(features))
        top_pred, medium_pred, all_pred = predictions

    # Resto della funzione uguale...
    top_avg = np.mean(top_pred, axis=0)
    medium_avg = np.mean(medium_pred, axis=0)
    all_avg = np.mean(all_pred, axis=0)

    top_idx = np.argmax(top_avg)
    medium_idx = np.argmax(medium_avg)
    all_indices = np.where(all_avg > 0.3)[0]

    return {
        'top_genre': (mappings['top_genres'][top_idx], float(top_avg[top_idx])),
        'specific_genre': (mappings['medium_genres'][medium_idx], float(medium_avg[medium_idx])),
        'all_genres': [(mappings['all_genres'][i], float(all_avg[i])) for i in all_indices][:10],
        'processing_device': device
    }

def main():
    print("🚀 SISTEMA CLASSIFICAZIONE GERARCHICA GENERI MUSICALI")
    print("=" * 70)
    print(f"TensorFlow: {tf.__version__}")
    print(f"GPU: {'✅ Attiva' if GPU_AVAILABLE else '❌ CPU only'}")
    print(f"Mixed Precision: {'✅ Attiva' if MIXED_PRECISION else '❌ Disattiva'}")
    
    # Se nessuna GPU, chiedi conferma
    if not GPU_AVAILABLE:
        risposta = input("\n⚠️ Continuare senza GPU? (s/n): ").lower()
        if risposta != 's':
            print("Training annullato. Configura la GPU e riprova.")
            return
    
    # Resto del codice uguale...
    processor = HierarchicalGenreProcessor()
    genre_top, genres_ids, tracks = processor.load_complete_metadata()
    hierarchy, top_level, mid_level, all_genres, id_to_name = processor.build_genre_hierarchy()
    processed_data = processor.process_genre_labels(genre_top, genres_ids, id_to_name)
    mappings = processor.create_hierarchical_encoders(processed_data, hierarchy)

    data_dir = "datasets/my_preprocessed_data"
    loader = DataLoader(data_dir)

    chunk_dir = "datasets/chunks"
    if not os.path.exists(chunk_dir) or len(os.listdir(chunk_dir)) == 0:
        print("⚙️ Preprocessing dati e creazione chunk...")
        prepare_hierarchical_dataset(processor, processed_data,
                                    limit=5000 if not GPU_AVAILABLE else None,
                                    save_dir=chunk_dir)
    else:
        print("📂 Chunk dataset già presente, salto preprocessing...")

    print("🔄 Creazione tf.data.Dataset dai chunk su disco...")
    dataset = create_tf_dataset_from_chunks(processor, save_dir=chunk_dir, batch_size=64)

    from check_data import diagnose_and_fix_dataset

    print("🔍 Diagnosi dataset prima del training...")
    dataset = diagnose_and_fix_dataset(dataset)  

    # Ottieni input shape
    example_chunk = next(iter(dataset.take(1)))
    example_features, example_labels = example_chunk
    input_shape = example_features.shape[1:]
    num_top = example_labels['top_genres'].shape[1]
    num_medium = example_labels['medium_genres'].shape[1]
    num_all = example_labels['all_genres'].shape[1]

    print(f"\n📐 Input shape: {input_shape}")
    print(f"🎵 Generi: Top={num_top}, Medium={num_medium}, All={num_all}")
    print("\n🎉 AVVIO TRAINING")

    # Crea modello con configurazione corretta
    model_manager = HierarchicalGenreModel(
        input_shape=input_shape,
        num_top=num_top,
        num_medium=num_medium,
        num_all=num_all,
        mixed_precision=MIXED_PRECISION,
        gpu_available=GPU_AVAILABLE
    )

    model = model_manager.build()
    model, history = model_manager.train(dataset)

    # Salva modello
    model.save("hierarchical_genre_model.h5")
    with open("hierarchical_genre_mappings.json", "w") as f:
        json.dump(mappings, f, indent=2)

    print("\n🎉 TRAINING COMPLETATO!")
    print("=" * 70)
    print(f"Dispositivo utilizzato: {'GPU' if GPU_AVAILABLE else 'CPU'}")
    print(f"Generi: Top={len(mappings['top_genres'])}, Medium={len(mappings['medium_genres'])}, All={len(mappings['all_genres'])}")
    print("📦 Modello salvato: hierarchical_genre_model.h5")

if __name__ == "__main__":
    main()