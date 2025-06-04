import tensorflow as tf
import numpy as np

def inspect_problematic_batch(dataset, batch_number=32):
    """
    Ispeziona specificamente il batch che causa problemi
    """
    print(f"🔍 ISPEZIONE BATCH {batch_number} (quello che causa NaN)")
    print("=" * 60)
    
    batch_count = 0
    for batch_features, batch_labels in dataset:
        batch_count += 1
        
        if batch_count == batch_number:
            print(f"📊 STATISTICHE BATCH {batch_number}:")
            
            # Statistiche features
            features_flat = tf.reshape(batch_features, [-1])
            
            print(f"Features shape: {batch_features.shape}")
            print(f"Features dtype: {batch_features.dtype}")
            
            # Valori problematici
            nan_count = tf.reduce_sum(tf.cast(tf.math.is_nan(batch_features), tf.int32))
            inf_count = tf.reduce_sum(tf.cast(tf.math.is_inf(batch_features), tf.int32))
            
            print(f"NaN values: {nan_count.numpy()}")
            print(f"Inf values: {inf_count.numpy()}")
            
            # Range valori
            min_val = tf.reduce_min(batch_features).numpy()
            max_val = tf.reduce_max(batch_features).numpy()
            mean_val = tf.reduce_mean(batch_features).numpy()
            std_val = tf.math.reduce_std(batch_features).numpy()
            
            print(f"Min: {min_val}")
            print(f"Max: {max_val}")
            print(f"Mean: {mean_val}")
            print(f"Std: {std_val}")
            
            # Controlla se ci sono valori estremi
            extreme_high = tf.reduce_sum(tf.cast(batch_features > 1000, tf.int32))
            extreme_low = tf.reduce_sum(tf.cast(batch_features < -1000, tf.int32))
            
            print(f"Valori > 1000: {extreme_high.numpy()}")
            print(f"Valori < -1000: {extreme_low.numpy()}")
            
            # Ispeziona labels
            print(f"\n🏷️ LABELS:")
            for label_name, label_tensor in batch_labels.items():
                label_nan = tf.reduce_sum(tf.cast(tf.math.is_nan(label_tensor), tf.int32))
                label_inf = tf.reduce_sum(tf.cast(tf.math.is_inf(label_tensor), tf.int32))
                label_sum = tf.reduce_sum(label_tensor)
                
                print(f"{label_name}: shape={label_tensor.shape}, NaN={label_nan.numpy()}, Inf={label_inf.numpy()}, Sum={label_sum.numpy()}")
            
            # Campioni specifici che potrebbero essere problematici
            print(f"\n🔬 CAMPIONI ESTREMI NEL BATCH:")
            for i in range(min(5, batch_features.shape[0])):
                sample = batch_features[i]
                sample_min = tf.reduce_min(sample).numpy()
                sample_max = tf.reduce_max(sample).numpy()
                sample_nan = tf.reduce_sum(tf.cast(tf.math.is_nan(sample), tf.int32)).numpy()
                
                if sample_nan > 0 or abs(sample_min) > 1000 or abs(sample_max) > 1000:
                    print(f"  Campione {i}: Min={sample_min}, Max={sample_max}, NaN={sample_nan}")
            
            break
    
    if batch_count < batch_number:
        print(f"⚠️ Batch {batch_number} non trovato. Dataset ha solo {batch_count} batch.")

def create_cleaned_dataset(original_dataset):
    """
    Crea una versione pulita del dataset
    """
    print("🧹 CREAZIONE DATASET PULITO...")
    
    def aggressive_cleaning(features, labels):
        print(f"Cleaning batch with shape: {features.shape}")
        
        # Pulizia features ultra-aggressiva
        features = tf.where(tf.math.is_nan(features), 0.0, features)
        features = tf.where(tf.math.is_inf(features), 0.0, features)
        features = tf.clip_by_value(features, -100.0, 100.0)
        
        # Normalizzazione per campione
        features_shape = tf.shape(features)
        features_flat = tf.reshape(features, [features_shape[0], -1])
        
        mean = tf.reduce_mean(features_flat, axis=1, keepdims=True)
        std = tf.math.reduce_std(features_flat, axis=1, keepdims=True)
        
        # Evita divisione per zero
        std = tf.maximum(std, 1e-8)
        
        features_normalized = (features_flat - mean) / std
        features = tf.reshape(features_normalized, features_shape)
        features = tf.clip_by_value(features, -10.0, 10.0)
        
        # Pulizia labels
        cleaned_labels = {}
        for key, value in labels.items():
            value = tf.where(tf.math.is_nan(value), 0.0, value)
            value = tf.where(tf.math.is_inf(value), 0.0, value)
            value = tf.clip_by_value(value, 0.0, 1.0)
            cleaned_labels[key] = value
            
        return features, cleaned_labels
    
    return original_dataset.map(aggressive_cleaning, num_parallel_calls=tf.data.AUTOTUNE)

# Funzione main per diagnosi
def diagnose_and_fix_dataset(dataset):
    """
    Diagnostica e corregge il dataset
    """
    print("🔧 DIAGNOSI E CORREZIONE DATASET")
    print("=" * 50)
    
    # 1. Ispeziona il batch problematico
    inspect_problematic_batch(dataset, 32)
    
    # 2. Crea dataset pulito
    cleaned_dataset = create_cleaned_dataset(dataset)
    
    # 3. Verifica che la pulizia abbia funzionato
    print("\n✅ VERIFICA DATASET PULITO...")
    inspect_problematic_batch(cleaned_dataset, 32)
    
    return cleaned_dataset

if __name__ == "__main__":
    print("🔍 Strumento per diagnosticare e correggere dataset problematici")
    print("Usa: cleaned_dataset = diagnose_and_fix_dataset(your_dataset)")