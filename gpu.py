import tensorflow as tf
from tensorflow.keras import mixed_precision

def setup_gpu():
    """
    Configura TensorFlow per utilizzare la GPU in modo ottimale:
    - Abilita la crescita della memoria GPU (allocazione dinamica VRAM)
    - Rileva GPU disponibili e restituisce True/False
    """
    gpus = tf.config.list_physical_devices('GPU')
    if not gpus:
        print("⚠️ Nessuna GPU trovata, utilizzo CPU")
        return False

    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✅ GPU configurata: {len(gpus)} dispositivi rilevati")
        return True
    except RuntimeError as e:
        print(f"❌ Errore configurazione GPU: {e}")
        return False

def setup_mixed_precision():
    """
    Abilita mixed precision (float16) per training più veloce su GPU compatibili.
    """
    try:
        policy = mixed_precision.Policy('mixed_float16')
        mixed_precision.set_global_policy(policy)
        print("✅ Mixed Precision abilitata (float16)")
        return True
    except Exception as e:
        print(f"⚠️ Mixed Precision non disponibile: {e}")
        return False