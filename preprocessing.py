import os 
import tensorflow as tf
import numpy as np

from processor import HierarchicalGenreProcessor
from loader import DataLoader
from utils.audio import audio_path,extract_features
from config import GPU_AVAILABLE,CHUNK_SIZE,BATCH_SIZE

def prepare_hierarchical_dataset(processor, processed_data, limit=None, chunk_size=5000, save_dir="datasets/chunks"):
    """Prepara dataset gerarchico salvando i dati in chunk per evitare OOM"""
    os.makedirs(save_dir, exist_ok=True)

    features = []
    top_labels = []
    medium_labels = []
    all_labels = []

    count = 0
    successful = 0
    chunk_id = 0

    print(f"Preparazione dataset gerarchico con chunk da {chunk_size} tracce...")

    for item in processed_data:
        if limit and count >= limit:
            break

        track_id = item['track_id']
        path = audio_path(track_id)

        if os.path.exists(path):
            track_features = extract_features(path)
            count += 1

            if track_features is not None:
                for segment in track_features:
                    features.append(segment)
                    top_labels.append(item['genre_top'])
                    medium_labels.append(item['primary_genre'])
                    all_labels.append(item['genres_all'])

                successful += 1

        if count % chunk_size == 0 and count > 0:
            print(f"💾 Salvando chunk {chunk_id} (tracce elaborate: {count})")
            _save_chunk(features, top_labels, medium_labels, all_labels, processor, save_dir, chunk_id)
            chunk_id += 1

            # Reset memoria
            features, top_labels, medium_labels, all_labels = [], [], [], []
            if GPU_AVAILABLE:
                tf.keras.backend.clear_session()

    # Salva l’ultimo chunk rimanente
    if features:
        print(f"💾 Salvando ultimo chunk {chunk_id} (tracce residue)")
        _save_chunk(features, top_labels, medium_labels, all_labels, processor, save_dir, chunk_id)

    print(f"✅ Dataset preparato: {successful} tracce totali.")
    return None  # I dati vengono salvati direttamente su disco


def _save_chunk(features, top_labels, medium_labels, all_labels, processor, save_dir, chunk_id):
    chunk_path = os.path.join(save_dir, f"chunk_{chunk_id}")
    os.makedirs(chunk_path, exist_ok=True)

    # Conversione in array e encoding
    features = np.array(features, dtype=np.float32)[..., np.newaxis]

    y_top = tf.keras.utils.to_categorical(
        processor.top_encoder.transform(top_labels),
        num_classes=len(processor.top_encoder.classes_)
    )

    y_medium = tf.keras.utils.to_categorical(
        processor.medium_encoder.transform(medium_labels),
        num_classes=len(processor.medium_encoder.classes_)
    )

    y_all = processor.all_encoder.transform(all_labels).astype(np.float32)

    # Salvataggio su disco
    np.save(os.path.join(chunk_path, "features.npy"), features)
    np.save(os.path.join(chunk_path, "y_top.npy"), y_top)
    np.save(os.path.join(chunk_path, "y_medium.npy"), y_medium)
    np.save(os.path.join(chunk_path, "y_all.npy"), y_all)


def chunk_generator(save_dir="datasets/chunks"):
    chunk_dirs = sorted([
        os.path.join(save_dir, d)
        for d in os.listdir(save_dir)
        if os.path.isdir(os.path.join(save_dir, d))
    ])
    
    for chunk_path in chunk_dirs:
        features = np.load(os.path.join(chunk_path, "features.npy"))
        y_top = np.load(os.path.join(chunk_path, "y_top.npy"))
        y_medium = np.load(os.path.join(chunk_path, "y_medium.npy"))
        y_all = np.load(os.path.join(chunk_path, "y_all.npy"))

        for i in range(features.shape[0]):
            yield (
                features[i],  # shape: (96, 64, 1)
                {
                    'top_genres': y_top[i],        # shape: (16,)
                    'medium_genres': y_medium[i],  # shape: (147,)
                    'all_genres': y_all[i]         # shape: (153,)
                }
            )





def create_tf_dataset_from_chunks(processor, save_dir="datasets/chunks", batch_size=BATCH_SIZE, shuffle_buffer=10000):
    # Prendi shape e dimensioni codificatori
    example_chunk = next(chunk_generator(save_dir))
    input_shape = example_chunk[0].shape  # esclude dimensione batch
    num_top_classes = len(processor.top_encoder.classes_)
    num_medium_classes = len(processor.medium_encoder.classes_)
    num_all_classes = len(processor.all_encoder.classes_)

    output_signature = (
        tf.TensorSpec(shape=input_shape, dtype=tf.float32),
        {
            'top_genres': tf.TensorSpec(shape=(num_top_classes,), dtype=tf.float32),
            'medium_genres': tf.TensorSpec(shape=(num_medium_classes,), dtype=tf.float32),
            'all_genres': tf.TensorSpec(shape=(num_all_classes,), dtype=tf.float32),
        }
    )

    dataset = tf.data.Dataset.from_generator(
        lambda: chunk_generator(save_dir),
        output_signature=output_signature
    )
    
    # Unbatch per avere singoli esempi e poi fai shuffle e batch
    #dataset = dataset.unbatch()
    dataset = dataset.shuffle(shuffle_buffer)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset
def preprocess_pipeline(features_path, metadata_path, limit=None, cache_dir="preprocessed_cache", chunk_dir="datasets/chunks", processor=None):
    os.makedirs(cache_dir, exist_ok=True)

    # Controllo se i chunk esistono già
    if os.path.exists(chunk_dir) and len(os.listdir(chunk_dir)) > 0:
        print("📦 Dataset chunkato già pronto su disco, evita preprocessing completo.")
    else:
        print("🔧 Preprocessing iniziale (prima volta)...")

        loader = DataLoader(features_path, metadata_path)
        features_raw, metadata = loader.load_all()

        if processor is None:
            processor = HierarchicalGenreProcessor()

        genre_top, genres_ids, _ = processor.load_complete_metadata()
        hierarchy, top_level, mid_level, all_genres, id_to_name = processor.build_genre_hierarchy()
        processed_data = processor.process_genre_labels(genre_top, genres_ids, id_to_name)
        processor.create_hierarchical_encoders(processed_data, hierarchy)

        prepare_hierarchical_dataset(processor, processed_data, limit=limit, save_dir=chunk_dir)

        print(f"✅ Dataset chunkato salvato in: {chunk_dir}")

    return processor  # Torna processor per usarlo nel tf.dataset



__all__ = ['prepare_hierarchical_dataset', 'preprocess_pipeline']

