#!/usr/bin/env python3
"""
Sistema di Training Bilanciato per Classificazione Generi Musicali FMA
Con implementazione avanzata di bilanciamento pesi e sampling
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, Model, callbacks
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from collections import Counter
import json
import ast
from typing import Dict, List, Tuple, Any
import librosa
import warnings

warnings.filterwarnings("ignore")

# ==================== CONFIGURAZIONE ====================
class Config:
    # Audio settings
    SAMPLE_RATE = 22050
    DURATION = 30.0
    N_MELS = 128
    HOP_LENGTH = 512
    N_FFT = 2048
    NUM_SEGMENTS = 10
    
    # Training settings
    BATCH_SIZE = 4
    NUM_EPOCHS = 100
    LEARNING_RATE = 1e-4
    PATIENCE = 15
    
    # Paths
    FMA_LARGE_PATH = 'modello3/fma_large'
    METADATA_PATH = 'modello3/fma_metadata'
    
    # Bilanciamento
    USE_CLASS_WEIGHTS = True
    USE_WEIGHTED_SAMPLING = True
    USE_FOCAL_LOSS = True
    OVERSAMPLE_THRESHOLD = 100  # Min samples per class
    UNDERSAMPLE_THRESHOLD = 5000  # Max samples per class
    
    # Device
    MIXED_PRECISION = True

config = Config()

# ==================== LOSS FUNCTIONS BILANCIATE ====================
class FocalLoss(tf.keras.losses.Loss):
    """
    Focal Loss per gestire classi sbilanciate
    """
    def __init__(self, alpha=1.0, gamma=2.0, from_logits=False, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.gamma = gamma
        self.from_logits = from_logits
    
    def call(self, y_true, y_pred):
        if self.from_logits:
            y_pred = tf.nn.softmax(y_pred, axis=-1)
        
        # Clip predictions per stabilità numerica
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        
        # Calcola cross entropy
        ce_loss = -y_true * tf.math.log(y_pred)
        
        # Calcola focal weight: (1 - p_t)^gamma
        p_t = tf.reduce_sum(y_true * y_pred, axis=-1, keepdims=True)
        focal_weight = tf.pow(1.0 - p_t, self.gamma)
        
        # Applica alpha weighting se necessario
        if isinstance(self.alpha, (float, int)):
            alpha_weight = self.alpha
        else:
            alpha_weight = tf.reduce_sum(y_true * self.alpha, axis=-1, keepdims=True)
        
        focal_loss = alpha_weight * focal_weight * ce_loss
        return tf.reduce_sum(focal_loss, axis=-1)

class WeightedBinaryCrossEntropy(tf.keras.losses.Loss):
    """
    Binary Cross Entropy con pesi per multi-label classification
    """
    def __init__(self, pos_weight=None, **kwargs):
        super().__init__(**kwargs)
        self.pos_weight = pos_weight
    
    def call(self, y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        
        if self.pos_weight is not None:
            # Applica pesi diversi per classi positive/negative
            loss = -self.pos_weight * y_true * tf.math.log(y_pred) - \
                   (1.0 - y_true) * tf.math.log(1.0 - y_pred)
        else:
            loss = -y_true * tf.math.log(y_pred) - \
                   (1.0 - y_true) * tf.math.log(1.0 - y_pred)
        
        return tf.reduce_mean(loss, axis=-1)

# ==================== ESTRATTORE FEATURES OTTIMIZZATO ====================
class AudioFeatureExtractor:
    """
    Estrattore features audio ottimizzato con segmentazione
    """
    def __init__(self):
        self.sample_rate = config.SAMPLE_RATE
        self.duration = config.DURATION
        self.n_mels = config.N_MELS
        self.hop_length = config.HOP_LENGTH
        self.n_fft = config.N_FFT
        self.num_segments = config.NUM_SEGMENTS
    
    def extract_features(self, file_path: str) -> np.ndarray:
        """Estrae features mel-spectrogram con segmentazione"""
        try:
            # Carica audio
            audio, sr = librosa.load(file_path, sr=self.sample_rate, duration=self.duration)
            
            # Calcola lunghezza segmenti
            samples_per_segment = int(self.duration * self.sample_rate / self.num_segments)
            features = []
            
            for i in range(self.num_segments):
                start = samples_per_segment * i
                end = start + samples_per_segment
                
                if end <= len(audio):
                    segment = audio[start:end]
                    
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
                    
                    features.append(log_mel_spec)
            
            return np.array(features, dtype=np.float32)
            
        except Exception as e:
            print(f"❌ Errore nell'estrazione features da {file_path}: {e}")
            return None
    
    def audio_path(self, track_id: int) -> str:
        """Genera il percorso del file audio"""
        tid_str = f'{track_id:06d}'
        return os.path.join(config.FMA_LARGE_PATH, tid_str[:3], tid_str + '.mp3')

# ==================== PROCESSORE GENERI BILANCIATO ====================
class BalancedGenreProcessor:
    """
    Processore generi con strategie di bilanciamento avanzate
    """
    def __init__(self):
        self.top_encoder = LabelEncoder()
        self.medium_encoder = LabelEncoder()
        self.all_encoder = MultiLabelBinarizer()
        self.feature_extractor = AudioFeatureExtractor()
        
        # Statistiche bilanciamento
        self.class_weights = {}
        self.genre_statistics = {}
        
    def load_fma_metadata(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Carica metadata FMA con verifica struttura"""
        print("📂 Caricamento metadata FMA...")
        
        # Carica tracks.csv
        tracks_path = os.path.join(config.METADATA_PATH, 'tracks.csv')
        tracks = pd.read_csv(tracks_path, index_col=0, header=[0, 1])
        
        # Carica genres.csv
        genres_path = os.path.join(config.METADATA_PATH, 'genres.csv')
        genres_df = pd.read_csv(genres_path, index_col=0)
        
        print(f"✅ Caricati {len(tracks)} tracce e {len(genres_df)} generi")
        return tracks, genres_df
    
    def analyze_genre_distribution(self, tracks: pd.DataFrame) -> Dict:
        """Analizza distribuzione generi per strategie di bilanciamento"""
        print("📊 Analisi distribuzione generi...")
        
        # Focus su subset large
        large_subset = tracks[tracks[('set', 'subset')] == 'large']
        genre_top = large_subset[('track', 'genre_top')].dropna()
        
        genre_counts = genre_top.value_counts()
        
        # Statistiche
        stats = {
            'total_tracks': len(genre_top),
            'num_genres': len(genre_counts),
            'mean_per_genre': genre_counts.mean(),
            'std_per_genre': genre_counts.std(),
            'median_per_genre': genre_counts.median(),
            'imbalance_ratio': genre_counts.max() / genre_counts.min(),
            'genre_counts': genre_counts.to_dict()
        }
        
        # Categorizza generi per bilanciamento
        q25, q75 = genre_counts.quantile([0.25, 0.75])
        
        stats['underrepresented'] = genre_counts[genre_counts < q25].index.tolist()
        stats['balanced'] = genre_counts[(genre_counts >= q25) & (genre_counts <= q75)].index.tolist()
        stats['overrepresented'] = genre_counts[genre_counts > q75].index.tolist()
        
        print(f"   📈 Distribuzione: {stats['num_genres']} generi")
        print(f"   ⚖️ Rapporto sbilanciamento: {stats['imbalance_ratio']:.1f}:1")
        print(f"   📊 Sotto-rappresentati: {len(stats['underrepresented'])}")
        print(f"   📊 Bilanciati: {len(stats['balanced'])}")
        print(f"   📊 Sovra-rappresentati: {len(stats['overrepresented'])}")
        
        self.genre_statistics = stats
        return stats
    
    def compute_hierarchical_class_weights(self, processed_data: List[Dict]) -> Dict:
        """Calcola class weights per tutti i livelli gerarchici"""
        print("⚖️ Calcolo class weights gerarchici...")
        
        weights = {}
        
        # Top level weights
        top_genres = [item['genre_top'] for item in processed_data]
        top_classes = np.unique(top_genres)
        top_weights = compute_class_weight('balanced', classes=top_classes, y=top_genres)
        weights['top'] = dict(zip(top_classes, top_weights))
        
        # Medium level weights
        medium_genres = [item['primary_genre'] for item in processed_data]
        medium_classes = np.unique(medium_genres)
        medium_weights = compute_class_weight('balanced', classes=medium_classes, y=medium_genres)
        weights['medium'] = dict(zip(medium_classes, medium_weights))
        
        # All genres weights (per ogni genere nella classificazione multi-label)
        all_genre_names = set()
        for item in processed_data:
            all_genre_names.update(item['genres_all'])
        
        all_weights = {}
        for genre in all_genre_names:
            # Conta presenza/assenza del genere
            genre_presence = [1 if genre in item['genres_all'] else 0 for item in processed_data]
            pos_count = sum(genre_presence)
            neg_count = len(genre_presence) - pos_count
            
            # Calcola peso per classe positiva (presenza del genere)
            if pos_count > 0:
                pos_weight = neg_count / pos_count
                all_weights[genre] = pos_weight
            else:
                all_weights[genre] = 1.0
        
        weights['all'] = all_weights
        
        # Salva statistiche
        print(f"   🔝 Top level: {len(weights['top'])} generi")
        print(f"      - Peso min: {min(weights['top'].values()):.3f}")
        print(f"      - Peso max: {max(weights['top'].values()):.3f}")
        
        print(f"   🎯 Medium level: {len(weights['medium'])} generi")
        print(f"      - Peso min: {min(weights['medium'].values()):.3f}")
        print(f"      - Peso max: {max(weights['medium'].values()):.3f}")
        
        print(f"   🎵 All genres: {len(weights['all'])} generi")
        print(f"      - Peso min: {min(weights['all'].values()):.3f}")
        print(f"      - Peso max: {max(weights['all'].values()):.3f}")
        
        self.class_weights = weights
        return weights
    
    def balance_dataset_sampling(self, processed_data: List[Dict]) -> List[Dict]:
        """Applica strategie di resampling per bilanciare il dataset"""
        print("🔄 Applicazione strategie di resampling...")
        
        # Raggruppa per genere top
        genre_groups = {}
        for item in processed_data:
            genre = item['genre_top']
            if genre not in genre_groups:
                genre_groups[genre] = []
            genre_groups[genre].append(item)
        
        balanced_data = []
        
        for genre, items in genre_groups.items():
            current_count = len(items)
            
            if current_count < config.OVERSAMPLE_THRESHOLD:
                # Oversample generi sotto-rappresentati
                needed = config.OVERSAMPLE_THRESHOLD - current_count
                oversampled = np.random.choice(items, size=needed, replace=True).tolist()
                balanced_data.extend(items + oversampled)
                print(f"   📈 {genre}: {current_count} → {len(items) + len(oversampled)} (+{len(oversampled)})")
                
            elif current_count > config.UNDERSAMPLE_THRESHOLD:
                # Undersample generi sovra-rappresentati
                undersampled = np.random.choice(items, size=config.UNDERSAMPLE_THRESHOLD, replace=False).tolist()
                balanced_data.extend(undersampled)
                print(f"   📉 {genre}: {current_count} → {len(undersampled)} (-{current_count - len(undersampled)})")
                
            else:
                # Mantieni generi bilanciati
                balanced_data.extend(items)
                print(f"   ✅ {genre}: {current_count} (mantenuto)")
        
        print(f"📊 Dataset bilanciato: {len(processed_data)} → {len(balanced_data)} tracce")
        return balanced_data
    
    def process_fma_labels(self, tracks: pd.DataFrame, genres_df: pd.DataFrame) -> List[Dict]:
        """Processa labels FMA per training gerarchico"""
        print("🏷️ Processamento labels gerarchiche...")
        
        # Mappa ID → Nome genere
        id_to_name = dict(zip(genres_df.index, genres_df['title']))
        
        # Focus su subset large
        large_subset = tracks[tracks[('set', 'subset')] == 'large']
        
        processed_data = []
        
        for track_id in large_subset.index:
            try:
                # Genere principale
                genre_top = large_subset.loc[track_id, ('track', 'genre_top')]
                if pd.isna(genre_top):
                    continue
                
                # Lista generi (se disponibile)
                genre_ids_raw = large_subset.loc[track_id, ('track', 'genres')]
                genre_names = [genre_top]  # Default al genere principale
                
                if not pd.isna(genre_ids_raw):
                    try:
                        # Parse lista ID generi
                        if isinstance(genre_ids_raw, str) and '[' in genre_ids_raw:
                            genre_ids = ast.literal_eval(genre_ids_raw)
                        else:
                            genre_ids = [int(x.strip()) for x in str(genre_ids_raw).split(',')]
                        
                        # Converti ID in nomi
                        genre_names = [id_to_name.get(gid, f"Unknown_{gid}") 
                                     for gid in genre_ids if gid in id_to_name]
                        
                        if not genre_names:
                            genre_names = [genre_top]
                            
                    except:
                        genre_names = [genre_top]
                
                processed_data.append({
                    'track_id': track_id,
                    'genre_top': genre_top,
                    'primary_genre': genre_names[0],
                    'genres_all': genre_names
                })
                
            except Exception as e:
                continue
        
        print(f"✅ Processate {len(processed_data)} tracce con labels valide")
        return processed_data
    
    def create_encoders(self, processed_data: List[Dict]) -> Dict:
        """Crea encoder per tutti i livelli gerarchici"""
        print("🔧 Creazione encoder gerarchici...")
        
        # Raccogli generi per livello
        top_genres = set()
        medium_genres = set()
        all_genre_combinations = []
        
        for item in processed_data:
            top_genres.add(item['genre_top'])
            medium_genres.add(item['primary_genre'])
            all_genre_combinations.append(item['genres_all'])
        
        # Fit encoder
        self.top_encoder.fit(list(top_genres))
        self.medium_encoder.fit(list(medium_genres))
        self.all_encoder.fit(all_genre_combinations)
        
        # Crea mappings
        mappings = {
            'top_genres': list(self.top_encoder.classes_),
            'medium_genres': list(self.medium_encoder.classes_),
            'all_genres': list(self.all_encoder.classes_)
        }
        
        print(f"✅ Encoder creati:")
        print(f"   🔝 Top: {len(mappings['top_genres'])} generi")
        print(f"   🎯 Medium: {len(mappings['medium_genres'])} generi")
        print(f"   🎵 All: {len(mappings['all_genres'])} generi")
        
        return mappings

# ==================== MODELLO CON BILANCIAMENTO ====================
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
        
        if pool_type == 'avg':
            x = layers.AveragePooling2D(pool_size)(x)
        elif pool_type == 'max':
            x = layers.MaxPooling2D(pool_size)(x)
        elif pool_type == 'avg+max':
            x1 = layers.AveragePooling2D(pool_size)(x)
            x2 = layers.MaxPooling2D(pool_size)(x)
            x = x1 + x2
        
        return x

class BalancedHierarchicalModel:
    """
    Modello PANNs CNN14 Multi-Task per classificazione gerarchica bilanciata
    """
    def __init__(self, input_shape: Tuple, num_top: int, num_medium: int, 
                 num_all: int, class_weights: Dict):
        self.input_shape = input_shape
        self.num_top = num_top
        self.num_medium = num_medium
        self.num_all = num_all
        self.class_weights = class_weights
        self.model = None
    
    def build_model(self) -> Model:
        """Costruisce modello PANNs CNN14 Multi-Task"""
        print("🏗️ Costruzione modello PANNs CNN14 Multi-Task...")
        
        inputs = layers.Input(shape=self.input_shape, name='audio_input')
        
        # PANNs CNN14 Backbone
        x = inputs
        
        # Preprocessing - BatchNorm iniziale
        x = layers.BatchNormalization(name='bn0')(x)
        
        # CNN14 Architecture - 6 ConvBlocks
        x = PANNsConvBlock(64, name='conv_block1')(x, pool_size=(2, 2), pool_type='avg')
        x = layers.Dropout(0.2)(x)
        
        x = PANNsConvBlock(128, name='conv_block2')(x, pool_size=(2, 2), pool_type='avg')
        x = layers.Dropout(0.2)(x)
        
        x = PANNsConvBlock(256, name='conv_block3')(x, pool_size=(2, 2), pool_type='avg')
        x = layers.Dropout(0.2)(x)
        
        x = PANNsConvBlock(512, name='conv_block4')(x, pool_size=(2, 2), pool_type='avg')
        x = layers.Dropout(0.2)(x)
        
        x = PANNsConvBlock(1024, name='conv_block5')(x, pool_size=(2, 2), pool_type='avg')
        x = layers.Dropout(0.2)(x)
        
        x = PANNsConvBlock(2048, name='conv_block6')(x, pool_size=(1, 1), pool_type='avg')
        x = layers.Dropout(0.2)(x)
        
        # PANNs Global Pooling Strategy (usando custom layer)
        x = PANNsGlobalPooling(name='panns_global_pooling')(x)
        
        # Dense layers finali
        x = layers.Dropout(0.5)(x)
        embeddings = layers.Dense(2048, activation='relu', name='fc1')(x)
        embeddings = layers.Dropout(0.5)(embeddings)
        
        # Multi-Task Heads per classificazione gerarchica
        print(f"   🎯 Creazione heads: Top({self.num_top}), Medium({self.num_medium}), All({self.num_all})")
        
        # Shared representation
        shared = layers.Dense(512, activation='relu', name='shared_features')(embeddings)
        shared = layers.BatchNormalization()(shared)
        shared = layers.Dropout(0.3)(shared)
        
        # Output heads bilanciati
        top_output = layers.Dense(self.num_top, activation='softmax', 
                                 name='top_genres', dtype='float32')(shared)
        medium_output = layers.Dense(self.num_medium, activation='softmax', 
                                   name='medium_genres', dtype='float32')(shared)
        all_output = layers.Dense(self.num_all, activation='sigmoid', 
                                name='all_genres', dtype='float32')(shared)
        
        self.model = Model(inputs, [top_output, medium_output, all_output])
        
        print(f"   📊 Parametri totali: {self.model.count_params():,}")
        return self.model
    
    def compile_with_balanced_losses(self) -> None:
        """Compila modello con loss functions bilanciate"""
        print("⚖️ Configurazione loss functions bilanciate...")
        
        # Prepara class weights per TensorFlow
        top_weights_array = np.ones(self.num_top)
        medium_weights_array = np.ones(self.num_medium)
        
        # Converti class weights in array
        if 'top' in self.class_weights:
            for i, genre in enumerate(self.class_weights['top'].keys()):
                if i < self.num_top:
                    top_weights_array[i] = self.class_weights['top'][genre]
        
        if 'medium' in self.class_weights:
            for i, genre in enumerate(self.class_weights['medium'].keys()):
                if i < self.num_medium:
                    medium_weights_array[i] = self.class_weights['medium'][genre]
        
        # Crea pos_weights per multi-label
        pos_weights = np.ones(self.num_all)
        if 'all' in self.class_weights:
            for i, genre in enumerate(self.class_weights['all'].keys()):
                if i < self.num_all:
                    pos_weights[i] = self.class_weights['all'][genre]
        
        # Loss functions
        if config.USE_FOCAL_LOSS:
            top_loss = FocalLoss(alpha=top_weights_array, gamma=2.0)
            medium_loss = FocalLoss(alpha=medium_weights_array, gamma=2.0)
        else:
            top_loss = 'categorical_crossentropy'
            medium_loss = 'categorical_crossentropy'
        
        all_loss = WeightedBinaryCrossEntropy(pos_weight=pos_weights)
        
        # Compilazione con ottimizzazioni PANNs
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=config.LEARNING_RATE,
                beta_1=0.9,
                beta_2=0.999,
                clipnorm=1.0  # Gradient clipping per stabilità
            ),
            loss={
                'top_genres': top_loss,
                'medium_genres': medium_loss,
                'all_genres': all_loss
            },
            loss_weights={
                'top_genres': 1.0,    # Massima priorità per generi principali
                'medium_genres': 0.8, # Priorità media per sottogeneri
                'all_genres': 0.6     # Priorità minore per multi-label
            },
            metrics={
                'top_genres': ['accuracy'],  # Rimosso top_k per mixed precision
                'medium_genres': ['accuracy'],  # Rimosso top_k per mixed precision
                'all_genres': ['binary_accuracy', 'precision', 'recall']
            }
        )
        
        print("✅ Modello compilato con strategie di bilanciamento")

# ==================== DATASET GENERATOR BILANCIATO ====================
class BalancedDataGenerator(tf.keras.utils.Sequence):
    """
    Generator bilanciato per training efficiente
    """
    def __init__(self, processed_data: List[Dict], processor: BalancedGenreProcessor,
                 batch_size: int = 32, shuffle: bool = True, weighted_sampling: bool = True):
        self.processed_data = processed_data
        self.processor = processor
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.weighted_sampling = weighted_sampling
        
        # Calcola pesi per sampling
        if weighted_sampling:
            self.sample_weights = self._calculate_sample_weights()
        
        self.on_epoch_end()
    
    def _calculate_sample_weights(self) -> np.ndarray:
        """Calcola pesi per weighted sampling"""
        genre_counts = Counter([item['genre_top'] for item in self.processed_data])
        total_samples = len(self.processed_data)
        num_classes = len(genre_counts)
        
        weights = []
        for item in self.processed_data:
            genre = item['genre_top']
            weight = total_samples / (num_classes * genre_counts[genre])
            weights.append(weight)
        
        return np.array(weights)
    
    def __len__(self):
        return int(np.floor(len(self.processed_data) / self.batch_size))
    
    def __getitem__(self, index):
        # Genera indici per il batch
        start_idx = index * self.batch_size
        end_idx = min((index + 1) * self.batch_size, len(self.indices))
        batch_indices = self.indices[start_idx:end_idx]
        
        return self._generate_batch(batch_indices)
    
    def _generate_batch(self, batch_indices):
        # Inizializza batch arrays
        batch_features = []
        batch_top_labels = []
        batch_medium_labels = []
        batch_all_labels = []
        
        for idx in batch_indices:
            item = self.processed_data[idx]
            
            # Estrai features audio
            audio_path = self.processor.feature_extractor.audio_path(item['track_id'])
            if os.path.exists(audio_path):
                features = self.processor.feature_extractor.extract_features(audio_path)
                
                if features is not None:
                    # Aggiungi ogni segmento come esempio separato
                    for segment in features:
                        batch_features.append(segment)
                        batch_top_labels.append(item['genre_top'])
                        batch_medium_labels.append(item['primary_genre'])
                        batch_all_labels.append(item['genres_all'])
        
        if not batch_features:
            # Fallback se nessuna feature estratta
            return self._empty_batch()
        
        # Converti in array
        X = np.array(batch_features, dtype=np.float32)
        
        # Encode labels
        y_top = to_categorical(
            self.processor.top_encoder.transform(batch_top_labels),
            num_classes=len(self.processor.top_encoder.classes_)
        )
        
        y_medium = to_categorical(
            self.processor.medium_encoder.transform(batch_medium_labels),
            num_classes=len(self.processor.medium_encoder.classes_)
        )
        
        y_all = self.processor.all_encoder.transform(batch_all_labels).astype(np.float32)
        
        return X, {
            'top_genres': y_top,
            'medium_genres': y_medium,
            'all_genres': y_all
        }
    
    def _empty_batch(self):
        """Crea batch vuoto in caso di errore"""
        X = np.zeros((1, self.processor.feature_extractor.n_mels, 64, 1), dtype=np.float32)
        y_top = np.zeros((1, len(self.processor.top_encoder.classes_)), dtype=np.float32)
        y_medium = np.zeros((1, len(self.processor.medium_encoder.classes_)), dtype=np.float32)
        y_all = np.zeros((1, len(self.processor.all_encoder.classes_)), dtype=np.float32)
        
        return X, {
            'top_genres': y_top,
            'medium_genres': y_medium,
            'all_genres': y_all
        }
    
    def on_epoch_end(self):
        """Aggiorna indici a fine epoca"""
        if self.weighted_sampling:
            # Weighted sampling
            self.indices = np.random.choice(
                len(self.processed_data),
                size=len(self.processed_data),
                p=self.sample_weights / np.sum(self.sample_weights),
                replace=True
            )
        else:
            self.indices = np.arange(len(self.processed_data))
            
        if self.shuffle:
            np.random.shuffle(self.indices)

# ==================== TRAINING PIPELINE ====================
def train_balanced_model():
    """Pipeline completo di training bilanciato"""
    print("🚀 AVVIO TRAINING BILANCIATO PER CLASSIFICAZIONE GENERI")
    print("=" * 70)
    
    # Setup GPU e Mixed Precision
    if config.MIXED_PRECISION:
        tf.keras.mixed_precision.set_global_policy('mixed_float16')
        print("✅ Mixed Precision abilitata")
    
    # Inizializza processore
    processor = BalancedGenreProcessor()
    
    # Carica metadata
    tracks, genres_df = processor.load_fma_metadata()
    
    # Analizza distribuzione
    distribution_stats = processor.analyze_genre_distribution(tracks)
    
    # Processa labels
    processed_data = processor.process_fma_labels(tracks, genres_df)
    
    # Crea encoder
    mappings = processor.create_encoders(processed_data)
    
    # Calcola class weights
    class_weights = processor.compute_hierarchical_class_weights(processed_data)
    
    # Bilancia dataset
    if config.USE_WEIGHTED_SAMPLING:
        balanced_data = processor.balance_dataset_sampling(processed_data)
    else:
        balanced_data = processed_data
    
    # Split dataset
    train_data, temp_data = train_test_split(
        balanced_data, test_size=0.3, random_state=42,
        stratify=[item['genre_top'] for item in balanced_data]
    )
    val_data, test_data = train_test_split(
        temp_data, test_size=0.5, random_state=42,
        stratify=[item['genre_top'] for item in temp_data]
    )
    
    print(f"📊 Split dataset:")
    print(f"   Training: {len(train_data)} tracce")
    print(f"   Validation: {len(val_data)} tracce")
    print(f"   Test: {len(test_data)} tracce")
    
    # Crea generatori dati
    train_generator = BalancedDataGenerator(
        train_data, processor, batch_size=config.BATCH_SIZE,
        shuffle=True, weighted_sampling=config.USE_WEIGHTED_SAMPLING
    )
    
    val_generator = BalancedDataGenerator(
        val_data, processor, batch_size=config.BATCH_SIZE,
        shuffle=False, weighted_sampling=False
    )
    
    # Crea modello
    input_shape = (config.N_MELS, 130, 1)  # (n_mels, time_steps, channels)
    model_builder = BalancedHierarchicalModel(
        input_shape=input_shape,
        num_top=len(mappings['top_genres']),
        num_medium=len(mappings['medium_genres']),
        num_all=len(mappings['all_genres']),
        class_weights=class_weights
    )
    
    model = model_builder.build_model()
    model_builder.compile_with_balanced_losses()
    
    print(f"📐 Modello creato:")
    print(f"   Input shape: {input_shape}")
    print(f"   Top genres: {len(mappings['top_genres'])}")
    print(f"   Medium genres: {len(mappings['medium_genres'])}")
    print(f"   All genres: {len(mappings['all_genres'])}")
    
    # Callbacks
    callbacks_list = [
        callbacks.ModelCheckpoint(
            'balanced_genre_model_best.h5',
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        callbacks.EarlyStopping(
            monitor='val_loss',
            patience=config.PATIENCE,
            restore_best_weights=True,
            verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=7,
            min_lr=1e-7,
            verbose=1
        ),
        callbacks.TerminateOnNaN()
    ]
    
    # Training
    print("\n🎯 INIZIO TRAINING")
    print("=" * 50)
    
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=config.NUM_EPOCHS,
        callbacks=callbacks_list,
        verbose=1
    )
    
    # Salva modello e configurazione
    model.save('balanced_hierarchical_genre_model.h5')
    
    # Salva mappings e configurazione
    full_config = {
        'mappings': mappings,
        'class_weights': class_weights,
        'distribution_stats': distribution_stats,
        'training_config': {
            'batch_size': config.BATCH_SIZE,
            'learning_rate': config.LEARNING_RATE,
            'epochs_trained': len(history.history['loss']),
            'use_class_weights': config.USE_CLASS_WEIGHTS,
            'use_weighted_sampling': config.USE_WEIGHTED_SAMPLING,
            'use_focal_loss': config.USE_FOCAL_LOSS
        }
    }
    
    with open('balanced_training_config.json', 'w', encoding='utf-8') as f:
        json.dump(full_config, f, indent=2, ensure_ascii=False, default=str)
    
    print("\n🎉 TRAINING COMPLETATO!")
    print("=" * 50)
    print("📦 File salvati:")
    print("   - balanced_hierarchical_genre_model.h5")
    print("   - balanced_training_config.json")
    print("   - balanced_genre_model_best.h5")
    
    return model, history, full_config

if __name__ == "__main__":
    # Verifica configurazione
    print("🔧 Configurazione:")
    print(f"   Dataset path: {config.FMA_LARGE_PATH}")
    print(f"   Metadata path: {config.METADATA_PATH}")
    print(f"   Batch size: {config.BATCH_SIZE}")
    print(f"   Use class weights: {config.USE_CLASS_WEIGHTS}")
    print(f"   Use weighted sampling: {config.USE_WEIGHTED_SAMPLING}")
    print(f"   Use focal loss: {config.USE_FOCAL_LOSS}")
    
    # Controlla se i path esistono
    if not os.path.exists(config.METADATA_PATH):
        print(f"❌ Path metadata non trovato: {config.METADATA_PATH}")
        exit(1)
    
    if not os.path.exists(config.FMA_LARGE_PATH):
        print(f"❌ Path dataset non trovato: {config.FMA_LARGE_PATH}")
        exit(1)
    
    # Avvia training
    try:
        model, history, config_data = train_balanced_model()
        
        # Stampa statistiche finali
        final_loss = history.history['val_loss'][-1]
        final_acc_top = history.history['val_top_genres_accuracy'][-1]
        
        print(f"\n📊 Risultati finali:")
        print(f"   Validation Loss: {final_loss:.4f}")
        print(f"   Top Genre Accuracy: {final_acc_top:.4f}")
        
    except Exception as e:
        print(f"❌ Errore durante il training: {e}")
        import traceback
        traceback.print_exc()