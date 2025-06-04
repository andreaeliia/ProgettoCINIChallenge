from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
import ast
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import json
from config import METADATA_PATH,GPU_AVAILABLE
from utils.audio import audio_path ,extract_features



class HierarchicalGenreProcessor:
    """
    Classe per processare la struttura gerarchica completa dei generi FMA
    """
    def __init__(self):
        self.genres_df = None
        self.top_encoder = LabelEncoder()
        self.medium_encoder = LabelEncoder()
        self.all_encoder = MultiLabelBinarizer()
        self.hierarchy_mapping = {}
        
    def load_complete_metadata(self):
        """Carica tutti i metadati dei generi"""
        print("Caricamento metadati completi...")
        
        # Carica tracks.csv
        tracks_path = os.path.join(METADATA_PATH, 'tracks.csv')
        tracks = pd.read_csv(tracks_path, index_col=0, header=[0, 1])
        
        # Carica genres.csv per la gerarchia
        genres_path = os.path.join(METADATA_PATH, 'genres.csv')
        self.genres_df = pd.read_csv(genres_path, index_col=0)
        
        print(f"Generi nel database: {len(self.genres_df)}")
        print("Struttura gerarchia generi:")
        print(self.genres_df.head(10))
        
        # Estrai tutte le informazioni sui generi
        genre_top = tracks[('track', 'genre_top')].dropna()
        genres_ids = tracks[('track', 'genres')].dropna()  # Lista di ID generi
        
        return genre_top, genres_ids, tracks
    
    def build_genre_hierarchy(self):
        """Costruisce la gerarchia completa dei generi"""
        print("Costruzione gerarchia generi...")
        
        # Mappa ID -> Nome genere
        id_to_name = dict(zip(self.genres_df.index, self.genres_df['title']))
        
        # Costruisce gerarchia: figlio -> genitore
        hierarchy = {}
        for idx, row in self.genres_df.iterrows():
            genre_name = row['title']
            parent_id = row.get('parent', 0)
            
            if parent_id != 0 and parent_id in id_to_name:
                parent_name = id_to_name[parent_id]
                hierarchy[genre_name] = parent_name
            else:
                hierarchy[genre_name] = None  # Genere top-level
        
        # Trova i livelli
        top_level = [g for g, p in hierarchy.items() if p is None]
        mid_level = [g for g, p in hierarchy.items() if p in top_level]
        all_genres = list(self.genres_df['title'])
        
        print(f"Generi top-level: {len(top_level)}")
        print(f"Generi mid-level: {len(mid_level)}")
        print(f"Tutti i generi: {len(all_genres)}")
        
        return hierarchy, top_level, mid_level, all_genres, id_to_name
    
    def process_genre_labels(self, genre_top, genres_ids, id_to_name):
        """Processa le etichette per tutti i livelli"""
        processed_data = []
        
        print("Processamento etichette multi-livello...")
        
        for track_id in genre_top.index:
            if track_id in genres_ids.index:
                # Genere principale
                top_genre = genre_top[track_id]
                
                # Lista di generi (da stringa di ID a nomi)
                genre_id_str = str(genres_ids[track_id])
                if genre_id_str and genre_id_str != 'nan':
                    try:
                        # Converti stringa di ID in lista
                        if '[' in genre_id_str:
                            genre_ids_list = ast.literal_eval(genre_id_str)
                        else:
                            genre_ids_list = [int(x.strip()) for x in genre_id_str.split(',')]
                        
                        # Converti ID in nomi
                        genre_names = [id_to_name.get(gid, f"Unknown_{gid}") 
                                     for gid in genre_ids_list if gid in id_to_name]
                        
                        if genre_names:  # Solo se abbiamo generi validi
                            processed_data.append({
                                'track_id': track_id,
                                'genre_top': top_genre,
                                'genres_all': genre_names,
                                'primary_genre': genre_names[0] if genre_names else top_genre
                            })
                    except:
                        # Fallback al genere principale
                        processed_data.append({
                            'track_id': track_id,
                            'genre_top': top_genre,
                            'genres_all': [top_genre],
                            'primary_genre': top_genre
                        })
        
        print(f"Processate {len(processed_data)} tracce con etichette multi-livello")
        return processed_data
    
    def prepare_hierarchical_dataset(processor, processed_data, limit=None):
        """Prepara dataset con etichette gerarchiche - OTTIMIZZATO PER GPU"""
        
        features = []
        top_labels = []
        medium_labels = []
        all_labels = []
        
        print(f"Preparazione dataset gerarchico...")
        count = 0
        successful = 0
        
        # ==========================================
        # ELABORAZIONE CON GESTIONE MEMORIA GPU
        # ==========================================
        for item in processed_data:
            if limit and count >= limit:
                break
                
            track_id = item['track_id']
            path = audio_path(track_id)
            
            if os.path.exists(path):
                count += 1
                if count % 100 == 0:
                    print(f"Processate {count} tracce, successo: {successful}")
                    # Libera memoria ogni 100 tracce se su GPU
                    if GPU_AVAILABLE:
                        tf.keras.backend.clear_session()
                
                track_features = extract_features(path)
                if track_features is not None:
                    for segment_features in track_features:
                        features.append(segment_features)
                        top_labels.append(item['genre_top'])
                        medium_labels.append(item['primary_genre'])
                        all_labels.append(item['genres_all'])
                    
                    successful += 1
        
        print(f"Dataset preparato: {len(features)} segmenti da {successful} tracce")
        
        # ==========================================
        # CONVERSIONE OTTIMIZZATA PER GPU
        # ==========================================
        print("Conversione dati per GPU...")
        
        # Converti in arrays con dtype ottimizzato per GPU
        with tf.device('/GPU:0' if GPU_AVAILABLE else '/CPU:0'):
            features = np.array(features, dtype=np.float32)[..., np.newaxis]  # float32 per GPU
            
            # Encode labels per ogni livello
            y_top = tf.keras.utils.to_categorical(
                processor.top_encoder.transform(top_labels), 
                num_classes=len(processor.top_encoder.classes_),
                dtype='float32'  # Importante per Mixed Precision
            )
            
            y_medium = tf.keras.utils.to_categorical(
                processor.medium_encoder.transform(medium_labels),
                num_classes=len(processor.medium_encoder.classes_),
                dtype='float32'
            )
            
            y_all = processor.all_encoder.transform(all_labels).astype(np.float32)
        
        print(f"✅ Dati preparati per {'GPU' if GPU_AVAILABLE else 'CPU'}")
        return features, {'top': y_top, 'medium': y_medium, 'all': y_all}
    def create_hierarchical_encoders(self, processed_data, hierarchy):
        """Crea encoder per ogni livello della gerarchia"""
        
        # Raccogli tutti i generi per livello
        top_genres = set()
        mid_genres = set()
        all_genre_combinations = []
        
        for item in processed_data:
            # Top level (genere principale)
            top_genres.add(item['genre_top'])
            
            # Mid level (genere primario specifico)
            mid_genres.add(item['primary_genre'])
            
            # All levels (multi-label)
            all_genre_combinations.append(item['genres_all'])
        
        # Fit degli encoder
        self.top_encoder.fit(list(top_genres))
        self.medium_encoder.fit(list(mid_genres))
        self.all_encoder.fit(all_genre_combinations)
        
        print(f"Encoder creati:")
        print(f"- Top level: {len(self.top_encoder.classes_)} generi")
        print(f"- Medium level: {len(self.medium_encoder.classes_)} generi")
        print(f"- All levels: {len(self.all_encoder.classes_)} generi")
        
        # Salva mapping
        mappings = {
            'top_genres': list(self.top_encoder.classes_),
            'medium_genres': list(self.medium_encoder.classes_),
            'all_genres': list(self.all_encoder.classes_),
            'hierarchy': hierarchy
        }
        
        with open('hierarchical_genre_mappings.json', 'w', encoding='utf-8') as f:
            json.dump(mappings, f, indent=2, ensure_ascii=False)
        
        return mappings