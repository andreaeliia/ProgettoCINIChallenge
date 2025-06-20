#!/usr/bin/env python3
"""
Script per estrarre i mappings corretti dai metadata FMA
Genera file di configurazione con nomi reali dei generi per tutti i livelli
"""

import os
import pandas as pd
import json
import numpy as np
from collections import Counter, defaultdict
import ast
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer

def load_fma_metadata(metadata_path='modello3/fma_metadata'):
    """Carica metadata FMA"""
    print("📂 Caricamento metadata FMA...")
    
    # Carica tracks.csv
    tracks_path = os.path.join(metadata_path, 'tracks.csv')
    if not os.path.exists(tracks_path):
        raise FileNotFoundError(f"File non trovato: {tracks_path}")
    
    tracks = pd.read_csv(tracks_path, index_col=0, header=[0, 1])
    
    # Carica genres.csv
    genres_path = os.path.join(metadata_path, 'genres.csv')
    if not os.path.exists(genres_path):
        raise FileNotFoundError(f"File non trovato: {genres_path}")
    
    genres_df = pd.read_csv(genres_path, index_col=0)
    
    print(f"✅ Caricati {len(tracks)} tracce e {len(genres_df)} generi")
    return tracks, genres_df

def analyze_genre_hierarchy(genres_df):
    """Analizza la gerarchia dei generi FMA"""
    print("🌳 Analisi gerarchia generi...")
    
    # Trova generi root (parent == 0)
    root_genres = genres_df[genres_df['parent'] == 0]['title'].tolist()
    
    # Analizza struttura gerarchica
    hierarchy = defaultdict(list)
    for _, row in genres_df.iterrows():
        parent_id = row['parent']
        if parent_id != 0:
            parent_name = genres_df.loc[parent_id, 'title']
            hierarchy[parent_name].append(row['title'])
    
    print(f"   🌲 Root genres: {len(root_genres)}")
    print(f"   🍃 Subgenres: {len(genres_df) - len(root_genres)}")
    
    # Mostra alcuni esempi
    print(f"\n📋 Root genres:")
    for i, genre in enumerate(sorted(root_genres)[:10]):
        subgenres_count = len(hierarchy.get(genre, []))
        print(f"   {i+1:2d}. {genre} ({subgenres_count} subgenres)")
    
    return root_genres, hierarchy

def extract_training_data_mappings(tracks, genres_df):
    """Estrae mappings basati sui dati di training effettivi"""
    print("🔍 Estrazione mappings dai dati di training...")
    
    # Mappa ID → Nome genere
    id_to_name = dict(zip(genres_df.index, genres_df['title']))
    
    # Focus su subset large (training data)
    large_subset = tracks[tracks[('set', 'subset')] == 'large']
    print(f"   📊 Dataset large: {len(large_subset)} tracce")
    
    # Raccogli dati per ogni livello
    top_genres = []
    all_genre_lists = []
    
    # Mappings per ogni livello
    top_genre_set = set()
    medium_genre_set = set()
    all_genre_set = set()
    
    processed_count = 0
    valid_tracks = 0
    
    for track_id in large_subset.index:
        try:
            # Genere principale (top level)
            genre_top = large_subset.loc[track_id, ('track', 'genre_top')]
            if pd.isna(genre_top):
                continue
            
            # Aggiungi a collezioni
            top_genres.append(genre_top)
            top_genre_set.add(genre_top)
            medium_genre_set.add(genre_top)  # Top genre è anche medium
            all_genre_set.add(genre_top)     # Top genre è anche all
            
            # Lista completa generi (all genres)
            track_genres = [genre_top]  # Almeno il genere principale
            
            genre_ids_raw = large_subset.loc[track_id, ('track', 'genres')]
            if not pd.isna(genre_ids_raw):
                try:
                    # Parse lista ID generi
                    if isinstance(genre_ids_raw, str) and '[' in genre_ids_raw:
                        genre_ids = ast.literal_eval(genre_ids_raw)
                    else:
                        genre_ids = [int(x.strip()) for x in str(genre_ids_raw).split(',') if x.strip().isdigit()]
                    
                    # Converti ID in nomi
                    for gid in genre_ids:
                        if gid in id_to_name:
                            genre_name = id_to_name[gid]
                            track_genres.append(genre_name)
                            medium_genre_set.add(genre_name)
                            all_genre_set.add(genre_name)
                    
                    # Rimuovi duplicati mantenendo ordine
                    track_genres = list(dict.fromkeys(track_genres))
                    
                except Exception as e:
                    # Se parsing fallisce, usa solo top genre
                    pass
            
            all_genre_lists.append(track_genres)
            valid_tracks += 1
            
            processed_count += 1
            if processed_count % 5000 == 0:
                print(f"   Processate {processed_count} tracce...")
                
        except Exception as e:
            continue
    
    print(f"✅ Processate {valid_tracks} tracce valide su {processed_count}")
    
    # Converti in liste ordinate
    top_genres_list = sorted(list(top_genre_set))
    medium_genres_list = sorted(list(medium_genre_set))
    all_genres_list = sorted(list(all_genre_set))
    
    print(f"\n📊 Mappings estratti:")
    print(f"   🔝 Top genres: {len(top_genres_list)}")
    print(f"   🎯 Medium genres: {len(medium_genres_list)}")
    print(f"   🎵 All genres: {len(all_genres_list)}")
    
    # Statistiche distribuzione
    top_counts = Counter(top_genres)
    print(f"\n📈 Distribuzione top genres:")
    for genre, count in top_counts.most_common(10):
        print(f"   {genre}: {count:,} tracce")
    
    return {
        'top_genres': top_genres_list,
        'medium_genres': medium_genres_list,
        'all_genres': all_genres_list
    }, {
        'top_genre_counts': dict(top_counts),
        'valid_tracks': valid_tracks,
        'total_processed': processed_count
    }

def create_encoder_mappings(mappings):
    """Crea mappings encoder→indice e indice→encoder"""
    print("🔧 Creazione mappings encoder...")
    
    encoder_mappings = {}
    
    # Top genres
    top_encoder = LabelEncoder()
    top_encoder.fit(mappings['top_genres'])
    encoder_mappings['top_encoder_classes'] = top_encoder.classes_.tolist()
    encoder_mappings['top_genre_to_idx'] = {genre: int(idx) for idx, genre in enumerate(top_encoder.classes_)}
    encoder_mappings['top_idx_to_genre'] = {int(idx): genre for idx, genre in enumerate(top_encoder.classes_)}
    
    # Medium genres
    medium_encoder = LabelEncoder()
    medium_encoder.fit(mappings['medium_genres'])
    encoder_mappings['medium_encoder_classes'] = medium_encoder.classes_.tolist()
    encoder_mappings['medium_genre_to_idx'] = {genre: int(idx) for idx, genre in enumerate(medium_encoder.classes_)}
    encoder_mappings['medium_idx_to_genre'] = {int(idx): genre for idx, genre in enumerate(medium_encoder.classes_)}
    
    # All genres (multi-label)
    all_encoder = MultiLabelBinarizer()
    # Crea dataset dummy per fit
    dummy_data = [[genre] for genre in mappings['all_genres']]
    all_encoder.fit(dummy_data)
    encoder_mappings['all_encoder_classes'] = all_encoder.classes_.tolist()
    encoder_mappings['all_genre_to_idx'] = {genre: int(idx) for idx, genre in enumerate(all_encoder.classes_)}
    encoder_mappings['all_idx_to_genre'] = {int(idx): genre for idx, genre in enumerate(all_encoder.classes_)}
    
    print(f"   ✅ Encoder mappings creati")
    print(f"      Top: {len(encoder_mappings['top_encoder_classes'])} classi")
    print(f"      Medium: {len(encoder_mappings['medium_encoder_classes'])} classi") 
    print(f"      All: {len(encoder_mappings['all_encoder_classes'])} classi")
    
    return encoder_mappings

def update_config_file(config_path, mappings, encoder_mappings, stats):
    """Aggiorna il file di configurazione"""
    print(f"💾 Aggiornamento configurazione: {config_path}")
    
    # Carica configurazione esistente se presente
    config_data = {}
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            print(f"   📂 Configurazione esistente caricata")
        except:
            print(f"   ⚠️ Errore lettura config esistente, creando nuovo file")
    
    # Aggiorna con nuovi mappings
    config_data.update({
        'mappings': mappings,
        'encoder_mappings': encoder_mappings,
        'extraction_stats': stats,
        'extraction_info': {
            'extraction_date': pd.Timestamp.now().isoformat(),
            'source': 'FMA metadata extraction',
            'total_genres_found': {
                'top': len(mappings['top_genres']),
                'medium': len(mappings['medium_genres']),
                'all': len(mappings['all_genres'])
            }
        }
    })
    
    # Salva configurazione aggiornata
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config_data, f, indent=2, ensure_ascii=False)
    
    print(f"   ✅ Configurazione salvata: {config_path}")

def create_test_script_update(mappings):
    """Crea patch per aggiornare il test script"""
    print("🔧 Creazione patch per test script...")
    
    patch_content = f'''
# PATCH PER test_mp3_fixed.py
# Sostituisci la funzione _create_default_mappings con:

def _create_default_mappings(self):
    """Crea mappings reali estratti dai metadata FMA"""
    self.mappings = {{
        'top_genres': {mappings['top_genres']},
        'medium_genres': {mappings['medium_genres']},
        'all_genres': {mappings['all_genres']}
    }}
    print("   📝 Usati mappings reali estratti dai metadata FMA")
    print(f"      Top: {{len(self.mappings['top_genres'])}} generi")
    print(f"      Medium: {{len(self.mappings['medium_genres'])}} generi") 
    print(f"      All: {{len(self.mappings['all_genres'])}} generi")
'''
    
    with open('test_script_patch.txt', 'w', encoding='utf-8') as f:
        f.write(patch_content)
    
    print("   📄 Patch salvata in: test_script_patch.txt")

def validate_mappings(mappings):
    """Valida i mappings estratti"""
    print("✅ Validazione mappings...")
    
    issues = []
    
    # Verifica che top genres sia subset di medium e all
    top_set = set(mappings['top_genres'])
    medium_set = set(mappings['medium_genres'])
    all_set = set(mappings['all_genres'])
    
    if not top_set.issubset(medium_set):
        issues.append("Top genres non è subset di medium genres")
    
    if not top_set.issubset(all_set):
        issues.append("Top genres non è subset di all genres")
    
    # Verifica dimensioni ragionevoli
    if len(mappings['top_genres']) < 5:
        issues.append(f"Troppo pochi top genres: {len(mappings['top_genres'])}")
    
    if len(mappings['top_genres']) > 50:
        issues.append(f"Troppi top genres: {len(mappings['top_genres'])}")
    
    if issues:
        print("   ⚠️ Issues trovati:")
        for issue in issues:
            print(f"      - {issue}")
    else:
        print("   ✅ Mappings validati con successo")
    
    return len(issues) == 0

def main():
    """Funzione principale"""
    print("🎵 ESTRAZIONE MAPPINGS FMA")
    print("=" * 50)
    
    # Configurazione path
    metadata_path = 'modello3/fma_metadata'
    config_path = 'balanced_training_config.json'
    
    # Verifica prerequisiti
    if not os.path.exists(metadata_path):
        print(f"❌ Path metadata non trovato: {metadata_path}")
        print("   Assicurati che i metadata FMA siano nella posizione corretta")
        return
    
    try:
        # 1. Carica metadata
        tracks, genres_df = load_fma_metadata(metadata_path)
        
        # 2. Analizza gerarchia
        root_genres, hierarchy = analyze_genre_hierarchy(genres_df)
        
        # 3. Estrai mappings dai dati di training
        mappings, stats = extract_training_data_mappings(tracks, genres_df)
        
        # 4. Valida mappings
        if not validate_mappings(mappings):
            print("❌ Validazione mappings fallita")
            return
        
        # 5. Crea encoder mappings
        encoder_mappings = create_encoder_mappings(mappings)
        
        # 6. Aggiorna configurazione
        update_config_file(config_path, mappings, encoder_mappings, stats)
        
        # 7. Crea patch per test script
        create_test_script_update(mappings)
        
        # 8. Salva backup dettagliato
        backup_data = {
            'mappings': mappings,
            'encoder_mappings': encoder_mappings,
            'hierarchy': dict(hierarchy),
            'root_genres': root_genres,
            'stats': stats
        }
        
        with open('fma_mappings_complete.json', 'w', encoding='utf-8') as f:
            json.dump(backup_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n🎉 ESTRAZIONE COMPLETATA!")
        print(f"📦 File creati/aggiornati:")
        print(f"   - {config_path} (configurazione aggiornata)")
        print(f"   - fma_mappings_complete.json (backup completo)")
        print(f"   - test_script_patch.txt (patch per test script)")
        
        print(f"\n🎯 Prossimi passi:")
        print(f"1. Verifica la configurazione aggiornata:")
        print(f"   python -c \"import json; print(json.load(open('{config_path}'))['mappings'].keys())\"")
        print(f"")
        print(f"2. Testa il modello aggiornato:")
        print(f"   python test_mp3_fixed.py --model balanced_genre_model_best.h5 \\")
        print(f"                             --config {config_path} \\")
        print(f"                             --file modello3/212.mp3 \\")
        print(f"                             --save results_final.json")
        
        print(f"\n📊 Riassunto:")
        print(f"   🔝 Top genres: {len(mappings['top_genres'])}")
        print(f"   🎯 Medium genres: {len(mappings['medium_genres'])}")
        print(f"   🎵 All genres: {len(mappings['all_genres'])}")
        print(f"   📈 Tracce processate: {stats['valid_tracks']:,}")
        
    except Exception as e:
        print(f"❌ Errore durante l'estrazione: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()