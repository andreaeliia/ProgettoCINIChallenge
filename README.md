# 🎵 Sistema di Classificazione Generi Musicali Bilanciato

Sistema completo per la classificazione gerarchica di generi musicali utilizzando il dataset **FMA Large** con tecniche avanzate di bilanciamento per dataset sbilanciati.

## 📋 Caratteristiche Principali

- **🎯 Classificazione Gerarchica**: 3 livelli di predizione (Top, Medium, All genres)
- **⚖️ Bilanciamento Avanzato**: Class weights, weighted sampling, focal loss
- **🔧 Pipeline Completa**: Setup, training, inferenza e valutazione
- **📊 Metriche Bilanciate**: Balanced accuracy, F1 macro/weighted, error analysis
- **🎨 Visualizzazioni**: Confusion matrix, confidence analysis, error patterns
- **💾 GPU Ottimizzato**: Mixed precision, memory management

## 🗂️ Struttura del Progetto

```
music-genre-classification/
├── 📄 README.md                    # Questa documentazione
├── 🔧 setup_fma_dataset.py        # Setup automatico dataset FMA
├── 🔍 fma_inspector.py             # Analisi struttura dataset
├── 🏋️ balanced_trainer.py          # Training bilanciato
├── 🎯 balanced_inference.py        # Sistema inferenza
├── 📊 model_evaluation.py          # Valutazione con metriche bilanciate
├── ⚙️ config.py                    # Configurazione base
├── 🚀 quick_start.py               # Script avvio rapido
├── 📋 requirements.txt             # Dipendenze Python
└── 📁 examples/                    # Esempi e tutorial
    ├── example_config.json
    ├── tutorial_training.py
    └── tutorial_inference.py
```

## 🚀 Quick Start

### 1. Installazione Dipendenze

```bash
# Clona il repository
git clone <repository-url>
cd music-genre-classification

# Installa dipendenze
pip install -r requirements.txt

# Verifica installazione GPU (opzionale)
python -c "import tensorflow as tf; print('GPU disponibile:', tf.config.list_physical_devices('GPU'))"
```

### 2. Setup Dataset FMA Large

```bash
# Setup automatico con metadata + FMA Large (richiede ~200GB spazio)
python setup_fma_dataset.py --subset all --symlinks --analyze

# Solo metadata per test (3GB)
python setup_fma_dataset.py --subset metadata --symlinks --analyze

# Setup con verifica integrità
python setup_fma_dataset.py --subset fma_large --no_verify --cleanup
```

### 3. Analisi Dataset

```bash
# Analisi completa struttura dataset
python fma_inspector.py

# Genera configurazione bilanciamento
python fma_inspector.py --generate_config
```

### 4. Training Modello

```bash
# Training completo con bilanciamento
python balanced_trainer.py

# Training con parametri personalizzati
python balanced_trainer.py --batch_size 64 --epochs 50 --focal_loss
```

### 5. Inferenza su File MP3

```bash
# Singolo file
python balanced_inference.py --model balanced_hierarchical_genre_model.h5 \
                              --config balanced_training_config.json \
                              --file example.mp3

# Batch prediction su cartella
python balanced_inference.py --model balanced_hierarchical_genre_model.h5 \
                              --config balanced_training_config.json \
                              --folder music_folder/ \
                              --output results.json
```

### 6. Valutazione Modello

```bash
# Valutazione completa con visualizzazioni
python model_evaluation.py --model balanced_hierarchical_genre_model.h5 \
                           --config balanced_training_config.json \
                           --metadata fma_data/extracted/metadata/fma_metadata \
                           --audio fma_data/extracted/fma_large/fma_large
```

## 📊 Dataset FMA Large

### Informazioni Dataset
- **Tracce**: 106,574 tracce di 30 secondi
- **Generi**: 161 generi sbilanciati
- **Dimensione**: ~93 GB (audio) + 342 MB (metadata)
- **Formato**: MP3, 22050 Hz
- **Licenza**: Creative Commons

### Distribuzione Generi (Esempi)
```
Rock:        13,579 tracce (12.7%)
Electronic:   11,229 tracce (10.5%)
Pop:          5,507 tracce (5.2%)
Hip-Hop:      5,431 tracce (5.1%)
Jazz:         4,832 tracce (4.5%)
...
Spoken:       156 tracce (0.1%)
```

### Gerarchia Generi
- **Top Level**: 16 generi principali (Rock, Electronic, Jazz, etc.)
- **Medium Level**: ~147 sottogeneri
- **All Genres**: 161 generi totali (multi-label)

## ⚖️ Strategie di Bilanciamento

### 1. Class Weights
Pesi automatici per bilanciare l'importanza delle classi:
```python
weight = n_samples / (n_classes * n_samples_per_class)
```

### 2. Weighted Sampling
Campionamento probabilistico che favorisce classi sotto-rappresentate:
- Oversample: generi con < 100 tracce
- Undersample: generi con > 5000 tracce

### 3. Focal Loss
Loss function che si concentra su esempi difficili:
```python
FL(p_t) = -α(1-p_t)^γ log(p_t)
```
- α: bilanciamento classi
- γ: focus su esempi difficili (default: 2.0)

### 4. Multi-Task Learning
Tre head di classificazione con pesi differenti:
- Top genres: peso 1.0 (più importante)
- Medium genres: peso 0.8
- All genres: peso 0.6

## 🏗️ Architettura Modello

### Backbone CNN
```
Input: (128, 64, 1) - Mel Spectrogram
├── Conv2D(64) + BatchNorm + MaxPool + Dropout(0.25)
├── Conv2D(128) + BatchNorm + MaxPool + Dropout(0.25)  
├── Conv2D(256) + BatchNorm + MaxPool + Dropout(0.25)
├── Conv2D(512) + BatchNorm + GlobalAvgPool + Dropout(0.5)
└── Dense(512) + BatchNorm + Dropout(0.5)
```

### Multi-Head Classification
```
Shared Features (512)
├── Top Genres: Dense(16) + Softmax
├── Medium Genres: Dense(147) + Softmax
└── All Genres: Dense(161) + Sigmoid
```

### Ottimizzazioni
- **Mixed Precision**: Training in float16 per velocità
- **Gradient Clipping**: Previene gradient explosion
- **Memory Growth**: Allocazione dinamica GPU
- **Data Pipeline**: tf.data ottimizzato

## 📈 Metriche di Valutazione

### Metriche Principali
- **Balanced Accuracy**: Corregge per sbilanciamento classi
- **F1 Macro**: Media F1 score per classe (senza pesatura)
- **F1 Weighted**: F1 score pesato per frequenza classe
- **Precision/Recall**: Per classe e aggregate

### Analisi Specifiche
- **Error Analysis**: Pattern di errori per genere
- **Confidence Distribution**: Distribuzione confidence predizioni
- **Temporal Stability**: Consistenza predizioni nel tempo
- **Confusion Matrix**: Visualizzazione errori specifici

### Metriche Multi-Label
- **Hamming Loss**: Media etichette incorrette
- **Jaccard Score**: Intersection over Union
- **Label Ranking**: Ordinamento etichette per probabilità

## 🎯 Risultati Attesi

### Performance Target
- **Balanced Accuracy**: >0.70 per top genres
- **F1 Macro**: >0.65 per top genres  
- **F1 Weighted**: >0.75 per top genres
- **Multi-label Jaccard**: >0.45

### Generi Più Sfidanti
- **Experimental**: Spesso confuso con Noise
- **International**: Sovrapposizioni geografiche
- **Instrumental**: Confuso con generi specifici
- **Folk** vs **Country**: Confusione sistematica

## 🔧 Configurazione Avanzata

### config.py - Parametri Principali
```python
# Audio Processing
SAMPLE_RATE = 22050
DURATION = 30.0
N_MELS = 128
HOP_LENGTH = 512
NUM_SEGMENTS = 10

# Training
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
NUM_EPOCHS = 100
PATIENCE = 15

# Bilanciamento
USE_CLASS_WEIGHTS = True
USE_WEIGHTED_SAMPLING = True
USE_FOCAL_LOSS = True
OVERSAMPLE_THRESHOLD = 100
UNDERSAMPLE_THRESHOLD = 5000

# GPU
MIXED_PRECISION = True
```

### Personalizzazione Loss Functions
```python
# Focal Loss personalizzata
focal_loss = FocalLoss(alpha=class_weights, gamma=2.0)

# Binary Cross Entropy pesata
bce_loss = WeightedBinaryCrossEntropy(pos_weight=pos_weights)
```

## 🐛 Troubleshooting

### Problemi Comuni

#### 1. Out of Memory (OOM)
```bash
# Riduce batch size
python balanced_trainer.py --batch_size 16

# Disabilita mixed precision
# In config.py: MIXED_PRECISION = False
```

#### 2. Dataset Non Trovato
```bash
# Verifica path in config.py
python fma_inspector.py --check_paths

# Re-setup dataset
python setup_fma_dataset.py --subset metadata --symlinks
```

#### 3. Modello Non Converge
```bash
# Controlla bilanciamento
python fma_inspector.py --analyze_balance

# Riduci learning rate
# In config.py: LEARNING_RATE = 1e-5
```

#### 4. Predizioni Uniformi
- Verifica che il modello sia addestrato
- Controlla normalizzazione features audio
- Verifica class weights
- Controlla gradient clipping

### Log di Debug
```python
# Abilita debug verbose
import tensorflow as tf
tf.debugging.set_log_device_placement(True)

# Controlla NaN/Inf
tf.debugging.enable_check_numerics()
```

## 📚 Tutorial e Esempi

### Esempio Training Personalizzato
```python
from balanced_trainer import BalancedGenreProcessor, BalancedHierarchicalModel

# Setup processore
processor = BalancedGenreProcessor()
tracks, genres_df = processor.load_fma_metadata()
processed_data = processor.process_fma_labels(tracks, genres_df)

# Calcola class weights
class_weights = processor.compute_hierarchical_class_weights(processed_data)

# Training personalizzato
model_builder = BalancedHierarchicalModel(
    input_shape=(128, 64, 1),
    num_top=16,
    num_medium=147, 
    num_all=161,
    class_weights=class_weights
)

model = model_builder.build_model()
model_builder.compile_with_balanced_losses()
```

### Esempio Inferenza Batch
```python
from balanced_inference import MusicGenreInference

# Inizializza sistema
inference = MusicGenreInference(
    model_path='balanced_hierarchical_genre_model.h5',
    config_path='balanced_training_config.json'
)

# Batch prediction
results = inference.batch_predict('music_folder/')

# Analizza risultati
for filename, result in results['results'].items():
    top_genre = result['aggregated_results']['top_genre']['voting']['name']
    confidence = result['aggregated_results']['top_genre']['average_confidence']['confidence']
    print(f"{filename}: {top_genre} ({confidence:.3f})")
```

## 🤝 Contributi

### Come Contribuire
1. Fork del repository
2. Crea feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push al branch: `git push origin feature/amazing-feature`
5. Apri Pull Request

### Aree di Miglioramento
- [ ] Augmentation audio avanzata
- [ ] Architetture transformer per audio
- [ ] Ensemble methods
- [ ] Transfer learning da modelli pre-addestrati
- [ ] Ottimizzazione iperparametri automatica
- [ ] Support per altri dataset (GTZAN, Million Song Dataset)

## 📄 Licenza

Questo progetto è rilasciato sotto licenza MIT. Vedi `LICENSE` per dettagli.

Il dataset FMA è rilasciato sotto Creative Commons. Cita appropriatamente:

```bibtex
@inproceedings{defferrard2017fma,
  title={FMA: A dataset for music analysis},
  author={Defferrard, Micha{\"e}l and Benzi, Kirell and Vandergheynst, Pierre and Bresson, Xavier},
  booktitle={18th International Society for Music Information Retrieval Conference},
  year={2017}
}
```

## 🔗 Link Utili

- [FMA Dataset](https://github.com/mdeff/fma) - Dataset originale
- [TensorFlow Audio](https://www.tensorflow.org/tutorials/audio) - Tutorial TensorFlow
- [Librosa Documentation](https://librosa.org/) - Processing audio
- [Scikit-learn Metrics](https://scikit-learn.org/stable/modules/model_evaluation.html) - Metriche evaluation

## 📞 Supporto

Per supporto, apri una issue su GitHub o contatta:
- 📧 Email: [your-email@example.com]
- 💬 Discussions: GitHub Discussions
- 📖 Wiki: GitHub Wiki per guide dettagliate

---

**🎵 Happy Music Classification! 🎵**