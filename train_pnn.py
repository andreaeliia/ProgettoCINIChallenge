# Music Genre Classification con PANNs CNN14
# Dataset: FMA Large
# Approccio: Fine-tuning di modello pre-addestrato

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import pandas as pd
import numpy as np
import librosa
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import audiomentations as A
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ==================== CONFIGURAZIONE ====================
class Config:
    # Audio settings
    SAMPLE_RATE = 32000
    DURATION = 10.0  # secondi
    N_SAMPLES = int(SAMPLE_RATE * DURATION)
    
    # Training settings
    BATCH_SIZE = 16
    NUM_EPOCHS = 15
    LEARNING_RATE = 1e-4
    NUM_WORKERS = 4
    
    # Model settings
    NUM_CLASSES = 16  # Numero generi principali FMA
    DROPOUT = 0.5
    
    # Paths
    DATA_PATH = "fma_large"
    METADATA_PATH = "/mnt/HDD500"
    MODEL_SAVE_PATH = "models/"
    
    # Device
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = Config()

# ==================== PREPROCESSING E DATASET ====================
class AudioPreprocessor:
    def __init__(self, sample_rate=32000, duration=10.0):
        self.sample_rate = sample_rate
        self.duration = duration
        self.n_samples = int(sample_rate * duration)
        
        # Data Augmentation pipeline
        self.augment = A.Compose([
            A.AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.3),
            A.TimeStretch(min_rate=0.8, max_rate=1.2, p=0.3),
            A.PitchShift(min_semitones=-4, max_semitones=4, p=0.3),
            A.Shift(min_shift=-0.4, max_shift=0.4, p=0.3),
        ])
    
    def load_audio(self, filepath):
        """Carica e preprocessa un file audio"""
        try:
            # Carica con librosa per maggiore compatibilità con MP3
            audio, sr = librosa.load(filepath, sr=self.sample_rate, duration=self.duration)
            
            # Pad o tronca a lunghezza fissa
            if len(audio) < self.n_samples:
                audio = np.pad(audio, (0, self.n_samples - len(audio)), mode='constant')
            else:
                audio = audio[:self.n_samples]
            
            # Normalizzazione
            audio = librosa.util.normalize(audio)
            
            return audio.astype(np.float32)
        except Exception as e:
            print(f"Errore nel caricamento di {filepath}: {e}")
            return np.zeros(self.n_samples, dtype=np.float32)
    
    def apply_augmentation(self, audio):
        """Applica data augmentation"""
        return self.augment(samples=audio, sample_rate=self.sample_rate)

class FMADataset(Dataset):
    def __init__(self, df, data_path, preprocessor, is_training=True):
        self.df = df.reset_index(drop=True)
        self.data_path = Path(data_path)
        self.preprocessor = preprocessor
        self.is_training = is_training
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Costruisci il path del file
        track_id = f"{row['track_id']:06d}"
        subdir = track_id[:3]
        filepath = self.data_path / subdir / f"{track_id}.mp3"
        
        # Carica l'audio
        audio = self.preprocessor.load_audio(str(filepath))
        
        # Applica augmentation solo durante training
        if self.is_training:
            audio = self.preprocessor.apply_augmentation(audio)
        
        # Converti a tensor
        audio = torch.tensor(audio, dtype=torch.float32)
        label = torch.tensor(row['genre_encoded'], dtype=torch.long)
        
        return audio, label

# ==================== MODELLO PANNs CNN14 ====================
class CNN14(nn.Module):
    """
    CNN14 architettura da PANNs
    Semplificata per focus su implementazione
    """
    def __init__(self, sample_rate=32000, window_size=1024, hop_size=320, 
                 mel_bins=64, fmin=50, fmax=14000, classes_num=527):
        super(CNN14, self).__init__()
        
        window = 'hann'
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None
        
        # Spectrogram extractor
        self.spectrogram_extractor = torchaudio.transforms.Spectrogram(
            n_fft=window_size, hop_length=hop_size, 
            win_length=window_size, window_fn=torch.hann_window,
            center=center, pad_mode=pad_mode, power=2.0)
        
        # Logmel feature extractor
        self.logmel_extractor = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate, n_fft=window_size, 
            win_length=window_size, hop_length=hop_size,
            f_min=fmin, f_max=fmax, n_mels=mel_bins)
        
        self.bn0 = nn.BatchNorm2d(mel_bins)
        
        # CNN blocks
        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)
        self.conv_block5 = ConvBlock(in_channels=512, out_channels=1024)
        self.conv_block6 = ConvBlock(in_channels=1024, out_channels=2048)
        
        self.fc1 = nn.Linear(2048, 2048, bias=True)
        self.fc_audioset = nn.Linear(2048, classes_num, bias=True)
        
    def forward(self, input):
        """
        Input: (batch_size, data_length)
        """
        x = self.logmel_extractor(input)  # (batch_size, mel_bins, time_steps)
        x = x.transpose(1, 2)
        x = x.unsqueeze(1)      # (batch_size, 1, time_steps, mel_bins)
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)
        
        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block5(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block6(x, pool_size=(1, 1), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        
        x = torch.mean(x, dim=3)
        
        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu_(self.fc1(x))
        embedding = F.dropout(x, p=0.5, training=self.training)
        clipwise_output = torch.sigmoid(self.fc_audioset(embedding))
        
        return clipwise_output, embedding

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=in_channels, 
                              out_channels=out_channels,
                              kernel_size=(3, 3), stride=(1, 1),
                              padding=(1, 1), bias=False)
                              
        self.conv2 = nn.Conv2d(in_channels=out_channels, 
                              out_channels=out_channels,
                              kernel_size=(3, 3), stride=(1, 1),
                              padding=(1, 1), bias=False)
                              
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
    def forward(self, input, pool_size=(2, 2), pool_type='avg'):
        x = input
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        if pool_type == 'max':
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg':
            x = F.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg+max':
            x1 = F.avg_pool2d(x, kernel_size=pool_size)
            x2 = F.max_pool2d(x, kernel_size=pool_size)
            x = x1 + x2
        else:
            raise Exception('Incorrect argument!')
        
        return x

# ==================== MODELLO PER FINE-TUNING ====================
class MusicGenreClassifier(pl.LightningModule):
    def __init__(self, num_classes, class_weights=None, learning_rate=1e-4):
        super().__init__()
        self.save_hyperparameters()
        
        # Carica CNN14 pre-addestrato
        self.backbone = CNN14(classes_num=527)  # AudioSet classes
        
        # Freeze initial layers per transfer learning
        self.freeze_backbone_layers()
        
        # Custom classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(config.DROPOUT),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
        # Loss function con class weights
        if class_weights is not None:
            self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        else:
            self.criterion = nn.CrossEntropyLoss()
            
        self.learning_rate = learning_rate
        
        # Metriche
        self.train_acc = pl.metrics.Accuracy(task='multiclass', num_classes=num_classes)
        self.val_acc = pl.metrics.Accuracy(task='multiclass', num_classes=num_classes)
        
    def freeze_backbone_layers(self):
        """Freeze i primi layer del backbone"""
        # Freeze tutti i conv blocks tranne gli ultimi 2
        modules_to_freeze = [
            self.backbone.conv_block1,
            self.backbone.conv_block2, 
            self.backbone.conv_block3,
            self.backbone.conv_block4
        ]
        
        for module in modules_to_freeze:
            for param in module.parameters():
                param.requires_grad = False
    
    def unfreeze_backbone(self):
        """Unfreeze backbone per fine-tuning completo"""
        for param in self.backbone.parameters():
            param.requires_grad = True
    
    def forward(self, x):
        _, embeddings = self.backbone(x)
        output = self.classifier(embeddings)
        return output
    
    def training_step(self, batch, batch_idx):
        audio, labels = batch
        outputs = self(audio)
        loss = self.criterion(outputs, labels)
        
        # Metriche
        acc = self.train_acc(outputs, labels)
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        audio, labels = batch
        outputs = self(audio)
        loss = self.criterion(outputs, labels)
        
        # Metriche
        acc = self.val_acc(outputs, labels)
        
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        self.log('val_acc', acc, on_epoch=True, prog_bar=True)
        
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=3, factor=0.5
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val_loss'
        }

# ==================== DATA LOADING E PREPROCESSING ====================
class FMADataModule(pl.LightningDataModule):
    def __init__(self, data_path, metadata_path, batch_size=16, num_workers=4):
        super().__init__()
        self.data_path = data_path
        self.metadata_path = metadata_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.preprocessor = AudioPreprocessor()
        
    def setup(self, stage=None):
        # Carica metadata


        # Aggiungi questo all'inizio di setup() in FMADataModule
        print("Ispezione struttura dataset FMA...")
        tracks = pd.read_csv(self.metadata_path, index_col=0, header=[0, 1])

        # Stampa tutte le colonne
        print("Colonne disponibili:")
        for col in tracks.columns:
            print(f"  {col}")

        # Controlla colonne generi specificamente
        genre_columns = [col for col in tracks.columns if 'genre' in str(col).lower()]
        print(f"\nColonne generi trovate: {genre_columns}")




        exit(0)

        # Esamina i primi esempi
        for col in genre_columns:
            print(f"\n{col} - primi 10 valori unici:")
            unique_vals = tracks[col].dropna().unique()[:10]
            print(f"  {unique_vals}")
            print(f"  Totale valori unici: {len(tracks[col].dropna().unique())}")
        print("Caricamento metadata FMA...")
        tracks = pd.read_csv(self.metadata_path, index_col=0, header=[0, 1])
        
        # Seleziona subset con generi principali
        subset = tracks[('set', 'subset')] == 'large'
        tracks_subset = tracks[subset].copy()
        
        # Estrai generi top-level
        genres = tracks_subset[('track', 'genre_top')].dropna()
        
        # Encode labels
        self.label_encoder = LabelEncoder()
        genre_encoded = self.label_encoder.fit_transform(genres)
        
        # Crea DataFrame per training
        df = pd.DataFrame({
            'track_id': genres.index,
            'genre': genres.values,
            'genre_encoded': genre_encoded
        })
        
        # Bilancia il dataset (opzionale: sample per classe)
        df = self.balance_dataset(df)
        
        # Split train/val/test
        train_df, temp_df = train_test_split(df, test_size=0.3, stratify=df['genre_encoded'], random_state=42)
        val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['genre_encoded'], random_state=42)
        
        # Calcola class weights
        self.class_weights = torch.tensor(
            compute_class_weight('balanced', 
                               classes=np.unique(genre_encoded), 
                               y=genre_encoded),
            dtype=torch.float32
        )
        
        # Crea datasets
        self.train_dataset = FMADataset(train_df, self.data_path, self.preprocessor, is_training=True)
        self.val_dataset = FMADataset(val_df, self.data_path, self.preprocessor, is_training=False)
        self.test_dataset = FMADataset(test_df, self.data_path, self.preprocessor, is_training=False)
        
        self.num_classes = len(self.label_encoder.classes_)
        print(f"Dataset caricato: {len(train_df)} train, {len(val_df)} val, {len(test_df)} test")
        print(f"Classi: {self.label_encoder.classes_}")
    
    def balance_dataset(self, df, max_samples_per_class=5000):
        """Bilancia il dataset limitando samples per classe"""
        balanced_dfs = []
        for genre in df['genre'].unique():
            genre_df = df[df['genre'] == genre]
            if len(genre_df) > max_samples_per_class:
                genre_df = genre_df.sample(n=max_samples_per_class, random_state=42)
            balanced_dfs.append(genre_df)
        return pd.concat(balanced_dfs, ignore_index=True)
    
    def train_dataloader(self):
        # Weighted sampling per bilanciamento
        sample_weights = [1.0 / self.class_weights[label] for label in self.train_dataset.df['genre_encoded']]
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
        
        return DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size,
            sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

# ==================== TRAINING PIPELINE ====================
def train_model():
    """Pipeline completa di training"""
    
    # Setup data
    print("Setup del dataset...")
    dm = FMADataModule(
        data_path=config.DATA_PATH,
        metadata_path=config.METADATA_PATH,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS
    )
    dm.setup()
    
    # Setup model
    print("Setup del modello...")
    model = MusicGenreClassifier(
        num_classes=dm.num_classes,
        class_weights=dm.class_weights.to(config.DEVICE),
        learning_rate=config.LEARNING_RATE
    )
    
    # Callbacks
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=config.MODEL_SAVE_PATH,
        filename='music-genre-{epoch:02d}-{val_acc:.3f}',
        monitor='val_acc',
        mode='max',
        save_top_k=3
    )
    
    early_stop_callback = pl.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        mode='min'
    )
    
    # Trainer
    trainer = pl.Trainer(
        max_epochs=config.NUM_EPOCHS,
        gpus=1 if torch.cuda.is_available() else 0,
        callbacks=[checkpoint_callback, early_stop_callback],
        precision=16,  # Mixed precision
        gradient_clip_val=1.0,
    )
    
    # Phase 1: Train classifier solo (backbone frozen)
    print("Phase 1: Training classifier (backbone frozen)...")
    trainer.fit(model, dm)
    
    # Phase 2: Fine-tune tutto il modello
    print("Phase 2: Fine-tuning completo...")
    model.unfreeze_backbone()
    model.learning_rate = config.LEARNING_RATE / 10  # Riduce learning rate
    trainer.fit(model, dm, ckpt_path=checkpoint_callback.best_model_path)
    
    # Test finale
    print("Valutazione finale...")
    trainer.test(model, dm, ckpt_path=checkpoint_callback.best_model_path)
    
    return model, dm

# ==================== VALUTAZIONE ====================
def evaluate_model(model, dataloader, label_encoder):
    """Valutazione dettagliata del modello"""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            audio, labels = batch
            outputs = model(audio.to(config.DEVICE))
            preds = torch.argmax(outputs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Report di classificazione
    report = classification_report(
        all_labels, all_preds,
        target_names=label_encoder.classes_,
        output_dict=True
    )
    
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=label_encoder.classes_))
    
    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    print(f"\nConfusion Matrix Shape: {cm.shape}")
    
    return report, cm

# ==================== MAIN ====================
if __name__ == "__main__":
    # Crea directory per salvataggio
    os.makedirs(config.MODEL_SAVE_PATH, exist_ok=True)
    
    print("Avvio training Music Genre Classification con PANNs...")
    print(f"Device: {config.DEVICE}")
    print(f"Configurazione: {config.BATCH_SIZE} batch size, {config.NUM_EPOCHS} epochs")
    
    # Training
    model, dm = train_model()
    
    # Valutazione finale
    print("\n" + "="*50)
    print("VALUTAZIONE FINALE")
    print("="*50)
    
    evaluate_model(model, dm.test_dataloader(), dm.label_encoder)
    
    print("\nTraining completato!")
    print(f"Modelli salvati in: {config.MODEL_SAVE_PATH}")