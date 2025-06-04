import json
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TerminateOnNaN
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Model
import tensorflow.keras.layers as layers
import numpy as np

from config import BATCH_SIZE

class HierarchicalGenreModel:
    def __init__(self, input_shape, num_top, num_medium, num_all, mixed_precision=False, gpu_available=True):
        self.input_shape = input_shape
        self.num_top = num_top
        self.num_medium = num_medium
        self.num_all = num_all
        self.mixed_precision = mixed_precision
        self.gpu_available = gpu_available
        self.model = None
        self.history = None

    def build(self):
        device = '/GPU:0' if self.gpu_available else '/CPU:0'

        with tf.device(device):
            inputs = Input(shape=self.input_shape, name='audio_input', dtype='float32')
            
            # PREPROCESSING MINIMO - i dati sono già puliti
            x = inputs  # I dati sono già normalizzati!
            
            # Architettura semplice e stabile
            x = layers.Conv2D(32, 3, padding='same', activation='relu',
                             kernel_initializer='he_normal')(x)
            x = layers.BatchNormalization()(x)
            x = layers.MaxPooling2D(2)(x)
            x = layers.Dropout(0.2)(x)

            x = layers.Conv2D(64, 3, padding='same', activation='relu',
                             kernel_initializer='he_normal')(x)
            x = layers.BatchNormalization()(x)  
            x = layers.MaxPooling2D(2)(x)
            x = layers.Dropout(0.3)(x)

            x = layers.Conv2D(128, 3, padding='same', activation='relu',
                             kernel_initializer='he_normal')(x)
            x = layers.BatchNormalization()(x)
            x = layers.GlobalAveragePooling2D()(x)
            x = layers.Dropout(0.4)(x)
            
            # Dense layers
            shared = Dense(512, activation='relu',
                          kernel_initializer='he_normal')(x)
            shared = BatchNormalization()(shared)
            shared = Dropout(0.5)(shared)

            # Output branches semplici
            top_output = Dense(self.num_top, activation='softmax', name='top_genres')(shared)
            med_output = Dense(self.num_medium, activation='softmax', name='medium_genres')(shared)
            all_output = Dense(self.num_all, activation='sigmoid', name='all_genres')(shared)

            self.model = Model(inputs, [top_output, med_output, all_output])
            
            # Optimizer semplice ma con gradient clipping
            opt = tf.keras.optimizers.Adam(
                learning_rate=1e-4,  # LR normale
                clipnorm=1.0         # Gradient clipping
            )
            
            self.model.compile(
                optimizer=opt,
                loss={
                    'top_genres': 'categorical_crossentropy',
                    'medium_genres': 'categorical_crossentropy', 
                    'all_genres': 'binary_crossentropy'
                },
                loss_weights={'top_genres': 1.0, 'medium_genres': 1.0, 'all_genres': 0.8},
                metrics={
                    'top_genres': ['accuracy'],
                    'medium_genres': ['accuracy'], 
                    'all_genres': ['binary_accuracy', 'precision', 'recall']
                }
            )
            
        return self.model

    def train(self, dataset):
        print(f"\n=== TRAINING SEMPLICE E STABILE SU {'GPU' if self.gpu_available else 'CPU'} ===")
        print(f"Mixed Precision: {'✅ Abilitata' if self.mixed_precision else '❌ Disabilitata'}")

        # Dataset preparation SENZA pulizia extra (i dati sono già puliti)
        dataset = dataset.unbatch()
        
        print("🔄 Preparazione dataset...")
        dataset = dataset.shuffle(10000, reshuffle_each_iteration=False)
        total_size = sum(1 for _ in dataset)
        train_size = int(0.7 * total_size)
        val_size = int(0.15 * total_size)
        
        print(f"📊 Dataset split: Train={train_size} | Val={val_size}")

        train_dataset = dataset.take(train_size).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
        val_dataset = dataset.skip(train_size).take(val_size).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
        test_dataset = dataset.skip(train_size + val_size).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

        steps_per_epoch = train_size // BATCH_SIZE
        validation_steps = val_size // BATCH_SIZE

        # Callback semplice ma efficace
        class SimpleStableCallback(tf.keras.callbacks.Callback):
            def on_epoch_begin(self, epoch, logs=None):
                print(f"\n🚀 Epoca {epoch + 1}")
                
            def on_batch_end(self, batch, logs=None):
                if batch % 1000 == 0 and logs:
                    loss = logs.get('loss', 0)
                    if np.isnan(loss) or np.isinf(loss):
                        print(f"\n🚨 NaN rilevato al batch {batch}!")
                        self.model.stop_training = True
                    else:
                        print(f"Batch {batch}: Loss={loss:.4f}")
                
            def on_epoch_end(self, epoch, logs=None):
                if logs:
                    # Controllo NaN semplice
                    has_nan = any(np.isnan(v) or np.isinf(v) for v in logs.values() if isinstance(v, (int, float)))
                    
                    if not has_nan:
                        print(f"✅ Epoca {epoch + 1} STABILE:")
                        print(f"   Loss: {logs.get('loss', 0):.6f} | Val Loss: {logs.get('val_loss', 0):.6f}")
                        print(f"   Top Acc: {logs.get('top_genres_accuracy', 0):.4f} | Val: {logs.get('val_top_genres_accuracy', 0):.4f}")
                        print(f"   Med Acc: {logs.get('medium_genres_accuracy', 0):.4f} | Val: {logs.get('val_medium_genres_accuracy', 0):.4f}")
                        print(f"   All Acc: {logs.get('all_genres_binary_accuracy', 0):.4f} | Val: {logs.get('val_all_genres_binary_accuracy', 0):.4f}")
                    else:
                        print(f"❌ Epoca {epoch + 1} - NaN rilevati!")
                        for k, v in logs.items():
                            if np.isnan(v) or np.isinf(v):
                                print(f"   {k}: {v}")

        # Callbacks essenziali
        callbacks = [
            TerminateOnNaN(),
            ModelCheckpoint('simple_stable_model.h5', save_best_only=True, monitor='val_loss', verbose=1),
            EarlyStopping(patience=15, monitor='val_loss', restore_best_weights=True, verbose=1),
            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1),
            SimpleStableCallback()
        ]

        print(f"\n🎯 Training: {steps_per_epoch} steps/epoca")
        print("🛡️ Protezioni: NaN detection, Gradient clipping")
        print("=" * 60)

        try:
            self.history = self.model.fit(
                train_dataset,
                validation_data=val_dataset,
                epochs=100,
                steps_per_epoch=steps_per_epoch,
                validation_steps=validation_steps,
                callbacks=callbacks,
                verbose=1
            )
            
            print("\n🎉 TRAINING COMPLETATO!")
            
        except Exception as e:
            print(f"\n❌ Errore: {e}")
            return None, None

        # Valutazione finale
        print("\n📊 Valutazione finale...")
        try:
            results = self.model.evaluate(test_dataset, verbose=0)
            
            print("🏆 RISULTATI FINALI:")
            print(f"   Top Accuracy: {results[4]:.4f}")
            print(f"   Medium Accuracy: {results[5]:.4f}")
            print(f"   All Binary Accuracy: {results[6]:.4f}")
            print(f"   Precision: {results[7]:.4f}")
            print(f"   Recall: {results[8]:.4f}")
                
        except Exception as e:
            print(f"❌ Errore valutazione: {e}")

        return self.model, self.history