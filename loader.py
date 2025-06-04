import os
import numpy as np

class DataLoader:
    def __init__(self, data_dir):
        self.data_dir = data_dir

    def load_features_and_labels(self):
        print(f"🔍 Caricamento dati da: {self.data_dir}")

        X = np.load(os.path.join(self.data_dir, 'features.npy'))
        y_top = np.load(os.path.join(self.data_dir, 'labels_top.npy'))
        y_medium = np.load(os.path.join(self.data_dir, 'labels_medium.npy'))
        y_all = np.load(os.path.join(self.data_dir, 'labels_all.npy'))

        print(f"✅ Caricati {X.shape[0]} esempi")
        return X, {'top': y_top, 'medium': y_medium, 'all': y_all}

    def save_features_and_labels(self, features, labels):
        print(f"💾 Salvataggio dataset in: {self.data_dir}")
        os.makedirs(self.data_dir, exist_ok=True)

        np.save(os.path.join(self.data_dir, 'features.npy'), features)
        np.save(os.path.join(self.data_dir, 'labels_top.npy'), labels['top'])
        np.save(os.path.join(self.data_dir, 'labels_medium.npy'), labels['medium'])
        np.save(os.path.join(self.data_dir, 'labels_all.npy'), labels['all'])

        print("✅ Dati salvati con successo.")
