#!/usr/bin/env python3
"""
Script Semplice per Avvio Training FMA
"""

import os
import sys

def main():
    print("🎵 AVVIO TRAINING FMA BILANCIATO")
    print("=" * 50)
    
    # Test configurazione
    try:
        import config
        print("✅ config.py caricato")
        print(f"   Metadata: {config.METADATA_PATH}")
        if hasattr(config, 'FMA_LARGE_PATH') and config.FMA_LARGE_PATH:
            print(f"   Audio: {config.FMA_LARGE_PATH}")
        print(f"   GPU: {'✅' if config.GPU_AVAILABLE else '❌'}")
        print(f"   Batch size: {config.BATCH_SIZE}")
    except Exception as e:
        print(f"❌ Errore config: {e}")
        print("💡 Verifica che config.py sia stato creato correttamente")
        return False
    
    # Controlla script training
    if not os.path.exists('balanced_trainer.py'):
        print("❌ balanced_trainer.py non trovato")
        print("💡 Assicurati di avere tutti gli script del progetto")
        return False
    
    print("\n🏋️ Avvio training...")
    print("   (Questo può richiedere molto tempo)")
    
    try:
        import subprocess
        result = subprocess.run([
            sys.executable, 'balanced_trainer.py'
        ], check=True)
        
        print("\n🎉 Training completato!")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Errore training: {e}")
        return False
    except KeyboardInterrupt:
        print("\n⚠️ Training interrotto dall'utente")
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        print("\n🔧 Controlla gli errori e riprova")
