from gpu import setup_gpu,setup_mixed_precision






GPU_AVAILABLE = setup_gpu()
MIXED_PRECISION = setup_mixed_precision() if GPU_AVAILABLE else False

FMA_LARGE_PATH = 'modello3/fma_large'
METADATA_PATH = 'modello3/fma_metadata'
SAMPLE_RATE = 22050
DURATION = 30
N_MELS = 128
HOP_LENGTH = 512
NUM_SEGMENTS = 10

##Diviso in chunk per problema di ram
CHUNK_SIZE=5000

BATCH_SIZE=64