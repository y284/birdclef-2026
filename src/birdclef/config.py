import os
from pathlib import Path


def _raw_dir():
    for p in (
        os.environ.get("BIRDCLEF_DATA"),
        "/kaggle/input/competitions/birdclef-2026",
        "/kaggle/input/birdclef-2026",
    ):
        if p and Path(p).exists():
            return Path(p)
    return Path("data/raw")


RAW_DIR = _raw_dir()
WAV_DIR = Path(os.environ.get("BIRDCLEF_WAV", "data/wav"))
SUBSETS = ("train_audio", "train_soundscapes")

SR = 32000  # Hz, every WAV in WAV_DIR is mono 16-bit at this rate
WINDOW = 5  # seconds per prediction row

PERCH_ONNX = Path(os.environ.get("PERCH_ONNX", "models/perch_v2_no_dft.onnx"))  # huggingface.co/justinchuby/Perch-onnx
PERCH_DIR = Path(os.environ.get("BIRDCLEF_PERCH", "data/perch"))
TRT_CACHE = Path(os.environ.get("BIRDCLEF_TRT_CACHE", "models/trt_cache"))  # TensorRT engines built from PERCH_ONNX
STRIDE = 1  # seconds between the starts of consecutive Perch windows
PERCH_LABELS = Path(os.environ.get("PERCH_LABELS", "models/labels.csv"))  # Perch v2 class list, one name per row
CURVES_DIR = Path(os.environ.get("BIRDCLEF_CURVES", "data/curves"))
SECOND_WEIGHTS = (0.1, 0.3, 0.6, 0.3, 0.1)  # weight of a second by its position inside a 5 s window
TRUST_AUC = 0.9  # Perch labels with a higher ROC-AUC on the training clips are used as pseudo-labels

FRAMES_PER_SECOND = 32  # mel frames; the backbone downsamples time by 32, so the model has one frame per second
N_FFT, HOP, N_MELS, F_MIN, F_MAX, TOP_DB = 2048, SR // FRAMES_PER_SECOND, 128, 20, 16000, 80  # log-mel front end
RARE = 10  # labels with this many training clips or fewer are left out of evaluation
PSEUDO_POWER = 2  # pseudo target of a trusted species: clip(mean normalized curve over the crop, 0, 1) ** this
MIXUP_LAMBDA = (0.3, 0.7)  # weight of the first crop in a mix, drawn uniformly in this range
