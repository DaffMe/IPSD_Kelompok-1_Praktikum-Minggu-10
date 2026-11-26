from pathlib import Path


PROYEK = Path(__file__).resolve().parent
DATASET = PROYEK / "Dataset"
FOLDER_ORI = str(DATASET / "Original Images" / "Original Images" / "FOLDS" / "fold{k}")
FOLDER_AUG = str(DATASET / "Augmented Images" / "Augmented Images" / "FOLDS_AUG" / "fold{k}_AUG" / "Train")

PAKAI_AUG = True
FOLDS = [1, 2, 3, 4, 5]
KELAS = ["Chickenpox", "Cowpox", "HFMD", "Healthy", "Measles", "Monkeypox"]
JUMLAH_KELAS = len(KELAS)
UKURAN = 100
BATCH_SIZE = 32
EPOCH = 20
LEARNING_RATE = 0.0001
SEED = 42


def folder_hasil(pakai_aug, cek_cepat=False):
    nama = "hasil_dengan_aug" if pakai_aug else "hasil_tanpa_aug"
    return PROYEK / ".cek_cepat" / nama if cek_cepat else PROYEK / nama
