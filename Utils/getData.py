from pathlib import Path

import cv2 as cv
import numpy as np
import torch
from torch.utils.data import Dataset

from config import JUMLAH_KELAS, KELAS, UKURAN


EKSTENSI = (".jpg", ".jpeg", ".png", ".bmp")


def daftarKelas(folder):
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"Folder dataset tidak ditemukan: {folder}")
    return sorted(p.name for p in folder.iterdir() if p.is_dir() and not p.name.startswith("."))


def daftar_gambar(folder):
    folder = Path(folder)
    kelas = daftarKelas(folder)
    if kelas != KELAS:
        raise ValueError(f"Kelas di {folder}: {kelas}. Kelas yang diharapkan: {KELAS}.")
    gambar = {}
    for nama in kelas:
        gambar[nama] = sorted(p for p in (folder / nama).iterdir()
                              if p.is_file() and p.suffix.lower() in EKSTENSI)
        if not gambar[nama]:
            raise ValueError(f"Tidak ada gambar di {folder / nama}")
    return gambar


def baca_gambar(file):
    # imdecode menangani path dengan karakter Unicode di Windows.
    gambar = cv.imdecode(np.fromfile(file, dtype=np.uint8), cv.IMREAD_COLOR)
    if gambar is None:
        raise ValueError(f"Gambar tidak bisa dibaca: {file}")
    return gambar


class getImageLabel(Dataset):
    def __init__(self, folder, jumlah_kelas=JUMLAH_KELAS, ukuran=UKURAN):
        berkas = daftar_gambar(folder)
        self.kelas = list(berkas)
        if len(self.kelas) != jumlah_kelas:
            raise ValueError(f"Jumlah folder kelas harus {jumlah_kelas}.")

        # Simpan hasil resize dalam uint8 untuk menghemat RAM. Label memakai one-hot.
        self.gambar = np.empty((sum(map(len, berkas.values())), ukuran, ukuran, 3), dtype=np.uint8)
        self.label = np.empty(len(self.gambar), dtype=np.int64)
        self.one_hot = torch.eye(jumlah_kelas, dtype=torch.float32)
        posisi = 0
        for label, files in enumerate(berkas.values()):
            for file in files:
                self.gambar[posisi] = cv.resize(baca_gambar(file), (ukuran, ukuran))
                self.label[posisi] = label
                posisi += 1

    def __getitem__(self, item):
        gambar = torch.from_numpy(self.gambar[item]).float().div_(255)
        return gambar, self.one_hot[self.label[item]]

    def __len__(self):
        return len(self.gambar)


def batch_ke_model(gambar, label, perangkat):
    # OpenCV menghasilkan HWC. Conv2d menerima batch NCHW.
    return gambar.permute(0, 3, 1, 2).to(perangkat), label.to(perangkat)
