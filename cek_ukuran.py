from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import FOLDER_ORI, PROYEK
from Utils.getData import baca_gambar, daftar_gambar, getImageLabel


def main():
    folder = Path(FOLDER_ORI.format(k=1)) / "Train"
    berkas = daftar_gambar(folder)
    ukuran_asli = {baca_gambar(file).shape for files in berkas.values() for file in files}
    print(f"SEBELUM: {len(ukuran_asli)} ukuran berbeda. Contoh: {sorted(ukuran_asli)[:3]}")

    data = getImageLabel(folder)
    ukuran_baru = {tuple(gambar.shape) for gambar, _ in data}
    gambar, _ = data[0]
    print(f"SESUDAH: {ukuran_baru} dari {len(data)} gambar")
    print(f"Rentang piksel contoh setelah normalisasi: {gambar.min():.3f}–{gambar.max():.3f}")

    # getImageLabel mengurutkan kelas dan nama file, sehingga posisi ini
    # menunjuk gambar Monkeypox pertama pada data yang sudah diproses.
    contoh = berkas["Monkeypox"][0]
    asli = baca_gambar(contoh)
    posisi = len(data) - len(berkas["Monkeypox"])
    sesudah, _ = data[posisi]
    fig, ax = plt.subplots(1, 2, figsize=(8, 3.5))
    # Balik kanal BGR menjadi RGB untuk ditampilkan oleh Matplotlib.
    ax[0].imshow(asli[:, :, ::-1])
    ax[0].set_title(f"Sebelum {asli.shape}")
    ax[1].imshow(sesudah.numpy()[:, :, ::-1])
    ax[1].set_title(f"Sesudah {tuple(sesudah.shape)}")
    for panel in ax:
        panel.axis("off")
    fig.tight_layout()
    fig.savefig(PROYEK / "cek_ukuran.png", dpi=150)
    plt.close(fig)
    print("Gambar perbandingan disimpan: cek_ukuran.png")


if __name__ == "__main__":
    main()
