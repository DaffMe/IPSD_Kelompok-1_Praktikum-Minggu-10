import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)
from torch.utils.data import DataLoader

from config import (BATCH_SIZE, EPOCH, FOLDER_ORI, FOLDS, JUMLAH_KELAS, KELAS,
                    PAKAI_AUG, folder_hasil)
from Models.CNN import SimpleCNN
from Utils.getData import batch_ke_model, getImageLabel


def simpan_confusion_matrix(matriks, judul, lokasi, format_angka="d"):
    fig, ax = plt.subplots(figsize=(8, 6))
    tabel = pd.DataFrame(matriks, index=KELAS, columns=KELAS)
    sns.heatmap(tabel, annot=True, fmt=format_angka, cmap="Blues", ax=ax)
    ax.set(xlabel="Kelas prediksi", ylabel="Kelas asli", title=judul)
    fig.tight_layout()
    fig.savefig(lokasi, dpi=150)
    plt.close(fig)


def uji_satu_fold(k, pakai_aug, hasil, perangkat):
    lokasi = hasil / f"SimpleCNN_fold{k}.pt"
    checkpoint = torch.load(lokasi, map_location=perangkat, weights_only=True)
    if (checkpoint["kelas"], checkpoint["total_epoch"], checkpoint["fold"], checkpoint["pakai_aug"]) != (
        KELAS, EPOCH, k, pakai_aug
    ):
        raise ValueError(f"Checkpoint {lokasi} tidak sesuai pengaturan percobaan.")

    data_uji = getImageLabel(Path(FOLDER_ORI.format(k=k)) / "Test")
    loader = DataLoader(data_uji, batch_size=BATCH_SIZE)
    model = SimpleCNN(output=JUMLAH_KELAS).to(perangkat, memory_format=torch.channels_last)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    aktual_batch, peluang_batch = [], []
    with torch.inference_mode():
        for gambar, label in loader:
            gambar, label = batch_ke_model(gambar, label, perangkat)
            peluang_batch.append(model(gambar).softmax(dim=1).cpu().numpy())
            aktual_batch.append(label.argmax(dim=1).cpu().numpy())

    aktual = np.concatenate(aktual_batch)
    peluang = np.concatenate(peluang_batch)
    prediksi = peluang.argmax(axis=1)
    label_kelas = list(range(JUMLAH_KELAS))
    matriks = confusion_matrix(aktual, prediksi, labels=label_kelas)
    simpan_confusion_matrix(matriks, f"Fold {k} | Epoch terbaik {checkpoint['epoch']}",
                            hasil / f"confusion_matrix_fold{k}.png")

    mpox = KELAS.index("Monkeypox")
    metrik = {
        "fold": k,
        "accuracy": accuracy_score(aktual, prediksi),
        "precision": precision_score(aktual, prediksi, average="weighted", zero_division=0),
        "recall": recall_score(aktual, prediksi, average="weighted", zero_division=0),
        "f1": f1_score(aktual, prediksi, average="weighted", zero_division=0),
        "roc_auc": roc_auc_score(aktual, peluang, labels=label_kelas, multi_class="ovr", average="weighted"),
        "recall_monkeypox": matriks[mpox, mpox] / matriks[mpox].sum(),
    }
    return metrik, matriks


def evaluasi(pakai_aug, hasil, perangkat):
    log = hasil / "train_log.txt"
    isi_log = log.read_text(encoding="utf-8").splitlines() if log.exists() else []
    if not isi_log or not isi_log[-1].startswith("Selesai:"):
        raise SystemExit(f"Pelatihan di {hasil.name} belum selesai. Jalankan train.py untuk mode ini.")

    evaluasi_folds = [uji_satu_fold(k, pakai_aug, hasil, perangkat) for k in FOLDS]
    tabel = pd.DataFrame([metrik for metrik, _ in evaluasi_folds]).set_index("fold")
    ringkasan = pd.DataFrame([tabel.mean(), tabel.std()], index=["rata-rata", "simpangan"])
    tabel = pd.concat([tabel, ringkasan]).rename_axis("fold")
    matriks_total = sum(matriks for _, matriks in evaluasi_folds)
    pd.DataFrame(matriks_total, index=KELAS, columns=KELAS).to_csv(
        hasil / "confusion_matrix_total.csv", index_label="kelas_asli"
    )
    simpan_confusion_matrix(matriks_total / len(FOLDS), "Confusion matrix rata-rata per fold",
                            hasil / "confusion_matrix_rata_rata.png", format_angka=".1f")
    csv_sementara = hasil / "hasil_5fold.tmp"
    tabel.to_csv(csv_sementara, float_format="%.6f")
    csv_sementara.replace(hasil / "hasil_5fold.csv")
    return tabel, matriks_total


def ringkas_kesalahan(matriks):
    benar = matriks.diagonal()
    total = matriks.sum(axis=1)
    salah = total - benar
    kelas_sering_salah = int(salah.argmax())
    kelas_persen_salah = int((salah / total).argmax())
    for i in dict.fromkeys((kelas_sering_salah, kelas_persen_salah)):
        baris_salah = matriks[i].copy()
        baris_salah[i] = 0
        j = int(baris_salah.argmax())
        print(f"{KELAS[i]}: {salah[i]}/{total[i]} salah ({salah[i] / total[i]:.1%}), "
              f"terbanyak diprediksi {KELAS[j]} ({baris_salah[j]} prediksi).")

    mpox = KELAS.index("Monkeypox")
    print(f"Recall Monkeypox gabungan: {benar[mpox]}/{total[mpox]} = {benar[mpox] / total[mpox]:.2%}")


def main():
    parser = argparse.ArgumentParser(description="Evaluasi checkpoint sesuai PAKAI_AUG di config.py.")
    parser.add_argument("--tampilkan", action="store_true", help="Tampilkan evaluasi yang sudah tersimpan.")
    args = parser.parse_args()

    perangkat = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if perangkat.type == "cpu":
        torch.set_num_threads(min(4, torch.get_num_threads()))

    hasil = folder_hasil(PAKAI_AUG)
    csv = hasil / "hasil_5fold.csv"
    total_csv = hasil / "confusion_matrix_total.csv"
    log = hasil / "train_log.txt"
    print(f"\n=== {hasil.name} ===")
    if csv.exists() and total_csv.exists() and (not log.exists() or csv.stat().st_mtime >= log.stat().st_mtime):
        print("Menampilkan evaluasi final yang sudah tersimpan.")
        tabel = pd.read_csv(csv, index_col="fold")
        matriks = pd.read_csv(total_csv, index_col="kelas_asli").to_numpy()
    else:
        if args.tampilkan:
            raise SystemExit(f"Evaluasi terbaru untuk {hasil.name} belum tersedia. Jalankan test.py untuk mode ini.")
        print("Menghitung evaluasi final pada Test.")
        tabel, matriks = evaluasi(PAKAI_AUG, hasil, perangkat)

    print(tabel.round(4).to_string())
    ringkas_kesalahan(matriks)


if __name__ == "__main__":
    main()
