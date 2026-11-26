import argparse
import platform
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import ConcatDataset, DataLoader

from config import (BATCH_SIZE, EPOCH, FOLDER_AUG, FOLDER_ORI, FOLDS, JUMLAH_KELAS,
                    KELAS, LEARNING_RATE, PAKAI_AUG, SEED, folder_hasil)
from Models.CNN import SimpleCNN
from Utils.getData import batch_ke_model, getImageLabel


def catat(log, teks):
    print(teks, flush=True)
    with log.open("a", encoding="utf-8") as file:
        file.write(teks + "\n")


def satu_epoch(loader, model, loss_function, perangkat, optimizer=None):
    sedang_latih = optimizer is not None
    model.train(sedang_latih)
    total_loss = torch.zeros((), device=perangkat)
    with torch.set_grad_enabled(sedang_latih):
        for gambar, label in loader:
            gambar, label = batch_ke_model(gambar, label, perangkat)
            if sedang_latih:
                optimizer.zero_grad()
            loss = loss_function(model(gambar), label)
            if sedang_latih:
                loss.backward()
                optimizer.step()
            # Bobot tiap batch mengikuti jumlah gambarnya, termasuk batch terakhir.
            total_loss += loss.detach() * len(label)
    return total_loss.item() / len(loader.dataset)


def latih_satu_fold(k, pakai_aug, epoch_total, hasil, log, perangkat):
    folder = Path(FOLDER_ORI.format(k=k))
    data_latih = getImageLabel(folder / "Train")
    if pakai_aug:
        data_aug = getImageLabel(FOLDER_AUG.format(k=k))
        data_latih = ConcatDataset([data_latih, data_aug])
    data_valid = getImageLabel(folder / "Valid")
    catat(log, f"=== FOLD {k}: {len(data_latih)} gambar latih, {len(data_valid)} gambar validasi ===")

    torch.manual_seed(SEED)
    train_loader = DataLoader(data_latih, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(data_valid, batch_size=BATCH_SIZE)
    model = SimpleCNN(output=JUMLAH_KELAS).to(perangkat, memory_format=torch.channels_last)
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    loss_function = nn.CrossEntropyLoss()
    loss_train_all, loss_valid_all = [], []
    loss_terbaik = float("inf")

    for epoch in range(1, epoch_total + 1):
        train_loss = satu_epoch(train_loader, model, loss_function, perangkat, optimizer)
        valid_loss = satu_epoch(valid_loader, model, loss_function, perangkat)
        loss_train_all.append(train_loss)
        loss_valid_all.append(valid_loss)
        catat(log, f"Fold {k} | epoch {epoch:>2}/{epoch_total} | "
                   f"train loss = {train_loss:.4f} | valid loss = {valid_loss:.4f}")
        if valid_loss < loss_terbaik:
            loss_terbaik = valid_loss
            epoch_terbaik = epoch
            checkpoint = {
                "epoch": epoch,
                "total_epoch": epoch_total,
                "fold": k,
                "pakai_aug": pakai_aug,
                "kelas": KELAS,
                "model_state_dict": model.state_dict(),
            }
            lokasi = hasil / f"SimpleCNN_fold{k}.pt"
            sementara = lokasi.with_suffix(".tmp.pt")
            torch.save(checkpoint, sementara)
            sementara.replace(lokasi)

    catat(log, f"Fold {k} | epoch terbaik = {epoch_terbaik} | valid loss = {loss_terbaik:.4f}")
    epochs = range(1, epoch_total + 1)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epochs, loss_train_all, color="#931a00", label="Train")
    ax.plot(epochs, loss_valid_all, color="#1674bf", label="Valid")
    ax.axvline(epoch_terbaik, color="#777777", linestyle="--", label="Epoch terbaik")
    ax.set(xlabel="Epoch", ylabel="Cross-entropy loss", title=f"Fold {k} | Augmentasi {pakai_aug}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(hasil / f"training_fold{k}.png", dpi=150)
    plt.close(fig)


def latih_percobaan(pakai_aug, cek_cepat, perangkat):
    folds = [1] if cek_cepat else FOLDS
    epoch_total = 3 if cek_cepat else EPOCH
    hasil = folder_hasil(pakai_aug, cek_cepat)
    hasil.mkdir(parents=True, exist_ok=True)
    log = hasil / "train_log.txt"
    log.write_text("", encoding="utf-8")
    mulai = datetime.now()
    nama_perangkat = torch.cuda.get_device_name(perangkat) if perangkat.type == "cuda" else "CPU"
    catat(log, f"Mulai: {mulai:%Y-%m-%d %H:%M:%S} | komputer: {platform.node()} | "
               f"perangkat: {nama_perangkat} | torch: {torch.__version__} | "
               f"PAKAI_AUG = {pakai_aug} | EPOCH = {epoch_total} | FOLDS = {folds}")
    for k in folds:
        latih_satu_fold(k, pakai_aug, epoch_total, hasil, log, perangkat)
    selesai = datetime.now()
    detik = int((selesai - mulai).total_seconds())
    catat(log, f"Selesai: {selesai:%Y-%m-%d %H:%M:%S} | durasi {detik // 60} menit {detik % 60} detik")


def main():
    parser = argparse.ArgumentParser(description="Latih SimpleCNN sesuai PAKAI_AUG di config.py.")
    parser.add_argument("--cek-cepat", action="store_true", help="Latih fold 1 selama 3 epoch di .cek_cepat/.")
    args = parser.parse_args()

    perangkat = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if perangkat.type == "cpu":
        torch.set_num_threads(min(4, torch.get_num_threads()))
    latih_percobaan(PAKAI_AUG, args.cek_cepat, perangkat)


if __name__ == "__main__":
    main()
