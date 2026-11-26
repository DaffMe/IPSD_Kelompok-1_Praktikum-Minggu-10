from pathlib import Path

from config import FOLDER_AUG, FOLDER_ORI, FOLDS
from Utils.getData import daftar_gambar


def main():
    ada_masalah = False
    for k in FOLDS:
        print(f"\nFOLD {k}")
        folder = Path(FOLDER_ORI.format(k=k))
        bagian = {nama: folder / nama for nama in ("Train", "Valid", "Test")}
        bagian["Augmented Train"] = Path(FOLDER_AUG.format(k=k))
        for nama, lokasi in bagian.items():
            try:
                isi = {kelas: len(foto) for kelas, foto in daftar_gambar(lokasi).items()}
            except (FileNotFoundError, ValueError) as error:
                print(f"  {nama}: {error}")
                ada_masalah = True
                continue
            print(f"  {nama:<16} {len(isi)} kelas, total {sum(isi.values()):>5} gambar")
            print(f"    {isi}")
    if ada_masalah:
        raise SystemExit("Periksa hasil ekstrak serta alamat folder di config.py.")
    print("\nSemua fold memiliki Train, Valid, Test, dan Augmented Train dengan enam kelas.")


if __name__ == "__main__":
    main()
