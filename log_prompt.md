# Log penggunaan GenAI

## 24 November 2026

- **Alat** OpenAI Codex.
- **Prompt** “brainstorming mengenai tugas kelompokku \"[lokasi folder lokal Konteks Soal]\" perlu ngapain. untuk penulisan code, aku mau kamu yang buatkan (setup sesuaikan dengan yang disuruh). Biar kelompokku tinggal presentasi dan pahami codenya”
- **Penggunaan** Codex membaca PDF dan modul, menyiapkan environment dan dataset, menulis kode pemeriksaan data, model, pelatihan dan evaluasi berdasarkan modul, serta menjalankan pelatihan dan evaluasi kedua mode di komputer ini. Codex menyusun tabel dari hasil aktual dan menyiapkan catatan interpretasi serta alur presentasi.
- **Penyuntingan log** Lokasi folder lokal disingkat di atas karena log ini bisa disertakan di GitHub publik.

### Naskah latihan live coding

- **Prompt** “jadi urutannya gimana dulu? detail. sekalian Bikinin naskah live coding 10–15 menit dari hasil yang tadi, sekalian jelasin bagian penting tiap file yang ditunjukin (menyesuaikan urutan di file soal) biar aku paham dan bisa jawab kalau ditanya dosen. gunakan skills ini [bahasa-skills](https://github.com/alvinindra/bahasa-skills) siapa tau naskahnya gak ai slop dan mudah kami mengerti”
- **Penggunaan** Codex menyusun urutan rekaman, naskah latihan berdasarkan hasil eksperimen yang tersimpan, dan bahan tanya jawab tentang kode. Panduan bahasa dari repo yang ditautkan dipakai untuk menyunting naskah latihan.

### Revisi sesuai Langkah 1 sampai 13 modul

- **Prompt** “gak kurang2. aku harus jelasinnya kayak dari langkah awal - akhir dari guide menyesuaikan file ini \"[lokasi lokal Modul_Praktikum_W10_Mpox_v8.docx]\" termasuk yang ditunujukin code, input, outputnya (di run dulu sebelum presentasi)”
- **Penggunaan** Codex memeriksa ulang modul dan keluaran lokal, lalu merevisi naskah per nomor langkah. Contoh format tensor dan bentuk keluaran CNN diperiksa ulang di Python proyek. Hasil pelatihan dan evaluasi akhir memakai artefak yang telah dibuat sebelumnya.

### Naskah dengan output yang sudah disiapkan

- **Prompt** Pengguna meminta bagian persiapan terpisah dan naskah dengan petunjuk layar untuk menunjukkan kode, masukan, serta keluaran yang sudah dijalankan. Narasi mengikuti langkah modul. Kutipan prompt diringkas.
- **Penggunaan** Codex merevisi naskah, menjalankan pemeriksaan singkat untuk merekam keluarannya, dan menyalin kembali keluaran evaluasi final yang tersimpan ke berkas latihan.

## 25 November 2026

- **Prompt** Pengguna meminta pilihan mode di terminal dihapus agar percobaan dijalankan satu per satu lewat `PAKAI_AUG` di `config.py`. Pengguna juga meminta naskah dan panduan operator disesuaikan.
- **Penggunaan** Codex mengubah pemilihan mode pelatihan dan evaluasi. Hasil eksperimen yang sudah tersimpan dipakai kembali untuk presentasi.

## 26 November 2026

- **Prompt** Pengguna meminta audit akhir, perapian isi repositori, dan unggah ke GitHub kelompok. Pengguna juga meminta README ditulis dengan bahasa yang lebih natural.
- **Penggunaan** Codex memeriksa kode dan hasil yang sudah tersimpan, merapikan README, serta menyiapkan berkas yang diminta untuk diunggah. Pelatihan dan evaluasi memakai hasil run sebelumnya.
