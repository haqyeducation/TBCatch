> Historical V5 methodology notes. For the current folder layout and setup, use [README.md](README.md). Older ZIP paths below describe the original training transfer.

# CODA TB V5 — paket training lokal

**Built with PriorLabs-TabPFN**

Paket ini menambahkan BEATs frozen dan TabPFN v2, seluruh 16 fitur klinis jurnal, baseline audio LibROSA, ensemble terbatas, XAI, serta jalur mobile. Dataset dan kernel V4 yang sudah berhasil tetap digunakan. Hasil V4 tidak ditimpa.

## Cara menjalankan di PC training

1. Ekstrak **seluruh** `CODA_TB_V5_PACKAGE.zip` ke folder proyek, misalnya `D:/haqy/`. Gabungkan folder `research_package` bila diminta. Berkas V5 baru ditambahkan; sumber V3/V4 dalam paket sama dengan sumber sebelumnya.
2. Pastikan `dataset/dataset` masih berisi data asli. Manifest audit disertakan paket.
3. Buka `D:/haqy/CODA_TB_V5_Research.ipynb`. Pilih kernel V4 yang sudah berhasil: **Python 3.11 x64, torch 2.6.0+cu118**. Restart kernel sebelum Run All.
4. Sel pertama default: `MODE="full"`, `RUN_NAME="experiment_v5_001"`, `INSTALL_MISSING_LOCAL=True`. Tidak perlu mengunduh library/bobot dari internet. Tambahan dipasang dari 29 wheel lokal; PyTorch tidak diganti.
5. Jika proyek tidak ditemukan otomatis, isi `PACKAGE_DIR=r"D:\haqy\research_package"` dan `PROJECT_DIR=r"D:\haqy"`.
6. Klik **Run All**. Jangan mengubah config/library setelah mulai. Bila terputus, buka kernel baru lalu Run All dengan `RUN_NAME` yang sama; konfigurasi/sumber/data/versi library harus identik.
7. Selesai training, salin `research_package/runs/experiment_v5_001_RETURN.zip` kembali ke laptop untuk audit.

Notebook saja tidak cukup. Paket memerlukan `src`, `pretrained/v5`, `wheels_v5`, `assets_v5.json`, `config_v5.json`, dan manifest. ZIP tidak menyertakan audio mentah atau seluruh library V4 berukuran besar. Instalasi offline mengandalkan library dasar V4 sudah ada. Jika lingkungan berbeda, setup berhenti dengan pesan yang jelas, bukan mengganti PyTorch atau diam-diam menghapus kandidat.

## Yang dibandingkan

| Keluaran | Input / pendekatan |
|---|---|
| clinical_full | Seluruh 16 variabel klinis/demografi termasuk HR; LR/CB/TabPFN |
| paper_full_lr / paper_full_cb | Audio fitur jurnal + seluruh klinis; per-cough LR/CB dan mean probabilitas peserta |
| paper_full_tabpfn | Audio fitur jurnal dipool per peserta + seluruh klinis; TabPFN |
| paper_pooled_lr / paper_pooled_cb | Kontrol audio jurnal dengan pooling pasien, PCA64, dan head yang sama dengan BEATs; mengisolasi manfaat representasi dari agregasi |
| beats_full_lr / beats_full_cb / beats_full_tabpfn | BEATs + denyut + klinis lengkap |
| ensemble_full | Empat campuran model dengan bobot/pasangan yang ditetapkan sebelum melihat test |
| beats_hr | BEATs + denyut saja |
| beats_mobile | BEATs + denyut + kuesioner praktis, tanpa tinggi/berat/temperatur atau subtipe riwayat TB |
| selected_full | Endpoint utama: pemilihan keluarga BEATs/ensemble berdasarkan inner CV sebelum test; memakai ulang model terpilih, tanpa fit tambahan |

Klinis lengkap: age, sex, height, weight, reported_cough_dur, tb_prior, tb_prior_Pul, tb_prior_Extrapul, tb_prior_Unknown, hemoptysis, heart_rate, temperature, weight_loss, smoke_lweek, fever, night_sweats.

Audio dan label tidak diperlakukan sebagai 9.281 pasien independen; unit evaluasi tetap peserta. Dataset lokal 1.040 peserta berbeda dari 1.105 pada paper. Belum ada klaim peningkatan atau novelty yang terbukti.

## Protokol dan efisiensi

- Split sama dengan V4: 10 outer × 5 inner, 20% outer-training untuk kalibrasi. Kode publik penulis Sensors memakai 15% dan StratifiedGroupKFold; paket ini tidak mengklaim reproduksi persis skor jurnal.
- Seluruh kandidat memilih mean inner UAR, lalu AUC. V4 mobile mengutamakan AUC; perubahan ini dilaporkan.
- 44 konfigurasi / 2.200 evaluasi konfigurasi-inner, termasuk enam kontrol pooling yang diperlukan untuk ablation adil; jauh lebih sedikit dari 18.100 pada V4. Pengurangan hitungan bukan janji pengurangan durasi yang sama.
- BEATs tidak di-fine-tune. Ekstraksi hanya sekali, label-free, disimpan per batch. PCA/imputasi/scaling belajar hanya pada training fold.
- Cache fitur jurnal yang lolos preflight Python 3.11 disertakan. Cache hanya digunakan bila hash sumber/manifest dan versi library cocok; jika berbeda, fitur dihitung ulang secara otomatis.
- Bobot BEATs adalah encoder 768 dimensi. Fitur pooled pasien: mean dan std, lalu PCA 32/64 bila dikonfigurasi. Jendela 0,5 detik; padding bukan informasi baru.
- Model klasik menggunakan CPU. TabPFN dan ekstraksi BEATs memakai CUDA pada training PC. Tidak ada alasan mengisi VRAM dengan duplikasi model yang tidak mempercepat pekerjaan.
- Batas **45 GB desimal (45.000.000.000 byte)**, headroom 3 GB. Batch dipilih melalui benchmark. Pemakaian total GPU diperiksa; allocator PyTorch dibatasi. Ini bukan hard cap sistem operasi terhadap proses lain yang tiba-tiba memakai GPU. Jalankan tanpa workload GPU berat lain.
- FP32 encoder adalah default untuk kestabilan; tidak menjalankan seluruh jaringan attention/encoder training hanya agar VRAM penuh.
- TabPFN memakai bobot lokal v2 dan library 2.0.9, bukan model default versi terbaru. Checkpoint menyimpan konteks training CPU; tidak menyimpan objek CUDA yang sulit dipindahkan.
- Prediksi test, model, dan cutoff tersimpan sebelum XAI. Resume postprocessing tidak mengulang training.

## Output

Folder run memuat konfigurasi dan environment, audit kohort/WAV, nested splits, candidate grid, fitur peserta, checkpoint per inner fold, pemilihan model, model/calibrator/cutoff, prediksi test, metrik fold dan gabungan, paired bootstrap, XAI, simulasi HR, cascade, serta PNG visualisasi.

`nested_mean_std.csv` adalah ringkasan utama; SD antar-fold bukan confidence interval. `pooled_oof_metrics.csv` menghitung keputusan dari cutoff fold masing-masing. Bootstrap berpasangan bersifat eksploratif, tanpa retraining uncertainty atau koreksi banyak perbandingan.

XAI kelompok memakai 4 peserta/fold dan individual 2 peserta/fold. Sampling terbatas ini wajib dilaporkan. Pergantian background dengan permutation seed tetap memeriksa sensitivitas penjelasan terhadap background, bukan validitas kausal/klinis. Simulasi HR ±5/±10 BPM bukan pengukuran error kamera.

Transformasi audio classifier klasik menggunakan float64 untuk mengurangi perbedaan numerik batch. XAI memeriksa probabilitas mentah all-present terhadap prediksi peserta tunggal sebelum kalibrasi, mencatat raw/calibrated batch delta, lalu memakai prediksi tunggal sebagai endpoint kanonis. Selisih raw lebih dari 1e-5 menghentikan postprocessing untuk diperiksa; completeness tetap 1e-8. Delta kalibrasi tetap dilaporkan karena isotonic dapat memperbesar roundoff di batas step.

Cascade: BEATs+HR dahulu, lalu BEATs+HR+kuesioner mobile. Model riset lengkap tidak dipakai dengan temperatur/tinggi/berat diam-diam dikosongkan. HR missing memaksa tahap lanjut, tetapi belum ada model khusus HR hilang; implementasi aplikasi nantinya perlu quality check dan pilihan pengukuran ulang. Hasil rendah bukan kepastian bebas TB.

## Sumber dan provenance

- Sensors: https://www.mdpi.com/1424-8220/26/4/1223 ; kode penulis: https://github.com/gpk1983/Reproducible-TB-Screening-from-Cough-Audio
- Fitur audio 261 dimensi menggunakan rumus LibROSA kode penulis; uji lokal mengeksekusi fungsi ekstraksi penulis secara terisolasi untuk pemeriksaan parity. Kesamaan rumus fitur tidak berarti seluruh pipeline/pencarian/split identik.
- BEATs: https://github.com/microsoft/unilm/tree/master/beats ; kode Microsoft/MIT disertakan, relative import disesuaikan. Upstream commit dan hash tersimpan pada `assets_v5.json`.
- Unduhan bobot penerbit BEATs tidak tersedia saat persiapan. Bobot diperoleh dari mirror `lpepino/beats_ckpts`; hash LFS cocok dengan mirror `mooneyko/BEATs`, namun belum diautentikasi terhadap unduhan penerbit. Provenance ini dipertahankan, bukan disebut bobot yang diverifikasi langsung oleh penerbit.
- TabPFN v2/Nature: https://www.nature.com/articles/s41586-024-08328-6 ; bobot resmi: https://huggingface.co/Prior-Labs/TabPFN-v2-clf . Lisensi bobot v2 disertakan di `pretrained/v5/TABPFN_LICENSE.txt`.

Jika model yang memakai TabPFN atau diturunkan dari outputnya dipublikasikan/didistribusikan, ikuti ketentuan atribusi lisensi: tampilkan “Built with PriorLabs-TabPFN”, sertakan lisensi, dan gunakan nama model berawalan “TabPFN”, misalnya `TabPFN-CODA-TB-V5`. ID eksperimen internal dalam tabel bukan penamaan produk publik. Ketentuan ini perlu dibawa ke dokumentasi/web jika TabPFN akhirnya dipakai.

## Batas klaim

Belum ada skor V5 sebelum training. Tidak ada jaminan melampaui AUC 0,81 atau baseline lokal. Menggunakan BEATs/TabPFN/ensemble/XAI bukan penemuan metode tersebut. Peningkatan harus ditunjukkan pada input dan evaluasi sama, dengan ablation yang mendukung nilai tambah audio.

Kohort telah dipakai berulang pada V1–V4. Nested CV tidak menghilangkan bias adaptasi peneliti. Tidak ada validasi eksternal Indonesia, pasangan video/PPG untuk validasi denyut kamera, atau model deployment final. Penelitian skrining, bukan diagnosis.
