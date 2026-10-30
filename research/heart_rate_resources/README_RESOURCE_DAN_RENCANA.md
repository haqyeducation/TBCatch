# Resource pengukur denyut kamera dan status model TB

Diperiksa 24 Oktober 2026. Ini paket persiapan resource, bukan hasil validasi akurasi alat. Rincian file, hash, lisensi dan status unduhan ada dalam `RESOURCE_MANIFEST.json`. Skrip `verify_resources.py` memakai Python lokal dengan NumPy, pandas dan openpyxl yang sudah tersedia; tidak mengeksekusi library pihak ketiga yang diunduh.

## Status aset deteksi TB

**Aset penelitian tersedia; aplikasi deteksi final belum tersedia.**

- `../model_bank_v5/`: 120 bundle classifier, scaler/PCA, calibrator, cutoff, metadata dan catatan pemilihan; 514 file diekstrak dari return ZIP dengan checksum diperiksa. Bukan 120 model yang harus dijalankan bersamaan.
- `../pretrained/v5/BEATs_iter3_plus_AS2M.pt`: encoder audio, 361.499.833 byte.
- `../pretrained/v5/tabpfn-v2-classifier-v2_default.ckpt`: pretrained TabPFN, 29.009.539 byte; lisensinya disimpan di folder sama.
- `../src/coda_v5_features.py`, `coda_v5_models.py`, `coda_v5.py`: ekstraksi fitur, model, protokol kalibrasi/evaluasi. Runtime inference per pasien vs batch masih perlu diselaraskan, sesuai audit V5.
- `../.venv_v5_py311/`: Python 3.11, torch/torchaudio 2.6.0+cu118, numpy 1.26.4, pandas 2.2.2, scipy 1.13.1, sklearn 1.6.1, catboost 1.2.8, tabpfn 2.0.9, librosa .10.2.post1, joblib 1.4.2.

Tidak tersedia sebagai produk final: satu model deployment terpilih, API prediksi, pipeline rekaman mentah → segmentasi batuk tervalidasi, dan web pengguna. Semua kelompok model V5 memasukkan fitur denyut; tersedianya encoder audio bukan berarti tersedia model TB audio-only V5 terkalibrasi. Jangan diam-diam memasukkan denyut nol/median seolah-olah terukur.

Konfigurasi arsip dalam bank model sengaja tidak diedit: path D:\haqy berasal dari PC training. Saat membuat inference loader, gunakan salinan konfigurasi di memori dengan path lokal dan device sesuai lingkungan. Jangan memanggil ulang konfigurasi full-training untuk mengoperasikan inference di laptop, dan jangan menimpa artefak penelitian agar terlihat cocok.

## Pendekatan yang dipilih untuk tahap pertama

**Contact camera PPG**: jari menutup kamera belakang; flash dipakai bila kemampuan perangkat memungkinkan. Ambil mean intensitas RGB dan timestamp frame aktual, lakukan detrending/filtering, estimasi periode denyut, dan penilaian kualitas. Ini pemrosesan sinyal; tidak memerlukan training neural network atau dataset pasien TB baru untuk memulai.

Dataset berlabel tetap diperlukan untuk mengevaluasi error dan kualitas, serta untuk melatih/tuning jika kelak kita memilih pendekatan machine learning. Data publik tidak menggantikan pengujian pada HP yang akan dipakai pengguna.

Jangan menambah face-rPPG, tekanan darah, SpO2, HRV, atau diagnosis aritmia pada versi pertama. Fitur tersebut memerlukan pembuktian tersendiri dan tidak diperlukan untuk memenuhi input denyut model V5.

## Data yang ditemukan dan sudah dicek

### 1. Welltory — langsung siap sebagai input sinyal

[Repository resmi](https://github.com/Welltory/welltory-ppg-dataset), dipin ke commit `3b506e9c97c45b89695e87f1d667fe8a033e24fa`, lisensi CC0.

Sudah diunduh dan diekstrak: `data/welltory/`. Ada 21 pasangan PPG.csv/RR.txt; repository melaporkan 13 relawan. RGB dan timestamp berasal dari kamera HP; RR pembanding berasal dari Polar H10. Tidak ada video mentah. Seluruh timestamp monoton, angka finite, dan interval RR positif dalam pemeriksaan ini. Gunakan untuk menguji jalur RGB/timestamp → estimasi denyut. Identitas folder rekaman bukan identitas pasien unik; metadata pemetaan ke 13 relawan tidak tersedia jelas pada paket ini. Penyelarasan awal waktu RR terhadap PPG perlu diperiksa sebelum evaluasi per-jendela; jangan menganggap sinkronisasi sampel sempurna hanya dari dua file.

Inventaris lokal: `metadata/welltory_inventory.csv`.

### 2. Fingertip Videos — langsung siap sebagai input video

[Halaman dataset dan lisensi CC BY 4.0](https://figshare.com/articles/dataset/Fingertip_Videos_for_Heart_Rate_estimation/17052410), DOI `10.6084/m9.figshare.17052410.v2`.

Sudah diunduh:

- `downloads/DatasetVideoFiles.zip`: 68.141.216 byte, **51 MP4**.
- `downloads/ProjectDatasetReadings.xlsx`: 9.935 byte, **51 baris label** yang cocok dengan seluruh nama video.
- Salinan label CSV: `metadata/figshare_video_labels.csv`.

Ukuran dan MD5 cocok dengan API publisher, CRC ZIP lolos. Deskripsi publisher melaporkan 24 peserta, perekaman Redmi Note 8 sekitar 20 detik/30 fps, dan BPM pembanding Andesfit pulse oximeter. Kita tidak menghitung ulang durasi/fps setiap video pada tahap inventaris ini. Ada 24 rekaman berusia ≥18 tahun dan 27 rekaman anak; **bukan 24 peserta dewasa**. Pemetaan eksplisit rekaman berulang ke individu tidak ada di Excel. Karena itu jangan melakukan random clip split lalu mengklaim evaluasi patient-independent. Cocok untuk uji video → mean RGB → estimasi BPM dengan algoritma yang dibekukan; label BPM skalar tidak memvalidasi ketepatan setiap beat/HRV.

### 3. BUT PPG 2.0.0 — sumber utama tambahan untuk penilaian kualitas

[PhysioNet resmi, akses terbuka CC BY 4.0](https://physionet.org/content/butppg/2.0.0/).

Versi ini berisi 3.888 rekaman 10 detik dari 50 peserta; dari metadata ada **1.926 rekaman finger**, sisanya ear. Tersedia PPG kamera yang telah diubah menjadi sinyal, ECG/HR referensi dan anotasi kualitas. Paket ini bukan koleksi video mentah. Gunakan subset finger untuk kesesuaian aplikasi; grouping individu tersedia dari tiga digit awal ID. Jangan menyambung klip 10 detik yang berbeda menjadi rekaman 30 detik palsu.

Metadata `quality-hr-ann.csv`, `subject-info.csv`, lisensi dan SHA256SUMS sudah diunduh ke `metadata/` dan diverifikasi terhadap hash publisher. **ZIP sinyal lengkap belum berhasil diunduh**: server timeout setelah sekitar 0,49 MB dari 90.951.417 byte. Berkas parsial bernama `.zip.part`, tidak boleh dipakai sebagai dataset lengkap.

[Unduh ZIP resmi sekitar 86,7 MiB](https://physionet.org/content/butppg/get-zip/2.0.0/). Unduhan ini berguna untuk tahap kualitas, tetapi tidak menghalangi memulai dari dua dataset yang sudah lengkap di atas. Sumber alternatif resmi: `s3://physionet-open/butppg/2.0.0/`; metadata telah berhasil diambil melalui endpoint S3 publik.

Klip BUT 10 detik tidak selalu memenuhi waktu settling/quality-gate aplikasi 30 detik. Pisahkan evaluasi estimator pada klip pendek dari evaluasi sesi aplikasi; laporkan kegagalan/coverage, jangan menurunkan quality threshold hanya agar hasil selalu keluar.

## Kode dan dokumentasi

### Kandidat browser: ppg-js 0.3.0, MIT

[Repository](https://github.com/sontakey/ppg-js). Paket npm **versi tetap .3.0**, bukan `latest` saat runtime, sudah disimpan di `downloads/ppg-js-0.3.0.tgz`; integrity SHA-512 cocok dengan registry. Seluruh berkas paket diekstrak ke `vendor/ppg-js-0.3.0/`, termasuk `dist/index.global.js` sehingga tidak perlu CDN untuk menjalankan build browser.

Ada jalur kamera, headless engine, fungsi DSP, quality gate dan replay. Sumber ini **kandidat implementasi**, bukan bukti klinis. Dokumentasinya sendiri menyatakan belum divalidasi pada populasi dengan ECG/chest strap. Default quality threshold dan rentang HR tidak boleh langsung dianggap cocok dengan kebutuhan V5. Versi npm yang disimpan dapat tertinggal dari branch GitHub; manifest menyimpan identitasnya.

Hal yang wajib diperhatikan integrasi: library dapat mengembalikan `heartRate=0` ketika `quality.good=false`. Adapter aplikasi wajib mengubah hasil ini menjadi **HR tidak tersedia**, bukan meneruskan 0 BPM ke model TB. Interpolasi frame ke grid lebih tinggi juga tidak menciptakan informasi temporal yang sebelumnya tidak direkam.

### Pembanding offline yang opsional

- [HeartPy, MIT](https://github.com/paulvangentcom/heartrate_analysis_python): pembanding estimasi pulse dari PPG.
- [NeuroKit2, MIT](https://github.com/neuropsychology/NeuroKit): pembanding pengolahan PPG dan evaluasi sinyal.
- SciPy sudah tersedia dalam environment V5 untuk baseline detrending, filtering dan analisis spektrum. Jangan menambah dua toolkit sekaligus bila satu baseline sudah cukup.
- `opencv-python` diperlukan bila video akan didekode pada notebook Python; `wfdb` bila membaca sinyal BUT dalam format WFDB. Keduanya belum terpasang dalam environment V5. Browser dapat mendekode video dan memakai package JS tanpa dependency Python tersebut. `openpyxl` tersedia di Python Anaconda lokal untuk verifikasi Excel, belum pada venv training V5; label sudah disediakan sebagai CSV agar benchmark berikutnya tidak perlu memasangnya.

Dokumentasi resmi browser:

- [getUserMedia dan kebutuhan secure context](https://developer.mozilla.org/en-US/docs/Web/API/MediaDevices/getUserMedia).
- [requestVideoFrameCallback dan metadata waktu frame](https://developer.mozilla.org/en-US/docs/Web/API/HTMLVideoElement/requestVideoFrameCallback).
- [Constraints/capabilities kamera](https://developer.mozilla.org/en-US/docs/Web/API/Media_Capture_and_Streams_API/Constraints).

Gunakan HTTPS untuk akses dari HP. `localhost` pada laptop bukan `localhost` pada HP; mengakses alamat LAN `http://192.168...` tidak otomatis memenuhi syarat kamera. Torch, exposure lock dan pemilihan lensa harus diperiksa dari capabilities aktual, bukan dijanjikan berfungsi pada semua browser.

Dasar medis-teknis: [Bae et al., Communications Medicine 2022](https://www.nature.com/articles/s43856-022-00102-x) memvalidasi pengukuran kamera jari pada 95 peserta dengan pembanding. Artikel ini mendukung kelayakan prinsip, bukan angka akurasi kode kita.

## Langkah kerja berikutnya dan kriteria keluar

1. **Benchmark offline terlebih dahulu.** Algoritma sederhana/dsp dan kandidat JS menerima data yang sama. Mulai Welltory lalu video Figshare. Ukur MAE/median absolute error/bias, error tinggi yang lolos, serta coverage (berapa sesi menghasilkan BPM sah). Laporkan semua kegagalan. Jangan memilih hanya rekaman mudah. Data tuning dan test harus terpisah; bila identitas peserta tidak lengkap, nyatakan keterbatasannya.
2. **Tetapkan quality gate dari bukti.** Gerakan, saturasi, timestamp/drop frame, amplitudo lemah, kestabilan antarjendela dan persetujuan estimator diperiksa. Tetapkan target galat sebelum mengevaluasi holdout. Contoh target teknis awal MAE≤5 BPM dan uji sensitivitas ±5/±10 BPM adalah pilihan proyek, bukan sertifikasi/standar klinis yang sudah dipenuhi. Selaraskan dengan hasil V5 bahwa error −10 BPM dapat menambah FN.
3. **Prototipe web pengukur denyut terpisah.** Sesi sekitar 30 detik setelah settling, dapat diperpanjang bila sinyal belum cukup. Tampilkan gelombang dan instruksi, BPM hanya jika kualitas lolos, alasan jika gagal. Simpan RGB/timestamp dan diagnostik bila pengguna setuju; video mentah tidak wajib dikirim/disimpan. Hentikan kamera dan torch setelah selesai.
4. **Pengujian perangkat nyata.** Mulai satu Android/Chrome dan satu iPhone/Safari bila tersedia, lalu perluas. Ukur keberhasilan, repeatability, waktu, flash dan camera compatibility. Hasil tidak stabil tidak boleh diteruskan ke model TB. HR diukur saat diam **sebelum** sesi batuk, bukan sambil batuk.
5. **Validasi pengukuran.** Dibutuhkan data berpasangan alat referensi yang layak pada perangkat/kondisi target untuk klaim akurasi HR baru. Bila belum memiliki alat, tahap 1–4 tetap bisa dilakukan; tahap ini belum selesai. Nadi manual hanya pemeriksaan kasar. Jangan memakai keluaran aplikasi HR lain sebagai gold standard.
6. **Integrasi riset TB.** Selesaikan konsistensi inferensi checkpoint terlebih dahulu; hubungkan HR sah ke `beats_hr`, lalu isian jika dibutuhkan. Pengukuran gagal menghasilkan `heart_rate=null` dan status ditolak, bukan nilai rekaan. Alur penanganan HR hilang harus dievaluasi tersendiri. Setelah model deployment dipilih dan dibekukan, lakukan evaluasi end-to-end baru.

Dataset PPG tidak perlu memiliki label TB: fungsinya menguji pengukur denyut. **Jangan menyatukan peserta PPG dan CODA secara acak menjadi dataset multimodal buatan.** Untuk membuktikan aplikasi TB dengan HR kamera pada populasi baru, akhirnya diperlukan rekaman kamera+batuk+klinis dari orang yang sama dan referensi diagnosis TB. Dataset publik di atas tidak menyediakan validasi gabungan tersebut.

Belum dilakukan: benchmark akurasi resource baru, eksekusi library JS, pembuatan UI/endpoint, pengujian HP nyata atau validasi alat. Sumber lokal sudah cukup untuk mulai tahap benchmark tanpa GPU training baru; kesiapan klinis dan akurasi pada HP pengguna masih terbuka.
