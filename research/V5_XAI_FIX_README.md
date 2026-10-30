> Historical V5 methodology notes. For the current folder layout and setup, use [README.md](README.md). Older ZIP paths below describe the original training transfer.

# V5 XAI — perbaikan runtime dan resume tanpa training ulang

**Built with PriorLabs-TabPFN**

Error `XAI raw batch discrepancy >1e-5` berarti prediksi all-present dalam batch berbeda dari prediksi satu peserta. Traceback belum membuktikan apakah model yang gagal TabPFN, model lain, atau besaran selisihnya. Patch ini mencatat diagnostik dan mengevaluasi XAI secara pointwise ketika perbedaan terdeteksi; tidak menaikkan toleransi asli dan tidak mengubah skor test.

## Cara termudah di PC training

1. Ekstrak seluruh `CODA_TB_V5_XAI_RESUME_FIX.zip` ke folder proyek V5, misalnya `D:/haqy/`. Paket kecil ini hanya menambahkan helper dan notebook; sumber training/config/model asli tidak disertakan untuk ditimpa.
2. Buka `D:/haqy/CODA_TB_V5_Resume_XAI.ipynb`, pilih kernel V5 yang sama, lalu restart kernel.
3. Pastikan `RUN_NAME="experiment_v5_001"` sama dengan nama run training. Jika menggunakan nama lain, isi nama tersebut. Bila paket tidak ditemukan otomatis, isi `PACKAGE_DIR=r"D:\haqy\research_package"`.
4. Jalankan **Run All pada notebook Resume XAI ini**. Ia membaca `runs/<RUN_NAME>/resolved_config.json`, tabel fitur, split, dan model tersimpan. Tidak memerlukan data audio/ekstraksi BEATs baru atau unduhan library.
5. Setelah selesai, bawa kembali `research_package/runs/<RUN_NAME>_RETURN.zip` ke laptop untuk audit.

Jangan mengganti RUN_NAME dengan nama run baru untuk perbaikan ini. Pertahankan folder `runs/<RUN_NAME>` dan checkpoint. Jika Anda sudah mengedit `coda_v5.py`, `coda_v5_models.py`, atau config secara manual, kembalikan berkas asli dari paket V5 sebelum resume; helper akan menolak sumber yang berbeda dari fingerprint training.

## Jika notebook lama masih terbuka

Sesudah ekstraksi patch, sebuah sel baru sebelum sel XAI dapat menjalankan:

```python
from src import v5_xai_resume as repair
protected = repair.apply(cfg, splits)
try:
    v5.finish_postprocessing(table, splits, cfg)
finally:
    repair.verify_unchanged(cfg, protected)
display(repair.summarize(cfg))
```

Kemudian lanjutkan visualisasi dan ekspor RETURN pada notebook lama. Notebook resume terpisah lebih mudah digunakan setelah restart kernel karena memuat variabel sendiri.

## Yang dipertahankan dan yang berubah

- Seluruh checksum artefak training/evaluasi yang tercatat pada `TRAIN_EVAL_COMPLETE.json` diverifikasi sebelum resume, dan diperiksa lagi setelahnya.
- Kode asli, bobot, hyperparameter, cutoff, serta prediksi test lama tetap digunakan.
- Helper bernama `v5_xai_resume.py` agar tidak masuk glob fingerprint source V5 yang asli. Sumber patch disalin ke `run/runtime_sources/` dan direkam hash-nya untuk audit terpisah.
- Perubahan predictor hanya berlaku selama satu pemanggilan XAI dan selalu dipulihkan melalui `finally`, termasuk bila terjadi error. Prediksi robustness/cascade tetap memakai predictor asli.
- Ketika selisih batch terdeteksi, semua koalisi XAI berikutnya pada bundle tersebut menggunakan inferensi satu baris, dengan cache input yang identik. Menggunakan satu fungsi prediksi pointwise membuat nilai koalisi konsisten.
- Kelengkapan kelompok input diperiksa. Koalisi all-present harus benar-benar sama dengan input peserta. Prediksi pointwise harus repeatable pada toleransi 1e-10; completeness tetap 1e-8. Mismatch nyata masih menghentikan proses.
- Postprocessing fold yang sudah lengkap tetap digunakan. Fold yang belum lengkap mengulang postprocessing saja. TabPFN memulihkan konteks training tersimpan untuk inferensi; ini bukan pencarian model atau pelatihan ulang bobot foundation model.

Inferensi per baris dapat lebih lambat daripada batch. Cache mengurangi prediksi duplikat, tetapi durasi pada GPU sasaran belum diukur.

## Diagnostik yang wajib dibaca saat audit

`nested_grouped_xai.csv` menyimpan `original_batch_raw_delta`, `original_batch_calibrated_delta`, `rowwise_fallback_used`, dan `pointwise_full_raw_delta`. Setelah fallback aktif, sebagian probe batch tidak diulang dan nilainya dapat kosong; `fallback_trigger_raw_delta` menyimpan selisih yang mengaktifkannya.

Kolom `archived_test_raw_delta`, `archived_test_calibrated_delta`, dan `archived_youden_decision_changed` membandingkan endpoint penjelasan pointwise dengan hasil test lama pada peserta sampel. Jika selisih besar atau keputusan berbeda, itu perlu diaudit. Patch tidak menimpa atau menghitung ulang skor test agar terlihat konsisten. Ini juga tidak membuktikan bahwa perbedaan GPU selalu hanya roundoff kecil.

Ringkasan tersimpan di `V5_XAI_DIAGNOSTIC_SUMMARY.json`. Pemeriksaan terbatas pada kasus XAI yang disampel, bukan replay seluruh test. Fold lama yang selesai sebelum patch dapat tidak memiliki kolom diagnostik tambahan. Completeness merupakan identitas matematis, bukan validasi klinis atau bukti kausal.

## Pengujian lokal

Lolos pada Python 3.11: reproduksi error dengan drift batch buatan 0,0002; hasil fallback cocok dengan inferensi pointwise referensi; cache digunakan kembali; input tidak lengkap dan predictor tidak repeatable tetap ditolak; predictor dipulihkan setelah exception; actual TabPFN v2 CPU; resume postprocessing tanpa pemanggilan training classifier; checksum training tetap identik; runtime source masuk RETURN.

Seluruh sel notebook resume juga telah dieksekusi pada run sintetis dalam kernel baru, termasuk pemuatan run, visualisasi, verifikasi artefak terlindungi, dan ekspor RETURN.

Belum direproduksi pada GPU atau model hasil training Anda karena artefaknya belum tersedia di laptop ini. Tidak ada training TB baru untuk perbaikan ini.
