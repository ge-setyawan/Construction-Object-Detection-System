# Construction Object Detection System

Aplikasi desktop berbasis Python untuk mendeteksi objek konstruksi pada gambar dan video menggunakan model YOLOv11. Antarmuka GUI dibuat dengan CustomTkinter dan menampilkan hasil deteksi, jumlah objek, analisis *counting region*, serta ekspor hasil ke Excel.

## Fitur

- Deteksi objek pada gambar (`.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`) dan video (`.mp4`, `.avi`, `.mov`, `.mkv`, `.webm`).
- Kelas objek: **Cangkul**, **Helm Proyek**, **Scaffolding Bambu**, **Semen**, dan **Tukang**.
- Pengaturan tingkat kepercayaan (*confidence threshold*) deteksi.
- Pilihan peningkatan citra: Original, Contrast Stretching, Histogram Equalization, CLAHE, dan Brightness.
- *Counting Region* untuk menghitung objek yang berada di area pilihan.
- Ringkasan jumlah objek dan rata-rata confidence secara langsung.
- Ekspor hasil deteksi gambar atau video ke file Excel (`.xlsx`).

## Prasyarat

- Python 3.10 atau lebih baru.
- Sistem operasi Windows, macOS, atau Linux dengan lingkungan desktop untuk menampilkan GUI.
- File model tersedia pada `models/best.pt`.

## Instalasi

1. Clone repository lalu masuk ke folder proyek.

2. Buat dan aktifkan virtual environment (disarankan).

   **Windows PowerShell**

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

   **macOS / Linux**

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. Pasang dependensi aplikasi dari file `requirements.txt`.

   ```bash
   pip install -r requirements.txt
   ```

4. Pastikan file model berada di lokasi berikut.

   ```text
   models/best.pt
   ```

## Menjalankan Aplikasi

Jalankan perintah berikut dari folder proyek:

```bash
python main.py
```

Jika model tidak ditemukan, aplikasi tetap terbuka, tetapi status model akan menampilkan bahwa model belum tersedia dan deteksi tidak dapat dijalankan.

## Membuat Ulang Virtual Environment

Folder `venv/` atau `.venv/` hanya menyimpan library Python secara lokal. Folder tersebut aman dihapus untuk menghemat ruang, tetapi Anda perlu membuat ulang environment dan memasang dependensi sebelum aplikasi dijalankan kembali.

**Windows PowerShell**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python main.py
```

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python main.py
```

Jika PowerShell menolak aktivasi script, jalankan ini sekali pada jendela PowerShell yang sama, lalu ulangi perintah aktivasi:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

## Cara Menggunakan GUI

### 1. Memuat media

- Klik **Buka Gambar** untuk memilih satu gambar. Deteksi akan diproses dan ditampilkan otomatis.
- Klik **Buka Video** untuk memilih video. Frame pertama akan muncul pada panel pratinjau.
- Untuk video, tekan **Play Video** untuk memulai pemrosesan atau **Stop** untuk menghentikannya.

### 2. Mengatur deteksi

- Gunakan **Image Enhancement** bila ingin menerapkan peningkatan kualitas citra. Untuk gambar, hasil deteksi akan diperbarui saat pilihan diubah.
- Geser **Confidence Threshold** untuk mengatur batas minimum confidence. Nilai lebih tinggi membuat hasil lebih selektif; nilai lebih rendah dapat menampilkan lebih banyak kandidat deteksi.

### 3. Membuat Counting Region

1. Muat gambar atau video.
2. Klik **Aktifkan Region**.
3. Klik lalu tarik (*click and drag*) pada area pratinjau untuk membuat kotak region.
4. Panel kanan akan memperbarui **Objects in Region** beserta jumlah setiap kelas di dalam kotak tersebut.
5. Klik **Hapus Region** untuk menghapus area dan kembali menghitung seluruh frame.

> Region diterapkan pada gambar/frame yang sedang ditampilkan. Saat membuka media baru, region sebelumnya akan dihapus.

### 4. Membaca hasil deteksi

Panel kanan menampilkan:

- **Total Objects**: jumlah semua objek yang terdeteksi.
- **Average Confidence**: rata-rata confidence dari deteksi yang tampil.
- **Classes Detected**: jumlah jenis kelas yang terdeteksi.
- **Objek Terdeteksi**: jumlah per kelas; nama berada di kiri dan nilai berada di kanan.
- **Objects in Region**: total serta rincian objek per kelas di dalam region aktif.

### 5. Mengekspor Excel

Klik **Export Excel**, pilih lokasi penyimpanan, lalu simpan sebagai `.xlsx`.

- Untuk gambar, file berisi satu ringkasan hasil deteksi saat ini.
- Untuk video, file berisi hasil per frame yang diproses untuk inferensi. Aplikasi menjalankan deteksi setiap tiga frame agar pemutaran tetap responsif.

Kolom ekspor meliputi nama file/frame, enhancement, total objek, rata-rata confidence, jumlah setiap kelas, serta statistik region.

## Troubleshooting

| Masalah | Solusi |
| --- | --- |
| `ModuleNotFoundError` | Aktifkan virtual environment lalu jalankan `pip install -r requirements.txt`. |
| Status model tidak ditemukan | Periksa bahwa `best.pt` berada di `models/best.pt`. |
| Video tidak dapat dibuka | Gunakan format video yang didukung dan pastikan codec tersedia di sistem. |
| Ekspor Excel gagal | Pastikan `openpyxl` sudah terpasang dan lokasi simpan memiliki izin tulis. |
| Tampilan video terasa lambat | Gunakan video beresolusi lebih rendah atau jalankan pada perangkat dengan GPU yang mendukung PyTorch. |

## Catatan

Hasil deteksi bergantung pada kualitas model `best.pt`, kondisi pencahayaan, sudut kamera, dan kesesuaian objek terhadap data pelatihan. Gunakan hasil sebagai bantuan analisis dan lakukan verifikasi manual untuk keputusan penting.
