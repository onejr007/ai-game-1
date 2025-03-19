# **Deskripsi Proyek: AI Batu Gunting Kertas**  

🔗 **URL Aplikasi:** [AI Game](https://ai-game-1-production.up.railway.app/)  

## **📌 Tentang Proyek**  
Proyek ini adalah sistem **AI cerdas** yang bermain **Batu-Gunting-Kertas** melawan pengguna. AI dalam game ini **tidak hanya memilih secara acak**, tetapi **belajar dari pola permainan pengguna** dan **menggunakan berbagai strategi untuk meningkatkan kemenangan**.  

## **🧠 Fitur Utama**  
✅ **AI Berbasis Pola & Strategi** – AI mengenali pola permainan pengguna dan memilih strategi terbaik berdasarkan pola yang ditemukan.  
✅ **Penyimpanan Data Pola** – AI menyimpan pola permainan pengguna di `datapola.json` untuk analisis lebih lanjut.  
✅ **Strategi Dinamis** – Menggunakan berbagai strategi dari `strategi.py`, seperti **Monte Carlo Simulation, Markov Chain, Bayesian Learning, Look-Ahead Strategy**, dan masih banyak lagi.  
✅ **Validasi Best Move** – Sebelum memilih langkah, AI mengevaluasi apakah langkah tersebut sudah pernah digunakan sebelumnya dan apakah efektif dalam pola yang sama.  
✅ **Optimasi Performa** – Menggunakan **file lock** untuk mengelola file JSON dengan efisien, serta optimasi eksekusi API agar respons lebih cepat.  

## **🛠️ Teknologi yang Digunakan**  
- **FastAPI** – Backend utama untuk menangani request permainan.  
- **Python** – Bahasa pemrograman utama.  
- **NumPy, Scikit-learn** – Untuk strategi berbasis pembelajaran mesin.  
- **FileLock** – Untuk mengelola data game tanpa konflik.  
- **Railway.app** – Deployment aplikasi AI secara online.  

## **🚀 Cara Kerja AI**  
1️⃣ **Mendeteksi Pola Pemain** – AI memeriksa history game dan mencari pola yang sering digunakan pemain.  
2️⃣ **Mengecek Data di `datapola.json`** – Jika pola ditemukan, AI melihat strategi yang pernah digunakan dan tingkat kemenangan/kekalahan pengguna.  
3️⃣ **Memilih Strategi dari `strategi.py`** – AI menggunakan strategi yang memiliki performa terbaik berdasarkan pola pemain.  
4️⃣ **Validasi Best Move** – AI memastikan langkah terbaik berdasarkan riwayat kemenangan.  
5️⃣ **Menyesuaikan Strategi Jika Kalah** – Jika AI sering kalah dengan pola tertentu, ia akan mengganti strategi atau memilih langkah voting tertinggi berikutnya.  
6️⃣ **Menyimpan Data untuk Pembelajaran Masa Depan** – AI terus belajar dari permainan dan meningkatkan akurasi prediksinya.  

## **📌 Cara Menggunakan**  
1️⃣ **Akses URL:** [https://ai-game-1-production.up.railway.app/](https://ai-game-1-production.up.railway.app/)  
2️⃣ **Mainkan Batu-Gunting-Kertas** – Pilih salah satu langkah dan lihat bagaimana AI merespons.  
3️⃣ **Perhatikan Pola AI** – Semakin sering bermain, semakin AI memahami pola permainan Anda.  
4️⃣ **Coba Kalahkan AI!** – AI akan terus beradaptasi dan meningkatkan strateginya berdasarkan cara Anda bermain.  

## **💡 Pengembangan Selanjutnya**  
🔹 **Lebih Banyak Pola & Strategi** – AI akan semakin pintar dengan menambah variasi pola dan strategi.  
🔹 **Peningkatan Performa API** – Mengoptimalkan kecepatan respons untuk pengalaman bermain yang lebih lancar.  
🔹 **Leaderboard & Statistik** – Menampilkan statistik kemenangan/kekalahan pengguna vs AI.  

⚡ **Apakah Anda bisa mengalahkan AI ini? Coba sekarang!** 🎮
