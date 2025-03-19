import random
import bcrypt
import datetime
import json
import jwt
import time
import threading
import os
import numpy as np
import logging
import sys
import orjson
import functools

from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.security import OAuth2PasswordBearer
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from collections import defaultdict
from filelock import FileLock, Timeout
from apscheduler.schedulers.background import BackgroundScheduler
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from starlette.middleware.base import BaseHTTPMiddleware


from strategi import STRATEGY_FUNCTIONS

# 🔐 Konfigurasi Keamanan
SECRET_KEY = "supersecretkey"
TOKEN_EXPIRY = 1800  # 30 menit
blacklisted_tokens = set()

# 📂 File JSON
USER_DATA_FILE = "datalogin.json"
STATS_DATA_FILE = "datastatistik.json"
POLA_DATA_FILE = "datapola.json"

label_encoder = LabelEncoder()
# Buat filter untuk menyaring request tertentu
class SuppressEndpointsFilter(logging.Filter):
    def filter(self, record):
        return all(endpoint not in record.getMessage() for endpoint in ["/play", "/","/leaderboard", "/ai_stats"])
    
class ProcessTimeMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        start_time = time.time()
        response = await call_next(request)
        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = str(process_time)
        return response

# Model untuk User
class UserRegister(BaseModel):
    username: str
    password: str

class UserLogin(BaseModel):
    username: str
    password: str

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Ganti dengan domain yang diizinkan, contoh: ["http://localhost:3000"]
    allow_credentials=True,
    allow_methods=["*"],  # Izinkan semua metode (POST, GET, OPTIONS, dll.)
    allow_headers=["*"],
)
app.add_middleware(ProcessTimeMiddleware)

# Mount folder static untuk frontend
app.mount("/static", StaticFiles(directory="static"), name="static")
# Konfigurasi logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.DEBUG,  # 🔥 Pastikan level DEBUG aktif
    handlers=[logging.StreamHandler(sys.stdout)]  # 🔥 Paksa log muncul di console Windows
)

logger = logging.getLogger(__name__)

# Terapkan filter ke logger Uvicorn Access
uvicorn_access_logger = logging.getLogger("uvicorn.access")
uvicorn_access_logger.addFilter(SuppressEndpointsFilter())

# Logging lebih cepat dengan NullHandler untuk komponen tidak penting
logging.getLogger("filelock").addHandler(logging.NullHandler())
logging.getLogger("filelock").setLevel(logging.WARNING)
logging.getLogger("uvicorn.error").setLevel(logging.WARNING)
logging.getLogger("uvicorn").setLevel(logging.WARNING)
logging.getLogger("uvicorn.access").propagate = False

logger.info(f"🔥 APP STARTED !")
sys.stdout.flush()  # 🔥 Paksa agar log langsung tampil

# 📥 Fungsi Baca & Tulis JSON dengan File Lock dan optimasi kecepatan
def read_json(file, default_data=None):
    lock = FileLock(file + ".lock")

    try:
        with lock.acquire(timeout=2):  # Timeout lebih cepat untuk menghindari deadlock
            if not os.path.exists(file):
                write_json(file, {} if default_data is None else dict(default_data))

            with open(file, "rb") as f:  # Mode 'rb' lebih cepat
                return orjson.loads(f.read())

    except (orjson.JSONDecodeError, FileNotFoundError):
        print(f"⚠️ File {file} rusak atau tidak ditemukan. Membuat ulang...")
        write_json(file, {} if default_data is None else dict(default_data))
        return {} if default_data is None else dict(default_data)

    except Timeout:
        print(f"❌ Timeout: Gagal mengunci {file}, melewati operasi ini.")
        return {} if default_data is None else dict(default_data)

def write_json(file, data):
    lock = FileLock(file + ".lock")

    try:
        with lock.acquire(timeout=2):  # Hindari deadlock dengan timeout kecil
            with open(file, "wb") as f:  # Gunakan 'wb' untuk menulis lebih cepat
                f.write(orjson.dumps(data))  # orjson lebih cepat daripada json
    except Timeout:
        print(f"❌ Timeout: Gagal mengunci {file}, operasi dilewati!")

# 📌 Pastikan file JSON selalu ada
def initialize_json_files():
    read_json(USER_DATA_FILE, default_data={})
    read_json(STATS_DATA_FILE, default_data={})
    read_json(POLA_DATA_FILE, default_data={})

initialize_json_files()


# **1️⃣ API Halaman Utama (Menyajikan index.html)**
@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    try:
        with open("static/index.html", "r", encoding="utf-8") as file:
            return HTMLResponse(content=file.read())
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="File index.html tidak ditemukan.")

# **2️⃣ Middleware untuk Mendapatkan User dari Token**
async def get_current_user(request: Request):
    token = request.headers.get("Authorization")
    if not token:
        raise HTTPException(status_code=401, detail="Token missing")
    token = token.split(" ")[1]
    if token in blacklisted_tokens:
        raise HTTPException(status_code=401, detail="Token blacklisted")
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        if datetime.datetime.utcnow().timestamp() > payload["exp"]:
            raise HTTPException(status_code=401, detail="Token expired")
        return payload["user_id"]
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

# **🔄 Reset Data Setiap Jam 12 Malam WIB
def reset_all_user_data():
    write_json(STATS_DATA_FILE, {})
    print("[INFO] Semua data pengguna telah direset!")

scheduler = BackgroundScheduler()
scheduler.add_job(reset_all_user_data, 'cron', hour=0, minute=0, timezone="Asia/Jakarta")
scheduler.start()

# **3️⃣ API Register User**
@app.post("/register")
async def register(user: UserRegister):
    users = read_json(USER_DATA_FILE)

    if user.username in users:
        return JSONResponse(content={"error": "Username sudah ada"}, status_code=400)

    hashed_password = bcrypt.hashpw(user.password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

    users[user.username] = {"password": hashed_password}
    write_json(USER_DATA_FILE, users)

    # Tambahkan statistik default untuk user
    stats = read_json(STATS_DATA_FILE)
    stats[user.username] = {"total_games": 0, "total_wins": 0, "total_losses": 0, "total_draws": 0, "score": 0, "win_streak": 0, "history": []}
    write_json(STATS_DATA_FILE, stats)

    return JSONResponse(content={"message": "Registrasi berhasil"})

# **4️⃣ API Login User & Dapatkan Token**
@app.post("/login")
async def login(user: UserLogin):
    users = read_json(USER_DATA_FILE)

    if user.username not in users:
        raise HTTPException(status_code=400, detail="Invalid username or password")

    stored_password = users[user.username]["password"]
    if not bcrypt.checkpw(user.password.encode('utf-8'), stored_password.encode('utf-8')):
        raise HTTPException(status_code=400, detail="Invalid username or password")

    token = jwt.encode({"user_id": user.username, "exp": datetime.datetime.utcnow() + datetime.timedelta(hours=12)}, SECRET_KEY, algorithm="HS256")

    return JSONResponse(content={"token": token})

# **5️⃣ AI Learning: Prediksi Berdasarkan Data User**
# 🔹 Fungsi mengenali pola dari history pemain
def detect_pattern(moves, max_length=15):
    for length in range(3, max_length + 1):
        if len(moves) >= length * 2 and moves[-length:] == moves[-(length * 2):-length]:
            return "".join(moves[-length:])
    return None

# 🔹 Fungsi validasi best move berdasarkan hasil sebelumnya
def validate_best_move(user_id, pola_str, move):
    stats = read_json(STATS_DATA_FILE)
    
    if user_id not in stats or "history" not in stats[user_id]:
        return True  # Jika tidak ada history, anggap valid
    
    history = stats[user_id]["history"]
    recent_results = [h["result"] for h in history[-10:]]
    
    # Jika dalam 10 game terakhir lebih banyak kalah, jangan gunakan move ini
    if recent_results.count("Kalah") > recent_results.count("Menang"):
        return False
    return True

# 🔹 Fungsi menyimpan pola baru jika belum ada
def save_pattern(user_id, pola_str):
    data_pola = read_json(POLA_DATA_FILE)

    if pola_str not in data_pola:
        data_pola[pola_str] = {}

    write_json(POLA_DATA_FILE, data_pola)

# 🔹 Fungsi update daftar user yang dikalahkan pada pola tertentu
def update_users_defeated(user_id, pola_str, strategy, result):
    data_pola = read_json(POLA_DATA_FILE)

    if pola_str not in data_pola:
        data_pola[pola_str] = {}

    if strategy not in data_pola[pola_str]:
        data_pola[pola_str][strategy] = {}

    if user_id not in data_pola[pola_str][strategy]:
        data_pola[pola_str][strategy][user_id] = {"wins": 0, "losses": 0}

    if result == "Menang":
        data_pola[pola_str][strategy][user_id]["wins"] += 1
    elif result == "Kalah":
        data_pola[pola_str][strategy][user_id]["losses"] += 1

    write_json(POLA_DATA_FILE, data_pola)

# 🔹 Fungsi utama prediksi move AI
def predict_next_move(user_id):
    stats = read_json(STATS_DATA_FILE)
    
    if user_id not in stats or "history" not in stats[user_id] or len(stats[user_id]["history"]) < 5:
        return random.choice(["batu", "gunting", "kertas"])  # Jika data minim, pilih acak
    
    history = stats[user_id]["history"]
    recent_moves = [h["user_move"] for h in history[-150:]]
    recent_results = [h["result"] for h in history[-150:]]
    
    # 🔍 1️⃣ Mencari pola history pemain
    pola_str = detect_pattern(recent_moves)
    
    if not pola_str:
        pola_str = "".join(recent_moves[-3:])  # Gunakan pola 3 langkah terakhir jika tidak ada pola berulang
    
    # 🔄 Simpan pola baru jika belum ada
    save_pattern(user_id, pola_str)

    # 📥 Baca strategi dari datapola.json
    pola_data = read_json(POLA_DATA_FILE)
    strategi_terpilih = []

    if pola_str in pola_data:
        strategi_terpilih = list(pola_data[pola_str].keys())  # Gunakan strategi yang sudah ada
    else:
        strategi_terpilih = ["random_choice"]  # Jika tidak ada strategi, gunakan random_choice
    
    # 📊 2️⃣ Gunakan strategi yang tersedia dan hitung voting move
    move_votes = defaultdict(int)
    
    for strategi in strategi_terpilih:
        if strategi in STRATEGY_FUNCTIONS:
            move = STRATEGY_FUNCTIONS[strategi](recent_moves)
            if move:
                move_votes[move] += 1

    # 🎯 3️⃣ Memilih move terbaik berdasarkan voting strategi
    if move_votes:
        best_move = max(move_votes, key=move_votes.get)
    else:
        best_move = random.choice(["batu", "gunting", "kertas"])

    # ✅ 4️⃣ Validasi best move sebelum digunakan
    if not validate_best_move(user_id, pola_str, best_move):
        sorted_votes = sorted(move_votes.items(), key=lambda x: x[1], reverse=True)
        best_move = sorted_votes[1][0] if len(sorted_votes) > 1 else random.choice(["batu", "gunting", "kertas"])

    # 🔄 5️⃣ Update daftar user yang dikalahkan dalam pola ini
    hasil_game_terakhir = history[-1]["result"] if history else "Kalah"
    update_users_defeated(user_id, pola_str, best_move, hasil_game_terakhir)

    return best_move

# **6️⃣ API Play Game**
@app.post("/play")
async def play(request: Request, user_choice: str, current_user: str = Depends(get_current_user)):
    logger.info(f"🛠️ [DEBUG] User {current_user} memilih: {user_choice}")
    sys.stdout.flush()  # 🔥 Paksa agar log langsung muncul di Windows console

    if user_choice not in ["batu", "gunting", "kertas"]:
        raise HTTPException(status_code=400, detail="Invalid choice")

    ai_choice = predict_next_move(current_user)

    # 🔥 Debugging: Tampilkan strategi yang tersedia
    logger.info(f"🛠️ [DEBUG] Hasil : {ai_choice}")
    sys.stdout.flush()  # 🔥 Paksa agar log langsung tampil

    result = "Seri" if user_choice == ai_choice else (
        "Menang" if (user_choice == "batu" and ai_choice == "gunting") or 
                    (user_choice == "gunting" and ai_choice == "kertas") or 
                    (user_choice == "kertas" and ai_choice == "batu")
        else "Kalah"
    )

    stats = read_json(STATS_DATA_FILE)

    # ✅ Pastikan user memiliki entri statistik jika belum ada
    if current_user not in stats:
        stats[current_user] = {
            "total_games": 0, "total_wins": 0, "total_losses": 0, "total_draws": 0,
            "score": 0, "win_streak": 0, "history": []
        }

    # ✅ Pastikan history selalu dalam format yang benar
    game_record = {
        "user_move": user_choice,
        "ai_move": ai_choice,
        "result": result
    }
    
    # ✅ Tambahkan history jika format benar
    if "history" in stats[current_user] and isinstance(stats[current_user]["history"], list):
        stats[current_user]["history"].append(game_record)
    else:
        stats[current_user]["history"] = [game_record]  # Perbaiki jika history tidak ada atau format salah

    # ✅ Update statistik user
    stats[current_user]["total_games"] += 1
    if result == "Menang":
        stats[current_user]["total_wins"] += 1
        stats[current_user]["score"] += 1
        stats[current_user]["win_streak"] += 1
    elif result == "Kalah":
        stats[current_user]["total_losses"] += 1
        stats[current_user]["win_streak"] = 0
    else:
        stats[current_user]["total_draws"] += 1

    write_json(STATS_DATA_FILE, stats)

    return JSONResponse(content={
        "user_choice": user_choice,
        "ai_choice": ai_choice,
        "result": result
    })

# **7️⃣ API Leaderboard dengan Win Rate (%)**
@app.get("/leaderboard")
async def leaderboard():
    stats = read_json(STATS_DATA_FILE)
    
    leaderboard_data = []
    for user, data in stats.items():
        total_games = data["total_games"]
        total_wins = data["total_wins"]
        
        win_rate = round((total_wins / total_games) * 100, 2) if total_games > 0 else 0
        
        leaderboard_data.append({
            "username": user,
            "total_games": total_games,
            "total_wins": total_wins,
            "total_losses": data["total_losses"],
            "total_draws": data["total_draws"],
            "win_rate": win_rate
        })
    
    # Urutkan berdasarkan win_rate (descending) & total_games (descending)
    leaderboard_data.sort(key=lambda x: (-x["win_rate"], -x["total_games"]))

    return JSONResponse(content={"leaderboard": leaderboard_data})

# **8️⃣ API Cek Statistik AI**
@app.get("/ai_stats")
async def ai_statistics():
    stats = read_json(STATS_DATA_FILE)

    total_users = len(stats)  # Hitung jumlah user yang ada dalam data
    total_games = sum(user_data["total_games"] for user_data in stats.values())
    ai_wins = sum(user_data["total_losses"] for user_data in stats.values())  # AI menang jika user kalah
    ai_losses = sum(user_data["total_wins"] for user_data in stats.values())  # AI kalah jika user menang
    ai_draws = sum(user_data["total_draws"] for user_data in stats.values())  # Seri dihitung langsung

    # **Hindari pembagian dengan nol**
    def safe_div(numerator, denominator):
        return round((numerator / denominator) * 100, 2) if denominator > 0 else 0

    ai_win_rate = safe_div(ai_wins, total_games)
    ai_loss_rate = safe_div(ai_losses, total_games)
    ai_draw_rate = safe_div(ai_draws, total_games)

    return JSONResponse(content={
        "total_users": total_users,
        "total_games": total_games,
        "ai_wins": ai_wins,
        "ai_losses": ai_losses,
        "ai_draws": ai_draws,
        "ai_win_rate": ai_win_rate,
        "ai_loss_rate": ai_loss_rate,
        "ai_draw_rate": ai_draw_rate
    })

# **9️⃣ API Reset Statistik Pemain**
@app.post("/reset_stats")
async def reset_stats(current_user: str = Depends(get_current_user)):
    stats = read_json(STATS_DATA_FILE)
    stats[current_user] = {"total_games": 0, "total_wins": 0, "total_losses": 0, "total_draws": 0, "score": 0, "win_streak": 0, "history": []}
    write_json(STATS_DATA_FILE, stats)
    return {"message": "Statistik berhasil direset!"}

# **🔟 Self-Training AI Tanpa Henti (1 pertandingan per detik)**
def continuous_ai_training():
    while True:
        stats = read_json(STATS_DATA_FILE)

        # **Pastikan AI memiliki entri statistik jika belum ada**
        if "AI_vs_AI" not in stats:
            stats["AI_vs_AI"] = {
                "total_games": 0, "total_wins": 0, "total_losses": 0, "total_draws": 0,
                "score": 0, "win_streak": 0, "history": []
            }

        ai1_choice = random.choice(["batu", "gunting", "kertas"])
        ai2_choice = random.choice(["batu", "gunting", "kertas"])

        # **Menentukan hasil pertandingan**
        if ai1_choice == ai2_choice:
            result = "Seri"
        elif (ai1_choice == "batu" and ai2_choice == "gunting") or \
             (ai1_choice == "gunting" and ai2_choice == "kertas") or \
             (ai1_choice == "kertas" and ai2_choice == "batu"):
            result = "Menang"
        else:
            result = "Kalah"

        # **Update statistik AI**
        stats["AI_vs_AI"]["total_games"] += 1
        if result == "Menang":
            stats["AI_vs_AI"]["total_wins"] += 1
            stats["AI_vs_AI"]["score"] += 1
            stats["AI_vs_AI"]["win_streak"] += 1
        elif result == "Kalah":
            stats["AI_vs_AI"]["total_losses"] += 1
            stats["AI_vs_AI"]["win_streak"] = 0
        else:
            stats["AI_vs_AI"]["total_draws"] += 1

        # **Simpan history pertandingan AI vs AI**
        stats["AI_vs_AI"]["history"].append({
            "ai1_move": ai1_choice,
            "ai2_move": ai2_choice,
            "result": result
        })

        # **Simpan ke JSON**
        write_json(STATS_DATA_FILE, stats)

        # **Tunggu 1 detik sebelum pertandingan berikutnya**
        time.sleep(1)

# **1️⃣1️⃣ Jalankan Self-Training AI di Threading Background**
training_thread = threading.Thread(target=continuous_ai_training, daemon=True)
training_thread.start()

# 🧠 AI Belajar dari Semua User
def get_global_move_distribution():
    stats = read_json(STATS_DATA_FILE)
    move_counts = {"batu": 0, "gunting": 0, "kertas": 0}

    for user, data in stats.items():
        for history in data.get("history", []):
            move_counts[history["user_move"]] += 1

    total_moves = sum(move_counts.values())
    if total_moves == 0:
        return random.choice(["batu", "gunting", "kertas"])

    most_used = max(move_counts, key=move_counts.get)
    counter_moves = {"batu": "kertas", "gunting": "batu", "kertas": "gunting"}

    return counter_moves[most_used]

# 🔑 Logout & Token Blacklist
@app.post("/logout")
async def logout(request: Request):
    token = request.headers.get("Authorization")
    if token:
        token = token.split(" ")[1]
        blacklisted_tokens.add(token)
    return {"message": "Logged out successfully"}

def fix_broken_history():
    stats = read_json(STATS_DATA_FILE)
    for user, data in stats.items():
        if "history" in data and isinstance(data["history"], list):
            fixed_history = []
            for record in data["history"]:
                if "user_move" in record and "ai_move" in record and "result" in record:
                    fixed_history.append(record)  # Hanya simpan data yang valid
            
            stats[user]["history"] = fixed_history  # Perbaiki history user

    write_json(STATS_DATA_FILE, stats)
    print("[INFO] Semua data history yang rusak telah diperbaiki!")

fix_broken_history()

def debug_stats_data():
    stats = read_json(STATS_DATA_FILE)
    for user, data in stats.items():
        if "history" not in data or not isinstance(data["history"], list):
            print(f"[ERROR] User: {user} tidak memiliki history yang benar: {data}")
        else:
            for i, record in enumerate(data["history"]):
                if "user_move" not in record or "ai_move" not in record or "result" not in record:
                    print(f"[ERROR] User: {user}, History Index: {i}, Data: {record}")

debug_stats_data()
