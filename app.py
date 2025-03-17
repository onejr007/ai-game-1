import random
import bcrypt
import datetime
import json
import jwt
import time
import threading
import os
from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.security import OAuth2PasswordBearer
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from collections import defaultdict

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Ganti dengan domain yang diizinkan, contoh: ["http://localhost:3000"]
    allow_credentials=True,
    allow_methods=["*"],  # Izinkan semua metode (POST, GET, OPTIONS, dll.)
    allow_headers=["*"],
)

# Konfigurasi JWT (Token Auth)
SECRET_KEY = "supersecretkey"
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Mount folder static untuk frontend
app.mount("/static", StaticFiles(directory="static"), name="static")

# File JSON untuk menyimpan data
USER_DATA_FILE = "datalogin.json"
STATS_DATA_FILE = "datastatistik.json"

# Fungsi untuk membaca data JSON dengan pengecekan otomatis
def read_json(file, default_data=None):
    if not os.path.exists(file):
        write_json(file, default_data if default_data is not None else {})
    try:
        with open(file, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        write_json(file, default_data if default_data is not None else {})
        return default_data if default_data is not None else {}

# Fungsi untuk menulis data ke JSON
def write_json(file, data):
    with open(file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

# Fungsi untuk memastikan file JSON selalu ada
def initialize_json_files():
    read_json(USER_DATA_FILE, default_data={})
    read_json(STATS_DATA_FILE, default_data={})

initialize_json_files()

# Model untuk User
class UserRegister(BaseModel):
    username: str
    password: str

class UserLogin(BaseModel):
    username: str
    password: str


# **1️⃣ API Halaman Utama (Menyajikan index.html)**
@app.get("/", response_class=HTMLResponse)
def serve_frontend():
    try:
        with open("static/index.html", "r", encoding="utf-8") as file:
            return HTMLResponse(content=file.read())
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="File index.html tidak ditemukan.")

# **2️⃣ Middleware untuk Mendapatkan User dari Token**
def get_current_user(request: Request):
    token = request.headers.get("Authorization")
    if not token:
        raise HTTPException(status_code=401, detail="Token missing")
    try:
        payload = jwt.decode(token.split(" ")[1], SECRET_KEY, algorithms=["HS256"])
        return payload["user_id"]
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

# **3️⃣ API Register User**
@app.post("/register")
def register(user: UserRegister):
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
def login(user: UserLogin):
    users = read_json(USER_DATA_FILE)

    if user.username not in users:
        raise HTTPException(status_code=400, detail="Invalid username or password")

    stored_password = users[user.username]["password"]
    if not bcrypt.checkpw(user.password.encode('utf-8'), stored_password.encode('utf-8')):
        raise HTTPException(status_code=400, detail="Invalid username or password")

    token = jwt.encode({"user_id": user.username, "exp": datetime.datetime.utcnow() + datetime.timedelta(hours=12)}, SECRET_KEY, algorithm="HS256")

    return JSONResponse(content={"token": token})

# **5️⃣ AI Learning: Prediksi Berdasarkan Data User**
move_map = {"batu": 0, "gunting": 1, "kertas": 2}
counter_moves = {"batu": "kertas", "gunting": "batu", "kertas": "gunting"}

def predict_next_move(user_id):
    stats = read_json(STATS_DATA_FILE)
    if user_id not in stats or "history" not in stats[user_id] or len(stats[user_id]["history"]) < 10:
        return random.choice(["batu", "gunting", "kertas"])

    history = stats[user_id]["history"]
    recent_moves = [h["user_move"] for h in history[-150:]]
    recent_results = [h["result"] for h in history[-150:]]

    # **1️⃣ Push-Hit Exploit Shield**
    if len(recent_results) >= 10 and all(r == "Menang" for r in recent_results[-10:]):
        return counter_moves[recent_moves[-1]]

    if len(recent_results) >= 5 and all(r == "Menang" for r in recent_results[-5:]):
        return counter_moves[recent_moves[-1]]

    # **2️⃣ Reverse Adaptive Momentum**
    if len(recent_results) >= 4 and recent_results[-3:] == ["Kalah", "Seri", "Seri"]:
        return counter_moves[recent_moves[-1]]

    # **3️⃣ Dynamic Push-Hit Switch**
    if len(recent_results) >= 6:
        last_six = recent_results[-6:]
        if last_six.count("Menang") >= 4 and last_six.count("Seri") <= 2:
            return counter_moves[recent_moves[-1]]

    # **4️⃣ Recursive Cycle Detection**
    def detect_cycle(moves, cycle_length):
        if len(moves) >= cycle_length * 2:
            return moves[-cycle_length:] == moves[-(cycle_length * 2):-cycle_length]
        return False

    cycle_lengths = [3, 5, 7, 10, 15, 20, 25, 30]
    for length in cycle_lengths:
        if detect_cycle(recent_moves, length):
            return counter_moves[recent_moves[-1]]

    # **5️⃣ Bayesian Learning**
    move_counts = {move: recent_moves.count(move) for move in ["batu", "gunting", "kertas"]}
    total_moves = sum(move_counts.values())

    if total_moves > 0:
        probabilities = {move: move_counts[move] / total_moves for move in move_counts}
        predicted_user_move = max(probabilities, key=probabilities.get)
        bayesian_move = counter_moves[predicted_user_move]
    else:
        bayesian_move = random.choice(["batu", "gunting", "kertas"])

    # **6️⃣ Advanced Markov Chain**
    transition_counts = defaultdict(lambda: defaultdict(int))
    for i in range(len(recent_moves) - 1):
        transition_counts[recent_moves[i]][recent_moves[i + 1]] += 1

    last_move = recent_moves[-1]
    if last_move in transition_counts:
        markov_move = max(transition_counts[last_move], key=transition_counts[last_move].get, default=random.choice(["batu", "gunting", "kertas"]))
    else:
        markov_move = random.choice(["batu", "gunting", "kertas"])

    markov_counter = counter_moves[markov_move]

    # **7️⃣ AI Adaptive Learning Rate**
    ai_win_rate = sum(1 for r in recent_results if r == "Menang") / len(recent_results) * 100
    user_win_rate = sum(1 for r in recent_results if r == "Kalah") / len(recent_results) * 100

    if user_win_rate < 25:
        return random.choice(["batu", "gunting", "kertas"])

    if ai_win_rate < 50:
        return counter_moves[recent_moves[-1]]

    # **8️⃣ Weighted Confidence Voting**
    decision_weights = {
        bayesian_move: 0.35,
        markov_counter: 0.3,
        counter_moves[recent_moves[-1]]: 0.25,
        recent_moves[-1]: 0.05,
        random.choice(["batu", "gunting", "kertas"]): 0.05,
    }

    return max(decision_weights, key=decision_weights.get)

# **6️⃣ API Play Game**
@app.post("/play")
def play(request: Request, user_choice: str, current_user: str = Depends(get_current_user)):
    if user_choice not in ["batu", "gunting", "kertas"]:
        raise HTTPException(status_code=400, detail="Invalid choice")

    ai_choice = predict_next_move(current_user)
    result = "Seri" if user_choice == ai_choice else (
        "Menang" if (user_choice == "batu" and ai_choice == "gunting") or 
                    (user_choice == "gunting" and ai_choice == "kertas") or 
                    (user_choice == "kertas" and ai_choice == "batu")
        else "Kalah"
    )

    stats = read_json(STATS_DATA_FILE)

    # **Pastikan user memiliki statistik jika belum ada**
    if current_user not in stats:
        stats[current_user] = {
            "total_games": 0, "total_wins": 0, "total_losses": 0, "total_draws": 0,
            "score": 0, "win_streak": 0, "history": []
        }

    # **Update statistik berdasarkan hasil permainan**
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

    # **Tambahkan history permainan**
    stats[current_user]["history"].append({
        "user_move": user_choice,
        "ai_move": ai_choice,
        "result": result
    })

    write_json(STATS_DATA_FILE, stats)

    return JSONResponse(content={
        "user_choice": user_choice,
        "ai_choice": ai_choice,
        "result": result
    })

# **7️⃣ API Leaderboard dengan Win Rate (%)**
@app.get("/leaderboard")
def leaderboard():
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
def ai_statistics():
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
def reset_stats(current_user: str = Depends(get_current_user)):
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
