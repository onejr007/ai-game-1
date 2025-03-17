import random
import bcrypt
import datetime
import json
import jwt
import time
import threading
import os
import numpy as np
from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.security import OAuth2PasswordBearer
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from collections import defaultdict
from filelock import FileLock
from apscheduler.schedulers.background import BackgroundScheduler
from sklearn.ensemble import RandomForestClassifier


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Ganti dengan domain yang diizinkan, contoh: ["http://localhost:3000"]
    allow_credentials=True,
    allow_methods=["*"],  # Izinkan semua metode (POST, GET, OPTIONS, dll.)
    allow_headers=["*"],
)

# 🔐 Konfigurasi Keamanan
SECRET_KEY = "supersecretkey"
TOKEN_EXPIRY = 1800  # 30 menit
blacklisted_tokens = set()

# 📂 File JSON
USER_DATA_FILE = "datalogin.json"
STATS_DATA_FILE = "datastatistik.json"

# Mount folder static untuk frontend
app.mount("/static", StaticFiles(directory="static"), name="static")

# 📥 Fungsi Baca & Tulis JSON dengan File Lock
def read_json(file, default_data=None):
    lock = FileLock(file + ".lock")
    with lock:
        if not os.path.exists(file):
            write_json(file, default_data if default_data is not None else {})
        try:
            with open(file, "r", encoding="utf-8") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            write_json(file, default_data if default_data is not None else {})
            return default_data if default_data is not None else {}

def write_json(file, data):
    lock = FileLock(file + ".lock")
    with lock:
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
def predict_next_move(user_id):
    stats = read_json(STATS_DATA_FILE)

    if user_id not in stats or "history" not in stats[user_id] or len(stats[user_id]["history"]) < 10:
        return random.choice(["batu", "gunting", "kertas"])

    history = stats[user_id]["history"]
    recent_moves = [h["user_move"] for h in history[-150:]]
    recent_results = [h["result"] for h in history[-150:]]

    move_map = {"batu": 0, "gunting": 1, "kertas": 2}
    counter_moves = {"batu": "kertas", "gunting": "batu", "kertas": "gunting"}

    # ✅ 1️⃣ Push-Hit Exploit Shield
    if len(recent_results) >= 10 and all(r == "Menang" for r in recent_results[-10:]):
        return counter_moves[recent_moves[-1]]

    # ✅ 2️⃣ Reverse Adaptive Momentum
    if len(recent_results) >= 4 and recent_results[-3:] == ["Kalah", "Seri", "Seri"]:
        return counter_moves[recent_moves[-1]]

    # ✅ 3️⃣ Dynamic Push-Hit Switch
    if len(recent_results) >= 6:
        last_six = recent_results[-6:]
        if last_six.count("Menang") >= 4 and last_six.count("Seri") <= 2:
            return counter_moves[recent_moves[-1]]

    # ✅ 4️⃣ Recursive Cycle Detection
    def detect_cycle(moves, cycle_length):
        if len(moves) >= cycle_length * 2:
            return moves[-cycle_length:] == moves[-(cycle_length * 2):-cycle_length]
        return False

    cycle_lengths = [3, 5, 7, 10, 15, 20]
    for length in cycle_lengths:
        if detect_cycle(recent_moves, length):
            return counter_moves[recent_moves[-1]]

    # ✅ 5️⃣ Bayesian Learning
    move_counts = {move: recent_moves.count(move) for move in ["batu", "gunting", "kertas"]}
    predicted_user_move = max(move_counts, key=move_counts.get, default="batu")
    bayesian_move = counter_moves[predicted_user_move]

    # ✅ 6️⃣ Markov Chain Learning
    transition_counts = defaultdict(lambda: defaultdict(int))
    for i in range(len(recent_moves) - 1):
        transition_counts[recent_moves[i]][recent_moves[i + 1]] += 1

    last_move = recent_moves[-1]
    markov_move = max(transition_counts[last_move], key=transition_counts[last_move].get, default=random.choice(["batu", "gunting", "kertas"]))

    # ✅ 7️⃣ AI Adaptive Learning Rate
    ai_win_rate = sum(1 for r in recent_results if r == "Menang") / len(recent_results) * 100
    if ai_win_rate < 50:
        return counter_moves[recent_moves[-1]]

    # ✅ 8️⃣ Weighted Confidence Voting
    decision_weights = {
        bayesian_move: 0.3,
        markov_move: 0.25,
        counter_moves[recent_moves[-1]]: 0.2,
        random.choice(["batu", "gunting", "kertas"]): 0.05
    }

    # ✅ 9️⃣ Neural Network Prediction
    if trained_model:
        neural_move = trained_model.predict(np.array([[random.choice(["batu", "gunting", "kertas"])]])).item()
        decision_weights[neural_move] = 0.3

    # ✅ 🔟 Monte Carlo Simulation
    def monte_carlo_simulation():
        move_scores = {move: 0 for move in ["batu", "gunting", "kertas"]}
        for _ in range(5000):
            ai_move = random.choice(["batu", "gunting", "kertas"])
            if ai_move == counter_moves[get_global_move_distribution()]:
                move_scores[ai_move] += 1
        return max(move_scores, key=move_scores.get)

    monte_carlo_move = monte_carlo_simulation()
    decision_weights[monte_carlo_move] = 0.4

    # ✅ 1️⃣1️⃣ Fuzzy Logic Adaptation
    def fuzzy_logic_adjustment(ai_move):
        win_count = recent_results.count("Menang")
        lose_count = recent_results.count("Kalah")
        return random.choice(["batu", "gunting", "kertas"]) if win_count > 7 else counter_moves[ai_move] if lose_count > 5 else ai_move

    fuzzy_move = fuzzy_logic_adjustment(monte_carlo_move)
    decision_weights[fuzzy_move] = 0.3

    # ✅ 1️⃣2️⃣ Global Move Learning
    global_move = get_global_move_distribution()

    # ✅ 1️⃣3️⃣ Entropy-Based Decision Making
    entropy_move = counter_moves[recent_moves[-1]]

    # ✅ 1️⃣4️⃣ Multi-Agent System
    def simulate_multiple_agents():
        agent_moves = [random.choice(["batu", "gunting", "kertas"]) for _ in range(3)]
        return counter_moves[max(set(agent_moves), key=agent_moves.count)]

    multi_agent_move = simulate_multiple_agents()
    decision_weights[multi_agent_move] = 0.2

    # **Pemilihan strategi akhir berdasarkan voting**
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

# 🤖 Training AI Neural Network
def train_ai_model():
    stats = read_json(STATS_DATA_FILE)
    X, y = [], []

    for user, data in stats.items():
        history = data.get("history", [])
        for i in range(len(history) - 1):
            # **Cek apakah 'user_move' ada dalam history**
            if "user_move" in history[i] and "user_move" in history[i + 1]:
                X.append([history[i]["user_move"]])
                y.append(history[i + 1]["user_move"])

    if len(X) > 5:
        model = RandomForestClassifier(n_estimators=10)
        model.fit(np.array(X).reshape(-1, 1), np.array(y))
        return model
    return None

trained_model = train_ai_model()

# 🔑 Logout & Token Blacklist
@app.post("/logout")
def logout(request: Request):
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
