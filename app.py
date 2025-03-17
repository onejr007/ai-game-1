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

    if user_id not in stats or "history" not in stats[user_id] or len(stats[user_id]["history"]) < 5:
        return random.choice(["batu", "gunting", "kertas"])
    
    history = stats[user_id]["history"]
    recent_moves = [h["user_move"] for h in history[-150:]]
    recent_results = [h["result"] for h in history[-150:]]
    timestamps = [h.get("timestamp", 0) for h in history[-150:]]  # ✅ FIXED


    counter_moves = {"batu": "kertas", "gunting": "batu", "kertas": "gunting"}


    # ✅ 1️⃣ Pattern Recognition
    def detect_pattern(moves, max_length=5):
        for length in range(2, max_length + 1):
            if len(moves) >= length * 2 and moves[-length:] == moves[-(length * 2):-length]:
                return counter_moves[moves[-1]]
        return None
    pattern_move = detect_pattern(recent_moves)

    # ✅ 2️⃣ Bayesian Learning
    move_counts = {move: recent_moves.count(move) for move in ["batu", "gunting", "kertas"]}
    bayesian_move = counter_moves[max(move_counts, key=move_counts.get)]

    # ✅ 3️⃣ Markov Chain Learning
    if len(recent_moves) >= 3:
        transition_counts = defaultdict(lambda: defaultdict(int))
        for i in range(len(recent_moves) - 1):
            transition_counts[recent_moves[i]][recent_moves[i + 1]] += 1
        last_move = recent_moves[-1]
        markov_move = counter_moves[max(transition_counts[last_move], key=transition_counts[last_move].get, default=random.choice(["batu", "gunting", "kertas"]))]
    else:
        markov_move = None

    # ✅ 4️⃣ Monte Carlo + Dynamic Simulation
    def monte_carlo_simulation():
        move_scores = {move: 0 for move in ["batu", "gunting", "kertas"]}
        for _ in range(5000):
            ai_move = random.choice(["batu", "gunting", "kertas"])
            if ai_move == counter_moves[bayesian_move]:
                move_scores[ai_move] += 1
        return max(move_scores, key=move_scores.get)
    monte_carlo_move = monte_carlo_simulation()

    # ✅ 5️⃣ Time-Based Pattern Recognition
    def filter_recent_moves(history, time_window=300):
        current_time = time.time()
        return [move for i, move in enumerate(history) if timestamps[i] >= current_time - time_window]
    time_based_move = detect_pattern(filter_recent_moves(recent_moves))

    # ✅ 6️⃣ Game-Theoretic Equilibrium
    equilibrium_move = counter_moves[random.choice(["batu", "gunting", "kertas"])]

    # ✅ 7️⃣ Bayesian Neural Estimation
    def bayesian_neural_prediction():
        transition_matrix = np.zeros((3, 3))
        move_map = {"batu": 0, "gunting": 1, "kertas": 2}

        for i in range(len(recent_moves) - 1):
            prev_move = move_map[recent_moves[i]]
            next_move = move_map[recent_moves[i + 1]]
            transition_matrix[prev_move][next_move] += 1

        probabilities = transition_matrix / (np.sum(transition_matrix, axis=1, keepdims=True) + 1e-9)
        predicted_move = np.argmax(probabilities[move_map[recent_moves[-1]]])

        return counter_moves[list(move_map.keys())[predicted_move]]
    neural_move = bayesian_neural_prediction()

    # ✅ 8️⃣ Multi-Tier Voting System
    def multi_tier_voting():
        short_term = recent_moves[-5:] if len(recent_moves) >= 5 else recent_moves
        mid_term = recent_moves[-20:] if len(recent_moves) >= 20 else recent_moves
        long_term = recent_moves

        tier_1 = counter_moves[max(set(short_term), key=short_term.count)]
        tier_2 = counter_moves[max(set(mid_term), key=mid_term.count)]
        tier_3 = counter_moves[max(set(long_term), key=long_term.count)]

        return random.choice([tier_1, tier_2, tier_3])
    tiered_move = multi_tier_voting()

    # ✅ 9️⃣ Reverse Exploit Strategy
    if len(recent_moves) >= 2 and recent_moves[-2] == counter_moves[recent_moves[-1]]:
        reverse_exploit = counter_moves[recent_moves[-1]]
    else:
        reverse_exploit = None

    # ✅ 🔟 Anti-Mirror Strategy
    if len(recent_moves) >= 2 and recent_moves[-2] == counter_moves[recent_moves[-1]]:
        anti_mirror = counter_moves[recent_moves[-1]]
    else:
        anti_mirror = None

    # ✅ 1️⃣1️⃣ Streak-Based Prediction
    if len(recent_results) >= 5:
        win_streak = recent_results[-5:].count("Menang")
        lose_streak = recent_results[-5:].count("Kalah")

        if win_streak >= 4:
            streak_prediction = counter_moves[recent_moves[-1]]
        elif lose_streak >= 4:
            streak_prediction = random.choice(["batu", "gunting", "kertas"])
        else:
            streak_prediction = None
    else:
        streak_prediction = None

    # ✅ 1️⃣2️⃣ Look-Ahead Simulation
    def look_ahead_simulation():
        simulated_moves = [counter_moves[recent_moves[-1]]]
        for _ in range(2):
            simulated_moves.append(counter_moves[simulated_moves[-1]])
        return simulated_moves[-1]
    look_ahead = look_ahead_simulation()

    # ✅ 1️⃣3️⃣ Psychological Counterplay
    if len(recent_results) >= 6 and recent_results[-6:].count("Menang") >= 4:
        psychological_counter = counter_moves[recent_moves[-1]]
    else:
        psychological_counter = None

    # ✅ 1️⃣4️⃣ Hybrid AI Switching
    if len(recent_results) >= 10:
        win_rate = recent_results[-10:].count("Menang") / 10
        if win_rate < 0.4:
            hybrid_switch = counter_moves[recent_moves[-1]]
        else:
            hybrid_switch = random.choice(["batu", "gunting", "kertas"])
    else:
        hybrid_switch = None

    # ✅ 1️⃣5️⃣ Opponent Conditioning
    if len(recent_results) >= 6 and recent_results[-6:].count("Kalah") >= 4:
        conditioning = counter_moves[recent_moves[-1]]
    else:
        conditioning = None

    # ✅ 1️⃣6️⃣ Dynamic Response Adjustment
    if len(recent_results) >= 8:
        recent_win_rate = recent_results[-8:].count("Menang") / 8
        if recent_win_rate < 0.5:
            dynamic_response = counter_moves[recent_moves[-1]]
        else:
            dynamic_response = random.choice(["batu", "gunting", "kertas"])
    else:
        dynamic_response = None

    # ✅ 1️⃣7️⃣ Entropy-Based Decision Making
    if len(recent_moves) >= 5:
        move_distribution = {move: recent_moves[-5:].count(move) for move in ["batu", "gunting", "kertas"]}
        most_common_move = max(move_distribution, key=move_distribution.get)
        entropy_move = counter_moves[most_common_move]
    else:
        entropy_move = None

    # ✅ 1️⃣8️⃣ Gradient Learning Strategy
    if len(recent_moves) >= 6:
        recent_trend = [move for move in recent_moves[-6:]]
        if recent_trend == ["batu", "gunting", "kertas", "batu", "gunting", "kertas"]:
            gradient_learning = counter_moves[recent_moves[-1]]
        else:
            gradient_learning = None
    else:
        gradient_learning = None

    # ✅ 1️⃣9️⃣ Reinforcement Learning Adaptation
    if len(recent_results) >= 10:
        ai_win_rate = recent_results[-10:].count("Menang") / 10
        if ai_win_rate < 0.4:
            reinforcement_learning = counter_moves[recent_moves[-1]]
        else:
            reinforcement_learning = random.choice(["batu", "gunting", "kertas"])
    else:
        reinforcement_learning = None

    # ✅ 2️⃣0️⃣ Exploration vs. Exploitation
    if len(recent_results) >= 10:
        ai_win_rate = recent_results[-10:].count("Menang") / 10
        if ai_win_rate < 0.5:
            exploration_exploitation = counter_moves[recent_moves[-1]]
        else:
            exploration_exploitation = random.choice(["batu", "gunting", "kertas"])
    else:
        exploration_exploitation = None

    # ✅ 2️⃣1️⃣ Neural Network-Based Decision Making
    if trained_model:
        input_data = np.array([[recent_moves.count("batu"), recent_moves.count("gunting"), recent_moves.count("kertas")]])
        predicted_move = trained_model.predict(input_data).item()
        neural_network_move = counter_moves[predicted_move]
    else:
        neural_network_move = None

    # ✅ 2️⃣2️⃣ Recursive Bayesian Updating
    if len(recent_moves) >= 5:
        move_probs = {move: (recent_moves[-5:].count(move) + 1) / 6 for move in ["batu", "gunting", "kertas"]}
        predicted_move = max(move_probs, key=move_probs.get)
        recursive_bayesian = counter_moves[predicted_move]
    else:
        recursive_bayesian = None

    # ✅ 2️⃣3️⃣ Look-Back Analysis
    if len(recent_moves) >= 10:
        most_frequent_move = max(set(recent_moves[-10:]), key=recent_moves[-10:].count)
        look_back_analysis = counter_moves[most_frequent_move]
    else:
        look_back_analysis = None

    # ✅ 2️⃣4️⃣ Delayed Counter Strategy
    if len(recent_moves) >= 4:
        delayed_counter = counter_moves[recent_moves[-4]]
    else:
        delayed_counter = None

    # ✅ 2️⃣5️⃣ Move Distribution Analysis
    if len(recent_moves) >= 10:
        move_distribution = {move: recent_moves.count(move) / len(recent_moves) for move in ["batu", "gunting", "kertas"]}
        predicted_move = max(move_distribution, key=move_distribution.get)
        move_distribution_analysis = counter_moves[predicted_move]
    else:
        move_distribution_analysis = None

    # ✅ 2️⃣6️⃣ Weighted Random Selection
    if len(recent_moves) >= 5:
        move_weights = {move: recent_moves[-5:].count(move) + 1 for move in ["batu", "gunting", "kertas"]}
        weighted_choices = random.choices(list(move_weights.keys()), weights=move_weights.values(), k=1)
        weighted_random_selection = counter_moves[weighted_choices[0]]
    else:
        weighted_random_selection = None

    # ✅ 2️⃣7️⃣ Meta-Learning Strategy
    if len(recent_results) >= 15:
        win_rate = recent_results[-15:].count("Menang") / 15
        if win_rate < 0.4:
            meta_learning = counter_moves[recent_moves[-1]]
        else:
            meta_learning = random.choice(["batu", "gunting", "kertas"])
    else:
        meta_learning = None

    # ✅ 2️⃣8️⃣ Adaptive Randomization
    if random.random() < 0.2:
        adaptive_randomization = random.choice(["batu", "gunting", "kertas"])
    else:
        adaptive_randomization = None

    # ✅ 2️⃣9️⃣ Probability Curve Prediction
    if len(recent_moves) >= 10:
        move_probabilities = {move: (recent_moves.count(move) + 1) / (len(recent_moves) + 3) for move in ["batu", "gunting", "kertas"]}
        predicted_move = max(move_probabilities, key=move_probabilities.get)
        probability_curve_prediction = counter_moves[predicted_move]
    else:
        probability_curve_prediction = None

    # ✅ 3️⃣0️⃣ AI Self-Optimization
    if len(recent_results) >= 20:
        ai_win_rate = recent_results[-20:].count("Menang") / 20
        if ai_win_rate < 0.5:
            ai_self_optimization = counter_moves[recent_moves[-1]]
        else:
            ai_self_optimization = random.choice(["batu", "gunting", "kertas"])
    else:
        ai_self_optimization = None

    # 🔥 Mengumpulkan semua strategi yang memiliki nilai
    active_strategies = {
        "pattern_move": pattern_move,
        "bayesian_move": bayesian_move,
        "markov_move": markov_move,
        "monte_carlo_move": monte_carlo_move,
        "time_based_move": time_based_move,
        "equilibrium_move": equilibrium_move,
        "neural_move": neural_network_move,
        "tiered_move": tiered_move,
        "reverse_exploit": reverse_exploit,
        "anti_mirror": anti_mirror,
        "streak_prediction": streak_prediction,
        "look_ahead": look_ahead,
        "psychological_counter": psychological_counter,
        "hybrid_switch": hybrid_switch,
        "conditioning": conditioning,
        "dynamic_response": dynamic_response,
        "entropy_move": entropy_move,
        "gradient_learning": gradient_learning,
        "reinforcement_learning": reinforcement_learning,
        "exploration_exploitation": exploration_exploitation,
        "recursive_bayesian": recursive_bayesian,
        "look_back_analysis": look_back_analysis,
        "delayed_counter": delayed_counter,
        "move_distribution_analysis": move_distribution_analysis,
        "weighted_random_selection": weighted_random_selection,
        "meta_learning": meta_learning,
        "adaptive_randomization": adaptive_randomization,
        "probability_curve_prediction": probability_curve_prediction,
        "ai_self_optimization": ai_self_optimization
    }

    # 🔥 Menghapus strategi yang bernilai None
    active_strategies = {key: value for key, value in active_strategies.items() if value is not None}

    # 🔥 Jika ada strategi yang aktif, gunakan voting berbobot
    if active_strategies:
        strategy_weights = {key: 1 for key in active_strategies}  # Memberikan bobot awal yang sama
        chosen_strategy = max(active_strategies, key=lambda k: strategy_weights[k])
        return active_strategies[chosen_strategy]

    # 🔥 Jika tidak ada strategi yang aktif, AI memilih secara acak
    return random.choice(["batu", "gunting", "kertas"])


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
