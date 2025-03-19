import random
import numpy as np
from collections import defaultdict

counter_moves = {"batu": "kertas", "gunting": "batu", "kertas": "gunting"}

# 🔹 1️⃣ Pattern Recognition
def detect_pattern(moves):
    if len(moves) < 3:
        return random.choice(["batu", "gunting", "kertas"])
    last_move = moves[-1]
    return counter_moves[last_move]

# 🔹 2️⃣ Monte Carlo Rollout
def monte_carlo_rollout(moves, simulations=5000):
    move_scores = defaultdict(int)
    for _ in range(simulations):
        move = random.choice(["batu", "gunting", "kertas"])
        if move == counter_moves[moves[-1]]:
            move_scores[move] += 1
    return max(move_scores, key=move_scores.get)

# 🔹 3️⃣ Markov Chain Probability
def markov_chain_prediction(moves):
    if len(moves) < 3:
        return random.choice(["batu", "gunting", "kertas"])
    transition_counts = defaultdict(lambda: defaultdict(int))
    for i in range(len(moves) - 1):
        transition_counts[moves[i]][moves[i + 1]] += 1
    last_move = moves[-1]
    next_move = max(transition_counts[last_move], key=transition_counts[last_move].get, default=random.choice(["batu", "gunting", "kertas"]))
    return counter_moves[next_move]

# 🔹 4️⃣ Bayesian Probability Estimation
def bayesian_estimation(moves):
    move_counts = {move: moves.count(move) for move in ["batu", "gunting", "kertas"]}
    predicted_move = max(move_counts, key=move_counts.get)
    return counter_moves[predicted_move]

# 🔹 5️⃣ Opponent Exploit Learning
def opponent_exploit_learning(moves):
    if len(moves) < 3:
        return random.choice(["batu", "gunting", "kertas"])
    last_two = moves[-2:]
    if last_two[0] == last_two[1]:  # Jika lawan berulang kali memilih langkah yang sama
        return counter_moves[last_two[1]]
    return random.choice(["batu", "gunting", "kertas"])

# 🔹 6️⃣ Cycle Breaker Strategy
def cycle_breaker(moves):
    if len(moves) < 4:
        return random.choice(["batu", "gunting", "kertas"])
    if moves[-4:] == moves[-8:-4]:  # Jika ada pola berulang dalam 4 langkah terakhir
        return counter_moves[moves[-1]]
    return random.choice(["batu", "gunting", "kertas"])

# 🔹 7️⃣ Nash Equilibrium Strategy
def nash_equilibrium():
    return random.choice(["batu", "gunting", "kertas"])

# 🔹 8️⃣ Psychological Trap Strategy
def psychological_trap(moves):
    if len(moves) < 5:
        return random.choice(["batu", "gunting", "kertas"])
    if moves[-1] == moves[-3]:  # Jika pemain kembali ke pola sebelumnya
        return counter_moves[moves[-1]]
    return random.choice(["batu", "gunting", "kertas"])

# 🔹 9️⃣ Meta-Learning Strategy
def meta_learning(moves, results):
    if len(moves) < 6:
        return random.choice(["batu", "gunting", "kertas"])
    win_rate = results[-6:].count("Menang") / 6
    if win_rate < 0.4:
        return counter_moves[moves[-1]]
    return random.choice(["batu", "gunting", "kertas"])

# 🔹 🔟 Reverse Psychology Move
def reverse_psychology(moves):
    if len(moves) < 2:
        return random.choice(["batu", "gunting", "kertas"])
    return counter_moves[counter_moves[moves[-1]]]

# 📌 Dictionary untuk memanggil strategi
STRATEGY_FUNCTIONS = {
    "detect_pattern": detect_pattern,
    "monte_carlo_rollout": monte_carlo_rollout,
    "markov_chain_prediction": markov_chain_prediction,
    "bayesian_estimation": bayesian_estimation,
    "opponent_exploit_learning": opponent_exploit_learning,
    "cycle_breaker": cycle_breaker,
    "nash_equilibrium": nash_equilibrium,
    "psychological_trap": psychological_trap,
    "meta_learning": meta_learning,
    "reverse_psychology": reverse_psychology
}
