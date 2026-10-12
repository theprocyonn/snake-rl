"""
Bölüm 2 eğitimi: ajan ham tahtadan öğrenir.

Kullanım:
    python train.py                              # 600.000 adım (GPU'da ~yarım saat, CPU'da saatler)
    python train.py --adim 1000000               # daha uzun
    python train.py --klasor /yol/kayit          # çıktıların yazılacağı klasör
    python train.py --devam                      # kesilen eğitime ara kayıttan devam et
    python train.py --sekillendirme-yok          # "yeme yaklaşma" ödülü olmadan (çok daha zor)

Eğitim sırasında her 20.000 adımda ajan, rastgele hamle olmadan 20 oyun oynar.
En iyi çıkan model `model_en_iyi.pt` olarak saklanır. Bu ölçüm, eğitim skorlarından
daha güvenilirdir (eğitim skorları rastgele keşif adımlarıyla gürültülüdür).
"""
import argparse
import os
import time

import matplotlib
import numpy as np

from agent import DEVICE, EPS_MIN, EPS_STEPS, LEARN_EVERY, DQNAgent
from snake_env import SnakeEnv

EVAL_EVERY = 20_000
EVAL_GAMES = 20
PRINT_EVERY = 5_000


def epsilon(step):
    return max(EPS_MIN, 1.0 - (1.0 - EPS_MIN) * step / EPS_STEPS)


def evaluate_greedy(agent, env, games):
    scores = []
    for _ in range(games):
        obs = env.reset()
        done = False
        while not done:
            obs, _, done = env.step(agent.act(obs, 0.0))
        scores.append(env.score)
    return float(np.mean(scores))


def moving_avg(x, w):
    x = np.asarray(x, dtype=float)
    if len(x) < w:
        return np.array([])
    return np.convolve(x, np.ones(w) / w, mode="valid")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--adim", type=int, default=600_000, help="toplam eğitim adımı")
    p.add_argument("--klasor", default=".", help="model, ara kayıt ve grafiğin yazılacağı klasör")
    p.add_argument("--devam", action="store_true", help="ara kayıttan devam et")
    p.add_argument("--sekillendirme-yok", action="store_true", help="yeme yaklaşma/uzaklaşma ödülünü kapat")
    p.add_argument("--goster", action="store_true", help="bitince grafiği pencerede göster")
    args = p.parse_args()

    if not args.goster:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(args.klasor, exist_ok=True)
    model_path = os.path.join(args.klasor, "model_en_iyi.pt")
    last_path = os.path.join(args.klasor, "model_son.pt")
    ckpt_path = os.path.join(args.klasor, "ara_kayit.pt")
    plot_path = os.path.join(args.klasor, "egitim_egrisi.png")

    env = SnakeEnv(shaping=not args.sekillendirme_yok)
    eval_env = SnakeEnv()
    agent = DQNAgent()

    scores, evals, best_eval, step0 = [], [], -1.0, 0
    if args.devam and os.path.exists(ckpt_path):
        extra = agent.load_checkpoint(ckpt_path)
        scores, evals = extra["scores"], extra["evals"]
        best_eval, step0 = extra["best_eval"], extra["step"]
        print(f"Ara kayıttan devam ediliyor: adım {step0}")
    elif args.devam:
        print("Ara kayıt bulunamadı, baştan başlanıyor.")

    print(f"Cihaz: {DEVICE}   Ödül şekillendirme: {'kapalı' if args.sekillendirme_yok else 'açık'}")
    obs = env.reset()
    t0 = time.time()

    def save_ckpt(step):
        agent.save_checkpoint(ckpt_path, {"step": step, "scores": scores, "evals": evals, "best_eval": best_eval})

    step = step0
    try:
        for step in range(step0 + 1, args.adim + 1):
            eps = epsilon(step)
            a = agent.act(obs, eps)
            obs2, r, done = env.step(a)
            agent.remember(obs, a, r, obs2, done)
            obs = obs2
            if step % LEARN_EVERY == 0:
                agent.learn()
            if done:
                scores.append(env.score)
                obs = env.reset()

            if step % EVAL_EVERY == 0:
                m = evaluate_greedy(agent, eval_env, EVAL_GAMES)
                evals.append([step, m])
                star = ""
                if m > best_eval:
                    best_eval = m
                    agent.save_model(model_path)
                    star = "  (yeni en iyi, kaydedildi)"
                save_ckpt(step)
                print(f">>> adım {step:>8}  rastgelesiz ölçüm ({EVAL_GAMES} oyun): {m:5.1f}{star}", flush=True)
            elif step % PRINT_EVERY == 0:
                last = scores[-100:]
                avg = np.mean(last) if last else 0.0
                print(f"adım {step:>8}  eps {eps:.2f}  bölüm {len(scores):>6}  "
                      f"son100 eğitim ort {avg:5.1f}  süre {time.time() - t0:6.0f}s", flush=True)
    except KeyboardInterrupt:
        print("\nEğitim durduruldu (Ctrl+C).")

    save_ckpt(step)
    agent.save_model(last_path)
    if best_eval < 0:  # hiç ölçüm yapılmadıysa sonuncuyu kaydet
        agent.save_model(model_path)
    print(f"En iyi ölçüm: {best_eval:.1f}   Dosyalar: {model_path}, {last_path}")

    # Grafik: rastgelesiz ölçümler (adıma göre) ve eğitim skoru (bölüme göre)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    if evals:
        ax[0].plot([e[0] for e in evals], [e[1] for e in evals], color="#2f7d5b", linewidth=2, marker="o", markersize=3)
    ax[0].set_xlabel("Adım")
    ax[0].set_ylabel("Ortalama yem (rastgele hamle yok)")
    ax[0].set_title("Gerçek performans")
    ax[1].plot(scores, color="#9bb0a4", alpha=0.35, linewidth=0.8)
    ma = moving_avg(scores, 100)
    if len(ma):
        ax[1].plot(np.arange(99, 99 + len(ma)), ma, color="#2f7d5b", linewidth=2)
    ax[1].set_xlabel("Bölüm")
    ax[1].set_ylabel("Yenen yem")
    ax[1].set_title("Eğitim skoru (keşif gürültülü)")
    fig.tight_layout()
    fig.savefig(plot_path, dpi=130)
    print(f"Grafik kaydedildi: {plot_path}")
    if args.goster:
        plt.show()


if __name__ == "__main__":
    main()
