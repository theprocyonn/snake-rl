"""
DQN eğitimi.

Kullanım:
    python train.py                     # 150.000 adım eğit (birkaç dakika), eğriyi göster
    python train.py --adim 300000       # daha uzun eğit
    python train.py --canli 50          # her 50 adımda bir pencereyi çizerek izle (yavaşlatır)

Eğitimi Ctrl+C ile istediğiniz an durdurabilirsiniz; o ana kadarki model kaydedilir.
"""
import argparse
import time

import numpy as np

from agent import DQNAgent, EPS_MIN, EPS_STEPS
from snake_env import SnakeEnv


def epsilon(step):
    """Başta 1.0 (tamamen rastgele), EPS_STEPS adımda EPS_MIN'e doğrusal düşer."""
    return max(EPS_MIN, 1.0 - (1.0 - EPS_MIN) * step / EPS_STEPS)


def moving_avg(x, w):
    x = np.asarray(x, dtype=float)
    if len(x) < w:
        return np.array([])
    return np.convolve(x, np.ones(w) / w, mode="valid")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--adim", type=int, default=150_000, help="toplam eğitim adımı")
    p.add_argument("--canli", type=int, default=0, help="N>0 ise her N adımda bir pencereyi çiz")
    p.add_argument("--cikis", default="model_en_iyi.pt", help="en iyi modelin kaydedileceği dosya")
    args = p.parse_args()

    env = SnakeEnv()
    agent = DQNAgent()
    obs = env.reset()

    view = None
    if args.canli > 0:
        from render import Renderer
        view = Renderer(env, "Eğitim")

    scores = []
    best_avg = -1.0
    t0 = time.time()
    step = 0

    try:
        for step in range(1, args.adim + 1):
            eps = epsilon(step)
            a = agent.act(obs, eps)
            obs2, r, done = env.step(a)
            agent.remember(obs, a, r, obs2, done)
            obs = obs2
            agent.learn()

            if view is not None and step % args.canli == 0:
                if not view.pump():
                    print("Pencere kapatıldı, eğitim durduruluyor.")
                    break
                last = np.mean(scores[-50:]) if scores else 0.0
                view.draw(env,
                          f"Adım {step}   Bölüm {len(scores)}   Skor {env.score}",
                          f"epsilon {eps:.2f}   son 50 bölüm ortalaması {last:.1f}")

            if done:
                scores.append(env.score)
                obs = env.reset()
                # Keşif bittikten sonra, son 100 bölümün ortalaması rekor kırdıysa kaydet
                if step > EPS_STEPS and len(scores) >= 100:
                    avg = float(np.mean(scores[-100:]))
                    if avg > best_avg:
                        best_avg = avg
                        agent.save(args.cikis)

            if step % 5000 == 0:
                last = scores[-100:]
                avg = np.mean(last) if last else 0.0
                print(f"adım {step:>7}  eps {eps:.2f}  bölüm {len(scores):>5}  "
                      f"son100 ort {avg:5.1f}  en yüksek {max(last) if last else 0:>3}  "
                      f"süre {time.time() - t0:5.0f}s", flush=True)
    except KeyboardInterrupt:
        print("\nEğitim durduruldu (Ctrl+C).")

    if view is not None:
        view.close()

    agent.save("model_son.pt")
    if best_avg < 0:  # hiç rekor kaydedilmediyse (kısa eğitim) sonuncuyu kullan
        agent.save(args.cikis)
        print(f"Model kaydedildi: {args.cikis}")
    else:
        print(f"En iyi model kaydedildi: {args.cikis} (son 100 bölüm ort. {best_avg:.1f})")
    print("Son model: model_son.pt")

    # Öğrenme eğrisi
    import matplotlib.pyplot as plt
    plt.figure(figsize=(9, 4.5))
    plt.plot(scores, color="#9bb0a4", alpha=0.35, linewidth=0.8, label="bölüm skoru")
    ma = moving_avg(scores, 100)
    if len(ma):
        plt.plot(np.arange(99, 99 + len(ma)), ma, color="#2f7d5b", linewidth=2.2, label="son 100 bölüm ortalaması")
    plt.xlabel("Bölüm")
    plt.ylabel("Yenen yem sayısı")
    plt.title("DQN öğrenme eğrisi")
    plt.legend()
    plt.tight_layout()
    plt.savefig("egitim_egrisi.png", dpi=130)
    print("Grafik kaydedildi: egitim_egrisi.png (pencereyi kapatınca program biter)")
    plt.show()


if __name__ == "__main__":
    main()
