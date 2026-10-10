"""
Kayıtlı modelleri ekransız, rastgele hamle olmadan (epsilon = 0) oynatıp karşılaştırır.

Kullanım:
    python evaluate.py model_en_iyi.pt model_en_iyi_gamma95.pt
    python evaluate.py model_en_iyi.pt --oyun 200
"""
import argparse
import statistics

from agent import DQNAgent
from snake_env import SnakeEnv


def evaluate(path, games):
    agent = DQNAgent()
    agent.load(path)
    env = SnakeEnv()
    scores = []
    for _ in range(games):
        obs = env.reset()
        done = False
        while not done:
            obs, _, done = env.step(agent.act(obs, 0.0))
        scores.append(env.score)
    return scores


def main():
    p = argparse.ArgumentParser()
    p.add_argument("modeller", nargs="+", help="karşılaştırılacak .pt dosyaları")
    p.add_argument("--oyun", type=int, default=100, help="her model için oynanacak oyun sayısı")
    args = p.parse_args()

    print(f"Her model {args.oyun} oyun oynuyor (rastgele hamle yok)...\n")
    for path in args.modeller:
        s = evaluate(path, args.oyun)
        print(f"{path}")
        print(f"   ortalama {statistics.mean(s):5.1f}   ortanca {statistics.median(s):5.1f}   "
              f"en düşük {min(s):3d}   en yüksek {max(s):3d}\n", flush=True)


if __name__ == "__main__":
    main()
