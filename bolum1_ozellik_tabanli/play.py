"""
Oyun penceresi: kendin oyna ya da eğitilmiş yapay zekayı izle.

Kullanım:
    python play.py                          # model varsa yapay zeka modunda açılır
    python play.py --model model_son.pt     # başka bir model dosyası

Tuşlar:
    1            insan modu            2            yapay zeka modu
    Ok tuşları / WASD    yönlendir (insan modu)
    Yukarı / Aşağı       hızı değiştir (yapay zeka modu)
    Boşluk               oyun bittikten sonra yeniden başla (insan modu)
    Esc                  çık
"""
import argparse
import os

import pygame

from render import Renderer
from snake_env import SnakeEnv

HUMAN_TICK = 0.16  # insan modunda adım süresi (saniye)

KEYMAP = {
    pygame.K_UP: 0, pygame.K_RIGHT: 1, pygame.K_DOWN: 2, pygame.K_LEFT: 3,
    pygame.K_w: 0, pygame.K_d: 1, pygame.K_s: 2, pygame.K_a: 3,
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="model_en_iyi.pt")
    args = p.parse_args()

    env = SnakeEnv()
    view = Renderer(env, "Yılan")

    agent = None
    if os.path.exists(args.model):
        from agent import DQNAgent
        agent = DQNAgent()
        agent.load(args.model)

    mode = "ai" if agent else "insan"
    note = "" if agent else f"'{args.model}' bulunamadı: önce train.py ile eğit."
    best = {"insan": 0, "ai": 0}
    games = {"insan": 0, "ai": 0}

    obs = env.reset()
    human_state = "ready"   # ready | playing | over
    queue = []
    ai_sps = 12.0           # yapay zeka modunda saniyedeki adım
    acc = 0.0
    pause = 0.0
    clock = pygame.time.Clock()

    def switch(m):
        nonlocal mode, obs, human_state, queue, acc, pause, note
        mode = m
        obs = env.reset()
        human_state, queue, acc, pause, note = "ready", [], 0.0, 0.0, ""

    running = True
    while running:
        dt = clock.tick(60) / 1000.0

        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
            elif e.type == pygame.KEYDOWN:
                if e.key == pygame.K_ESCAPE:
                    running = False
                elif e.key == pygame.K_1:
                    switch("insan")
                elif e.key == pygame.K_2:
                    if agent:
                        switch("ai")
                    else:
                        note = "Eğitilmiş model yok: önce train.py çalıştır."
                elif mode == "insan":
                    if e.key in KEYMAP and human_state != "over":
                        if human_state == "ready":
                            human_state, acc = "playing", 0.0
                        d = KEYMAP[e.key]
                        last = queue[-1] if queue else env.dir
                        if d != last and d != (last + 2) % 4 and len(queue) < 2:
                            queue.append(d)
                    elif e.key in (pygame.K_SPACE, pygame.K_RETURN) and human_state == "over":
                        obs = env.reset()
                        human_state, queue = "ready", []
                else:
                    if e.key == pygame.K_UP:
                        ai_sps = min(300.0, ai_sps * 1.5)
                    elif e.key == pygame.K_DOWN:
                        ai_sps = max(2.0, ai_sps / 1.5)

        # --- oyun mantığı ---
        if mode == "insan":
            if human_state == "playing":
                acc += dt
                while acc >= HUMAN_TICK and not env.over:
                    acc -= HUMAN_TICK
                    d = queue.pop(0) if queue else env.dir
                    env.step_abs(d)
                if env.over:
                    human_state = "over"
                    games["insan"] += 1
                    best["insan"] = max(best["insan"], env.score)
        else:
            if env.over:
                pause -= dt
                if pause <= 0:
                    obs = env.reset()
            else:
                acc += dt
                interval = 1.0 / ai_sps
                n = 0
                while acc >= interval and not env.over and n < 60:
                    acc -= interval
                    n += 1
                    obs, _, done = env.step(agent.act(obs, 0.0))
                if env.over:
                    games["ai"] += 1
                    best["ai"] = max(best["ai"], env.score)
                    pause = 0.8
                    acc = 0.0

        # --- çizim ---
        name = "İnsan" if mode == "insan" else "Yapay zeka"
        line1 = f"{name}   Skor {env.score}   En yüksek {max(best[mode], env.score)}   Oyun {games[mode]}"
        if note:
            line2 = note
        elif mode == "insan":
            line2 = "1: insan   2: yapay zeka   Ok tuşları / WASD: yönlendir   Esc: çık"
        else:
            line2 = f"1: insan   2: yapay zeka   Yukarı/Aşağı: hız ({ai_sps:.0f} adım/sn)   Esc: çık"

        banner = banner2 = None
        if mode == "insan":
            if human_state == "ready":
                banner, banner2 = "Hazır", "Başlamak için bir ok tuşuna bas"
            elif human_state == "over":
                banner, banner2 = "Oyun bitti", "Boşluk: yeniden başla"
        view.draw(env, line1, line2, banner or "", banner2 or "")

    view.close()


if __name__ == "__main__":
    main()
