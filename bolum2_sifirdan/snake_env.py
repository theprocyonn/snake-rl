"""
Yılan oyunu motoru, Bölüm 2: ajan HAM TAHTAYI görür.

Bölüm 1'den farkı: elle tasarlanmış özellik yok (uzaklık, boş alan, kuyruğa yol...).
Ajan tahtayı 3 katmanlı bir "görüntü" olarak alır (15x15 hücre):
    katman 0: yılanın başı        (1 olan hücre)
    katman 1: yılanın gövdesi     (1 olan hücreler)
    katman 2: yem                 (1 olan hücre)
Ne yapılacağını, hangi hamlenin tehlikeli olduğunu, yemin nerede olduğunu
bu görüntüden kendisi çıkarmak zorundadır.

Hamleler mutlak yöndür: 0 yukarı, 1 sağ, 2 aşağı, 3 sol.
Geri dönüş (180 derece) yok sayılır, yılan düz devam eder.
"""
import random
from collections import deque

import numpy as np

GRID = 15
DIRS = [(0, -1), (1, 0), (0, 1), (-1, 0)]  # yukarı, sağ, aşağı, sol
N_ACTIONS = 4
CHANNELS = 3

R_EAT = 1.0
R_DIE = -1.0
R_CLOSER = 0.02    # sadece ödül şekillendirme açıkken
R_FARTHER = -0.025


class SnakeEnv:
    def __init__(self, grid=GRID, seed=None, shaping=True):
        self.n = grid
        self.shaping = shaping
        self.rng = random.Random(seed)
        self.reset()

    def reset(self):
        m = self.n // 2
        self.snake = deque([(m, m), (m - 1, m), (m - 2, m)])  # ilk eleman = baş
        self.body = set(self.snake)
        self.dir = 1
        self.score = 0
        self.since_food = 0
        self.over = False
        self._place_food()
        return self.obs()

    def _place_food(self):
        free = [(x, y) for x in range(self.n) for y in range(self.n) if (x, y) not in self.body]
        if not free:
            self.food = None
            return False
        self.food = self.rng.choice(free)
        return True

    def blocked(self, x, y):
        return x < 0 or y < 0 or x >= self.n or y >= self.n or (x, y) in self.body

    def obs(self):
        """Ham tahta: (3, n, n) boyutunda 0/1 değerli uint8 dizisi."""
        o = np.zeros((CHANNELS, self.n, self.n), np.uint8)
        for i, (x, y) in enumerate(self.snake):
            o[0 if i == 0 else 1, y, x] = 1
        if self.food:
            o[2, self.food[1], self.food[0]] = 1
        return o

    def step(self, action):
        """action: 0 yukarı, 1 sağ, 2 aşağı, 3 sol. Döner: (gözlem, ödül, bitti_mi)."""
        return self.step_abs(action)

    def step_abs(self, new_dir):
        if new_dir == (self.dir + 2) % 4:
            new_dir = self.dir
        self.dir = new_dir
        hx, hy = self.snake[0]
        dx, dy = DIRS[new_dir]
        nx, ny = hx + dx, hy + dy
        self.since_food += 1

        if self.blocked(nx, ny):
            self.over = True
            return self.obs(), R_DIE, True

        before = abs(self.food[0] - hx) + abs(self.food[1] - hy) if self.food else 0
        self.snake.appendleft((nx, ny))
        self.body.add((nx, ny))

        if self.food and (nx, ny) == self.food:
            self.score += 1
            self.since_food = 0
            if not self._place_food():  # tahta doldu
                self.over = True
                return self.obs(), R_EAT, True
            return self.obs(), R_EAT, False

        tail = self.snake.pop()
        self.body.discard(tail)

        if self.since_food > 100 + 8 * len(self.snake):  # sonsuz daire engeli
            self.over = True
            return self.obs(), R_DIE, True

        reward = 0.0
        if self.shaping and self.food:
            after = abs(self.food[0] - nx) + abs(self.food[1] - ny)
            reward = R_CLOSER if after < before else R_FARTHER
        return self.obs(), reward, False
