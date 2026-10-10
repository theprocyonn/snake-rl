"""
Yılan oyunu motoru (ekran yok, sadece mantık).

Ajan için hamleler "göreli"dir: 0 = düz, 1 = sağa dön, 2 = sola dön.
Ajanın gördüğü durum 17 sayıdan oluşan bir vektördür (aşağıda açıklandı).
"""
import random
from collections import deque

GRID = 15
DIRS = [(0, -1), (1, 0), (0, 1), (-1, 0)]  # yukarı, sağ, aşağı, sol
N_ACTIONS = 3
STATE_SIZE = 17

# Işın yönleri, yılanın baktığı yöne göre (ileri, sağ) bileşenleri:
# ileri, ileri-sağ, sağ, geri-sağ, geri, geri-sol, sol, ileri-sol
RAYS = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]

# Ödüller
R_EAT = 1.0
R_DIE = -1.0
R_CLOSER = 0.02
R_FARTHER = -0.025


class SnakeEnv:
    def __init__(self, grid=GRID, seed=None):
        self.n = grid
        self.rng = random.Random(seed)
        self.reset()

    # ------------------------------------------------------------------
    def reset(self):
        m = self.n // 2
        self.snake = deque([(m, m), (m - 1, m), (m - 2, m)])  # ilk eleman = baş
        self.body = set(self.snake)
        self.dir = 1  # sağa bakıyor
        self.score = 0
        self.since_food = 0
        self.over = False
        self._place_food()
        return self.state()

    def _place_food(self):
        free = [(x, y) for x in range(self.n) for y in range(self.n) if (x, y) not in self.body]
        if not free:
            self.food = None
            return False
        self.food = self.rng.choice(free)
        return True

    def blocked(self, x, y):
        return x < 0 or y < 0 or x >= self.n or y >= self.n or (x, y) in self.body

    # ------------------------------------------------------------------
    def step(self, action):
        """action: 0 düz, 1 sağa, 2 sola. Döner: (durum, ödül, bitti_mi)."""
        new_dir = (self.dir + (0, 1, 3)[action]) % 4
        return self.step_abs(new_dir)

    def step_abs(self, new_dir):
        """Mutlak yön ile hareket (insan modu için). Geri dönüş yok sayılır."""
        if new_dir == (self.dir + 2) % 4:
            new_dir = self.dir
        self.dir = new_dir
        hx, hy = self.snake[0]
        dx, dy = DIRS[new_dir]
        nx, ny = hx + dx, hy + dy
        self.since_food += 1

        if self.blocked(nx, ny):
            self.over = True
            return self.state(), R_DIE, True

        before = abs(self.food[0] - hx) + abs(self.food[1] - hy) if self.food else 0
        self.snake.appendleft((nx, ny))
        self.body.add((nx, ny))

        if self.food and (nx, ny) == self.food:
            self.score += 1
            self.since_food = 0
            if not self._place_food():  # tahta doldu: oyunu kazandı
                self.over = True
                return self.state(), R_EAT, True
            return self.state(), R_EAT, False

        tail = self.snake.pop()
        self.body.discard(tail)

        # Yem yemeden çok uzun süre geçerse bölümü bitir (sonsuz daire engeli)
        if self.since_food > 100 + 8 * len(self.snake):
            self.over = True
            return self.state(), R_DIE, True

        after = abs(self.food[0] - nx) + abs(self.food[1] - ny) if self.food else 0
        return self.state(), (R_CLOSER if after < before else R_FARTHER), False

    # ------------------------------------------------------------------
    def _flood(self, x, y):
        """(x, y)'den ulaşılabilen boş alan ve kuyruğa ulaşılıp ulaşılamadığı."""
        if self.blocked(x, y):
            return 0, False
        tail = self.snake[-1]
        seen = {(x, y)}
        stack = [(x, y)]
        reaches_tail = False
        while stack:
            cx, cy = stack.pop()
            for dx, dy in DIRS:
                nx, ny = cx + dx, cy + dy
                if (nx, ny) == tail:
                    reaches_tail = True
                    continue
                if (nx, ny) not in seen and not self.blocked(nx, ny):
                    seen.add((nx, ny))
                    stack.append((nx, ny))
        return len(seen), reaches_tail

    def state(self):
        """
        17 özellik:
          [0-7]  8 yönde (ileri, ileri-sağ, sağ, ... ileri-sol) en yakın engele
                 uzaklığın tersi (1/uzaklık). Yakın engel = 1'e yakın değer.
          [8-9]  Yem, yılanın bakış yönüne göre: (ileri, sağ) bileşeni, -1..1
          [10]   Yılan uzunluğu / toplam hücre sayısı
          [11-13] Üç hamlenin (düz, sağ, sol) sonrası ulaşılabilir boş alan oranı.
          [14-16] Üç hamlenin her biri için: o hamleden sonra kuyruğa hâlâ bir yol
                 var mı (1 = var, 0 = yok). Yol yoksa yılan kendini hapsetmiş demektir.
        """
        n = self.n
        hx, hy = self.snake[0]
        fwd = DIRS[self.dir]
        right = DIRS[(self.dir + 1) % 4]
        s = []

        for f, r in RAYS:
            vx = f * fwd[0] + r * right[0]
            vy = f * fwd[1] + r * right[1]
            dist = 1
            x, y = hx + vx, hy + vy
            while not self.blocked(x, y):
                x += vx
                y += vy
                dist += 1
            s.append(1.0 / dist)

        if self.food:
            ddx, ddy = self.food[0] - hx, self.food[1] - hy
        else:
            ddx = ddy = 0
        s.append((ddx * fwd[0] + ddy * fwd[1]) / n)
        s.append((ddx * right[0] + ddy * right[1]) / n)

        s.append(len(self.snake) / (n * n))

        free_cells = max(1, n * n - len(self.snake))
        tails = []
        for d in (self.dir, (self.dir + 1) % 4, (self.dir + 3) % 4):
            dx, dy = DIRS[d]
            area, rt = self._flood(hx + dx, hy + dy)
            s.append(area / free_cells)
            tails.append(1.0 if rt else 0.0)
        s.extend(tails)

        return s
