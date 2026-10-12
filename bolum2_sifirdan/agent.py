"""
Bölüm 2 ajanı: evrişimli sinir ağı (CNN) ile Double DQN.

Ağ, 3x15x15'lik ham tahtayı girdi olarak alır ve 4 hamlenin (yukarı, sağ, aşağı, sol)
beklenen toplam ödülünü tahmin eder. Elle yazılmış hiçbir özellik yoktur.

Bölüm 1'den farklar:
  - Girdi: ham tahta, ağ: evrişim katmanları (GPU varsa otomatik kullanılır)
  - Double DQN: sonraki hamleyi ana ağ seçer, değerini hedef ağ ölçer
    (Q değerlerinin gereğinden iyimser tahmin edilmesini azaltır)
  - Öğrenme her 4 adımda bir yapılır (standart DQN uygulaması, hızlandırır)
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from snake_env import CHANNELS, GRID, N_ACTIONS

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

GAMMA = 0.99
LR = 1e-4
BATCH = 64
BUFFER_SIZE = 100_000
START_LEARNING = 5_000   # havuzda en az bu kadar deneyim birikince öğrenmeye başla
LEARN_EVERY = 4          # kaç adımda bir öğrenme güncellemesi
TARGET_EVERY = 1_000     # kaç güncellemede bir hedef ağ eşitlenir
EPS_STEPS = 200_000      # epsilon kaç adımda 1.0'dan EPS_MIN'e düşer
EPS_MIN = 0.02


class QNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(CHANNELS, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * GRID * GRID, 256), nn.ReLU(),
            nn.Linear(256, N_ACTIONS),
        )

    def forward(self, x):
        return self.fc(self.conv(x))


class ReplayBuffer:
    def __init__(self, capacity=BUFFER_SIZE):
        self.cap = capacity
        self.i = 0
        self.size = 0
        shape = (capacity, CHANNELS, GRID, GRID)
        self.s = np.zeros(shape, np.uint8)
        self.s2 = np.zeros(shape, np.uint8)
        self.a = np.zeros(capacity, np.int64)
        self.r = np.zeros(capacity, np.float32)
        self.d = np.zeros(capacity, np.float32)

    def add(self, s, a, r, s2, done):
        i = self.i
        self.s[i], self.a[i], self.r[i], self.s2[i], self.d[i] = s, a, r, s2, float(done)
        self.i = (i + 1) % self.cap
        self.size = min(self.size + 1, self.cap)

    def sample(self, n):
        idx = np.random.randint(0, self.size, n)
        return (torch.from_numpy(self.s[idx]).to(DEVICE).float(),
                torch.from_numpy(self.a[idx]).to(DEVICE),
                torch.from_numpy(self.r[idx]).to(DEVICE),
                torch.from_numpy(self.s2[idx]).to(DEVICE).float(),
                torch.from_numpy(self.d[idx]).to(DEVICE))


class DQNAgent:
    def __init__(self):
        self.q = QNet().to(DEVICE)
        self.target = QNet().to(DEVICE)
        self.target.load_state_dict(self.q.state_dict())
        self.opt = torch.optim.Adam(self.q.parameters(), lr=LR)
        self.buffer = ReplayBuffer()
        self.updates = 0

    def act(self, obs, eps=0.0):
        if np.random.rand() < eps:
            return int(np.random.randint(N_ACTIONS))
        with torch.no_grad():
            x = torch.from_numpy(obs).to(DEVICE).float().unsqueeze(0)
            return int(self.q(x).argmax(dim=1).item())

    def remember(self, s, a, r, s2, done):
        self.buffer.add(s, a, r, s2, done)

    def learn(self):
        if self.buffer.size < START_LEARNING:
            return
        s, a, r, s2, d = self.buffer.sample(BATCH)
        pred = self.q(s).gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            best = self.q(s2).argmax(dim=1, keepdim=True)          # ana ağ seçer
            nxt = self.target(s2).gather(1, best).squeeze(1)       # hedef ağ değerlendirir
            y = r + GAMMA * (1.0 - d) * nxt
        loss = F.smooth_l1_loss(pred, y)
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), 10.0)
        self.opt.step()

        self.updates += 1
        if self.updates % TARGET_EVERY == 0:
            self.target.load_state_dict(self.q.state_dict())

    # --- sadece model ağırlıkları (oynamak ve ölçmek için) ---
    def save_model(self, path):
        torch.save(self.q.state_dict(), path)

    def save(self, path):  # evaluate.py ve play.py ile uyumlu isim
        self.save_model(path)

    def load(self, path):
        self.q.load_state_dict(torch.load(path, map_location=DEVICE))
        self.target.load_state_dict(self.q.state_dict())
        self.q.eval()

    # --- eğitimi sürdürebilmek için ara kayıt (deneyim havuzu kaydedilmez) ---
    def save_checkpoint(self, path, extra):
        torch.save({"q": self.q.state_dict(), "target": self.target.state_dict(),
                    "opt": self.opt.state_dict(), "updates": self.updates, "extra": extra}, path)

    def load_checkpoint(self, path):
        ck = torch.load(path, map_location=DEVICE)
        self.q.load_state_dict(ck["q"])
        self.target.load_state_dict(ck["target"])
        self.opt.load_state_dict(ck["opt"])
        self.updates = ck["updates"]
        return ck["extra"]
