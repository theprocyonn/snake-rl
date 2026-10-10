"""
DQN ajanı: küçük bir sinir ağı, deneyim havuzu ve hedef ağ.

Fikir: Q tablosu yerine bir sinir ağı kullanıyoruz. Ağ, durum vektörünü (14 sayı)
alır ve 3 hamlenin (düz, sağ, sol) beklenen toplam ödülünü tahmin eder.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from snake_env import STATE_SIZE, N_ACTIONS

# Hiperparametreler (denemek için bunlarla oynayabilirsiniz)
GAMMA = 0.99          # gelecek ödüllerin ağırlığı
LR = 1e-3             # öğrenme hızı
BATCH = 64            # her güncellemede kaç deneyimden öğrenilir
BUFFER_SIZE = 50_000  # deneyim havuzu boyutu
START_LEARNING = 1000 # havuzda en az bu kadar deneyim birikince öğrenmeye başla
TARGET_EVERY = 500    # hedef ağ kaç adımda bir güncellenir
EPS_STEPS = 30_000    # epsilon kaç adımda 1.0'dan EPS_MIN'e düşer
EPS_MIN = 0.02


class QNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(STATE_SIZE, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, N_ACTIONS),
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    """Geçmiş deneyimleri saklar; öğrenirken rastgele örnek çekilir."""

    def __init__(self, capacity=BUFFER_SIZE):
        self.cap = capacity
        self.i = 0
        self.size = 0
        self.s = np.zeros((capacity, STATE_SIZE), np.float32)
        self.a = np.zeros(capacity, np.int64)
        self.r = np.zeros(capacity, np.float32)
        self.s2 = np.zeros((capacity, STATE_SIZE), np.float32)
        self.d = np.zeros(capacity, np.float32)

    def add(self, s, a, r, s2, done):
        i = self.i
        self.s[i], self.a[i], self.r[i], self.s2[i], self.d[i] = s, a, r, s2, float(done)
        self.i = (i + 1) % self.cap
        self.size = min(self.size + 1, self.cap)

    def sample(self, n):
        idx = np.random.randint(0, self.size, n)
        return (torch.from_numpy(self.s[idx]), torch.from_numpy(self.a[idx]),
                torch.from_numpy(self.r[idx]), torch.from_numpy(self.s2[idx]),
                torch.from_numpy(self.d[idx]))


class DQNAgent:
    def __init__(self):
        self.q = QNet()
        self.target = QNet()
        self.target.load_state_dict(self.q.state_dict())
        self.opt = torch.optim.Adam(self.q.parameters(), lr=LR)
        self.buffer = ReplayBuffer()
        self.updates = 0

    def act(self, state, eps=0.0):
        """Epsilon-açgözlü hamle: eps olasılıkla rastgele, yoksa ağın en iyi bulduğu."""
        if np.random.rand() < eps:
            return int(np.random.randint(N_ACTIONS))
        with torch.no_grad():
            x = torch.as_tensor(np.asarray(state, dtype=np.float32)).unsqueeze(0)
            return int(self.q(x).argmax(dim=1).item())

    def remember(self, s, a, r, s2, done):
        self.buffer.add(s, a, r, s2, done)

    def learn(self):
        if self.buffer.size < START_LEARNING:
            return
        s, a, r, s2, d = self.buffer.sample(BATCH)
        pred = self.q(s).gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            nxt = self.target(s2).max(dim=1).values
            y = r + GAMMA * (1.0 - d) * nxt
        loss = F.smooth_l1_loss(pred, y)
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), 10.0)
        self.opt.step()

        self.updates += 1
        if self.updates % TARGET_EVERY == 0:
            self.target.load_state_dict(self.q.state_dict())

    def save(self, path):
        torch.save(self.q.state_dict(), path)

    def load(self, path):
        self.q.load_state_dict(torch.load(path, map_location="cpu"))
        self.target.load_state_dict(self.q.state_dict())
        self.q.eval()
