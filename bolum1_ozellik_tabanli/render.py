"""pygame ile çizim. Oyun mantığından bağımsızdır: sadece env'in durumunu çizer."""
import pygame

CELL = 32
PANEL = 56

BG = (24, 33, 29)
BOARD = (31, 44, 38)
GRID = (38, 54, 46)
SNAKE = (79, 184, 138)
HEAD = (154, 232, 195)
FOOD = (255, 122, 92)
TEXT = (228, 236, 230)
MUTED = (155, 176, 164)


class Renderer:
    def __init__(self, env, title="Yılan"):
        pygame.init()
        self.n = env.n
        self.size = self.n * CELL
        self.screen = pygame.display.set_mode((self.size, self.size + PANEL))
        pygame.display.set_caption(title)
        self.font = pygame.font.SysFont("arial", 18)
        self.big = pygame.font.SysFont("arial", 34, bold=True)

    def pump(self):
        """Pencere olaylarını işler. Pencere kapatıldıysa False döner."""
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                return False
        return True

    def close(self):
        pygame.quit()

    def draw(self, env, line1="", line2="", banner=None, banner2=""):
        scr = self.screen
        scr.fill(BG)
        scr.blit(self.font.render(line1, True, TEXT), (10, 6))
        scr.blit(self.font.render(line2, True, MUTED), (10, 30))

        oy = PANEL  # tahtanın dikey başlangıcı
        pygame.draw.rect(scr, BOARD, (0, oy, self.size, self.size))
        for i in range(1, self.n):
            pygame.draw.line(scr, GRID, (i * CELL, oy), (i * CELL, oy + self.size))
            pygame.draw.line(scr, GRID, (0, oy + i * CELL), (self.size, oy + i * CELL))

        if env.food:
            fx, fy = env.food
            pygame.draw.circle(scr, FOOD, (fx * CELL + CELL // 2, oy + fy * CELL + CELL // 2), int(CELL * 0.32))

        for i, (x, y) in enumerate(env.snake):
            rect = pygame.Rect(x * CELL + 3, oy + y * CELL + 3, CELL - 6, CELL - 6)
            pygame.draw.rect(scr, HEAD if i == 0 else SNAKE, rect, border_radius=8)

        if banner:
            veil = pygame.Surface((self.size, self.size), pygame.SRCALPHA)
            veil.fill((24, 33, 29, 180))
            scr.blit(veil, (0, oy))
            t = self.big.render(banner, True, TEXT)
            scr.blit(t, t.get_rect(center=(self.size // 2, oy + self.size // 2 - 14)))
            if banner2:
                t2 = self.font.render(banner2, True, MUTED)
                scr.blit(t2, t2.get_rect(center=(self.size // 2, oy + self.size // 2 + 24)))

        pygame.display.flip()
