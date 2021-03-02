from math import atan, atan2, cos, sin, radians
from random import randint, choice, choices
import numpy as np

import pygame
import re


class Board:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.board = [[[255, 255, 255] for _ in range(self.width + 3)] for _ in range(self.height + 3)]
        self.left = 5
        self.top = 5
        self.cell_size = 20
        self.cross = True
        self.colors = [(255, 255, 255), (0, 0, 0)]

    def set_view(self, left, top, cell_size):
        self.left = left
        self.top = top
        self.cell_size = cell_size

    def on_start(self):
        for i in range(self.height):
            for j in range(self.width):
                self.board[i][j] = choice([0, 1])

    def render(self, screen):
        for i in range(self.height):
            for j in range(self.width):
                x, y = self.left + j * self.cell_size, self.top + i * self.cell_size
                pygame.draw.rect(screen, self.board[i][j], ((x, y), (self.cell_size, self.cell_size)))
        pygame.draw.rect(screen, (255, 255, 255), ((self.left, self.top),
                                                   (self.cell_size * self.width, self.cell_size * self.height)), 1)

    @staticmethod
    def distance(x1, y1, x2, y2):
        return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5

    def get_cell(self, mouse_pos):
        m_x, m_y = mouse_pos
        for i in range(self.height):
            for j in range(self.width):
                x, y = self.left + j * self.cell_size, self.top + i * self.cell_size
                if x <= m_x <= x + self.cell_size and y <= m_y <= y + self.cell_size:
                    # cx, cy = self.left + j * self.cell_size + self.cell_size // 2, self.top + i * self.cell_size + self.cell_size // 2
                    # d = self.distance(m_x, m_y, cx, cy) / (2 ** 0.5 * self.cell_size)
                    return i, j
        return None

    def on_click(self, cell_coords):
        if cell_coords:
            i, j = cell_coords
            self.board[i][j] = list(np.array(self.board[i][j]) / 1.3)
            self.board[i + 1][j] = list(np.array(self.board[i][j]) / 1.07)
            self.board[i][j + 1] = list(np.array(self.board[i][j]) / 1.07)
            self.board[i - 1][j] = list(np.array(self.board[i][j]) / 1.07)
            self.board[i][j - 1] = list(np.array(self.board[i][j]) / 1.07)

    def get_click(self, mouse_pos):
        i, j = self.get_cell(mouse_pos)
        self.on_click((i, j))


class Life(Board):
    def __init__(self, width, height):
        super().__init__(width, height)
        self.editing = True
        self.colors = [None, (255, 255, 255)]

    def get_cell(self, mouse_pos):
        m_x, m_y = mouse_pos
        for i in range(self.height):
            for j in range(self.width):
                x, y = self.left + j * self.cell_size, self.top + i * self.cell_size
                if x <= m_x <= x + self.cell_size and y <= m_y <= y + self.cell_size:
                    return i, j
        return None

    def neighbours(self, i, j):
        # print("\n".join([" ".join([str(i) for i in j]) for j in self.board]))
        n_1 = self.board[i - 1][j - 1]
        n_2 = self.board[i - 1][j]
        n_3 = self.board[i - 1][j + 1]
        n_4 = self.board[i][j - 1]
        n_5 = self.board[i][j + 1]
        n_6 = self.board[i + 1][j - 1]
        n_7 = self.board[i + 1][j]
        n_8 = self.board[i + 1][j + 1]
#         print(f"""[ {n_1} ] [ {n_2} ] [ {n_3} ]
# [ {n_4} ] [ c ] [ {n_5} ]
# [ {n_6} ] [ {n_7} ] [ {n_8} ]""")
        return sum([n_1, n_2, n_3, n_4, n_5, n_6, n_7, n_8])

    def next_move(self):
        copy_board = [[0] * (self.width + 3) for _ in range(self.height + 3)]
        for i in range(self.height):
            for j in range(self.width):
                n = self.neighbours(i, j)
                # print(i, j, n)
                if self.board[i][j] == 0:
                    if n == 3:
                        copy_board[i][j] = 1
                else:
                    if n in (2, 3):
                        copy_board[i][j] = 1
        return copy_board

    def on_click(self, cell_coords):
        if cell_coords and self.editing:
            i, j = cell_coords
            if self.board[i][j] == 1:
                self.board[i][j] = 0
            else:
                self.board[i][j] = 1
        elif not self.editing:
            self.board = self.next_move()

    def set_editing(self, boolean):
        self.editing = boolean

    def render(self, screen):
        for i in range(self.height + 3):
            for j in range(self.width + 3):
                x, y = self.left + j * self.cell_size, self.top + i * self.cell_size
                color = self.colors[self.board[i][j]]
                if color:
                    pygame.draw.rect(screen, color, ((x, y), (self.cell_size, self.cell_size)))
                else:
                    pygame.draw.rect(screen, (255, 255, 255), ((x, y), (self.cell_size, self.cell_size)), 1)

        pygame.draw.rect(screen, (255, 255, 255), ((self.left, self.top),
                                                   (self.cell_size * self.width, self.cell_size * self.height)), 1)


class Minesweeper(Board):
    def __init__(self, width, height, n_mines):
        super().__init__(width, height)
        self.n_mines = n_mines
        self.board = [[-1] * (self.width + 4) for _ in range(self.height + 4)]
        mines = choices(range(self.width * self.height), k=self.n_mines)
        print(mines)
        for mine in mines:
            i, j = mine // self.height, mine % self.width
            self.board[i][j] = 10

    def neighbours(self, i, j):
        n_1 = self.board[i - 1][j - 1]
        n_2 = self.board[i - 1][j]
        n_3 = self.board[i - 1][j + 1]
        n_4 = self.board[i][j - 1]
        n_5 = self.board[i][j + 1]
        n_6 = self.board[i + 1][j - 1]
        n_7 = self.board[i + 1][j]
        n_8 = self.board[i + 1][j + 1]
        return [n_1, n_2, n_3, n_4, n_5, n_6, n_7, n_8].count(10)

    def open_cell(self, i, j):
        self.board[i][j] = self.neighbours(i, j)
        if self.is_valid(i - 1, j - 1) and self.neighbours(i - 1, j - 1) == 0 and self.board[i - 1][j - 1] == -1:
            self.open_cell(i - 1, j - 1)
        else:
            if self.board[i - 1][j - 1] == -1 and self.neighbours(i, j) == 0:
                self.board[i - 1][j - 1] = self.neighbours(i - 1, j - 1)
        if self.is_valid(i - 1, j) and self.neighbours(i - 1, j) == 0 and self.board[i - 1][j] == -1:
            self.open_cell(i - 1, j)
        else:
            if self.board[i - 1][j] == -1 and self.neighbours(i, j) == 0:
                self.board[i - 1][j] = self.neighbours(i - 1, j)
        if self.is_valid(i - 1, j + 1) and self.neighbours(i - 1, j + 1) == 0 and self.board[i - 1][j + 1] == -1:
            self.open_cell(i - 1, j + 1)
        else:
            if self.board[i - 1][j + 1] == -1 and self.neighbours(i, j) == 0:
                self.board[i - 1][j + 1] = self.neighbours(i - 1, j + 1)
        if self.is_valid(i, j - 1) and self.neighbours(i, j - 1) == 0 and self.board[i][j - 1] == -1:
            self.open_cell(i, j - 1)
        else:
            if self.board[i][j - 1] == -1 and self.neighbours(i, j) == 0:
                self.board[i][j - 1] = self.neighbours(i, j - 1)
        if self.is_valid(i, j + 1) and self.neighbours(i, j + 1) == 0 and self.board[i][j + 1] == -1:
            self.open_cell(i, j + 1)
        else:
            if self.board[i][j + 1] == -1 and self.neighbours(i, j) == 0:
                self.board[i][j + 1] = self.neighbours(i, j + 1)
        if self.is_valid(i + 1, j - 1) and self.neighbours(i + 1, j - 1) == 0 and self.board[i + 1][j - 1] == -1:
            self.open_cell(i + 1, j - 1)
        else:
            if self.board[i + 1][j - 1] == -1 and self.neighbours(i, j) == 0:
                self.board[i + 1][j - 1] = self.neighbours(i + 1, j - 1)
        if self.is_valid(i + 1, j) and self.neighbours(i + 1, j) == 0 and self.board[i + 1][j] == -1:
            self.open_cell(i + 1, j)
        else:
            if self.board[i + 1][j] == -1 and self.neighbours(i, j) == 0:
                self.board[i + 1][j] = self.neighbours(i + 1, j)
        if self.is_valid(i + 1, j + 1) and self.neighbours(i + 1, j + 1) == 0 and self.board[i + 1][j + 1] == -1:
            self.open_cell(i + 1, j + 1)
        else:
            if self.board[i + 1][j + 1] == -1 and self.neighbours(i, j) == 0:
                self.board[i + 1][j + 1] = self.neighbours(i + 1, j + 1)

    def render(self, screen):
        for i in range(self.height):
            for j in range(self.width):
                x, y = self.left + j * self.cell_size, self.top + i * self.cell_size
                cell = self.board[i][j]
                if cell == 10:
                    pygame.draw.rect(screen, (255, 0, 0), ((x, y), (self.cell_size, self.cell_size)))
                elif cell == -1:
                    pygame.draw.rect(screen, (255, 255, 255), ((x, y), (self.cell_size, self.cell_size)), 1)
                else:
                    pygame.draw.rect(screen, (255, 255, 255), ((x, y), (self.cell_size, self.cell_size)), 1)
                    font = pygame.font.Font(None, 50)
                    text = font.render(f"{cell}", True, (100, 255, 100))
                    text_x, text_y = x + 5, y + 5
                    screen.blit(text, (text_x, text_y))

        pygame.draw.rect(screen, (255, 255, 255), ((self.left, self.top),
                                                   (self.cell_size * self.width, self.cell_size * self.height)), 1)

    def on_click(self, cell_coords):
        if cell_coords:
            i, j = cell_coords
            if self.board[i][j] != 10:
                self.open_cell(i, j)

    def get_click(self, mouse_pos):
        cell = self.get_cell(mouse_pos)
        self.on_click(cell)

    def is_valid(self, i, j):
        if 0 <= i < self.height and 0 <= j < self.width:
            return True
        return False


class Lines(Board):
    def __init__(self, width, height):
        super().__init__(width, height)
        self.red = False
        self.red_x, self.red_y = 0, 0
        self.colors = [None, (0, 0, 255), (255, 0, 0)]
        self.path = []

    def has_path(self, x1, y1, x2, y2):
        INF = 10 ** 6
        dist = [[INF] * self.width for _ in range(self.height)]
        prev = [[None] * self.width for _ in range(self.height)]
        dist[y1][x1] = 0
        queue = [(x1, y1)]
        while queue:
            x, y = queue.pop(0)
            for dx, dy in (1, 0), (0, 1), (-1, 0), (0, -1):
                next_x, next_y = x + dx, y + dy
                if 0 <= next_x < self.width and 0 <= next_y < self.height and \
                        self.board[next_y][next_x] == 0 and dist[next_y][next_x] == INF:
                    dist[next_y][next_x] = dist[y][x] + 1
                    prev[next_y][next_x] = (x, y)
                    queue.append((next_x, next_y))
        if dist[y2][x2] == INF or (x2 == x1 and y2 == y1):
            return False
        x, y = x2, y2
        while prev[y][x] is not None:
            self.path.append((y, x))
            x, y = prev[y][x]
        return True

    def render(self, screen):
        for i in range(self.height):
            for j in range(self.width):
                x, y = self.left + j * self.cell_size, self.top + i * self.cell_size
                color = self.colors[self.board[i][j]]
                if color:
                    pygame.draw.rect(screen, (255, 255, 255), ((x, y), (self.cell_size, self.cell_size)), 1)
                    cell_2 = self.cell_size // 2
                    pygame.draw.circle(screen, color, (x + cell_2, y + cell_2), cell_2 - 2)
                else:
                    pygame.draw.rect(screen, (255, 255, 255), ((x, y), (self.cell_size, self.cell_size)), 1)

        pygame.draw.rect(screen, (255, 255, 255), ((self.left, self.top),
                                                   (self.cell_size * self.width, self.cell_size * self.height)), 1)

    def on_click(self, cell_coords):
        if cell_coords:
            i, j = cell_coords
            if not self.red and self.board[i][j] == 0:
                self.board[i][j] = 1
            elif not self.red and self.board[i][j] == 1:
                self.board[i][j] = 2
                self.red = True
                self.red_x = j
                self.red_y = i
            elif self.red:
                if i == self.red_y and j == self.red_x:
                    self.red = False
                    self.path = []
                    self.board[i][j] = 1
                elif self.has_path(self.red_x, self.red_y, j, i):
                    self.red = False

    def get_click(self, mouse_pos):
        cell = self.get_cell(mouse_pos)
        self.on_click(cell)

    def step(self):
        if self.path:
            self.board[self.red_y][self.red_x] = 0
            i, j = self.path.pop()
            self.board[i][j] = 1
            self.red_y, self.red_x = i, j


if __name__ == '__main__':
    pygame.init()

    pygame.display.set_caption('Board')
    w, h = 28, 28
    top, left, cell_size = 5, 5, 20
    board = Board(w, h)
    board.set_view(left, top, cell_size)
    size = width, height = w * cell_size + 2 * left, h * cell_size + 2 * top
    screen = pygame.display.set_mode(size)
    screen.fill(pygame.Color('black'))
    clock = pygame.time.Clock()
    fps = 30
    z = 1

    running = True
    i, j = w, h
    mouse_button_down = False
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.MOUSEBUTTONDOWN:
                mouse_button_down = True
            if event.type == pygame.MOUSEBUTTONUP:
                mouse_button_down = False
            if mouse_button_down:
                board.get_click(event.pos)

        screen.fill((0, 0, 0))
        clock.tick(fps)
        board.render(screen)
        pygame.display.flip()
