from math import atan, atan2, cos, sin, radians
from random import randint, choice, choices
import numpy as np
from PIL import Image
import pickle

import pygame
import re
from scipy.special import expit, softmax


def f_part(x):
    return x - int(x)


def rf_part(x):
    return 1 - f_part(x)


def brush_point(x, y, alpha, image):
    a = image[y][x][3]
    coef = 1
    arr0 = [0, 0, 0] + [min(int((alpha * coef) * 255) + a, 255)]
    image[y][x] = arr0


def brush_line(x0, y0, x1, y1, image, width=1):
    steep = abs(y1 - y0) > abs(x1 - x0)

    if steep:
        x0, y0 = y0, x0
        x1, y1 = y1, x1
    if x0 > x1:
        x0, x1 = x1, x0
        y0, y1 = y1, y0

    dx = x1 - x0
    dy = y1 - y0

    if dx == 0.0:
        gradient = 1.0
    else:
        gradient = dy / dx

    x_end = round(x0)
    y_end = y0 + gradient * (x_end - x0)
    x_gap = rf_part(x0 + 0.5)
    x_pxl1 = x_end
    y_pxl1 = int(y_end)
    if steep:
        brush_point(y_pxl1, x_pxl1, rf_part(y_end) * x_gap, image)
        brush_point(y_pxl1 + 1, x_pxl1, f_part(y_end) * x_gap, image)
    else:
        brush_point(x_pxl1, y_pxl1, rf_part(y_end) * x_gap, image)
        brush_point(x_pxl1, y_pxl1 + 1, f_part(y_end) * x_gap, image)

    intery = y_end + gradient - width // 2
    # Image.fromarray(image).convert("RGBA").save(f"raster_img1.png")

    x_end = round(x1)
    y_end = y1 + gradient * (x_end - x1)
    x_gap = f_part(x1 + 0.5)
    x_pxl2 = x_end

    y_pxl2 = int(y_end)
    if steep:
        brush_point(y_pxl2, x_pxl2, rf_part(y_end) * x_gap, image)
        brush_point(y_pxl2 + 1, x_pxl2, f_part(y_end) * x_gap, image)
    else:
        brush_point(x_pxl2, y_pxl2, rf_part(y_end) * x_gap, image)
        brush_point(x_pxl2, y_pxl2 + 1, f_part(y_end) * x_gap, image)

    if steep:
        for x in range(x_pxl1 + 1, x_pxl2):
            brush_point(int(intery), x, rf_part(intery), image)
            for y in range(width):
                brush_point(int(intery) + y + 1, x, 1, image)
            brush_point(int(intery) + 1 + width, x, f_part(intery), image)
            intery = intery + gradient
    else:
        for x in range(x_pxl1 + 1, x_pxl2):
            brush_point(x, int(intery), rf_part(intery), image)
            for y in range(width):
                brush_point(x, int(intery) + y + 1, 1, image)
            brush_point(x, int(intery) + 1 + width, f_part(intery), image)
            intery = intery + gradient


class Board:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.board = [[[0, 0, 0, 0] for _ in range(self.width + 3)] for _ in range(self.height + 3)]
        self.left = 5
        self.top = 5
        self.cell_size = 20
        self.cross = True
        self.prev_x, self.prev_y = 0, 0
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
                # print(self.board[i][j])
                self.draw_rect_alpha(screen, self.board[i][j], ((x, y), (self.cell_size, self.cell_size)))

    def save_img(self, file_name):
        b = [[[] for _ in range(self.width)] for _ in range(self.height)]
        for i in range(self.height):
            for j in range(self.width):
                b[i][j] = self.board[i][j][3]
        img = np.array(b, dtype=np.uint8)
        img = Image.fromarray(img).convert("L")
        img.save(f"{file_name}.png")

    @staticmethod
    def draw_rect_alpha(surface, color, rect):
        shape_surf = pygame.Surface(pygame.Rect(rect).size, pygame.SRCALPHA)
        pygame.draw.rect(shape_surf, color, shape_surf.get_rect())
        surface.blit(shape_surf, rect)

    @staticmethod
    def distance(x1, y1, x2, y2):
        return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5

    def get_cell(self, mouse_pos):
        m_x, m_y = mouse_pos
        for i in range(self.height):
            for j in range(self.width):
                x, y = self.left + j * self.cell_size, self.top + i * self.cell_size
                if x <= m_x <= x + self.cell_size and y <= m_y <= y + self.cell_size:
                    return i, j
        return None

    def on_click(self, cell_coords):
        if cell_coords:
            i, j = cell_coords
            brush_line(self.prev_x, self.prev_y, j, i, self.board)
            self.prev_x, self.prev_y = j, i

    def get_click(self, mouse_pos):
        i, j = self.get_cell(mouse_pos)
        self.on_click((i, j))

    def save(self, ans):
        b = [[[] for _ in range(self.width)] for _ in range(self.height)]
        for i in range(self.height):
            for j in range(self.width):
                b[i][j] = self.board[i][j][3]
        img = np.array(b, dtype=np.uint8)
        img = Image.fromarray(img).convert("L")
        d = {"0": "circle", "1": "not filled without tail", "2": "filled without tail", "3": "filled with tail"}
        print(f"saving {d[ans]}")
        with open("images.csv", "a+")as file:
            file.write(f"{ans}," + ",".join(list(map(str, list(np.asarray(img).flatten())))))

    def clear(self):
        self.board = [[[0, 0, 0, 0] for _ in range(self.width + 3)] for _ in range(self.height + 3)]

    def predict(self):
        with open('weights.pkl', 'rb') as f:
            w = pickle.load(f)
        b = [[[] for _ in range(self.width)] for _ in range(self.height)]
        for i in range(self.height):
            for j in range(self.width):
                b[i][j] = self.board[i][j][3]
        img = np.array(b, dtype=np.uint8)
        img = Image.fromarray(img).convert("L")
        inputs_list = list(np.asarray(img).flatten())
        inputs = np.array(inputs_list).reshape(len(inputs_list), 1)
        for weights_matrix in w[:1]:
            inputs = self.activation1(weights_matrix @ inputs)
        inputs = self.activation2(w[1] @ inputs)
        return inputs

    def activation1(self, x):
        return expit(x)

    def activation2(self, x):
        return softmax(x)



if __name__ == '__main__':
    pygame.init()

    pygame.display.set_caption('Board')
    w, h = 28, 28
    top, left, cell_size = 5, 5, 20
    board = Board(w, h)
    board.set_view(left, top, cell_size)
    size = width, height = w * cell_size + 2 * left, h * cell_size + 2 * top
    screen = pygame.display.set_mode(size, pygame.SRCALPHA)
    screen.fill(pygame.Color('black'))
    clock = pygame.time.Clock()
    fps = 20
    z = 1

    running = True
    i, j = w, h
    f = True
    mouse_button_down = False
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.MOUSEMOTION and pygame.mouse.get_focused() and event.buttons[0]:
                if f:
                    board.prev_x, board.prev_y = board.get_cell(pygame.mouse.get_pos())[::-1]
                    f = False
                board.get_click(pygame.mouse.get_pos())
            if event.type == pygame.MOUSEBUTTONUP:
                f = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_s:
                    board.save(input())
                if event.key == pygame.K_c:
                    board.clear()
                if event.key == pygame.K_p:
                    print(list(map(lambda x: '{0:.10f}'.format(x), board.predict().flatten())))

        screen.fill((255, 255, 255, 0))
        clock.tick(fps)
        board.render(screen)
        # board.save_img("image")
        pygame.display.flip()
