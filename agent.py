import cv2
import pygame
import time
from copy import deepcopy
import torch
import random
import numpy as np
from TetrisBattle.envs.tetris_env import TetrisDoubleEnv, TetrisSingleEnv
debug = False


def convert_state(state):

    # Đảm bảo state có đúng định dạng 2D (20, 34)
    if state.ndim == 3 and state.shape[2] == 1:
        state = state.squeeze(axis=2)  # Loại bỏ chiều thứ 3

    board = state[:, :10].astype(int).tolist()
    offsetX = get_offset(state)
    piece_index, next_piece_index = get_block_type(state)
    tetris_shapes = [
        [[1, 1, 1, 1]],
        [[1, 1], [1, 1]],
        [[1, 0, 0], [1, 1, 1]],
        [[0, 0, 1], [1, 1, 1]],
        [[1, 1, 0], [0, 1, 1]],
        [[0, 1, 1], [1, 1, 0]],
        [[0, 1, 0, 0], [1, 1, 1, 1]],
    ]

    # Chuyển chỉ số block thành hình dạng
    state = np.array(state)
    piece = tetris_shapes[piece_index - 1]
    next_piece = tetris_shapes[next_piece_index - 1]

    if (debug):
        print("piece_index, next_piece_index", piece_index, next_piece_index)
        print("piece")
        print(np.array(piece))

    return board, piece, next_piece, offsetX


def get_offset(state):
    offsetX = -1
    for row in state[:, :10]:  # Chỉ xem xét phần bản đồ 20x10
        for col_idx, cell in enumerate(row):

            if abs(cell - 0.3) < 0.001:
                if offsetX == -1:
                    offsetX = col_idx
                else:
                    offsetX = min(offsetX, col_idx)  # Tìm offset nhỏ nhất

    if (debug):
        print("XXXXXXXXXXXXXXXXX")
        print(state[:, :10])
        print(offsetX)
        print("XXXXXXXXXXXXXXXXX")

    # return offsetX if offsetX != -1 else 3  # Mặc định là 3 nếu không tìm thấy
    return offsetX


def get_block_type(state):
    add_info = state[:, 10:17]
    block_current = np.argmax(add_info[6])
    block_next = np.argmax(add_info[1])

    return block_current + 1, block_next + 1


def rotate_clockwise(shape):
    new_shape = list(zip(*shape))[::-1]
    update_offset = np.array(shape).shape[1] - np.array(new_shape).shape[1]
    return new_shape, update_offset


class Field:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.field = [[0] * self.width for _ in range(self.height)]

    def updateField(self, field):
        self.field = field
        # Update height based on the actual field data
        self.height = len(field)

    def projectPieceDown(self, piece, offsetX, workingPieceIndex):

        offsetY = self.height
        for y in range(self.height):
            if self.check_collision(self.field, piece, (offsetX, y)):
                offsetY = y
                break

        # Nếu không thể đặt khối, trả về None
        if offsetY == self.height:
            return None

        # Đặt khối vào field
        for x in range(len(piece[0])):
            for y in range(len(piece)):
                if piece[y][x] > 0:
                    # Kiểm tra xem tọa độ nằm trong giới hạn bản đồ
                    if 0 <= offsetX + x < self.width and 0 <= offsetY - 1 + y < self.height:
                        self.field[offsetY - 1 + y][offsetX +
                                                    x] = -workingPieceIndex

        return self

    @staticmethod
    def check_collision(field, shape, offset):
        off_x, off_y = offset
        for cy, row in enumerate(shape):
            for cx, cell in enumerate(row):
                try:
                    if cell and field[cy + off_y][cx + off_x]:
                        return True
                except IndexError:
                    return True
        return False

    def undo(self, workingPieceIndex):
        self.field = [[0 if el == -workingPieceIndex else el for el in row]
                      for row in self.field]

    ############################################################
    #                   HEURISTICS                            #
    ############################################################

    def heuristics(self):
        heights = self.heights()  # Chiều cao các cột

        # ,self.well_depth()]
        return [max(heights), self.lines_cleared(), self.num_holes(), self.bumpiness()]

    def heights(self):
        """
        Chiều cao của từng cột.
        """
        return [max([i for i, cell in enumerate(col[::-1]) if cell] + [0]) for col in zip(*self.field)]

    def aggregate_height(self):
        height = 0
        for col in range(10):
            for row in range(20):
                if self.field[row][col] != 0:
                    height += 20 - row
                    break
        return height

    def num_holes(self):
        holes = 0
        for col in range(10):
            block_found = False
            for row in range(20):
                if self.field[row][col] != 0:
                    block_found = True
                elif block_found:
                    holes += 1
        return holes

    def completLine(self):
        return sum(all(cell != 0 for cell in row) for row in self.field)

    def lines_cleared(self):
        return sum(all(cell != 0 for cell in row) for row in self.field)

    def bumpiness(self):
        heights = []
        for col in range(10):
            for row in range(20):
                if self.field[row][col] != 0:
                    heights.append(20 - row)
                    break
            else:
                heights.append(0)
        return sum(abs(heights[i] - heights[i + 1]) for i in range(9))

    def numberOfHoles(self, heights):
        """
        Số lỗ (holes) trong mỗi cột.
        Lỗ là các ô trống nằm bên dưới một ô không trống trong cùng cột.
        """
        results = []
        for col_idx in range(self.width):
            result = 0
            for row_idx in range(self.height):
                if self.field[row_idx][col_idx] == 0 and row_idx < self.height - heights[col_idx]:
                    result += 1
            results.append(result)
        return results

    def maxHeightColumns(self, heights):
        """
        Chiều cao cột cao nhất.
        """
        return max(heights)

    def minHeightColumns(self, heights):
        """
        Chiều cao cột thấp nhất.
        """
        return min(heights)

    def maxPitDepth(self, heights):
        """
        Độ sâu lớn nhất giữa cột cao nhất và cột thấp nhất.
        """
        return max(heights) - min(heights)

    def emptySpaces(self, heights):
        """
        Số ô trống chưa được lấp trên toàn bộ lưới.
        """
        total_blocks = sum(heights)  # Tổng số khối đã xếp
        total_cells = self.width * self.height  # Tổng số ô trên lưới
        return total_cells - total_blocks

    def well_depth(self,):
        depths = 0
        for col in range(10):
            for row in range(20):
                if self.field[row][col] == 0:
                    if (col == 0 or self.field[row][col - 1] != 0) and (col == 9 or self.field[row][col + 1] != 0):
                        depths += 1
                else:
                    break
        return depths


class Ai:
    @staticmethod
    def best(field, workingPieces, workingPieceIndex, weights, level, offsetX, env=None):
        bestRotation = None
        bestOffset = None
        bestScore = None
        best_move = None
        best_offset = None

        workingPieceIndex = deepcopy(workingPieceIndex)
        workingPiece = workingPieces[workingPieceIndex]

        # Thử tất cả các góc xoay (4 lần xoay tối đa)
        for rotation in range(4):
            # Duyệt qua tất cả các cột hợp lệ
            for offset in range(field.width - len(workingPiece[0]) + 1):

                # Tạo một bản sao của field để kiểm tra
                simulated_field = deepcopy(field)
                projected_field = simulated_field.projectPieceDown(
                    workingPiece, offset, level)

                # Nếu khối được đặt thành công, tính điểm heuristics
                if projected_field:
                    heuristics = projected_field.heuristics()  # Lấy đặc trưng heuristics
                    # score = sum(a * b for a, b in zip(heuristics, weights))  # Tính điểm với trọng số

                    score = 0

                    for i in range(min(len(heuristics), len(weights))):
                        score += heuristics[i] * weights[i]

                    if (debug):
                        print(np.array(projected_field.field)[10:])
                        print("max_h", heuristics[0], " lines_cleared:", heuristics[1], " num_holes:",
                              heuristics[2])
                        print("score", score, "bestScore", bestScore)
                        print("rotation", rotation, "offset", offset)
                        print("--------------")

                    # Cập nhật vị trí tốt nhất nếu tìm được điểm cao hơn
                    if bestScore is None or score > bestScore:
                        bestScore = score
                        bestOffset = offset
                        bestRotation = rotation
                        best_offsetX = offsetX

                # Hoàn tác trạng thái sau mỗi thử nghiệm
                # simulated_field.undo(level)

            # Xoay khối để thử hướng tiếp theo
            workingPiece, update_offset = rotate_clockwise(workingPiece)
            state, reward, done, infos = env.step(4)
            board, piece, next_piece, offsetX = convert_state(state)
            # print (piece, next_piece)

            if (debug):
                print("workingPiece", workingPiece)
                print(update_offset, "New ofset", offsetX)

        if (debug):
            print("FINAL", "bestScore", bestScore)
            print("bestRotation", bestRotation, "bestOffset",
                  bestOffset, "offsetX", offsetX)
            moves = []  # Khởi tạo danh sách hành động
            moves.extend(["UP"] * (bestRotation))  # Thêm hành động xoay
            moves.extend(["RIGHT" if bestOffset > offsetX else "LEFT"]
                         * abs(bestOffset - offsetX))  # Thêm di chuyển
            print("moves", moves)
            print("*****************")
        return bestOffset, bestRotation, bestScore, best_offsetX

    @staticmethod
    def choose(initialField, piece, next_piece, offsetX, weights, env=None):
        field = Field(initialField.width, initialField.height)
        field.updateField(deepcopy(initialField.field))

        offset, rotation, _, newOffsetX = Ai.best(
            field, [piece, next_piece], 0, weights, 1, offsetX, env)

        moves = []  # Khởi tạo danh sách hành động
        moves.extend(["UP"] * (rotation))  # Thêm hành động xoay
        moves.extend(["RIGHT" if offset > newOffsetX else "LEFT"]
                     * abs(offset - newOffsetX))  # Thêm di chuyển
        return moves


class Agent:
    def __init__(self, turn):
        pass
        # self.weights = np.random.normal(0, 1, 34)  # Trọng số ban đầu ngẫu nhiên

    def choose_action(self, obs, env=None):
        board, piece, next_piece, offsetx = convert_state(obs)
        # print(np.array(board))
        field = Field(len(board[0]), len(board))
        # field.updateField(deepcopy(board))
        field.updateField(board.copy())
        moves = Ai.choose(field, piece, next_piece, offsetx, self.weights, env)
        numerical_actions = []
        for move in moves:
            if move == "UP":
                numerical_actions.append(4)  # Rotate
            elif move == "LEFT":
                numerical_actions.append(6)  # Move left
            elif move == "RIGHT":
                numerical_actions.append(5)  # Move right
        # If no moves, drop the piece
        numerical_actions.append(2)  # Drop

        return numerical_actions


env = TetrisSingleEnv(gridchoice="none", obs_type="grid", mode="human")

agent = Agent(0)
agent.weights = [-0.17213171976777253, 0.5217413003245039, -
                 0.6185933808608899, -0.06244111306508593]

done = False
total_reward = 0
total_count_step = 0
cleard_line = 0
used_block = 0

state = env.reset()
count_step = 0
images = []
pygame.event.get()
while not done:
    pygame.event.pump()

    actions = agent.choose_action(state, env)
    count_step += 1
    for action in actions:
        state, reward, done, infos = env.step(action)

    total_reward += reward
    total_count_step += 1
    used_block += 1

print("total_reward:", total_reward, " || ", "used_block:", used_block)
