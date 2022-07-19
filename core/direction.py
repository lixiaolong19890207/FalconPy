import math
import numpy as np


class Direction:
    def __init__(self, x: [int, float] = 1, y: [int, float] = 0):
        self.coords = [x, y]

    @property
    def x(self):
        return self.coords[0]

    @property
    def y(self):
        return self.coords[1]

    def __mul__(self, scale):
        return Direction(self.x * scale, self.y * scale)

    def __getitem__(self, index):
        return self.coords[index]

    def dot(self, other):
        return np.dot(self.coords, other.coords)

    def length(self):
        _sum = 0
        for i in self.coords:
            _sum += math.pow(i, 2)
        return math.sqrt(_sum)

    def normalize(self):
        len = self.length()
        self.coords = [x * (1.0 / self.length()) for x in self.coords]


class Direction2(Direction):
    def __init__(self, x: [int, float], y: [int, float], z: [int, float]):
        super().__init__(x, y)
        self.normalize()


class Direction3(Direction):
    def __init__(self, x: [int, float], y: [int, float], z: [int, float]):
        super().__init__(x, y)
        self.coords.append(z)
        self.normalize()

    @property
    def z(self):
        return self.coords[2]

    def __mul__(self, scale):
        return Direction3(self.x * scale, self.y * scale, self.z * scale)

    def __neg__(self):
        return self * -1

    def cross(self, other):
        return np.cross(self.coords, other.coords)

