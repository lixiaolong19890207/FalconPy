import math


class Point:
    def __init__(self, x: [int, float], y: [int, float]):
        self.coords = [x, y]

    @property
    def x(self):
        return self.coords[0]

    @property
    def y(self):
        return self.coords[1]

    @x.setter
    def x(self, v: [int, float]):
        self.coords[0] = v

    @y.setter
    def y(self, v: [int, float]):
        self.coords[1] = v

    def distance(self, other):
        assert len(self.coords) == len(other.coords), 'Unmatched point size!'
        return math.dist(self.coords, other.coords)

    def distance_patient(self, other, spacing):
        assert len(self.coords) == len(other.coords), 'Unmatched point size!'
        return math.sqrt(
            math.pow((self.x - other.x) * spacing.x, 2) +
            math.pow((self.y - other.y) * spacing.y, 2)
        )

    def __eq__(self, other, tolerance=0.05):
        return self.distance(other) <= tolerance

    def __add__(self, other):
        return Point(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        return Point(self.x - other.x, self.y - other.y)

    def __mul__(self, scale):
        return Point(self.x * scale, self.y * scale)

    def __getitem__(self, index):
        return self.coords[index]


class Point2(Point):
    pass


class Point3(Point):
    def __init__(self, x: [int, float], y: [int, float], z: [int, float]):
        super().__init__(x, y)
        self.coords.append(z)

    @property
    def z(self):
        return self.coords[2]

    @z.setter
    def z(self, v: [int, float]):
        self.coords[2] = v

    def distance_patient(self, other, spacing):
        assert len(self.coords) == len(other.coords), 'Unmatched point3 size!'
        return math.sqrt(
            math.pow((self.x - other.x) * spacing.x, 2) +
            math.pow((self.y - other.y) * spacing.y, 2) +
            math.pow((self.z - other.z) * spacing.z, 2)
        )

    def __add__(self, other):
        return Point3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other):
        return Point3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, scale):
        return Point3(self.x * scale, self.y * scale, self.z * scale)


