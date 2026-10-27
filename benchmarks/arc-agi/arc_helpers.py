from dataclasses import dataclass
from enum import IntEnum
from typing import List, Optional, Set, Tuple, Union


class Color(IntEnum):
    BLACK = 0
    BLUE = 1
    RED = 2
    GREEN = 3
    YELLOW = 4
    GREY = 5
    MAGENTA = 6
    ORANGE = 7
    CYAN = 8
    MAROON = 9


@dataclass(frozen=True)
class Point:
    r: int
    c: int


@dataclass
class GridObject:
    pixels: Set[Tuple[Point, int]]

    @property
    def bounding_box(self) -> Tuple[int, int, int, int]:
        rows = [p.r for p, _ in self.pixels]
        cols = [p.c for p, _ in self.pixels]
        if not rows:
            return (0, 0, 0, 0)
        return (min(rows), min(cols), max(rows), max(cols))

    def move(self, dr: int, dc: int) -> 'GridObject':
        new_pixels = {(Point(p.r + dr, p.c + dc), color) for p, color in self.pixels}
        return GridObject(pixels=new_pixels)

    def change_color(self, new_color: int) -> 'GridObject':
        new_pixels = {(p, new_color) for p, _ in self.pixels}
        return GridObject(pixels=new_pixels)


class Grid:
    def __init__(self, matrix: List[List[int]]):
        self.matrix = [list(row) for row in matrix]
        self.height = len(matrix)
        self.width = len(matrix[0]) if self.height > 0 else 0

    def __array__(self, dtype=None, copy=None):
        import numpy
        return numpy.array(self.matrix, dtype=dtype)

    def extract_objects(self, connectivity: int = 4, ignore_background: bool = True) -> List[GridObject]:
        visited = set()
        objects = []
        dr = [-1, 1, 0, 0, -1, -1, 1, 1]
        dc = [0, 0, -1, 1, -1, 1, -1, 1]
        num_dirs = 8 if connectivity == 8 else 4

        for r in range(self.height):
            for c in range(self.width):
                val = self.matrix[r][c]
                if (r, c) in visited or (ignore_background and val == 0):
                    continue
                pixels = set()
                queue = [(r, c)]
                visited.add((r, c))
                target_color = val

                while queue:
                    curr_r, curr_c = queue.pop(0)
                    pixels.add((Point(curr_r, curr_c), target_color))
                    for d in range(num_dirs):
                        nr, nc = curr_r + dr[d], curr_c + dc[d]
                        if 0 <= nr < self.height and 0 <= nc < self.width:
                            if (nr, nc) not in visited and self.matrix[nr][nc] == target_color:
                                visited.add((nr, nc))
                                queue.append((nr, nc))
                objects.append(GridObject(pixels=pixels))
        return objects

    def filter_by_color(self, color: int) -> List[GridObject]:
        return [obj for obj in self.extract_objects(ignore_background=False)
                if any(c == color for _, c in obj.pixels)]

    def to_matrix(self) -> List[List[int]]:
        return self.matrix


DSL_NAMES = {
    "Color": Color,
    "Point": Point,
    "GridObject": GridObject,
    "Grid": Grid,
    "List": List,
    "Tuple": Tuple,
    "Set": Set,
    "Optional": Optional,
    "Union": Union,
}
