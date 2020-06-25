from math import sqrt

import math


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y


class Segment:
    def __init__(self, point1, point2, line=None, intersection_point=None):
        self.a = point1
        self.b = point2
        self.line = line
        self.intersection_point = intersection_point

    def sq_shortest_dist_to_point(self, other_point):
        dx = self.b.x - self.a.x
        dy = self.b.y - self.a.y
        dr2 = float(dx ** 2 + dy ** 2)

        lerp = ((other_point.x - self.a.x) * dx + (other_point.y - self.a.y) * dy) / dr2
        if lerp < 0:
            lerp = 0
        elif lerp > 1:
            lerp = 1

        x = lerp * dx + self.a.x
        y = lerp * dy + self.a.y

        _dx = x - other_point.x
        _dy = y - other_point.y
        square_dist = _dx ** 2 + _dy ** 2
        return square_dist

    def shortest_dist_to_point(self, other_point):
        if self.a.x == self.b.x and self.a.y == self.b.y:
            return math.sqrt((other_point.x-self.a.x)**2 + (other_point.y-self.a.y)**2)
        dist = None
        try:
            dist = math.sqrt(self.sq_shortest_dist_to_point(other_point))
        except:
            pass
        return dist


def get_dist_from_point_to_segment(point, line):
    x0 = point[0]
    y0 = point[1]
    a = line.coefficients[0]
    b = -1
    c = line.coefficients[1]
    dist = abs(a * x0 + b * y0 + c) / sqrt(a ** 2 + b ** 2)
    return dist


def timing(f):
    def wrap(*args):
        import time
        time1 = time.time()
        ret = f(*args)
        time2 = time.time()
        print('%s function took %0.3f ms' % (f.__name__, (time2-time1)*1000.0))
        return ret
    return wrap


if __name__ == "__main__":
    segments_and_point_and_answer = [
        [Segment(Point(1.0, 1.0), Point(1.0, 1.0)), Point(2.0, 2.0), math.sqrt(2.0)],
        [Segment(Point(1.0, 1.0), Point(1.0, 3.0)), Point(2.0, 3.0), 1.0],
        [Segment(Point(0.0, 0.0), Point(0.0, 3.0)), Point(1.0, 1.0), 1.0],
        [Segment(Point(1.0, 1.0), Point(3.0, 3.0)), Point(2.0, 2.0), 0.0],
        [Segment(Point(-1.0, -1.0), Point(3.0, 3.0)), Point(2.0, 2.0), 0.0],
        [Segment(Point(1.0, 1.0), Point(1.0, 3.0)), Point(2.0, 3.0), 1.0],
        [Segment(Point(1.0, 1.0), Point(1.0, 3.0)), Point(2.0, 4.0), math.sqrt(2.0)],
        [Segment(Point(1.0, 1.0), Point(-3.0, -3.0)), Point(-3.0, -4.0), 1],
        [Segment(Point(1.0, 1.0), Point(-3.0, -3.0)), Point(-4.0, -3.0), 1],
        [Segment(Point(1.0, 1.0), Point(-3.0, -3.0)), Point(1, 2), 1],
        [Segment(Point(1.0, 1.0), Point(-3.0, -3.0)), Point(2, 1), 1],
        [Segment(Point(1.0, 1.0), Point(-3.0, -3.0)), Point(-3, -1), math.sqrt(2.0)],
        [Segment(Point(1.0, 1.0), Point(-3.0, -3.0)), Point(-1, -3), math.sqrt(2.0)],
        [Segment(Point(-1.0, -1.0), Point(3.0, 3.0)), Point(3, 1), math.sqrt(2.0)],
        [Segment(Point(-1.0, -1.0), Point(3.0, 3.0)), Point(1, 3), math.sqrt(2.0)],
        [Segment(Point(1.0, 1.0), Point(3.0, 3.0)), Point(3, 1), math.sqrt(2.0)],
        [Segment(Point(1.0, 1.0), Point(3.0, 3.0)), Point(1, 3), math.sqrt(2.0)],
        [Segment(Point(3.0, 0.0), Point(0.0, -5.0)), Point(-4, 5), math.sqrt(74)],
        [Segment(Point(3.0, 0.0), Point(0.0, -5.0)), Point(0, -6), 1],
        [Segment(Point(3.0, 0.0), Point(0.0, -5.0)), Point(0, -5), 0],
        [Segment(Point(0.0, 0.0), Point(1.0, 0.0)), Point(0, -5), 5],
        [Segment(Point(0.0, 0.0), Point(1.0, 0.0)), Point(-1, -5), math.sqrt(26)],
        [Segment(Point(0.0, 1.0), Point(0.0, 0.0)), Point(-1, -5), math.sqrt(26)],
        [Segment(Point(0.0, 1.0), Point(0.0, 0.0)), Point(3, 2), math.sqrt(10)]
    ]

    for i, (segment, point, answer) in enumerate(segments_and_point_and_answer):
        result = segment.shortest_dist_to_point(point)
        print(result)
        assert (abs(result - answer) <= 0.001)
