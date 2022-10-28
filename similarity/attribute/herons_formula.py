import math


class HeronsFormula:
    @staticmethod
    def herons(histogram):
        max_gradient_herons = list()
        min_gradient_herons = list()

        for peak in histogram.max_peaks:
            max_gradient_herons.append(HeronsFormula.heron(peak))

        for peak in histogram.min_peaks:
            min_gradient_herons.append(HeronsFormula.heron(peak))

        return max_gradient_herons, min_gradient_herons

    @staticmethod
    def heron(peak):
        herons = Herons()
        herons.apply_herons(peak)

        return herons


class Herons:
    def __init__(self):
        self.A = 0
        self.s = 0
        self.a_distance = 0
        self.b_distance = 0
        self.c_distance = 0
        self.l = 0

    def apply_herons(self, peak):
        self._calc_distances(peak)
        self._calc_s()
        self._calc_A()

        self.l = (2 / self.c_distance) * self.A

    def _calc_distances(self, peak):
        self.a_distance = self._get_manhattan_distance((peak.x.index, peak.x.value),
                                                       (peak.x_right.index, peak.x_right.value))

        self.b_distance = self._get_manhattan_distance((peak.x.index, peak.x.value),
                                                       (peak.x_left.index, peak.x_left.value))

        self.c_distance = self._get_manhattan_distance((peak.x_left.index, peak.x_left.value),
                                                       (peak.x_right.index, peak.x_right.value))

    def _calc_s(self):
        self.s = (self.a_distance + self.b_distance + self.c_distance) / 2

    def _calc_A(self):
        calc_a = self.s - self.a_distance
        calc_b = self.s - self.b_distance
        calc_c = self.s - self.c_distance

        ret = self.s * calc_a * calc_b * calc_c
        ret = abs(ret)

        self.A = math.sqrt(ret)

    def _get_manhattan_distance(self, start, end):
        sx, sy = start
        ex, ey = end

        return abs(ex - sx) + abs(ey - sy)
