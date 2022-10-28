import numpy as np


class SimilarityCalculator:
    def __init__(self, h_histogram, l_histogram, peak_closeness_list, threshold_weight=0.1, threshold_closeness=0.1):
        self.h_histogram = h_histogram
        self.l_histogram = l_histogram
        self.peak_closeness_list = peak_closeness_list

        self.threshold_weight = threshold_weight
        self.threshold_closeness = threshold_closeness

    def get_similarity(self):
        similarity = Similarity(threshold_weight=self.threshold_weight, threshold_closeness=self.threshold_closeness)
        similarity.calc_similarity(self.h_histogram, self.l_histogram, self.peak_closeness_list)

        return similarity


class Similarity:
    def __init__(self, threshold_weight=0.1, threshold_closeness=0.1):
        self.threshold_weight = threshold_weight
        self.threshold_closeness = threshold_closeness

        self.r = 0

    def calc_similarity(self, h_histogram, l_histogram, peak_closeness_list):
        k1 = self.calc_k1(h_histogram.data, l_histogram.data)
        k2 = self.calc_k2(h_histogram.max_peaks, peak_closeness_list)
        k3 = self.calc_k3(h_histogram.max_peaks, peak_closeness_list)

        self.r = k1 * k2 * k3

    def calc_k1(self, h_data, l_data):
        x_length_h = len(h_data)
        x_length_l = len(l_data)

        area = 0
        max_length = max(x_length_h, x_length_l)
        for index in range(0, max_length):
            y_h = 0
            y_l = 0

            if x_length_h > index:
                y_h = h_data[index]

            if x_length_l > index:
                y_l = h_data[index]

            area += abs(y_h - y_l)

        area_sum_h = 0
        for y in h_data:
            area_sum_h += y

        return np.exp(area / (area_sum_h * max(h_data)) * -1)

    def calc_k2(self, peaks, peak_closeness_list):
        k2 = 0

        length = len(peaks)
        for i in range(0, length):
            k2 += peaks[i].weight.u * peak_closeness_list[i].c

        return k2

    def calc_k3(self, peaks, peak_closeness_list):
        k3 = 1

        length = len(peaks)
        for i in range(0, length):
            p = 1

            if (peaks[i].weight.u > self.threshold_weight) and (peak_closeness_list[i].c < self.threshold_closeness):
                p = 0

            k3 += p

        return k3
