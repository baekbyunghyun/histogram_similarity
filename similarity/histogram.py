import numpy as np

from similarity.attribute.herons_formula import *
from similarity.attribute.histogram_weight import *


class Histogram:
    def __init__(self, data, threshold_first_derivation=0.001, threshold_second_derivation=0.00001):
        self.data = data
        self.threshold_first_derivation = threshold_first_derivation
        self.threshold_second_derivation = threshold_second_derivation

        self.min_peaks = list()
        self.max_peaks = list()

    def new_set_peaks(self):
        gradients = np.gradient(self.data)
        inflection_points = np.gradient(gradients)

        max_gradient_indexes, min_gradient_indexes = self._get_zero_gradient_indexes(gradients)
        inflection_point_indexes = self._get_inflection_point_indexes(inflection_points)

        # bind first and second derivation
        binder = Binder(self.data, gradients, inflection_points)
        self.min_peaks, self.max_peaks = binder.bind(
            min_gradient_indexes, max_gradient_indexes, inflection_point_indexes)

        # set herons
        for peak in self.min_peaks:
            heron = HeronsFormula.heron(peak)
            peak.heron = heron

        for peak in self.max_peaks:
            heron = HeronsFormula.heron(peak)
            peak.heron = heron

        # set weight
        for peak in self.min_peaks:
            weight = HistogramWeight.weight(self.min_peaks, peak)
            peak.weight = weight

        for peak in self.max_peaks:
            weight = HistogramWeight.weight(self.max_peaks, peak)
            peak.weight = weight

    def _get_zero_gradient_indexes(self, gradients):
        max_gradient_indexes = list()
        min_gradient_indexes = list()

        index = 0
        is_positive_gradient = False
        is_negative_gradient = False
        for gradient in gradients:
            if gradient > 0:
                if is_negative_gradient is True:
                    min_gradient_indexes.append(index)

                is_positive_gradient = True
                is_negative_gradient = False

            else:
                if is_positive_gradient is True:
                    max_gradient_indexes.append(index)

                is_positive_gradient = False
                is_negative_gradient = True

            index += 1

        return max_gradient_indexes, min_gradient_indexes

    def _get_inflection_point_indexes(self, gradients):
        inflection_point_indexes = list()

        index = 0
        is_positive_gradient = False
        is_negative_gradient = False
        for gradient in gradients:
            if gradient > 0:
                if is_negative_gradient is True:
                    inflection_point_indexes.append(index)

                is_positive_gradient = True
                is_negative_gradient = False

            else:
                if is_positive_gradient is True:
                    inflection_point_indexes.append(index)

                is_positive_gradient = False
                is_negative_gradient = True

            index += 1

        return inflection_point_indexes

    def set_peaks(self):
        gradients = np.gradient(self.data)
        inflection_points = np.gradient(gradients)

        abs_gradients = [abs(gradient) for gradient in gradients]
        abs_gradients = np.array(abs_gradients)

        min_gradient_indexes = self._get_index_to_zero_gradient(abs_gradients,
                                                                self.threshold_first_derivation)

        min_first_derivation_indexes, max_first_derivation_indexes \
            = self._separate_min_max_gradients(self.data, min_gradient_indexes)

        abs_inflection_points = [abs(inflection_point) for inflection_point in inflection_points]
        abs_inflection_points = np.array(abs_inflection_points)

        inflection_point_indexes = self._get_index_to_zero_gradient(abs_inflection_points,
                                                                    self.threshold_second_derivation)

        # bind first and second derivation
        binder = Binder(self.data, abs_gradients, abs_inflection_points)
        self.min_peaks, self.max_peaks = binder.bind(
            min_first_derivation_indexes, max_first_derivation_indexes, inflection_point_indexes)

        # set herons
        for peak in self.min_peaks:
            heron = HeronsFormula.heron(peak)
            peak.heron = heron

        for peak in self.max_peaks:
            heron = HeronsFormula.heron(peak)
            peak.heron = heron

        # set weight
        for peak in self.min_peaks:
            weight = HistogramWeight.weight(self.min_peaks, peak)
            peak.weight = weight

        for peak in self.max_peaks:
            weight = HistogramWeight.weight(self.max_peaks, peak)
            peak.weight = weight

    def _get_index_to_zero_gradient(self, abs_gradients, threshold):
        minimum_gradient_indexes = np.where(np.asarray(abs_gradients) <= threshold)[0]
        minimum_gradient_indexes = self._remove_duplicate_indexes(abs_gradients, minimum_gradient_indexes)

        return minimum_gradient_indexes

    def _remove_duplicate_indexes(self, abs_gradients, minimum_gradient_indexes):
        removed_duplicate_indexes = []
        duplicate_indexes = []

        for index in minimum_gradient_indexes:
            if len(duplicate_indexes) < 1:
                duplicate_indexes.append(index)
                continue

            if duplicate_indexes[-1] != (index - 1):
                picked_index = self._pick_minimum_value(abs_gradients, duplicate_indexes)
                removed_duplicate_indexes.append(picked_index)

                del duplicate_indexes[:]

            duplicate_indexes.append(index)

        if len(duplicate_indexes) > 0:
            picked_index = self._pick_minimum_value(abs_gradients, duplicate_indexes)
            removed_duplicate_indexes.append(picked_index)

        return removed_duplicate_indexes

    def _pick_minimum_value(self, abs_gradients, indexes):
        gradients_from_indexes = abs_gradients[indexes]
        min_gradient_index = np.where(gradients_from_indexes == min(gradients_from_indexes))[0]

        return indexes[min_gradient_index[0]]

    def _separate_min_max_gradients(self, data, gradient_indexes):
        min_indexes = []
        max_indexes = []

        for index in gradient_indexes:
            before_index = index - 1
            after_index = index + 1

            if before_index < 0:
                before_index = 0

            if after_index > len(gradient_indexes) - 1:
                after_index = len(gradient_indexes) - 1

            before_value = data[before_index]
            after_value = data[after_index]
            value = data[index]

            if (value > before_value) and (value > after_value):
                max_indexes.append(index)

            else:
                min_indexes.append(index)

        return min_indexes, max_indexes


class Peak:
    def __init__(self):
        self.x = None
        self.x_left = None
        self.x_right = None
        self.heron = None
        self.weight = None


class Position:
    def __init__(self, index=-1, value=-1, first_derivation=-1, second_derivation=-1):
        self.index = index
        self.value = value
        self.first_derivation = first_derivation
        self.second_derivation = second_derivation


class Binder:
    def __init__(self, data, gradients, inflection_points):
        self.data = data
        self.gradients = gradients
        self.inflection_points = inflection_points

    def bind(self, min_peak_indexes, max_peak_indexes, inflection_point_indexes):
        min_peaks = list()
        max_peaks = list()

        for index in min_peak_indexes:
            peak = self._get_peak(index, inflection_point_indexes)
            if peak is None:
                continue

            min_peaks.append(peak)

        for index in max_peak_indexes:
            peak = self._get_peak(index, inflection_point_indexes)
            if peak is None:
                continue

            max_peaks.append(peak)

        return min_peaks, max_peaks

    def _get_peak(self, peak_index, inflection_point_indexes):
        left_inflection_point_index = -1
        right_inflection_point_index = -1

        for inflection_point_index in inflection_point_indexes:
            if inflection_point_index < peak_index:
                left_inflection_point_index = inflection_point_index

            else:
                right_inflection_point_index = inflection_point_index
                break

        if (left_inflection_point_index == -1) or (right_inflection_point_index == -1):
            return None

        peak = Peak()
        peak.x = Position(index=peak_index, value=self.data[peak_index],
                          first_derivation=self.gradients[peak_index],
                          second_derivation=self.inflection_points[peak_index])
        peak.x_left = Position(index=left_inflection_point_index, value=self.data[left_inflection_point_index],
                               first_derivation=self.gradients[left_inflection_point_index],
                               second_derivation=self.inflection_points[left_inflection_point_index])
        peak.x_right = Position(index=right_inflection_point_index, value=self.data[right_inflection_point_index],
                                first_derivation=self.gradients[right_inflection_point_index],
                                second_derivation=self.inflection_points[right_inflection_point_index])

        return peak
