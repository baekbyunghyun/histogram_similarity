import numpy as np


class ClosenessCalculator:
    def __init__(self, target_histogram, parameter_delta_x=0.1, parameter_delta_y=0.1,
                 parameter_n=0.1, parameter_epsilon=0.0001):
        self.target_histogram = target_histogram
        self.parameter_delta_x = parameter_delta_x
        self.parameter_delta_y = parameter_delta_y
        self.parameter_n = parameter_n
        self.parameter_epsilon = parameter_epsilon

    def get_closeness_histogram(self, histogram):
        max_peak_closeness_list = list()
        min_peak_closeness_list = list()

        for peak in self.target_histogram.max_peaks:
            closeness = self.get_closeness_peak(peak, histogram.max_peaks)

            max_peak_closeness_list.append(closeness)

        for peak in self.target_histogram.min_peaks:
            closeness = self.get_closeness_peak(peak, histogram.min_peaks)

            min_peak_closeness_list.append(closeness)

        return max_peak_closeness_list, min_peak_closeness_list

    def get_closeness_peak(self, target_peak, peaks):
        closeness_list = list()

        for peak in peaks:
            closeness = Closeness(self.parameter_delta_x, self.parameter_delta_y, self.parameter_n, self.parameter_epsilon)
            closeness.calc_closeness(target_peak, peak, len(self.target_histogram.max_peaks), len(peaks))

            closeness_list.append(closeness)

        max_closeness = closeness_list.pop(0)
        for closeness in closeness_list:
            if max_closeness.c < closeness.c:
                max_closeness = closeness

        return max_closeness


class Closeness:
    def __init__(self, parameter_delta_x=0.1, parameter_delta_y=0.1, parameter_n=0.1, parameter_epsilon=0.0001):
        self.parameter_delta_x = parameter_delta_x
        self.parameter_delta_y = parameter_delta_y
        self.parameter_n = parameter_n
        self.parameter_epsilon = parameter_epsilon

        self.c = 0

    def calc_closeness(self, h_peak, l_peak, num_of_h_peak, num_of_l_peak):
        closeness_of_x = self.calc_closeness_of_x(h_peak, l_peak)
        closeness_of_y = self.calc_closeness_of_y(h_peak, l_peak)
        closeness_of_form = self.calc_closeness_of_form(h_peak, l_peak, num_of_h_peak, num_of_l_peak)

        self.c = closeness_of_x * closeness_of_y * closeness_of_form

    def calc_closeness_of_x(self, h_peak, l_peak):
        x_of_h_peak = h_peak.x.index
        x_of_l_peak = l_peak.x.index

        difference = x_of_h_peak - x_of_l_peak

        return np.exp(pow(difference / self.parameter_delta_x, 2) * -1)

    def calc_closeness_of_y(self, h_peak, l_peak):
        y_of_h_peak = h_peak.x.value
        y_of_l_peak = l_peak.x.value

        difference = y_of_h_peak - y_of_l_peak

        return np.exp(pow(difference / self.parameter_delta_y, 2) * -1)

    def calc_closeness_of_form(self, h_peak, l_peak, num_of_h_peak, num_of_l_peak):
        f = self.get_f(h_peak, l_peak)

        return np.exp((f * self.parameter_n) / (num_of_h_peak + num_of_l_peak) * -1)

    def get_f(self, h_peak, l_peak):
        f = (abs(h_peak.x_left.first_derivation - l_peak.x_left.first_derivation)
             / (abs(h_peak.x_left.first_derivation - l_peak.x_left.first_derivation) + self.parameter_epsilon))

        f += (abs(h_peak.x_right.first_derivation - l_peak.x_right.first_derivation)
              / (abs(h_peak.x_right.first_derivation - l_peak.x_right.first_derivation) + self.parameter_epsilon))

        f += abs((h_peak.x_right.index - h_peak.x_left.index) - (l_peak.x_right.index - l_peak.x_left.index)) \
             / (abs(h_peak.x_right.index - h_peak.x_left.index) - (l_peak.x_right.index - l_peak.x_left.index) + self.parameter_epsilon)

        return f
