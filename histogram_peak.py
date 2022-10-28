

class HistogramPeak:
    def __init__(self, peaks=None, range_length=100, parameter_theta=0.1):
        self.histogram = None
        self.peaks = peaks
        self.parameter_theta = parameter_theta

        self.valid_peak_counts = [0] * range_length
        self.value_ranges = self._init_value_range(range_length)

    def _init_value_range(self, range_length):
        value_ranges = list()

        interval_value = 1.0 / range_length
        for i in range(0, range_length):
            value_ranges.append(round(interval_value * (i + 1), 2))

        return value_ranges

    def extract_peak_count(self):
        for peak in self.peaks:
            value = peak.x.value

            if value < self.parameter_theta:
                continue

            index = -1
            for point in self.value_ranges:
                index += 1

                if value < point:
                    break

            self.valid_peak_counts[index] += 1

    def extract_feature(self):
        for value in self.histogram:
            if value < self.parameter_theta:
                continue

            index = -1
            for point in self.value_ranges:
                index += 1

                if value < point:
                    break

            self.valid_peak_counts[index] += 1

    def to_string(self, label):
        feature = '{0}'.format(label)

        index = 0
        for count in self.valid_peak_counts:
            feature += ' {0}:{1}'.format(index, count)

            index += 1

        return feature
