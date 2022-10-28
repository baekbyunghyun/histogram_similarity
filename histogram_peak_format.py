

class HistogramPeakFormat:
    TYPE_OF_NORMAL = 'normal'
    TYPE_OF_ALL_PEAKS = 'all'
    TYPE_OF_MAXIMUM_PEAKS = 'maximum'
    TYPE_OF_HERONS = 'herons'
    TYPE_OF_WEIGHT = 'weight'

    def __init__(self, histogram=None, map_level=100, format_type='normal', threshold=0.0):
        self.histogram = histogram
        self.map_level = map_level
        self.format_type = format_type
        self.threshold = threshold

        self.features = list()

    def _init_value_range(self, range_length):
        value_ranges = list()

        interval_value = 1.0 / range_length
        for i in range(0, range_length):
            value_ranges.append(round(interval_value * (i + 1), 2))

        return value_ranges

    def extract_feature(self):
        if self.format_type == HistogramPeakFormat.TYPE_OF_ALL_PEAKS:
            self.features = self._extract_all_feature()

        elif self.format_type == HistogramPeakFormat.TYPE_OF_MAXIMUM_PEAKS:
            self.features = self._extract_maximum_feature()

        elif self.format_type == HistogramPeakFormat.TYPE_OF_HERONS:
            self.features = self._extract_herons_feature()

        elif self.format_type == HistogramPeakFormat.TYPE_OF_WEIGHT:
            self.features = self._extract_weight_feature()

        else:
            self.features = self._extract_normal_feature()

    def _extract_all_feature(self):
        features = list()
        for _ in range(0, self.map_level):
            features.append([0] * len(self.histogram.data))

        value_range = self._init_value_range(self.map_level)

        max_peaks = self.histogram.max_peaks
        for peak in max_peaks:
            index_x = peak.x.index
            value = peak.x.value

            index_y = -1
            for point in value_range:
                index_y += 1

                if value < point:
                    break

            features[index_y][index_x] += value

        return features

    def _extract_maximum_feature(self):
        features = list()
        for _ in range(0, self.map_level):
            features.append([0] * len(self.histogram.data))

        value_range = self._init_value_range(self.map_level)

        max_peaks = self.histogram.max_peaks
        for peak in max_peaks:
            index_x = peak.x.index
            value = peak.x.value

            if value < self.threshold:
                continue

            index_y = -1
            for point in value_range:
                index_y += 1

                if value < point:
                    break

            features[index_y][index_x] += value

        return features

    def _extract_herons_feature(self):
        features = list()
        for _ in range(0, self.map_level):
            features.append([0] * len(self.histogram.data))

        value_range = self._init_value_range(self.map_level)

        max_peaks = self.histogram.max_peaks
        for peak in max_peaks:
            index_x = peak.x.index
            heron_l = peak.heron.l

            if heron_l < self.threshold:
                continue

            index_y = -1
            for point in value_range:
                index_y += 1

                if heron_l < point:
                    break

            features[index_y][index_x] += heron_l

        return features

    def _extract_weight_feature(self):
        features = list()
        for _ in range(0, self.map_level):
            features.append([0] * len(self.histogram.data))

        value_range = self._init_value_range(self.map_level)

        max_peaks = self.histogram.max_peaks
        for peak in max_peaks:
            index_x = peak.x.index
            weight = peak.weight.u

            if weight < self.threshold:
                continue

            index_y = -1
            for point in value_range:
                index_y += 1

                if weight < point:
                    break

            features[index_y][index_x] += weight

        return features

    def _extract_normal_feature(self):
        features = list()
        features.append(self.histogram.data)

        return features

    def to_string(self, label):
        feature_str = '{0}'.format(label)

        index = 0
        for feature in self.features:
            for count in feature:
                feature_str += ' {0}:{1}'.format(index, count)

                index += 1

        return feature_str
