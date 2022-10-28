
class HistogramWeight:
    @staticmethod
    def weight(peaks, target_peak):
        denominator = HistogramWeight.get_denominator(peaks)

        weight = Weight()
        weight.calc_weight(target_peak, denominator)

        return weight

    @staticmethod
    def get_denominator(peaks):
        denominator = 0

        for peak in peaks:
            denominator += abs(peak.x.second_derivation) * peak.heron.l

        return denominator


class Weight:
    def __init__(self):
        self.u = 0

    def calc_weight(self, peak, denominator):
        molcule = abs(peak.x.second_derivation) * peak.heron.l

        self.u = molcule / denominator
