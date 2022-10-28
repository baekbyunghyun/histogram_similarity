from scipy.stats import gaussian_kde
from similarity.histogram import *
from similarity.calculator.closeness_calculator import *
from similarity.calculator.similarity_calculator import *


class Data:
    def __init__(self, label, indexes, values):
        self.label = label
        self.indexes = indexes
        self.values = values

    def __str__(self):
        return 'Label: {0} \nindexes: {1} \nvalues: {2}\n'.format(self.label, self.indexes, self.values)

    @staticmethod
    def load_data(file_path):
        with open(file_path, 'rb') as file_descriptor:
            lines = file_descriptor.readlines()

        datas = list()

        for line in lines:
            line = line.strip()

            attributes = line.split(' ')
            label = attributes[0]

            indexes = list()
            values = list()

            for i in range(1, len(attributes)):
                index = attributes[i].split(':')[0]
                value = attributes[i].split(':')[1]

                indexes.append(index)
                values.append(value)

            datas.append(Data(label, indexes, values))

        return datas


def convert_smooth_graph(data, num_of_attr):
    graph_src = []

    for i in range(0, num_of_attr):
        value = int(float(data.values[i]) * 100)
        value = abs(value)

        graph_src += ([i] * value)

    density = gaussian_kde(graph_src)
    density.covariance_factor = lambda: .25
    density._compute_covariance()

    xs = np.linspace(0, 10, 256)

    return density(xs), xs


if __name__ == '__main__':
    data_file_path = 'dataset/breast_cancer.spa'
    delta_x = 0.01
    delta_y = 0.01
    n = 0.1
    epsilon = 0.0001
    threshold_weight = 0.1
    threshold_closeness = 0.1
    result_dir_path = 'result'

    datas = Data.load_data(data_file_path)

    count_similarity = 0
    count_different = 0

    for i in range(0, len(datas)):
        src_data = datas[i]
        src_label = src_data.label

        for j in range(0, len(datas)):
            dest_data = datas[j]
            dest_label = dest_data.label

            if src_label != dest_label:
                continue

            h_data, h_xs = convert_smooth_graph(src_data, 10)
            h_histogram = Histogram(h_data)
            h_histogram.new_set_peaks()

            l_data, l_xs = convert_smooth_graph(dest_data, 10)
            l_histogram = Histogram(l_data)
            l_histogram.new_set_peaks()

            closenessCalculator = ClosenessCalculator(h_histogram, parameter_delta_x=delta_x,
                                                      parameter_delta_y=delta_y, parameter_n=n,
                                                      parameter_epsilon=epsilon)
            max_peak_closeness_list, min_peak_closeness_list = closenessCalculator.get_closeness_histogram(l_histogram)

            similarityCalculator = SimilarityCalculator(h_histogram, l_histogram, max_peak_closeness_list,
                                                        threshold_weight=threshold_weight,
                                                        threshold_closeness=threshold_closeness)
            similarity = similarityCalculator.get_similarity()

            if similarity.r <= 0.01:
                count_different += 1

            else:
                count_similarity += 1

        print('\n--------------- RESULT ----------------')
        print('Number of test:          {0}'.format(count_similarity + count_different))
        print('Number of similarity:    {0}'.format(count_similarity))
        print('Number of different:     {0}'.format(count_different))
        print()
