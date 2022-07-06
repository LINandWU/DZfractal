
import os
import logging
import math
import pandas as pd

import numpy as np
from sklearn.linear_model import LinearRegression

# ------------ #
# Setup logger #
# ------------ #
logger = logging.getLogger("dtrzrcstat")
formatter = logging.Formatter("[%(levelname)s] %(message)s")
logger.setLevel(logging.INFO)
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.ERROR)
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)


SIGNIFICANT_PEARSON_R = [
    1.000, 1.000, 1.000, 1.000, 1.000,
    1.000, 0.929, 0.881, 0.833, 0.794,
    0.755, 0.727, 0.703, 0.679, 0.654,
    0.635, 0.618, 0.600, 0.584, 0.570,
    0.556, 0.544, 0.532, 0.521, 0.511,
    0.501, 0.492, 0.483, 0.475, 0.467,
    0.459, 0.452, 0.446, 0.439, 0.433,
    0.427, 0.421, 0.415, 0.410, 0.405,
    0.400, 0.396, 0.391, 0.386, 0.382,
    0.378, 0.374, 0.370, 0.366, 0.363,
    0.363, 0.356, 0.356, 0.349, 0.349,
    0.343, 0.343, 0.337, 0.337, 0.331,
    0.331, 0.326, 0.326, 0.321, 0.321,
    0.316, 0.316, 0.311, 0.311, 0.307,
    0.307, 0.303, 0.303, 0.299, 0.299,
    0.295, 0.295, 0.291, 0.291, 0.287,
    0.287, 0.284, 0.284, 0.280, 0.280,
    0.277, 0.277, 0.274, 0.274, 0.271,
    0.271, 0.268, 0.268, 0.265, 0.265,
    0.262, 0.262, 0.260, 0.260, 0.257]

FLOATING_POINT_TOL = 1.0e-12
def linear_regression(x, y):
    '''
    Do linear regression of 1d-array of x and y data
    Return the slope, y-intercept and R-squared score
    '''
    if isinstance(x, np.ndarray):
        xx = x[:]
    elif isinstance(x, list):
        xx = np.array(x)
    else:
        raise RuntimeError("Unexpected input type for x {}".format(type(x)))

    xx = x.reshape(-1, 1)
    model = LinearRegression().fit(xx, y)
    k = model.coef_[0]
    intercept = model.intercept_
    Rsq = model.score(xx, y)
    Rsq = int(Rsq*10000)/10000

    return k, intercept, Rsq


class DetritalAgeData:
    '''
    This class is a wrapper over an np.array of detrital zircon best ages.
    Analytical errors are ignored.
    The age spectrum will be sorted upon loading.
    '''

    def __init__(self):
        self._name = ''
        self._data = []
        self._count_nodes = []
        self._counts = []
        self._counting_method = "Undefined"

    @property
    def name(self):
        return self._name

    @property
    def data(self):
        return self._data

    @property
    def counts(self):
        return self._counts

    @property
    def count_nodes(self):
        return self._count_nodes

    @data.setter
    def data(self, array):
        if len(array):
            self._data = array
            self._data.sort()  # ensure ages are sorted.
        else:
            raise RuntimeError("Input array is empty.")

    @count_nodes.setter
    def count_nodes(self, value):
        if isinstance(value, list):
            self._count_nodes = np.array(value)
        elif isinstance(value, np.ndarray):
            self._count_nodes = value
        else:
            raise RuntimeError("Unknown count nodes data")

    def load_data_txt(self, path, sample_name=None):
        '''
        Load data directly from txt file.
        '''
        if sample_name is None:
            sample_name = os.path.splitext(os.path.basename(path))[0]
        self.data = np.loadtxt(path).flatten()
        self._name = sample_name
        logger.info("Dataset %s is loaeded with %d ages from txt file %s.",
                    self._name, len(self.data), path)
        return self

    def load_data_xls(self, path, sheet_name,
                      header="Best Age", sample_name=None):
        '''
        Load data from excel spreadsheet
        '''
        if sample_name is None:
            sample_name = sheet_name
        array = pd.read_excel(path, sheet_name)[header].values
        self.data = array
        self._name = sample_name
        logger.info("Dataset [%s] is loaded with %d ages from excel file [%s] sheet [%s].",
                    self._name, len(self.data), path, sheet_name)
        return self

    def min(self):
        '''
        Return the minimum age of the dataset
        '''
        return self.data[0]

    def max(self):
        '''
        Return the maximum age of the dataset
        '''
        return self.data[-1]

    def generate_count_nodes(self, bin_size, age_minmax=None):
        '''
        Partition the data into counting nodes by bin_size.
        age_minmax should be an array of [age_min, age_max]
        If age_minmax is not given, paritition will be done automatically.
        '''
        if age_minmax is None:
            min_age = int(self.data[0]/bin_size) * bin_size
            max_age = int(self.data[-1]/bin_size) * bin_size
        else:
            min_age, max_age = min(age_minmax), max(age_minmax)
        self._count_nodes = np.arange(min_age, max_age + bin_size, bin_size)
        return self._count_nodes

    def count_cumulative(self, normalize=True):
        '''
        Do counting of data at each counts nodes
        '''
        counts = []
        self._counting_method = "cumulative"
        for node in self.count_nodes:
            counts.append(len([x for x in self.data if x >= node]))

        if normalize:
            self._counts = np.array(counts) / len(self.data)
        else:
            self._counts = np.array(counts)
        return self._counts

    def count_between_nodes(self, normalize=True):
        self._counting_method = "between_nodes"
        counts = []

        for age_min, age_max in zip(
                self.count_nodes[:-1], self.count_nodes[1:]):

            counts.append(
                len([x for x in self.data if age_min <= x < age_max]))

        if normalize:
            self._counts = np.array(counts) / len(self.data)
        else:
            self._counts = np.array(counts)
        return self._counts

    def uniform_log_linear_regression(self,
                                      n_nodes_per_segment):
        '''
        Perform linear regression of log(nodal age)-log(count)
        at every n_nodes_per_segment nodes.
        Then return the slope, intercept and R^2 at each counting node.
        Counting nodes belonging to the same segment
        will have the same slope, intercept, and R^2.
        '''
        n_count_nodes = len(self.count_nodes)
        n_data_points = len(self.data)
        n_segments = math.ceil((n_count_nodes - 1) / (n_nodes_per_segment - 1))

        Rsq = []
        slope = []
        intercept = []

        logger.debug("Linear regression will be performed on "
                     "%d intervals for sample [%s]", n_segments, self._name)

        # nodes with 0-count are reduced to 1 to avoid -inf when taking log
        log_counts = np.array(self.counts)
        log_counts[log_counts == 0] = 1
        log_counts = np.log(log_counts)
        log_count_nodes = np.log(self.count_nodes)

        for i in range(0, n_segments):
            idx_low = i * (n_nodes_per_segment - 1)
            idx_upp = min((i + 1) * (n_nodes_per_segment - 1) +
                          1, n_count_nodes)
            n_intervals = idx_upp - idx_low - 1

            try:
                segment_slope, segment_intercept, segment_Rsq =  \
                    linear_regression(
                        log_count_nodes[idx_low: idx_upp], log_counts[idx_low: idx_upp])
            except ValueError as err:
                print(i, n_segments, idx_low, idx_upp, n_nodes_per_segment,
                      n_data_points)  # print current state for debugging
                raise err

            # append the data for this segment to the global list
            slope += list(np.ones((n_intervals,)) * segment_slope)
            intercept += list(np.ones((n_intervals,)) * segment_intercept)
            Rsq += list(np.ones((n_intervals,)) * segment_Rsq)

        return slope, intercept, Rsq

    @staticmethod
    def _segmental_adaptive_regression(idx0,
                                       log_count_nodes, log_counts,
                                       n_nodes_min, delta_R, R_lower_threshold):
        '''
        Do adaptive log-log regression for one segment.
        Return the slope, intercept, R and current index
        '''
        assert len(log_count_nodes) == len(log_counts)
        n_all_nodes = len(log_count_nodes)

        # lambda function to do linear regression
        # for points [left, right)
        def regress(left, right): 
            return linear_regression(
                log_count_nodes[left:right], log_counts[left:right])

        #  idx1 = min(n_nodes_min + idx0, n_all_nodes)
        idx1 = min(n_nodes_min + idx0 + 1, n_all_nodes)
        n_intervals = idx1 - idx0 - 1

        # preliminary linear regression
        slope_old, intercept_old, R_old = regress(idx0, idx1)

        R_middle = R_old
        while idx1 < n_all_nodes:
            # include one more point and do the linear regression again
            slope_new, intercept_new, R_new = regress(idx0, idx1+1)

            if (R_new < R_lower_threshold and R_middle < R_lower_threshold) or \
                    R_old < R_lower_threshold:
                break
            if R_middle - R_new > delta_R and \
                    R_middle - R_new < 0.99 and \
                    idx0 < 40:
                break
            if (R_old > R_middle and R_middle > R_new
                    and R_new > 0.900 and R_old - R_middle >= 0.0021
                    and R_old - R_new < 0.03) \
                    or (R_old > R_middle and R_middle > R_new and R_new < 0.9 and R_new > 0.85 and R_old - R_middle < R_middle - R_new and R_middle - R_new > 0.015):
                break
            if idx0 >= 10 and idx0 < 47 and \
                    R_middle - R_new > delta_R and \
                    log_counts[idx0+2] != log_counts[idx1]:
                break
            if log_counts[idx1] == 0 or \
                (R_old < R_middle and R_middle > R_new and R_old - R_new > 0.00051 and idx0 != 35) or \
                    (R_old < R_middle and R_middle > R_new and R_middle - R_new < 0.00071):
                break
            if R_old < R_middle and R_middle > R_new and \
                    R_middle > 0.985 and R_middle - R_old > 0.004 and \
                    R_middle - R_old < 0.005:
                break

            R_old = R_middle
            R_middle = R_new

            # move on to the next point
            idx1 += 1
            slope_old = slope_new
            intercept_old = intercept_new
            n_intervals += 1

        # append to the global list
        slope = np.ones((n_intervals,)) * slope_old
        intercept = np.ones((n_intervals,)) * intercept_old
        R = np.ones((n_intervals, 1)) * R_middle
        return slope, intercept, R, idx1-1

    def adaptive_log_linear_regression(self, n_nodes_min=3, delta_R=0.05, R_lower_threshold=0.9):
        '''
        Adaptive linear regression on the log-log plot.
        n_nodes_min: the regression will take place for at least that many nodes
        delta_R: if by adding more points the change in R^2 is over this limit,
                stop adding more points.
        R_lower_thredshold: if R^2 is below this value, stop adding more points.
        '''
        R = []
        slope = []
        intercept = []
        log_count_nodes = np.log(np.array(self.count_nodes))
        log_counts = np.array(self.counts)
        log_counts[log_counts == 0] = 1
        log_counts = np.log(log_counts)

        idx = 0
        while idx < len(self.count_nodes) - 1:
            local_slope, local_intercept, local_R, idx = \
                DetritalAgeData._segmental_adaptive_regression(
                    idx, log_count_nodes, log_counts, n_nodes_min, delta_R, R_lower_threshold)

            # append to the list of slopes and R
            slope += list(local_slope)
            R += list(local_R)
            intercept += list(local_intercept)


        assert len(slope) == len(log_count_nodes) - 1
        assert len(intercept) == len(log_count_nodes) - 1
        assert len(R) == len(log_count_nodes) - 1

        return slope, intercept, R
