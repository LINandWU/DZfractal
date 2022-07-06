import pandas as pd
import dzfractal as dz
import numpy as np

# the file path of the input excel file
excel_file_name = "DZRawData.xlsx"

# list all samples in the file
dataset_names = pd.ExcelFile(excel_file_name).sheet_names

bin_size = 50  # count data every bin_size Ma
regression_method = "adaptive"  # adaptive or uniform
normalize_data = False

start_age = 10
end_age = 2560
count_nodes = np.arange(start_age, end_age + bin_size, bin_size)

# -----------------------------------
# if using uniform regression grid
# -----------------------------------
# this is No. of intervals in each segment
uniform_regression_interval = 5

# ----------------------------------
# if using adaptive regression grid
# ----------------------------------
# each regression segment have at least that many nodes
min_nodes_for_regression = 2
# if changes in R^2 is greater than this,
# do not add more points and finish regression for this segment
delta_R = 0.03
# if value of R^2 is smaller than this,
# do not add more points and finish regression for this segment
R_lower_threshold = 0.5


data_slopes = {}
# slopes
for data_name in dataset_names:
    data = dz.DetritalAgeData()
    data.count_nodes = count_nodes
    data.load_data_xls(excel_file_name, data_name)
    data.count_cumulative(normalize_data)
    negative_slopes, _, _ = data.adaptive_log_linear_regression(
        min_nodes_for_regression, delta_R, R_lower_threshold)
    data_slopes[data_name] = [-round(1000*x)/1000 for x in negative_slopes]


df = pd.DataFrame(data_slopes, columns = dataset_names)
df.to_csv("DZSlopes.csv")