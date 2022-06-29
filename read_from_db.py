import mysql.connector
import numpy
import pandas
from matplotlib import pyplot as plt

config = {
    'user': 'root',
    'password': 'admin',
    'host': 'localhost',
    'database': 'project_csci620',
    'raise_on_warnings': True
}


def connect_to_db(config):
    cnx = mysql.connector.connect(**config)
    return cnx


def readfile(filename):
    """
    reads the file and returns the data frame
    :return: the data frame
    """
    path = r"D:\YugiRIT\Courses\CSCI620-Intro To Big Data\Project\projStart\DataSet\bank (1)"
    path = path + r"\\"
    print(path + filename)
    data = pandas.read_csv(path + filename,
                           sep=";")
    return data


def insert_rec_into_db(row, connection):
    add_bank_data = "INSERT INTO bank_data_set " + \
                    "( age, job, marital_status, education," + \
                    "default_value, balance, housing, loan, contact, " + \
                    "day, month, duration, campaign, pdays, " + \
                    "previous, poutcome, outcome) " + \
                    "VALUES ( %s, %s, %s, %s, " + \
                    "%s, %s, %s, %s, %s, " + \
                    "%s, %s, %s, %s, %s, " + \
                    "%s, %s, %s) "
    cur = connection.cursor()
    default = False
    if row["default"] == "yes":
        default = True

    housing = False
    if row["housing"] == "yes":
        housing = True

    loan = False
    if row["loan"] == "yes":
        loan = True

    y = False
    if row["y"] == "yes":
        y = True

    type_parameters = (row["age"].item(), row["job"], row["marital"], row["education"],
                       default, row["balance"].item(), housing, loan, row["contact"],
                       row["day"].item(), row["month"], row["duration"].item(), row["campaign"].item(),
                       row["pdays"].item(),
                       row["previous"].item(), row["poutcome"], y)
    cur.execute(add_bank_data, type_parameters)


def main():
    bank_data = readfile()
    # print(bank_data)-
    cnx = connect_to_db(config)
    print(cnx.cmd_statistics())
    print(bank_data.iloc[1])
    print(len(bank_data))
    for i in range(len(bank_data)):
        insert_rec_into_db(bank_data.iloc[i], cnx)
        if i % 1000 == 0:
            cnx.commit()
    cnx.commit()
    cnx.close()


def plain_data_process():
    bank_data = readfile("bank-full.csv")
    print(bank_data.describe())
    descriptive_stats_analysis(bank_data)


def ouliers_in_data_frame(data):
    quartile1 = data.quantile(0.25)
    quartile3 = data.quantile(0.75)
    inter_quartile_range = quartile3 - quartile1
    lower_quart_outliers = quartile1 - (inter_quartile_range * 1.5)
    upper_quart_outliers = quartile3 + (inter_quartile_range * 1.5)
    return lower_quart_outliers, upper_quart_outliers


def descriptive_stats_analysis(data):
    # Find all numerical columns
    numeric_columns = data.select_dtypes(include=numpy.number).columns
    print(numeric_columns)
    # Find all categorical columns
    # All non numerical columns are considered as categorical columns
    categorical_columns = data.select_dtypes(include=object).columns
    print(categorical_columns)

    # Listing all categories
    for column in categorical_columns:
        print('Number of categories in {} are {}'.format(column, len(data[column].unique())))
    lower_quart_outliers, upper_quart_outliers = ouliers_in_data_frame(data)
    print("The outliers values in Lower Quartile are ", "\n", lower_quart_outliers)
    print("The outliers values in Upper Quartile are ", "\n", upper_quart_outliers)
    axis_index = 0
    fig, axes = plt.subplots(7, 1, figsize=(8, 25))
    for column in numeric_columns:
        print_outliers(upper_quart_outliers, lower_quart_outliers, data, column)
        f = data[[column]].boxplot(ax=axes[axis_index], vert=False)
        axis_index += 1

    plt.show()


def print_outliers(upper_quart_outliers, lower_quart_outliers, data, column_name):
    upper_quart_outliers_age = numpy.where(data[column_name] >= upper_quart_outliers[column_name])
    lower_quart_outliers_age = numpy.where(data[column_name] <= lower_quart_outliers[column_name])
    print("Lower Quartile outliers for {} are {}:".format(column_name, lower_quart_outliers_age))
    print("Upper Quartile outliers for {} are {}:".format(column_name, upper_quart_outliers_age))


if __name__ == '__main__':
    plain_data_process()
