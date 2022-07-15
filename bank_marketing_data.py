# import mysql.connector
import numpy
import pandas
import seaborn
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import  confusion_matrix, ConfusionMatrixDisplay
from sklearn import metrics

config = {
    'user': 'root',
    'password': 'admin',
    'host': 'localhost',
    'database': 'project_csci620',
    'raise_on_warnings': True
}


# def connect_to_db(config):
#     cnx = mysql.connector.connect(**config)
#     return cnx


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
    numeric_columns, categorical_columns, data = descriptive_stats_analysis(bank_data)
    data = data_pre_process(numeric_columns, categorical_columns, bank_data)
    split_the_data(data, numeric_columns)


def split_the_data(data, numeric_columns):
    features = [feat for feat in data.columns if feat != 'y']

    x = data[numeric_columns]  # feature set
    y = data['y']  # target

    # Splitting data into train and test
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)

    # train and test datasets dimensions
    print(x_train.shape, x_test.shape)

    train_decision_tree_classifier(x_train, x_test, y_train, y_test)
    train_logistic_regression(x_train, x_test, y_train, y_test)
    train_k_neighbors_classifier(x_train, x_test, y_train, y_test)

def train_decision_tree_classifier(x_train, x_test, y_train, y_test):
    from sklearn.tree import DecisionTreeClassifier
    from sklearn import metrics

    deseciontree_model = DecisionTreeClassifier(max_depth=10, random_state=40)
    deseciontree_model.fit(x_train, y_train)
    y_predicted_deseciontree = deseciontree_model.predict(x_test)
    y_predicted_deseciontree
    deseciontree_model.score(x_test, y_test)
    dtc_accuracy = metrics.accuracy_score(y_test, y_predicted_deseciontree)
    dtc_precision = metrics.precision_score(y_test, y_predicted_deseciontree)
    dtc_recall = metrics.recall_score(y_test, y_predicted_deseciontree)
    print("Accuracy:", metrics.accuracy_score(y_test, y_predicted_deseciontree))
    print("Precision:", metrics.precision_score(y_test, y_predicted_deseciontree))
    print("Recall:", metrics.recall_score(y_test, y_predicted_deseciontree))

    cm = confusion_matrix(y_test, y_predicted_deseciontree)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()

    plt.show()


def train_logistic_regression(x_train, x_test, y_train, y_test):
    from sklearn.linear_model import LogisticRegression

    lgr_model = LogisticRegression(C=10, random_state=40)
    lgr_model.fit(x_train, y_train)
    y_predicted_lgr = lgr_model.predict(x_test)
    y_predicted_lgr
    lgr_model.score(x_test, y_test)
    lgr_accuracy = metrics.accuracy_score(y_test, y_predicted_lgr)
    lgr_precision = metrics.precision_score(y_test, y_predicted_lgr)
    lgr_recall = metrics.recall_score(y_test, y_predicted_lgr)
    print("Accuracy:", metrics.accuracy_score(y_test, y_predicted_lgr))
    print("Precision:", metrics.precision_score(y_test, y_predicted_lgr))
    print("Recall:", metrics.recall_score(y_test, y_predicted_lgr))

    cm = confusion_matrix(y_test, y_predicted_lgr)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()

    plt.show()


def train_k_neighbors_classifier(x_train, x_test, y_train, y_test):
    from sklearn.neighbors import KNeighborsClassifier

    KNN_model = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
    KNN_model.fit(x_train, y_train)
    y_predicted_knn = KNN_model.predict(x_test)
    y_predicted_knn
    KNN_model.score(x_test, y_test)
    knn_accuracy = metrics.accuracy_score(y_test, y_predicted_knn)
    knn_precision = metrics.precision_score(y_test, y_predicted_knn)
    knn_recall = metrics.recall_score(y_test, y_predicted_knn)
    print("Accuracy:", metrics.accuracy_score(y_test, y_predicted_knn))
    print("Precision:", metrics.precision_score(y_test, y_predicted_knn))
    print("Recall:", metrics.recall_score(y_test, y_predicted_knn))

    cm = confusion_matrix(y_test, y_predicted_knn)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()

    plt.show()

def ouliers_in_data_frame(data):
    quartile1 = data.quantile(0.25)
    quartile3 = data.quantile(0.75)
    inter_quartile_range = quartile3 - quartile1
    lower_quart_outliers = quartile1 - (inter_quartile_range * 1.5)
    upper_quart_outliers = quartile3 + (inter_quartile_range * 1.5)
    return lower_quart_outliers, upper_quart_outliers


def data_pre_process(numeric_columns, categorical_columns, data):
    count_of_term_deposits(data)
    print(data.isnull().sum())
    print(data.duplicated().sum())
    correlation = data.corr()
    print(correlation)
    figure = plt.figure(figsize=(18, 10))
    seaborn.heatmap(correlation, annot=True)
    plt.show()
    data = encode_data(data)
    print(data)
    data = normalization(data, numeric_columns)
    return data


def normalization(data, numerical_columns):
    min_max_scaler = MinMaxScaler(feature_range=(0, 1))
    data[numerical_columns] = min_max_scaler.fit_transform(data[numerical_columns])
    print(data[numerical_columns])
    return data


def encode_data(data):
    label_encoding = {
        "y": {"no": 0, "yes": 1},
        "poutcome": {"unknown": 0, "failure": 1, "success": 2, "other": 3},
        "month": {"jan": 0, "feb": 1, "mar": 2, "apr": 3, "may": 4, "jun": 5, "jul": 6, "aug": 7, "sep": 8, "oct": 9,
                  "nov": 10, "dec": 11},
        "contact": {"unknown": 0, "cellular": 1, "telephone": 2},
        "loan": {"no": 0, "yes": 1},
        "housing": {"no": 0, "yes": 1},
        "default": {"no": 0, "yes": 1},
        "education": {"unknown": 0, "primary": 1, "secondary": 2, "tertiary": 3},
        "marital": {"single": 0, "married": 1, "divorced": 2},
        "job": {"unknown": 0, "technician": 1, "entrepreneur": 2, "blue-collar": 3, "management": 4, "retired": 5,
                "admin.": 6, "services": 7, "self-employed": 8, "unemployed": 9, "housemaid": 10, "student": 11}
    }
    return data.replace(label_encoding)


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
    # for column in numeric_columns:
    #     print_outliers(upper_quart_outliers, lower_quart_outliers, data, column)
    #     f = data[[column]].boxplot(ax=axes[axis_index], vert=False)
    #     axis_index += 1
    # plt.show()
    # seaborn.pairplot(data, hue='y', corner=True)
    # target_variable_numerical_features_graph(numeric_columns, data)
    # kernel_denstiy_estimation(numeric_columns, data)
    # count_based_on_categorical_features(data, categorical_columns)
    return numeric_columns, categorical_columns, data


def count_of_term_deposits(data):
    counts = data.y.value_counts()
    print("Total term deposits opened:", counts["yes"])
    print("Total term deposits not opened:", counts["no"])
    total_rec = counts['yes'] + counts['no']
    print("percentage of term deposits opened: ", counts['yes'] / total_rec * 100)
    print("percentage of term deposits not opened: ", counts['no'] / total_rec * 100)


def count_based_on_categorical_features(data, categorical_columns):
    plt.figure(figsize=(15, 80), facecolor='white')
    plotnum = 1
    plotnum = 1
    for cat in categorical_columns:
        axis = plt.subplot(12, 3, plotnum)
        axis.tick_params(axis='x', rotation=90)
        seaborn.countplot(x=cat, data=data)
        plt.xlabel(cat)
        plt.title(cat)
        plotnum += 1

    plt.subplots_adjust(hspace=0.5)
    plt.show()


def kernel_denstiy_estimation(numeric_columns, data):
    # fig, axes = plt.subplots(3, 1, figsize=(8, 25))
    # for index, column in enumerate(numeric_columns):
    #     if index == 3:
    #         break
    #     f = data[[column]].plot(kind='kde', ax=axes[index])
    # plt.show()

    fig, axes = plt.subplots(7, 1, figsize=(8, 25))
    for index, column in enumerate(numeric_columns):
        # if index > 3:
        f = data[[column]].plot(kind='kde', ax=axes[index - 4])
    plt.show()


def categorical_columns_plot(data, categorical_columns):
    print("hi")


def target_variable_numerical_features_graph(numeric_columns, data):
    plt.figure(figsize=(20, 60))
    plotnumber = 1
    for feature in numeric_columns:
        ax = plt.subplot(12, 3, plotnumber)
        seaborn.boxplot(x="y", y=data[feature], data=data)
        plt.xlabel(feature)
        plotnumber += 1
    plt.show()


def print_outliers(upper_quart_outliers, lower_quart_outliers, data, column_name):
    upper_quart_outliers_age = numpy.where(data[column_name] >= upper_quart_outliers[column_name])
    lower_quart_outliers_age = numpy.where(data[column_name] <= lower_quart_outliers[column_name])
    print("Lower Quartile outliers for {} are {}:".format(column_name, lower_quart_outliers_age))
    print("Upper Quartile outliers for {} are {}:".format(column_name, upper_quart_outliers_age))


if __name__ == '__main__':
    plain_data_process()
