# import mysql.connector
from collections import Counter

import numpy
import pandas
import seaborn
from imblearn.over_sampling import SMOTE
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import plot_roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

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


def oversample_the_data(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=42)
    smote = SMOTE()
    x_train_os, y_train_os = smote.fit_resample(x_train, y_train)
    print("The number of Classes before fit {}".format(Counter(y_train)))
    print("The number of Classes after fit {}".format(Counter(y_train_os)))
    print('not_deposited :', y_train_os.value_counts()[0] / len(y_train_os) * 100, '%')
    print('deposited: ', y_train_os.value_counts()[1] / len(y_train_os) * 100, '%')
    return x_train_os, x_test, y_train_os, y_test


def split_the_data(data, numeric_columns):
    features = [feat for feat in data.columns if feat != 'y']

    x = data[numeric_columns]  # feature set
    y = data['y']  # target

    # Splitting data into train and test
    # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)
    x_train, x_test, y_train, y_test = oversample_the_data(x, y)
    # train and test datasets dimensions
    print(x_train.shape, x_test.shape)
    models_list = []

    models_list.append(train_gaussianNB(x_train, x_test, y_train, y_test))
    models_list.append(train_decision_tree_classifier(x_train, x_test, y_train, y_test))
    models_list.append(train_random_forest_classifier(x_train, x_test, y_train, y_test))
    models_list.append(train_k_neighbors_classifier(x_train, x_test, y_train, y_test))
    models_list.append(train_logistic_regression(x_train, x_test, y_train, y_test))
    models_list.append(train_stochastic_gradient_descent(x_train, x_test, y_train, y_test))

    models = []
    model_data_frame = pandas.DataFrame()
    for i in models_list:
        models.append(i[0])
        dictionary = {}
        dictionary['Model'] = i[4]
        dictionary['Accuracy'] = i[1]
        dictionary['Precision'] = i[2]
        dictionary['Recall'] = i[3]
        model_data_frame = model_data_frame.append(dictionary, ignore_index=True)
    print(model_data_frame)

    ax = plt.gca()
    for i in models:
        plot_roc_curve(i, x_test, y_test, ax=ax)
    plt.show()

    cm = seaborn.light_palette('seagreen', as_cmap=True)
    s = model_data_frame.style.background_gradient(cmap=cm)
    print(s)

    plt.figure(figsize=(20, 5))
    seaborn.set(style="whitegrid")
    ax = seaborn.barplot(y='Accuracy', x='Model', data=model_data_frame)
    plt.show()


def train_stochastic_gradient_descent(x_train, x_test, y_train, y_test):
    from sklearn.linear_model import SGDClassifier

    sgd_model = SGDClassifier(loss='modified_huber', shuffle=True, random_state=101)
    sgd_model.fit(x_train, y_train)
    y_predicted_sgd = sgd_model.predict(x_test)
    y_predicted_sgd
    sgd_model.score(x_test, y_test)
    sgd_accuracy = metrics.accuracy_score(y_test, y_predicted_sgd)
    sgd_precision = metrics.precision_score(y_test, y_predicted_sgd)
    sgd_recall = metrics.recall_score(y_test, y_predicted_sgd)
    print("Accuracy of the StochasticGradient model:", metrics.accuracy_score(y_test, y_predicted_sgd))
    print("Precision of the StochasticGradient model:", metrics.precision_score(y_test, y_predicted_sgd))
    print("Recall of the StochasticGradient model:", metrics.recall_score(y_test, y_predicted_sgd))

    cm = confusion_matrix(y_test, y_predicted_sgd)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()

    plt.show()
    return sgd_model, sgd_accuracy, sgd_precision, sgd_recall, "StochasticGradient"


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
    print("Accuracy of the DescisionTree model:", metrics.accuracy_score(y_test, y_predicted_deseciontree))
    print("Precision of the DescisionTree model:", metrics.precision_score(y_test, y_predicted_deseciontree))
    print("Recall of the DescisionTree model:", metrics.recall_score(y_test, y_predicted_deseciontree))

    cm = confusion_matrix(y_test, y_predicted_deseciontree)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()

    plt.show()
    return deseciontree_model, dtc_accuracy, dtc_precision, dtc_recall, "DescisionTree"


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
    print("Accuracy of the LogisticRegression model:", metrics.accuracy_score(y_test, y_predicted_lgr))
    print("Precision of the LogisticRegression model:", metrics.precision_score(y_test, y_predicted_lgr))
    print("Recall of the LogisticRegression model:", metrics.recall_score(y_test, y_predicted_lgr))

    cm = confusion_matrix(y_test, y_predicted_lgr)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()

    plt.show()
    return lgr_model, lgr_accuracy, lgr_precision, lgr_recall, "LogisticRegression"


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
    print("Accuracy of the KNeighborsClassifier model:", metrics.accuracy_score(y_test, y_predicted_knn))
    print("Precision of the KNeighborsClassifier model:", metrics.precision_score(y_test, y_predicted_knn))
    print("Recall of the KNeighborsClassifier model:", metrics.recall_score(y_test, y_predicted_knn))

    cm = confusion_matrix(y_test, y_predicted_knn)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()

    plt.show()
    return KNN_model, knn_accuracy, knn_precision, knn_recall, "KNeighborsClassifier"


def train_gaussianNB(x_train, x_test, y_train, y_test):
    from sklearn.naive_bayes import GaussianNB

    gnb_model = GaussianNB()
    gnb_model.fit(x_train, y_train)
    y_predicted_gnb = gnb_model.predict(x_test)
    y_predicted_gnb
    gnb_model.score(x_test, y_test)
    gnb_accuracy = metrics.accuracy_score(y_test, y_predicted_gnb)
    gnb_precision = metrics.precision_score(y_test, y_predicted_gnb)
    gnb_recall = metrics.recall_score(y_test, y_predicted_gnb)
    print("Accuracy of the GaussianNB model:", metrics.accuracy_score(y_test, y_predicted_gnb))
    print("Precision of the GaussianNB model:", metrics.precision_score(y_test, y_predicted_gnb))
    print("Recall of the GaussianNB model:", metrics.recall_score(y_test, y_predicted_gnb))

    cm = confusion_matrix(y_test, y_predicted_gnb)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()

    plt.show()
    return gnb_model, gnb_accuracy, gnb_precision, gnb_recall, "GaussianNB"


def train_random_forest_classifier(x_train, x_test, y_train, y_test):
    from sklearn.ensemble import RandomForestClassifier

    rf_model = RandomForestClassifier(n_estimators=10)
    rf_model.fit(x_train, y_train)
    y_predicted_rf = rf_model.predict(x_test)
    y_predicted_rf
    rf_model.score(x_test, y_test)
    rf_accuracy = metrics.accuracy_score(y_test, y_predicted_rf)
    rf_precision = metrics.precision_score(y_test, y_predicted_rf)
    rf_recall = metrics.recall_score(y_test, y_predicted_rf)
    print("Accuracy of the RandomForestClassifier model:", metrics.accuracy_score(y_test, y_predicted_rf))
    print("Precision of the RandomForestClassifier model:", metrics.precision_score(y_test, y_predicted_rf))
    print("Recall of the RandomForestClassifier model:", metrics.recall_score(y_test, y_predicted_rf))

    cm = confusion_matrix(y_test, y_predicted_rf)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()

    plt.show()
    return rf_model, rf_accuracy, rf_precision, rf_recall, "RandomForestClassifier"


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
    print(data)
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
    for column in numeric_columns:
        print_outliers(upper_quart_outliers, lower_quart_outliers, data, column)
        f = data[[column]].boxplot(ax=axes[axis_index], vert=False)
        axis_index += 1
    plt.show()
    seaborn.pairplot(data, hue='y', corner=True)
    target_variable_numerical_features_graph(numeric_columns, data)
    kernel_denstiy_estimation(numeric_columns, data)
    count_based_on_categorical_features(data, categorical_columns)
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
    fig, axes = plt.subplots(7, 1, figsize=(8, 25))
    for index, column in enumerate(numeric_columns):
        # if index > 3:
        f = data[[column]].plot(kind='kde', ax=axes[index - 4])
    plt.show()


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
