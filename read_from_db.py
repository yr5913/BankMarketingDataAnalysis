import mysql.connector
import pandas

config = {
    'user': 'root',
    'password': 'admin',
    'host': 'localhost',
    'database': 'project_csci620',
    'raise_on_warnings': True
}


def connect_to_db(config):
    cnx = mysql.connector.connect(**config)
    return cnx;


def readfile():
    """
    reads the file and returns the data frame
    :return: the data frame
    """
    data = pandas.read_csv(r"D:\YugiRIT\Courses\CSCI620-Intro To Big Data\Project\projStart\DataSet\bank (1)\bank.csv",
                           sep=";")
    return data


def insert_rec_into_db(row, connection):
    add_bank_data("INSERT INTO bank_data_set "
               "(record_id, age, job, marital_status, education, default_value,"
                  ") "
               "VALUES (%s, %s, %s, %s, %s)")


def main():
    bank_data = readfile()
    print(bank_data)
    cnx = connect_to_db(config)
    print(cnx.cmd_statistics())
    cnx.close()


if __name__ == '__main__':
    main()
