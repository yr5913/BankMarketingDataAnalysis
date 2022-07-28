import pandas
from sqlalchemy import create_engine

db_connect_string = 'mysql+pymysql://root:admin@localhost/project_csci620'


def insert_into_db():
    table_name = "bank_data"
    bank_data = readfile("bank-full.csv")
    sql_engine = create_engine(db_connect_string, pool_recycle=3600)
    db_connection = sql_engine.connect()
    frame = bank_data.to_sql(table_name, db_connection, if_exists='fail')
    db_connection.close()


def readfile(filename):
    """
    reads the file and returns the data frame
    :return: the data frame
    """
    # path = r"D:\Courses\CSCI620-Intro To Big Data\Project\projStart\DataSet\bank (1)"
    # path = path + r"\\"
    # print(path + filename)
    data = pandas.read_csv(filename,
                           sep=";")
    return data


if __name__ == '__main__':
    insert_into_db()
