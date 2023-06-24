# Bank Marketing Data Analysis

This project performs several machine learning classification techniques on the bank marketing data to pick the best classification technique

## Setup

Download and install the latest [python](https://www.python.org/downloads/)

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install mysql, numpy, pandas, seaborn, matplotlib, imblearn and sklearn.

```bash
pip install numpy
pip install pandas
pip install seaborn
pip install matplotlib
pip install sklearn
pip install imblearn
pip install sqlalchemy
pip install pymysql
```

## Usage

We strongly recommend the usage of jupiter notebook to load the script bank_marketing_data.py into jupyter notebook and run it

If Jupyter notebook is not available, Please use below commands to run the script

After installing all the required python libraries, run the python script bank_marketing_data.py

```python
python3 bank_marketing_data.py
```
If above doesn't work, if alias is configured for python use the alias as shown below
```python
py bank_marketing_data.py
```

Once the script execution starts, 2 options are given to user
1) Option 1, to read data from CSV
    for this option the data set file bank.csv should be present in the same path as the script path
    if script wants to be run on full data set, please place bank-full.csv in the same path as script path and change the file name in line 16 of the script
2) Option 2, to read data from DB
    for this option mysql database should be up and running and connection string should be provided in line 15 of the script
    and also data should be present in the database table "bank_data"(for this script store_data_to_db.py can be run)
    Running script store_data_to_db.py
        1) Update the connection string in line 4
        2) Place the bank-full.csv in the same path as the script path and use below command
        ```python
            py bank_marketing_data.py
        ```

once the data is loaded into application, data analysis starts, and the results are printed on the console which identifies and  provides the best classification technique for the bank marketing data
