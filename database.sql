

#drop table project_csci620.bank_data_set;

CREATE TABLE project_csci620.bank_data_set (
    record_id int primary key,
    age int,
    job varchar(255),
    marital_status varchar(255),
    education varchar(255),
    default_value boolean,
    balance float,
    housing boolean,
    loan boolean,
    contact varchar(255),
    day int,
    month varchar(255),
    duration int,
    campaign int,
    pdays int, 
    previous int,
    poutcome varchar(255),
    outcome boolean
    
);
