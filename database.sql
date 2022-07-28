

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
CREATE TABLE `test5` (
  `index` bigint DEFAULT NULL,
  `age` bigint DEFAULT NULL,
  `job` text,
  `marital` text,
  `education` text,
  `default` text,
  `balance` bigint DEFAULT NULL,
  `housing` text,
  `loan` text,
  `contact` text,
  `day` bigint DEFAULT NULL,
  `month` text,
  `duration` bigint DEFAULT NULL,
  `campaign` bigint DEFAULT NULL,
  `pdays` bigint DEFAULT NULL,
  `previous` bigint DEFAULT NULL,
  `poutcome` text,
  `y` text,
  KEY `ix_test5_index` (`index`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
