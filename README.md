# San Francisco Salaries

#### Author: Ignacio Cobas
#### Dataset: https://www.kaggle.com/kaggle/sf-salaries

### Summary
I have explored this dataset regarding how much some employees in San Francisco made a year and then I fit a model to the data to try to predict the annual earnings. Since the only information available is the job's title and the employee's name, it was an NLP problem, so after cleaning up the data I performed a tfidf vectorization. After that, a number of models were tried and the best three were then taken for a hyperparameter tuning. 

### Dataset description
The dataset has 140k rows and 13 columns, although after the clean up and feature extraction, it ended up with 22k rows and 14 columns. Most of the columns have
