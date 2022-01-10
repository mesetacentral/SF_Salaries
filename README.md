# San Francisco Salaries

#### Author: Ignacio Cobas
#### Dataset: https://www.kaggle.com/kaggle/sf-salaries

### Summary
I have explored this dataset regarding how much some employees in San Francisco made a year and then I fit a model to the data to try to predict the annual earnings. Since the only information available is the job's title and the employee's name, it was an NLP problem, so after cleaning up the data I performed a tfidf vectorization. After that, a number of models were tried and the best three were then taken for a hyperparameter tuning. 

### Dataset description
The dataset has 140k rows and 13 columns, although after the clean up and feature extraction, it ended up with 22k rows and 14 columns. 

### Objective
Our goal was to predict column TotalPay using JobTitle and/or EmployeeName.

### Preprocessing
Useless columns like 'Notes', 'Agency', 'Id' or 'Status' have been dropped, but others such as 'FirstName', 'JobArea' or 'Gender' have been added after the feature extraction. All employees with a salary lower than 21k have been disregarded, along with employees who share the job with less than 200 coworkers or whose gender could not be guessed. A tfidf vectorization was performed on the remaining data, which was then normalized.

### Demo
To try out the code, go to 'demo'. The model right now uses column JobTitle for prediction, but any of the 4 work, it just has to be changed in src/generate_features.py line 105/106.

### Conclusions
It is surprising that many of the models that we tried got about the same r2 score (0.6, 0.65), and what is even more surprising is that the best model hasn't improved one bit after the tuning, it's just as good as many others with the default parameters. This could mean that it can't be perfected any more with this kind of vectorization (tfidf), so another one should be used instead. I'd like to try that with a more powerful library, like pytorch, where I can build deep neural networks that could maybe help me get to at leat 0.7 r2 score.

### License 
This project has been developed under the license Attribution 4.0 International (CC BY 4.0).