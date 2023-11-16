# Predict Creditworthiness of Personal Loan Borrowers Using Machine Learning Models
## Objective
Develop a machine learning model to predict credit worthiness of borrowers.

## Business Understanding
Creditworthiness prediction is of paramount importance in the financial industry as it plays a crucial role in mitigating risks and ensuring the stability of lending institutions. By harnessing advanced analytics and machine learning algorithms, financial institutions can assess the creditworthiness of potential borrowers more accurately. This proactive approach enables lenders to identify and flag high-risk individuals or businesses, reducing the likelihood of defaults. Predictive models analyze a numerous variables, such as credit history, income stability, loan premium, LTV etc. and providing a comprehensive evaluation of a borrower's ability to repay a loan. Timely prediction of loan defaults not only protects financial institutions from potential financial losses but also fosters responsible lending practices. Ultimately, the ability to forecast loan defaults empowers lenders to make informed decisions, maintain a healthy loan portfolio, and contribute to the overall stability of the financial system.


## Data Understanding and Analysis

**Source of Data**
The source for data is personal loan portfolio provided in the [Kaggle](https://www.kaggle.com/datasets/gauravduttakiit/bank-fears-loanliness).

**Instruction to save the database and run notebooks**
1. Download the database from [Kaggle](https://www.kaggle.com/datasets/gauravduttakiit/bank-fears-loanliness)
2. There are three files in the download. Out of which select **train_indessa.csv** file and save the file in the **Data folder** in the name of **Loan_Default.csv**

**Description of Data**

The dataset contained 532K records of customer loan portfolio. The dataset contained 26 variables covering loan amount, interest rate,term, loan purpose, application status, verification status,dti etc. Also contained applicants charactoristics such as credit gradings, state, home ownership, no of inquiries, revolving balances etc. The dataset contained 16 numerical variables, 9 categorical and 1 boolean data column. 

**Feature Engineering**

The dataset didnot contained loan installment. Therefore calculated the loan installment using numpy method. Also derived different related ratios annual loan installment to annual income and loan to annual income ratios.

##Method
**Data Clensing**

Started the analysis understanding the data and premilinary EDA for better understanding the behaviour of the data set. Variable wise dig through to understand data behaviour and identified strategies like mean, median, most frequent to handle missing values based on the distributions of data. The imputations were done using pipeline methodology after segregating the data frame into train and test subsets. Then clense train and test data set exported to Data folder.

The detail process of data cleansing and imputation can access through the [EDA and Data_Clensing Notebook](https://github.com/yasiSriLanka/dsc-capstone-loan-default-prediction/blob/main/EDA%20and%20Data_Clensing.ipynb)

** Model Development






## Fitted models are
1. Dummy Model
2. Logistic Regression
3. Logistic Regression with polinomial features
4. Gradient Boost
5. XGBoost
6. XGBoost with SMOTE
7. Random Forest
8. Random Forest with SMOTE
9. K-Nearest Neighbours
10. Stacking(XG Boost & Logistic)
11. Stacking(XG Boost & Random Forest)
12. Stacking(XG Boost, Random Forest & Logistic)

The detail coding on python of Model training, hyper parameter tunning, evaluation can be access with the [link](https://github.com/yasiSriLanka/dsc-capstone-loan-default-prediction/blob/main/Loan%20Default%20Prediction%20Model.ipynb). 

## Results
1. Comparison of Model Performance
<img width="536" alt="Model Comparison" src="https://github.com/yasiSriLanka/dsc-capstone-loan-default-prediction/assets/141664072/fefa10c5-02c3-4112-bc10-3fe4b5e404b8">

2. DefaultShield : XG Boost Classifier Model Performance on Unseen Data
   
   <img width="560" alt="Final Model Performance" src="https://github.com/yasiSriLanka/dsc-capstone-loan-default-prediction/assets/141664072/4c161705-a9fa-4e9f-b8b9-2de16308fb38">

   

## Conclusion
It is recommended XGBoost as the best model with test score of 0.819 and logloss of 0.402 on unseen data for creditworthiness prediction. The stacking models had perfomed slightly higher with test 0.821 and logloss of 0.405. Considering the complexitity of stacking model it was decided to conclude XGBoost as the best model algorithm.

###Final Recommendation : DefaultShield
XGBoost classifier with following parameters

booster='gbtree'

learning_rate=0.1

max_depth=10

n_estimators = 175

reg_lambda =1

colsample_bytree=0.5

subsample=0.8

The model achieved accuracy of 0.82 with f1 score on crossvalidation of 0.531. The reported logloss on unseen data was 0.40.
