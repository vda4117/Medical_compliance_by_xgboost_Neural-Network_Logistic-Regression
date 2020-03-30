# Medical_compliance
It is Machine Learning project. In this we train the model to predict, is patients following prescription or not. In this project, data is given about patient, that includes patient id, Age, Gender, is patients diabetes, Alcoholism, Hypertension, Smokes, Tuberculosis and the number of remainder sms per day. The targeted variable is Adherence.
 
patient_id: Unique ID of each patient.
Age: Age of patient (in years).
Gender: Gender of patient (F- Female, M- Male).
Prescription_period: The number of days, patient have to follow the prescription. 
Diabetes: Is patient have diabetes or not (0: No, 1: Yes).
Alcoholism: Is patient alcoholic (0: No, 1: Yes).
HyperTension: Is patient have hypertension (0: No, 1: Yes). 
Smokes: Is patient have addiction of smoking (0: No, 1: Yes).
Tuberculosis: Is patient have TB (0: No, 1: Yes). 
Sms_Remainder: The number of remainder sms send to patients per day.
Adherence: It is targeted variable. We have to find out is patients following prescription or not. 


The operation we have done as below:

Data Cleaning: Data was already in clean state.

Data Transformation: Transfer the object data type to int. Here only two columns (Gender and Adherence) are of object data type.

Correlation: Check the correlated columns and remove the highly correlated columns. Here we have ‘Diabetes’ and ‘Hypertension’ are highly correlated with ‘Age’ column so I removed ‘Diabetes’ and ‘Hypertension’ columns. Also, similar for ‘Alcoholism’. 

Data Splitting: Data is split into Train and Dev set.  We train a model on Train set and cross-verified on Dev set and Test set.
 
Model formation: We have use three different algorithms for modeling.
    1. Logistic Regression
    2. Xgboost
    3. Neural Network (That algorithm is developed by our team at Vchip Technology)


Prediction: 
    • On the basis of accuracy, recall, precision and f1-score, we decide which algorithm is doing better for this task.

Code file name:  
1. Medical_compliance_using_LogisticRegression.ipynb for Logistic Regression modeling.
2. Medical_compliance_using_xgboost.ipynb for xgboost modeling.
3. Medical_compliance_using_NN.ipynb for Neural network modeling. 

Thank you.
