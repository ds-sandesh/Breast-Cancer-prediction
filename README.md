            PREDICTING BREAST CANCER DIAGNOSIS USING TUNED SOFT VOTING MACHINE LEARNING ENSEMBLE


Breast cancer is a major issue worldwide. Finding it in an early stage can save a life. This paper looks at using machine learning to predict if the breast tumor is benign (Not cancer) or malignant (cancer). For this project the dataset used is Wisconsin Breast cancer dataset. At first, basic Machine learning models: Naïve Bayes, Logistic Regression, K-Nearest Neighbors (KNN) and support vector machine (SVM) were tested. Then we combined them into a soft voting ensemble. To make the model even better we used Grid Search to tune the internal setting (hyperparameters) of the ensemble. The tuned model reached a cross-validation accuracy of about 96.7% and a test accuracy of 94.7% accuracy. While the tuned modal performed very well, it showed a slight drop in test accuracy compared to the default ensemble, which is a sign of minor overfitting. Still the modal achieved 100% precision for malignant cases in our base tests meaning it is very reliable at avoiding false alarms.
Keywords: Breast Cancer, Machine Learning, Soft Tuning, Hyperparameters Tuning, Ensemble Learning.


Problem Statement:
Breast cancer is one of the most common cancers in women. Doctors use tests like biopsies to check tumors, but reading these cells images can sometimes be tricky. Human error or fatigue can lead to wrong diagnosis or delayed treatment.

Motivation:
We need reliable computers tools to help doctors make faster and more accurate decision A good machine learning model can act as a second opinion reducing mistakes and mistakes for patients.

Objectives:
The main objective of this study is to build and compare different machine learning models to classify breast cancer tumors. We also want to see if combining these modals into an ensemble and tuning their settings gives better result than using them with default setting.

Research Question:
Does tuning the hyperparameters of a soft voting ensemble improve its performance in predicting breast cancer diagnosis compared to using default modal setting?


We used the Breast Cancer Wisconsin (Diagnostic) Dataset. It consists of 569 records of breast mass sample. Each record has 32 columns. The first one is the Id number, second one is the diagnosis (M for malignant and B for benign) and the remaining are numerical values.
These 30 features are calculated from digitized image of a Fine Needle Aspirate (FNA) of a breast mass. They describe characteristics of the cell nuclei, such as radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry and fractal dimensions. For each of these 10 features the mean, standard error and worst (largest) value is provided.
Before modelling we did simple preprocessing:
1.	We dropped ID column as it doesn’t help us in prediction.
2.	We separated diagnosis column as our target variable.
3.	We checked for duplicate rows and found none, so no data was removed.
4.	We used Simple Imputer filling missing values with median also we checked for missing values too and found none.
5.	We applied StandardScaler to make sure that all features have similar range. This helps modals like KNN and SVM to work better.



We tested four individual Machine learning algorithms:
1.	Naïve Bayes (NB):
A simple probabilistic modal which assumes that the features are dependent. It works fast and well for small data.
2.	Logistic Regression (LR):
A linear modal that predicts the probability of a tumor to be malignant.
3.	K-Nearest Neighbors:
A modal that classifies the tumor based on majority of its closest neighbor in its feature space.
4.	Support Vector Machine (SVM):
A powerful modal that finds the best boundary to separate the benign and malignant tumors.

To improve the productivity of these modal, we introduced soft voting classifier and combined all these four models in it. Here each modal predicts the probability of tumor being malignant and the final prediction is the average of these probabilities.
To find the best possible version of this ensemble, we used GridSearch cross-validation. These modal tests many different combinations of settings for each model to see which performs best. We built a machine learning pipeline that first applies the imputer, then the standard scaler and finally the classifier. This prevents the data leakage from the test set into the training process.


Before building our modals, we checked data visually to understand what we are working with. It helps us to see patterns and problems that number alone might not show.

1.	Class Distribution:
First, we checked how many benign and malignant data are there in our dataset Figure 1 shows this distribution. The dataset has more benign (357) set than malignant (212). This imbalance is common in medical dataset because benign tumors are more frequently diagnosed than malignant.
This imbalance is important to note because machine learning can sometimes be biased towards the majority section. That’s why we use stratified sampling while splitting our data which ensures both our training and testing dataset have the same proportion of benign and malignant cases as the original dataset.

 <img width="3600" height="1800" alt="feature_boxplots" src="https://github.com/user-attachments/assets/12dacc69-4c1c-4584-9dd9-b032c41534e2" />


				
2.	Feature comparison between classes:
Here, we looked at how the key feature differ between benign and malignant tumors. We focused on four important features: radius_mean, texture_mean, area_mean and concave¬_points_mean. Figure shows the box plots comparing these features between the two diagnosis classes.
The results are quite clear:
•	Area_mean shows the biggest difference between the malignant and benign cases. Malignant tumor has much larger areas, with median values between 900–1000 whereas benign tumors have median values between 450-500. There are also some outliers in malignant group with very large area (2000).
•	Radius_mean and texture_mean also show differences, though they are less dramatic than area. Malignant tumors tend to have slightly larger radius and different texture patterns. 
•	Concave_points_mean shows that malignant tumor has more concave points as cancerous cells tend to have more irregular, intended shapes.
These visualizations confirm that the features we’re using are meaningful for distinguishing between benign and malignant tumors. The clear separation, especially in area segments suggests that machine learning modals should be able to learn useful patterns from this data. However, there is some overlap between the classes, which means the classification task is not trivial and requires careful modal selection and tuning.

 <img width="1800" height="1200" alt="target_distribution" src="https://github.com/user-attachments/assets/b245ef5f-1d74-4b20-a2eb-91e99ce44616" />
                   


Data Split:
We split the data into 80% for training (455 rows) and 20% for testing (114 rows). We used stratified sampling to make sure the ratio of benign to malignant cases stayed the same in both sets. We set the random state to 42 so the result can be repeated.

Validation:
We used 5-fold stratified cross-validation to check how well the modal generalize to unseen data.

Evaluation Metrics:
We looked at accuracy, precision, recall, f1-score and roc-auc. For medical diagnosis Recall (finding all actual cancer cases) and Precision (avoiding false alarms) are very important.

Hyperparameter Tuning Grid:
We tested the following settings for the tuned modal:
•	SVM: C values of [0.1, 1, 10], gamma of [scale, 0.01, 0.1], and kernel of [rbf, linear]
•	Naive Bayes: var_smoothing of [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]
•	Logistic Regression: C values of [0.01, 0.1, 1, 10, 100]
•	KNN: n_neighbors of [3, 5, 7, 9]



First, we evaluated the base modals and then default soft voting ensemble on the 20% test set (114 samples).

                      
0.	Naïve Bayes	          Accuracy    93.8%	  Precision   100.0%  Recall  83.3%	 F1 Score 90.9%	  ROC - AUC 0.993
1.	Logistic Regression	  Accuracy    93.8%	  Precision   97.3%   Recall  85.7%	 F1 Score 91.1%	  ROC - AUC 0.992
2.	KNN	                  Accuracy    91.2%	  Precision   97.1%   Recall  78.6%	 F1 Score 86.8%	  ROC - AUC 0.954
3.	SVM	                  Accuracy    92.1%	  Precision   94.6%   Recall  83.3%	 F1 Score 88.6%	  ROC - AUC 0.991
4.	Default Voting	      Accuracy    95.6%	  Precision   100.0%  Recall  88.1%	 F1 Score 93.6%	  ROC - AUC >0.99

The confusion matrix for the Default Voting Ensemble on the test is:
•	True Negatives (Benign predicted as Benign): 72
•	False Positive (Benign predicted as Malignant): 0
•	False Negative (Malignant predicted as Benign): 5
•	True Positives (Malignant predicted as Malignant): 37

Next, we ran the hyperparameter tuning using GridSearchCV. The search tested 1800 different combinations.
The GridSearch found the following combination of setting:
•	KNN neighbors: 3
•	Logistic Regression C: 100
•	Naïve Bayes var_smoothing: 1e-09
•	SVM C: 10, gamma: scale, kernel: linear


1.	Best Cross-Validation Accuracy	score-96.7%
2.	Final Test Set Accuracy	        score-95.7%


The result shows that combining models into a soft voting ensemble is highly effective. The default ensemble reached accuracy of 95.6% on the test set while tuned ensemble reached accuracy of 96.7% during cross-validation. Though the increase in percentage (1%) seems low but in health sector this increase in accuracy plays a vital role in prediction.

Interpretation and Strength:
The ensemble got 100% precision for malignant cases in the base test. This is a huge strength because if a model predicts the tumor as cancer than it is almost certainly sure. This prevents unnecessary stress and invasive procedures for patients who are actually healthy. The hyperparameter tuning successfully found a strong configuration, particularly by reducing the KNN neighbors to 3 (making it more sensitive to local patterns) and increasing the Logistic Regression regularization strength (C=100).

Limitations:
The tuned model’s test accuracy (94.7%) was slightly lower than the default ensemble’s test accuracy (95.6%), even though its cross-validation score was higher. This is a classic sign of overfitting. The model learned the training fold a little too well and did not generalized perfectly to the unseen test set.
Also, the model had 5 false negatives (it missed 5 cancer tumors). In medicine, missing a cancer tumor or case is dangerous. While the precision is perfect, we must be careful about the recall. Another limitation can be dataset size. With only 569 records, the model might not capture all the complex variation of breast cancer. We also used tabular data extracted from images, not the images themselves. 

	

This study aimed to answer whether tuning the hyperparameter of a soft voting ensemble can improve the breast cancer diagnosis. The tuning process successfully found an optimal set of parameters that achieved a 96.7% cross-validation accuracy. However, the slight drop in final accuracy (94.7%) compared to the default ensemble shows that default setting can sometimes be surprisingly robust on small dataset.
The model perfect precision for malignant cases makes it very reliable tool for giving a second opinion for doctors. Future work should focus on improving the recall rate to ensure no cancer cases are missed. This could be done by trying different ensemble techniques, collecting more diverse patient data, or exploring deep learning models that can analyze the actual medical images directly.
