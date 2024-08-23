-----------------------------------------------------------------------------------------------------------------------------------------------------------------
--
-- This file contains the workflow diagram, prerequisite softwares, packages used for the project
-- This file is subjected to copyrights.
--
-----------------------------------------------------------------------------------------------------------------------------------------------------------------
--
-- List of Softwares
--
-----------------------------------------------------------------------------------------------------------------------------------------------------------------
--
S.No	Software				Version				Bit				OS type
--
 1)		Python 				3.12.0				64				Windows 11
 2)		Microsoft VS Code		1.84.1				64				Windows 11
-----------------------------------------------------------------------------------------------------------------------------------------------------------------
--
-- List of VS Code Extensions
--
-----------------------------------------------------------------------------------------------------------------------------------------------------------------
--
S.No	Addin					Version
--
 1)		Jupyter				   v2024.1.0
 2)		Python				   v2024.0.0
 3)		Pylance				   v2023.12.1
-----------------------------------------------------------------------------------------------------------------------------------------------------------------
--
-- List of Python Packages
--
----------------------------------------------------------------------------------------------------------------------------------------------------------------- 
--
S.No	Package
--
 1)		streamlit
 2)		pandas
 3)		numpy
 4)		re
 5)		pickle
 6)		matplotlib
 7)		seaborn
 8)		sklearn
 9)		xgboost
 10)		time
 11)		datetime
 12)		PIL
-----------------------------------------------------------------------------------------------------------------------------------------------------------------
--
-- Project Work Flow
-- Major Steps : 1) Exploratory Data Analysis (Python)
--		 2) Label Encoding (Python)
--		 3) Outliers Detection and Removal after Label Encoding (Python)
--		 4) Feature Selection (Python)
--		 5) Scalar Transformation of Dataset (Python)
--		 6) Developing ML Model (Python)
--		 8) Inverse Scalar Transformation of trained Dataset (Python)
--		 9) Dumping the ML Model as a binary file (Python)
--		 10) Deploying the ML Model using Streamlit (Python & Streamlit)
--
----------------------------------------------------------------------------------------------------------------------------------------------------------------- 
|
|
|--- Developing Regression Model
|		|
|		|
|		|--- 1) The Dataset is read using Pandas to get the dataframe
|		|
|		|
|		|--- 2) The dataframe is checked for the missing values, erratic values and is cleaned up by dropping them
|		|
|		|
|		|--- 3) The dataframe is checked for the datatypes of the columns and is converted suitably to 'int64' (or) 'float64'
|		|
|		|
|		|--- 4) The dataframe is visualized for boxplot using seaborn & matplotlib to detect the outliers
|		|
|		|
|		|--- 5) The outliers are treated using IQR method for each column in the dataframe separately
|		|
|		|
|		|--- 6) The categorical columns in the dataframe are converted into numerical columns using Label Encoding
|		|
|		|
|		|--- 7) The dataframe is once checked for outliers and are treated using IQR method
|		|
|		|
|		|--- 8) The dataframe undergoes Feature selection method and the best features are identified for ML model. Correlation Matrix is used here
|		|
|		|
|		|--- 9) Once the features are identified, rest of the columns are dropped from the dataframe and is again treated for outliers by dropping them
|		|
|		|
|		|--- 10) The data is now free from outliers & skewness. The data is scaled down using Standard Scaler method to reduce the higher range data
|		|
|		|
|		|--- 11) The data is now split separately with 'X' as all the Independent Variables and 'Y' as the Dependent Variable (selling_price)
|		|
|		|
|		|--- 12) The data is split into training and testing datasets and is trained on various ML models:
|		|         |
|		|         |--- Multiple Linear Regression
|		|         |--- Polynomial Regression
|		|         |--- KNN Regression
|		|         |--- Decision Tree Regression
|		|         |--- Random Forest Regression
|		|         |--- Gradient Boosting Regression
|		|         |--- Extreme Gradient Boosting Regression
|		|         |
|		|
|		|
|		|--- 13) The best performing ML model is then chosen based on the Regression metrics (MSE, MAE, RMSE, R2 score). In this case, Random Forest Regression is chosen
|		|
|		|
|		|--- 14) The dataset is now inverse transformed retrieve the original dataset
|		|
|		|
|		|--- 15) This original dataset is again splitted into training & testing datasets and is fitted in the Random Forest Regression model
|		|
|		|
|		|--- 16) This new model is then dumped as a binary file using Pickling 
|		|
|		|
|
|--- Developing Classification Model
|		|
|		|
|		|--- 1) The Dataset is read using Pandas to get the dataframe
|		|
|		|
|		|--- 2) The dataframe is checked for the missing values, erratic values and is cleaned up by dropping them
|		|
|		|
|		|--- 3) The dataframe is checked for the datatypes of the columns and is converted suitably to 'int64' (or) 'float64'
|		|
|		|
|		|--- 4) The dataframe is visualized for boxplot using seaborn & matplotlib to detect the outliers
|		|
|		|
|		|--- 5) The outliers are treated using IQR method for each column in the dataframe separately
|		|
|		|
|		|--- 6) The categorical columns in the dataframe are converted into numerical columns using Label Encoding
|		|
|		|
|		|--- 7) The dataframe is once checked for outliers and are treated using IQR method
|		|
|		|
|		|--- 8) The data is now free from outliers & skewness. The data is scaled down using Standard Scaler method to reduce the higher range data
|		|
|		|
|		|--- 9) The 'status' column in the dataframe is now encoded by replacing 'Won' with '1' and 'Lost' with '0'.
|		|
|		|
|		|--- 10) The dataframe undergoes Feature selection method and the best features are identified for ML model. ANOVA Table is used here
|		|
|		|
|		|--- 11) Once the features are identified, rest of the columns are dropped from the dataframe
|		|
|		|
|		|--- 12) The data is now split separately with 'X' as all the Independent Variables and 'Y' as the Dependent Variable (status)
|		|
|		|
|		|--- 13) The dataframe is checked for class imbalance and is handled for Class Imbalancing using SMOTE method
|		|
|		|
|		|--- 14) The data is split into training and testing datasets and is trained on various ML models:
|		|         |
|		|         |--- KNN Classifier
|		|         |--- Logistic Regression
|		|         |--- Decision Tree Classifier
|		|         |--- Random Forest Classifier
|		|         |--- Gradient Boosting Classifier
|		|         |--- Extreme Gradient Boosting Classifier
|		|         |
|		|
|		|
|		|--- 15) The best performing ML model is then chosen based on the Classification metrics (Accuracy, Recall, Precision, F1 score). In this case, Random Forest Classifier is chosen
|		|
|		|
|		|--- 16) The dataset is now inverse transformed retrieve the original dataset
|		|
|		|
|		|--- 17) This original dataset is again splitted into training & testing datasets and is fitted in the Random Forest Classifier model
|		|
|		|
|		|--- 18) This new model is then dumped as a binary file using Pickling 
|		|
|		|
|
|
|--- Displaying in Streamlit application 
|		|
|		|
|		|--- 1) An option is provided to select the Model type as 'Regression' (or) 'Classification')
|		|
|		|
|		|--- 2) A streamlit GUI is developed with Three Stages as below:
|		|         |
|		|         |--- Stage 1 : Input Collection
|		|         |--- Stage 2 : Loading Trained Model
|		|         |--- Stage 3 : Displaying the results in the GUI
|		|
|		|
|		|--- 3) Based on the selected model & input, the results are predicted and displayed
|		|
|		|
|
|
----------------------------------------------------------------------------------------------------------------------------------------------------------------- 