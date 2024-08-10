# Pump-It-Up-Competition
My Submission for the Pump It Up Machine Learning Competition.

Details of the challenge and download of the dataset can be found here:
https://www.drivendata.org/competitions/7/pump-it-up-data-mining-the-water-table/

This was completed using sklearn.

## Files

### part1.py
This file performs various preprocessing steps on the training and test data. Afterwhich the training set results were gathered with 5 different models: Linear Regression, MLP, Random Forest Classifier, HGBC, GBC, as well as other preprocessing techniques such as: OneHotEncoder, OrdinalEncoder, TargetEncoder, StandardScaler. 
![acc_overview](https://github.com/user-attachments/assets/46a6624a-d522-4dd6-aa02-99c7cd46611a)

### part2.py 
This file uses Optuna to tune the best models from part1. Little to no increase in accuracy was found on the test set.
