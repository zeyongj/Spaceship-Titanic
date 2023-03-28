# Spaceship Titanic: Predict which passengers are transported to an alternate dimension

This repository contains the code for my submission to the Spaceship Titanic: Predict which passengers are transported to an alternate dimension competition on Kaggle.

## Project Overview

The Spaceship Titanic Kaggle competition provides a dataset containing information on passengers aboard the Titanic spaceship, including their names, ages, genders, and other relevant features. The goal of the competition is to build a machine learning model that can predict which passengers survived the disaster.

## File Descriptions

The repository contains the following files:

- `train.csv`: the training data, which includes information about the passengers and whether or not they survived.
- `test.csv`: the test data, which includes information about the passengers but does not include information about whether or not they survived.
- `submission.csv`: an submission file showing the format of the submission file required by Kaggle.
- `sample_submission.csv`: an sample submission file provided by Kaggle.
- `titanic.ipynb`: a Jupyter Notebook containing the code for the machine learning model and the data preprocessing steps.
- `titanic.py`: a Python file containing the code for the machine learning model and the data preprocessing steps.
- `README.md`: this file, which provides an overview of the project.

## Data Preprocessing

The `titanic.ipynb` notebook contains the code for the data preprocessing steps, which include filling in missing values and encoding categorical variables. Specifically, the following preprocessing steps were performed:

- Missing values in the `Age` column were filled in using the median age of passengers in the same ticket class and gender.
- Missing values in the `Embarked` column were filled in using the mode (i.e., most common value) of the `Embarked` column.
- The `Cabin` column was dropped since it contained a large number of missing values and was not deemed to be a useful predictor of survival.
- Categorical variables (i.e., `Sex` and `Embarked`) were encoded using one-hot encoding.

## Machine Learning Model

The machine learning model used in this project is a random forest classifier. The code for the model is contained in the `titanic.ipynb` notebook. The model achieved an accuracy of approximately 80% on the test data.

## Dependencies

- [ ] Python 3.9
- [ ] pandas
- [ ] scikit-learn

## Usage

- To run the code in this repository, follow these steps:

1. Clone the repository to your local machine.
2. Open the `titanic.ipynb` notebook in Jupyter Notebook or JupyterLab.
3. Run the code in the notebook to preprocess the data and train the machine learning model.
4. Generate predictions on the test data using the trained model.
5. Submit your predictions to the competition on Kaggle.

- Or you can execute the python file by running `python titanic.py`.

## Discussion

Some potential methods to improve the score:

- Feature Engineering: try creating new features from the existing ones, for example, combining 'Age' and 'Pclass' to create a new feature called 'AgeClass'. This could help the model learn better patterns in the data.
- Hyperparameter Tuning: try different hyperparameters for your model, such as the number of estimators, max depth, and min samples leaf, and see how they affect the performance of the model.
- Try Different Models: try different models and compare their performances. Some other models that you could try are XGBoost, LightGBM, and CatBoost.
- Ensemble Methods: try combining multiple models together to improve the overall performance. For example, you can train multiple random forest models and combine their predictions using a voting classifier.
- Regularization: try adding regularization to your model, such as L1 and L2 regularization, and see how it affects the performance.

## Conclusion

The classifier achieved an accuracy score of *0.7797584818861415* on the validation set. The predictions made on the test set were submitted to Kaggle, resulting in a score of *0.79471*.

In conclusion, the Titanic competition provides a great opportunity to practice data preprocessing and machine learning skills. By building a machine learning model to predict survival on the Titanic, we can gain insights into the factors that may have contributed to survival and develop a deeper understanding of the tragedy that occurred in 1912.

## Citation

A. Howard, A. Chow, and R. Holbrook, "Spaceship Titanic," Kaggle, 2022. [Online]. Available: https://kaggle.com/competitions/spaceship-titanic. [Accessed: March 27, 2023].


## License
This project is licensed under the Apache 2.0 or later License - see the `LICENSE` file for details.

## Author
Zeyong Jin

March 27th, 2023
