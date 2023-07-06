# Scratch Algorithm Machine Learning - Pacmann Project

## Step #1 - Select Machine Learning Algorithm
Search for algorithm references from valid sources. Examples of lecture notes/slides, papers, conferences, and books.
Lecture notes from CMU Statistics (https://www.stat.cmu.edu/~cshalizi/uADA/12/lectures/ch12.pdf).

## Step #2 - Create Algorithms from Scratch

To understand the learning components of the Logistic Regression algorithm, I will refer to the provided source, "Logistic Regression" Lecture notes from CMU Statistics (https://www.stat.cmu.edu/~cshalizi/uADA/12/lectures/ch12.pdf).

1. For fitting:
    * Optimization: The cost function that is optimized is the Cross Entropy Loss.
    * Optimization objective: Minimize the cost function.
    * Varied parameters for cost function optimization: Model parameters, i.e., weights and bias.
    * Optimization algorithm: Gradient Descent.

2. For making predictions:
    * Predictions are made by calculating the sigmoid function of the dot product of features and weights, plus the bias term.
    Pseudocode for the Algorithm:
    
    Here is the pseudocode for fitting the Logistic Regression model:
    C:\ProjectAdvancedML\scratch-algorithm-ML\pseudocode_fitting._logistic_reegression_model.py

    Here is the pseudocode for making predictions with the trained Logistic Regression model:
    C:\ProjectAdvancedML\scratch-algorithm-ML\pseudocode_making_predictions_trained_logistic_regression_model.py

    Code Implementation:
    
    Here is the implementation of the Logistic Regression Algorithm in a file named logistic_regression.py:
    C:\ProjectAdvancedML\scratch-algorithm-ML\logistic_regression.py

## Step #3 - Analysis, Conclusion, and References

Once I have implemented the Logistic Regression algorithm from scratch, the next step is to apply it to solve simple data problems. These steps to apply the code:
1. Prepare a relevant dataset for classification problems.
2. Perform any necessary data preprocessing, such as normalization or one-hot encoding.
3. Split the dataset into training and testing data.
4. Create an instance of the LogisticRegression class from the implemented code.
5. Call the fit method on the LogisticRegression object, providing the training data as arguments.
6. Make predictions on the testing data using the predict method.
7. Evaluate the performance of the model using appropriate evaluation metrics, such as accuracy or a confusion matrix.
8. Experiment with hyperparameters, such as learning rate and the number of iterations, to improve the model's performance.

## References:

* "Logistic Regression" lecture notes from CMU Statistics by Cosma Rohilla Shalizi (https://www.stat.cmu.edu/~cshalizi/uADA/12/lectures/ch12.pdf)