1.  Regression Model Metrics

Mean Squared Error (MSE)
- it is the squared of the predicted and actual values.
- Penalizes large error more strongly

Root Mean Squared Error (RMSE)
- Square root of MSE, brings error back to original value
- Easier to interpret (e.g, dollars, temperature)

Mean Absolute Error (MAE)
- Average of Absolute difference
- Less sensative to outlier than MSE

R2 Score
- Proportion of variance explained by the model
- Higher R2 = Better fit


2. Classification Model Metrics

Accuracy
- Percentage of correct predictions
- Works well if classes are balanced

Precision
- Of all predicted positive, how many are corrected?
- Important when false positives are costly (e.g, spam filter)

Recall
- Of all actual positives, How many did we catch?
- Important when false negative are costly (e.g, disease detection)

F1-Score
- Harmonic mean of precision and recall
- Balances both Metrics

Confusion Matrix
- table showing true positive, false positive, true negative, false negatives.
- Gives full picture of classification performance.

## Example
Imagine a predicting whether an email is spam:
- Accuracy = 95% (most predictions correct).
- Precision = 90% (few false alarms).
- Recall = 85% (Some spam missed).
- F1 = 87% (Balanced measure).

### Advanced Regression Topics
Multiple Regression
- Extend Simple regression to include multiple imput features (e.g, house size, number of rooms, location)
- Learn how coefficients represent the impact of each features

Polynomial Regression
- When data shows non-linear trends, fit curvs instead of straight lines.
- Example: predicting growth rates that accelerate  over time.

Regularization
- Techniques like Ridge, Lasso, ElasticNet prevent overfitting by penalizing large coefficients.
- Ridge shrinks coefficients, lasso can eliminate irrelevant features.

Residual Analysis
- Diagnose model errors by analyzing residual plots.
- Random Scatter = good fit; patterns = model misspecification.

### these are Matter Because of the following
Multiple Regression -> Handles Real World complexity.
Polynomial Regression -> Captures Nonlinear Relationships.
Regularization -> ensures model generalize wells
Residual Analysis -> validates assumptions and detects problems

