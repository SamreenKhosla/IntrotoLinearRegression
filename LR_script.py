# Load libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import codecademylib3

# Read in the data
codecademy = pd.read_csv('codecademy.csv')

# 1) Print the first five rows
print(codecademy.head())

# 2) Scatter plot of score (y) vs completed (x)
plt.scatter(codecademy.completed, codecademy.score)
plt.show()
plt.clf()

# 3) Fit linear regression: score ~ completed
model = sm.OLS.from_formula('score ~ completed', data=codecademy)
results = model.fit()
print(results.params)

# 4) Interpretations (commented out)
# Intercept: A learner who completed 0 items is expected to score about 13.21 points.
# Slope: Each additional completed item is associated with about a 1.31 point increase in expected score.

# 5) Scatter plot with regression line
plt.scatter(codecademy.completed, codecademy.score)
plt.plot(codecademy.completed, results.predict(codecademy))
plt.show()
plt.clf()

# 6) Predicted score for completed = 20
newdata = {'completed': [20]}
pred20 = results.predict(newdata)
print("predicted score for learner who has completed 20 prior lessons: ", pred20)

# 7) Fitted values
fitted_values = results.predict(codecademy)

# 8) Residuals
residuals = codecademy.score - fitted_values

# 9) Histogram of residuals (normality check)
plt.hist(residuals)
plt.show()
plt.clf()

# 10) Residuals vs fitted values (homoscedasticity check)
plt.scatter(fitted_values, residuals)
plt.show()
plt.clf()

# 11) Boxplot: score by lesson
sns.boxplot(x='lesson', y='score', data=codecademy)
plt.show()
plt.clf()

# 12) Regression: score ~ lesson
model2 = sm.OLS.from_formula('score ~ lesson', data=codecademy)
results2 = model2.fit()
print(results2.params)

# 13) Group means + mean difference
mean_A = np.mean(codecademy.score[codecademy.lesson == 'Lesson A'])
mean_B = np.mean(codecademy.score[codecademy.lesson == 'Lesson B'])
print("Mean score (A): ", mean_A)
print("Mean score (B): ", mean_B)
print("Mean score difference (A - B): ", mean_A - mean_B)

# 14) lmplot: score vs completed, colored by lesson
sns.lmplot(x='completed', y='score', hue='lesson', data=codecademy)
plt.show()
