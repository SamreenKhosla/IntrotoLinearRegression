#import libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
import numpy as np
import codecademylib3

#load data
forests = pd.read_csv('forests.csv')

#check multicollinearity with a heatmap
corr_grid = forests.corr()
sns.heatmap(
    corr_grid,
    xticklabels=corr_grid.columns,
    yticklabels=corr_grid.columns,
    annot=True
)
plt.show()
plt.clf()

#plot humidity vs temperature
sns.lmplot(x='temp', y='humid', hue='region', data=forests, fit_reg=False)
plt.show()
plt.clf()

#model predicting humidity
modelH = sm.OLS.from_formula('humid ~ temp + region', data=forests).fit()
print(modelH.params)

#equations
# (Numbers will match whatever prints above)
# Full equation:
# humid = b0 + b1*temp + b2*region
#
# If region is coded 0/1:
# Bejaia (region=0): humid = b0 + b1*temp
# Sidi Bel-abbes (region=1): humid = (b0+b2) + b1*temp

#interpretations
## Coefficient on temp:
# Holding region constant, a 1°C increase in temp changes humidity by b1 percent points (usually decreases if b1 is negative).

## For Bejaia equation:
# When region=0, predicted humid starts at b0 and changes by b1 per 1°C.

## For Sidi Bel-abbes equation:
# When region=1, intercept shifts by b2 compared to Bejaia, slope stays b1.

#plot regression lines
sns.lmplot(x='temp', y='humid', hue='region', data=forests, fit_reg=False)

# line for region=0
plt.plot(
    forests.temp,
    modelH.params[0] + modelH.params[2]*0 + modelH.params[1]*forests.temp,
    linewidth=5,
    label='Region 0'
)

# line for region=1
plt.plot(
    forests.temp,
    modelH.params[0] + modelH.params[2]*1 + modelH.params[1]*forests.temp,
    linewidth=5,
    label='Region 1'
)

plt.legend()
plt.show()
plt.clf()

#plot FFMC vs temperature
sns.lmplot(x='temp', y='FFMC', hue='fire', data=forests, fit_reg=False)
plt.show()
plt.clf()

#model predicting FFMC with interaction
modelF = sm.OLS.from_formula('FFMC ~ temp + fire + temp:fire', data=forests).fit()
print(modelF.params)

#equations
# Full equation:
# FFMC = a0 + a1*temp + a2*fire + a3*(temp*fire)
#
# No fire (fire=0): FFMC = a0 + a1*temp
# Fire (fire=1):    FFMC = (a0+a2) + (a1+a3)*temp

#interpretations
## For locations without fire:
# Each +1°C changes FFMC by a1.

## For locations with fire:
# Intercept shifts by a2 and slope changes by a3, so slope becomes (a1+a3).

#plot regression lines
sns.lmplot(x='temp', y='FFMC', hue='fire', data=forests, fit_reg=False)

# fire=0 line
plt.plot(
    forests.temp,
    modelF.params[0] + modelF.params[2]*0 + modelF.params[1]*forests.temp + modelF.params[3]*forests.temp*0,
    linewidth=5,
    label='No Fire'
)

# fire=1 line
plt.plot(
    forests.temp,
    modelF.params[0] + modelF.params[2]*1 + modelF.params[1]*forests.temp + modelF.params[3]*forests.temp*1,
    linewidth=5,
    label='Fire'
)

plt.legend()
plt.show()
plt.clf()

#plot FFMC vs humid
sns.lmplot(x='humid', y='FFMC', data=forests, fit_reg=False)
plt.show()
plt.clf()

#polynomial model predicting FFMC
modelP = sm.OLS.from_formula('FFMC ~ humid + np.power(humid, 2)', data=forests).fit()
print(modelP.params)

#regression equation
# FFMC = c0 + c1*humid + c2*humid^2

#sample predicted values
for h in [25, 35, 60, 70]:
    pred = modelP.params[0] + modelP.params[1]*h + modelP.params[2]*np.power(h, 2)
    print(h, pred)

#interpretation of relationship
# Because of the humid^2 term, the relationship is curved (not a straight line).

#multiple variables to predict FFMC
modelFFMC = sm.OLS.from_formula('FFMC ~ temp + rain + wind + humid', data=forests).fit()
print(modelFFMC.params)

#predict FWI from ISI and BUI
modelFWI = sm.OLS.from_formula('FWI ~ ISI + BUI', data=forests).fit()
print(modelFWI.params)
