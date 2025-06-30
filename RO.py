import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import plotly.graph_objs as go
import catboost as cb
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import scatter_matrix
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import xgboost as xgb
from lightgbm import LGBMRegressor
import lightgbm as lgb
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.linear_model import LinearRegression
import plotly.graph_objs as go
from matplotlib.markers import MarkerStyle
from matplotlib.lines import Line2D
from matplotlib.transforms import Affine2D
import seaborn as sns
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from lightgbm import LGBMRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
import lightgbm as lgb
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel

#################################################################################

# Load data from CSV file
data = pd.read_csv('data.csv')
data.head()


# Calculate the R-squared for the training and testing data
train_r2 = r2_score(df_label, y_pred_xg_train)
xgtest_r2 = r2_score(ts_label, y_pred_xg_test)


# Print the R-squared values
print("Training R-squared: {:.6f}".format(train_r2))
print("Testing R-squared: {:.6f}".format(xgtest_r2))

##################################################################################

# Calculate the RMSE for the training and testing data
train_rmse =np.sqrt(mean_squared_error(df_label,  y_pred_xg_train))
xgtest_rmse = np.sqrt(mean_squared_error(ts_label,  y_pred_xg_test))


# Print the RMSE values
print("Training RMSE: {:.6f} ".format(train_rmse))
print("Testing RMSE: {:.6f} ".format(xgtest_rmse))

#####################################################################################

fig = plt.figure(figsize=(8, 4.5), dpi=100)
ax = fig.add_subplot()

plt.scatter(df_label, y_pred_xg_train , s=90, color='orange', marker='o', label='Train', edgecolors='black')

plt.scatter(ts_label, y_pred_xg_test , s=80, color='lime', marker='h', label='Test', edgecolors='black')




xmin, xmax = plt.xlim()
ymin, ymax = plt.ylim()
plt.plot([xmin, xmax], [ymin, ymax], color='black', linestyle='-', label='45 degree line')
plt.xlabel('Experimental permeation water flux (L/m^2.h)')
plt.ylabel('Predicted permeation water flux (L/m^2.h)')
ax.grid(False)
ax.text(0.02,0.9, 'XGBoost',
        verticalalignment='bottom', horizontalalignment='left',
        transform=ax.transAxes,
        color='black', fontsize=15)
plt.legend(['Train set','Test set','x=y'],loc='lower right')
# Save the figure with 800 dpi resolution
fig.savefig('cross_per_xg.png', dpi=800)
plt.show()

#######################################################################################

fig = plt.figure(figsize=(8, 4), dpi=100)
ax = fig.add_subplot()

error_1 = df_label - y_pred_xg_train
error_2 = ts_label - y_pred_xg_test

plt.scatter(df_label ,error_1 ,  s=70, c="springgreen", marker='^', label='Train', edgecolors='black')

plt.scatter(ts_label, error_2 , s=80, c='fuchsia', marker='d', label='Test', edgecolors='black')


xmin, xmax = plt.xlim()
ymin, ymax = plt.ylim([-2,2])
plt.plot([0,xmax], [0, 0], color='black', linestyle='-')
plt.xlabel('Experimental permeation water flux (L/m^2.h)')
plt.ylabel('Error (Exp-Pred)')

# Set the origin of the coordinates to the middle of the vertical axis

ax.grid(False)
ax.text(0.02,0.9, 'XGBoost',
        verticalalignment='bottom', horizontalalignment='left',
        transform=ax.transAxes,
        color='black', fontsize=15)
plt.legend(['Train set','Test set'],loc='lower right')
# Save the figure with 800 dpi resolution
fig.savefig('Error_per_xg.png', dpi=800)
plt.show()

################################################################################################

# Calculate the R-squared for the training and testing data
train_r2 = r2_score(df_label, y_pred_cb_train)
cbtest_r2 = r2_score(ts_label, y_pred_cb_test)


# Print the R-squared values
print("Training R-squared: {:.6f}".format(train_r2))
print("Testing R-squared: {:.6f}".format(cbtest_r2))


# Calculate the RMSE for the training and testing data
train_rmse =np.sqrt(mean_squared_error(df_label, y_pred_cb_train))
cbtest_rmse = np.sqrt(mean_squared_error(ts_label, y_pred_cb_test))


# Print the RMSE values
print("Training RMSE: {:.6f} ".format(train_rmse))
print("Testing RMSE: {:.6f} ".format(cbtest_rmse))

###############################################################################################

fig = plt.figure(figsize=(8, 4.5), dpi=100)
ax = fig.add_subplot()

plt.scatter(df_label, y_pred_cb_train, s=90, color='orange', marker='o', label='Train', edgecolors='black')

plt.scatter(ts_label, y_pred_cb_test , s=80, color='lime', marker='h', label='Test', edgecolors='black')




xmin, xmax = plt.xlim()
ymin, ymax = plt.ylim()
plt.plot([xmin, xmax], [ymin, ymax], color='black', linestyle='-', label='45 degree line')
plt.xlabel('Experimental permeation water flux (L/m^2.h)')
plt.ylabel('Predicted permeation water flux (L/m^2.h)')
ax.grid(False)
ax.text(0.02,0.9, 'CatBoost',
        verticalalignment='bottom', horizontalalignment='left',
        transform=ax.transAxes,
        color='black', fontsize=15)
plt.legend(['Train set','Test set','x=y'],loc='lower right')
# Save the figure with 800 dpi resolution
fig.savefig('cross_per_cat.png', dpi=800)
plt.show()

#############################################################################################

fig = plt.figure(figsize=(8, 4.5), dpi=100)
ax = fig.add_subplot()

error_1 = df_label - y_pred_cb_train
error_2 = ts_label - y_pred_cb_test

plt.scatter(df_label ,error_1 ,  s=70, c="springgreen", marker='^', label='Train', edgecolors='black')

plt.scatter(ts_label, error_2 , s=80, c='fuchsia', marker='d', label='Test', edgecolors='black')


xmin, xmax = plt.xlim()
ymin, ymax = plt.ylim([-2,2])
plt.plot([0,xmax], [0, 0], color='black', linestyle='-')
plt.xlabel('Experimental permeation water flux (L/m^2.h)')
plt.ylabel('Error (Exp-Pred)')

# Set the origin of the coordinates to the middle of the vertical axis

ax.grid(False)
ax.text(0.02,0.9, 'CatBoost',
        verticalalignment='bottom', horizontalalignment='left',
        transform=ax.transAxes,
        color='black', fontsize=15)
plt.legend(['Train set','Test set'],loc='lower right')
# Save the figure with 800 dpi resolution
fig.savefig('erorr_per_cat.png', dpi=800)
plt.show()

####################################################################################

# Calculate the R-squared for the training and testing data
train_r2 = r2_score(df_label, y_pred_ab_train)
abtest_r2 = r2_score(ts_label, y_pred_ab_test)


# Print the R-squared values
print("Training R-squared: {:.6f}".format(train_r2))
print("Testing R-squared: {:.6f}".format(abtest_r2))

# Calculate the RMSE for the training and testing data
train_rmse =np.sqrt(mean_squared_error(df_label, y_pred_ab_train))
abtest_rmse = np.sqrt(mean_squared_error(ts_label, y_pred_ab_test))


# Print the RMSE values
print("Training RMSE: {:.6f} ".format(train_rmse))
print("Testing RMSE: {:.6f} ".format(abtest_rmse))

####################################################################################

fig = plt.figure(figsize=(8, 4.5), dpi=100)
ax = fig.add_subplot()

plt.scatter(df_label, y_pred_ab_train, s=90, color='orange', marker='o', label='Train', edgecolors='black')

plt.scatter(ts_label, y_pred_ab_test , s=80, color='lime', marker='h', label='Test', edgecolors='black')




xmin, xmax = plt.xlim()
ymin, ymax = plt.ylim()
plt.plot([xmin, xmax], [ymin, ymax], color='black', linestyle='-', label='45 degree line')
plt.xlabel('Experimental permeation water flux (L/m^2.h)')
plt.ylabel('Predicted permeation water flux (L/m^2.h)')
ax.grid(False)
ax.text(0.02,0.9, 'AdaBoost',
        verticalalignment='bottom', horizontalalignment='left',
        transform=ax.transAxes,
        color='black', fontsize=15)
plt.legend(['Train set','Test set','x=y'],loc='lower right')
# Save the figure with 800 dpi resolution
fig.savefig('cross_per_ab.png', dpi=800)
plt.show()


####################################################################

fig = plt.figure(figsize=(8, 4.5), dpi=100)
ax = fig.add_subplot()

error_1 = df_label - y_pred_ab_train
error_2 = ts_label - y_pred_ab_test

plt.scatter(df_label ,error_1 ,  s=70, c="springgreen", marker='^', label='Train', edgecolors='black')

plt.scatter(ts_label, error_2 , s=80, c='fuchsia', marker='d', label='Test', edgecolors='black')


xmin, xmax = plt.xlim()
ymin, ymax = plt.ylim([-6,6])
plt.plot([0,xmax], [0, 0], color='black', linestyle='-')
plt.xlabel('Experimental permeation water flux (L/m^2.h)')
plt.ylabel('Error (Exp-Pred)')

# Set the origin of the coordinates to the middle of the vertical axis

ax.grid(False)
ax.text(0.02,0.9, 'AdaBoost',
        verticalalignment='bottom', horizontalalignment='left',
        transform=ax.transAxes,
        color='black', fontsize=15)
plt.legend(['Train set','Test set'],loc='lower right')
# Save the figure with 800 dpi resolution
fig.savefig('erorr_per_ab.png', dpi=800)
plt.show()

######################################################################

# Calculate the R-squared for the training and testing data
train_r2 = r2_score(df_label, y_pred_svr_train)
svrtest_r2 = r2_score(ts_label, y_pred_svr_test)


# Print the R-squared values
print("Training R-squared: {:.6f}".format(train_r2))
print("Testing R-squared: {:.6f}".format(svrtest_r2))


# Calculate the RMSE for the training and testing data
train_rmse =np.sqrt(mean_squared_error(df_label, y_pred_svr_train))
svrtest_rmse = np.sqrt(mean_squared_error(ts_label, y_pred_svr_test))


# Print the RMSE values
print("Training RMSE: {:.6f} ".format(train_rmse))
print("Testing RMSE: {:.6f} ".format(svrtest_rmse))

########################################################################


fig = plt.figure(figsize=(8, 4.5), dpi=100)
ax = fig.add_subplot()

plt.scatter(df_label, y_pred_svr_train, s=90, color='orange', marker='o', label='Train', edgecolors='black')

plt.scatter(ts_label, y_pred_svr_test , s=80, color='lime', marker='h', label='Test', edgecolors='black')




xmin, xmax = plt.xlim()
ymin, ymax = plt.ylim()
plt.plot([xmin, xmax], [ymin, ymax], color='black', linestyle='-', label='45 degree line')
plt.xlabel('Experimental permeation water flux (L/m^2.h)')
plt.ylabel('Predicted permeation water flux (L/m^2.h)')
ax.grid(False)
ax.text(0.02,0.9, 'SVR',
        verticalalignment='bottom', horizontalalignment='left',
        transform=ax.transAxes,
        color='black', fontsize=15)
plt.legend(['Train set','Test set','x=y'],loc='lower right')
# Save the figure with 800 dpi resolution
fig.savefig('cross_per_svr.png', dpi=800)
plt.show()

########################################################################

fig = plt.figure(figsize=(8, 4), dpi=100)
ax = fig.add_subplot()

error_1 = df_label - y_pred_svr_train
error_2 = ts_label - y_pred_svr_test

plt.scatter(df_label ,error_1 ,  s=70, c="springgreen", marker='^', label='Train', edgecolors='black')

plt.scatter(ts_label, error_2 , s=80, c='fuchsia', marker='d', label='Test', edgecolors='black')


xmin, xmax = plt.xlim()
ymin, ymax = plt.ylim([-6,6])
plt.plot([0,xmax], [0, 0], color='black', linestyle='-')
plt.xlabel('Experimental permeation water flux (L/m^2.h)')
plt.ylabel('Error (Exp-Pred)')

# Set the origin of the coordinates to the middle of the vertical axis

ax.grid(False)
ax.text(0.02,0.9, 'SVR',
        verticalalignment='bottom', horizontalalignment='left',
        transform=ax.transAxes,
        color='black', fontsize=15)
plt.legend(['Train set','Test set'],loc='lower right')
# Save the figure with 800 dpi resolution
fig.savefig('erorr_per_svr.png', dpi=800)
plt.show()

