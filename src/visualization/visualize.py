# for data manipulation
import pandas as pd, numpy as np, random
#for graphs
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc
import train_model as tm
from sklearn.model_selection import learning_curve

sns.set()

# Train the models
fit_knn, fit_rf, fit_nn = tm.train()

#figure
fig1 = plt.figure(figsize=(8,6))
ax1 = fig1.add_subplot()
#figure2
fig2 = plt.figure(figsize=(8,6))
ax2 = fig2.add_subplot()
#figure3
fig3 = plt.figure(figsize=(8,6))
ax3 = fig3.add_subplot()

# Plot ROC curves for each model on the first subplot
ax1.plot([0, 1], [0, 1], 'k--')
knn_y_pred_proba = fit_knn.predict_proba()
fpr_knn, tpr_knn, _ = roc_curve(fit_knn.y_test, knn_y_pred_proba)
roc_auc_knn = auc(fpr_knn, tpr_knn)
ax1.plot(fpr_knn, tpr_knn, label='KNN (AUC = %0.2f)' % roc_auc_knn)

rf_y_pred_proba = fit_rf.predict_proba()
fpr_rf, tpr_rf, _ = roc_curve(fit_rf.y_test, rf_y_pred_proba)
roc_auc_rf = auc(fpr_rf, tpr_rf)
ax1.plot(fpr_rf, tpr_rf, label='Random Forest (AUC = %0.2f)' % roc_auc_rf)

nn_y_pred_proba = fit_nn.predict_proba()
fpr_nn, tpr_nn, _ = roc_curve(fit_nn.y_test, nn_y_pred_proba)
roc_auc_nn = auc(fpr_nn, tpr_nn)
ax1.plot(fpr_nn, tpr_nn, label='Neural Network (AUC = %0.2f)' % roc_auc_nn)

ax1.set_xlabel('False Positive Rate')
ax1.set_ylabel('True Positive Rate')
ax1.set_title('Receiver Operating Characteristic')
ax1.legend(loc='lower right')

# Create a heatmap of feature correlation scores on the second subplot
top_features = ['those', 'will', 'now', 'boy', 'kid', 'she', 'these', 'what', 'come', 'about', 'start', 'would', 'didnt', 'our', 'who', 'her', 'yeah', 'sad', 'were', 'said', 'but', 'night', 'nice', 'play']
dfX = tm.X[top_features]
corr= dfX.corr()
mask = np.zeros_like(corr, dtype=bool)
mask[np.triu_indices_from(mask)] = True
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5}, ax=ax2)
ax2.set_title('Top Twitter User Most Importance Feature Correlation Heatmap')

# Plot learning curve for KNN model on the third subplot
train_sizes, train_scores, test_scores = learning_curve(fit_knn.model, fit_knn.X_train, fit_knn.y_train, cv=5, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10))

# cal mean and std of train and test scores
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

ax3.set_title("Learning Curve (KNN)")
ax3.set_xlabel("Training examples")
ax3.set_ylabel("Score")
ax3.grid()

ax3.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="r")
ax3.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color="g")
ax3.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
ax3.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
ax3.legend(loc="best")

# Show the figure with all subplots
plt.show()

#plot feature importance
var=fit_rf.plot_feature_importance(top=24)