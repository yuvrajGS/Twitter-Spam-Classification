# import classes from the train_model.py in the same folder
import pandas as pd
from train_model import train


fit_knn, fit_rf, fit_nn = train()

pred_knn = fit_knn.predict()
accuracy_knn, classification_report_knn = fit_knn.score()
print("Finished running KNN")

pred_rf = fit_rf.predict()
accuracy_rf, classification_report_rf = fit_rf.score()
print("Finished running RF")


pred_nn = fit_nn.predict()
accuracy_nn, classification_report_nn = fit_nn.score()
print("Finished running NN")

print("The accuracy achieved by knn model:", accuracy_knn)
print('#'*60)
print("The classification report of knn model: \n", classification_report_knn)
print()
print("The accuracy achieved by random forest model:", accuracy_rf)
print('#'*60)
print("The classification report of random forest model: \n",
      classification_report_rf)
print()
print("The accuracy achieved by neural network model:", accuracy_nn)
print('#'*60)
print("The classification report of neural network model: \n",
      classification_report_nn)
