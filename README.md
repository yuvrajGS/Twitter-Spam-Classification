***
# **I. Abstract**
Social media has become an integral part of our lives, and Twitter is one of the leaders as a platform for sharing ideas, art, and general information. The increasing use of malicious bot accounts has tainted the platform, as these accounts can be used for malicious activities such as spamming, spreading false information, and targetted harassment. Therefore, there is a need for a reliable method to identify bot accounts on Twitter. To begin, the objective of the project is to correctly classify whether or not a Twitter account is a "bot" account (automated Twitter account) based on the account data (i.e. user tweets). By examining the tweets and user data produced by these accounts made available through the Twitter developer API & enhanced by the research article "Evidence of Spam and Bot Activity in Stock Microblogs on Twitter.", a machine-learning model will be trained to classify bot accounts. Finally, We will use various machine learning algorithms to develop our model, including random forest, K Nearest Neighbours and a Sequential Neural Network. These algorithms will prove effective in the classification of Twitter accounts, and we will evaluate the performance of the models via the accuracy, precision, recall and F1 score of the algorithms. Moreover, with hyperparameter tuning, we will select the best parameters from the user data to efficiently classify bot accounts. Additionally, we will analyze the content of the tweets themselves, looking for patterns in the language used, the types of links shared, and the sentiment expressed. We will also conduct a cross-validation analysis to ensure that the model is robust and not overfitting the data. 


# **II. Requirements**
### **1. Data**
The data is available in the following repository and google drive links:
- [Github Repository](https://github.com/meeroTheo/tweetsandusers)
  - clone the repository to your local machine, and then put the raw data in the data/raw folder and the processed data in the data/processed folder.
- [Google Drive](https://drive.google.com/drive/folders/1dFdmKEqhQo1FVCzIkDqTZov2FCgBQ4PM?usp=sharing)
### **2. Packages**
The following packages are required to run the code in this repository and can be installed with the following command(s):
```
pip install pandas
pip install sklearn
pip install tensorflow
pip install matplotlib
pip install seaborn
pip install numpy
```

Project Instructions
==============================

This repo contains the instructions for a machine learning project.

Project Organization
------------

    ├── LICENSE
    ├── README.md          <- The top-level README for describing highlights for using this ML project.
    ├── data
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── notebooks          <- Jupyter notebooks.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │   └── README.md      <- Youtube Video Link
    │   └── final_project_report <- final report .pdf format and supporting files
    │   └── presentation   <-  final power point presentation 
    |
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── src                <- Source code for use in this project.
       ├── __init__.py    <- Makes src a Python module
       │
       ├── preprocessing data           <- Scripts to download or generate data as well as pre-process the data and perform feature engineering
       │   └── pre-processing.py
       │
       ├── models         <- Scripts to train models and then use trained models to make
       │   │                 predictions
       │   ├── predict_model.py
       │   └── train_model.py
       │
       └── visualization  <- Scripts to create exploratory and results oriented visualizations
           │  
           ├── visualize.py 
           ├── train_model.py



