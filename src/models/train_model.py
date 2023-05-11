# Run this cell to suppress all FutureWarnings
import random
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from tensorflow import keras
from tensorflow.python.keras import layers
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
RANDOM_STATE = 42
random.seed(RANDOM_STATE)


# Load Dataset
users = pd.read_csv('data/processed/users.csv',
                    encoding='utf-8', low_memory=False)

# Shuffle Data
users = users.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

# Define X and y
X = users.drop(['Name', 'Screen_name', 'Label',
               'Description', 'Id', 'Location'], axis=1)
y = users['Label']

# Drop Unimportant Features
X = X.drop(['forget', 'cuando', 'color', 'di', 'strong', 'save', 'son', 'over',
            'año', 'hand', 'march', 'make', 'mucho', 'share', 'qué', 'take', 'war',
            'apart', 'foi', 'market', 'still', 'so', 'totalWords', 'been', 'okay',
            'hoy', 'Followers_count', 'bad', 'eso', 'free', '5', 'san', 'sleep',
            'twitter', 'Geo_enabled', 'rt', 'wrong', '2013', 'tonight', 'last',
            'can', 'fuck', 'Friends_count', 'light', 'say', 'noch', 'be', 'down',
            'Total_urls', 'oh', 'we', 'sure', '—', 'tomorrow', 'talk', 'where',
            'team', '2014', 'around', 'time', 'doesnt', 'Total_mentions', 'Favourites_count',
            'it', 'watch', 'could', 'need', 'off', 'ever', 'birthday', 'right', 'world',
            'better', 'know', 'thing', 'an', 'he', 'miss', 'had', 'keep', 'let', 'happen',
            'did', 'next', 'us', 'thank', 'stop', 'after', 'tell', 'call', 'same', 'their',
            'wait', 'back', 'amp', 'Total_favorite_count', 'hate', 'Total_retweet_count',
            'hope', 'even', 'berkalikali', 'enterprise', 'passenger', 'km', 'raiders',
            'hater', 'aos', '79', 'chin', 'aim', 'beliefs', 'healing', 'rains', 'handed',
            'hoo', 'woo', 'edit', 'bee', 'darling', 'whatsapp', 'downloaded', 'dot',
            'professor', 'clouds', 'counts', 'study', 'wit', 'sec', 'arrive', '1b',
            'potter', 'harry', 'jk', 'theyve', 'beware', 'rice', 'brb', 'crown',
            'den', 'corrupt', 'obamas', 'restart', 'district', 'chris', 'stomach', 'olicity', 'tale', 'explode', 'anti', 'vice', 'karma', 'kindness', 'queen', 'caramel', 'kaibigan', 'discovered', 'officer', 'yemen', 'split', 'toll', 'nepal', 'revolution', 'chill', 'policies', 'mysterious', 'buddies', 'villa', 'roommate', 'idk', 'medal', 'md', 'management', 'opportunity', 'daniel', 'manage', 'relevant', 'followers', 'microsoft', 'poverty', 'hosting', 'stress', 'percent', 'massage', 'weekly', 'oz', 'hanggang', '2k', 'function', 'thankyou', 'tay', 'ohhhh', 'disrespectful', 'training', 'asses', 'hated', 'spectacular', 'valid', 'guilty', 'proven', 'unnecessary', 'fridge', 'tiene', 'equipo', 'olympic', 'lab', 'backed', 'tennis', 'murray', 'yesterdays', 'duke', 'follower', 'happiest', 'sé', 'rated', 'clever', 'restaurants', 'visits', 'teachers', 'mais', 'ex', 'sessions', 'kentucky', 'located', 'islands', 'greece', 'discovery', 'gr8', 'toys', 'tres', 'muy', 'statistics', 'pros', 'mens', 'waves', 'deliver', 'spill', 'giants', 'joined', 'reported', 'carbon', 'api', 'petition', 'thrown', 'dump', 'forecast', 'cameron', 'palace', 'nonsense', 'champions', 'att', 'properly', 'luis', 'sao', 'problema', 'resort', 'importante', 'featuring', 'sponsors', 'xp', 'focused', 'title', 'aggressive', 'passive', 'cuts', 'jimmy', 'rooney', 'photographers', 'afghanistan', 'mentorship', 'cotton', 'vids', 'agenda', 'neymar', 'intelligent', 'attend', 'expecting', 'regret', 'vez', 'cada', 'consumers', 'explains', 'commission', 'handling', 'elite', 'according', 'spin', 'bees', 'yard', 'oscar', 'require', 'hobby', 'facing', 'sides', 'keen', 'rail', 'uploaded', 'sai', 'period', 'triple', 'messenger', 'garcia', 'kim', 'boxes', 'aug', 'lesson', 'segment', 'warns', '120', 'trading', 'est', '19th', 'puppies', 'kardashian', 'lg', 'pun', 'releasing', 'genuinely', 'rs', 'halo', 'listed', 'cos', '54', 'wallet', 'consumer', 'visa', 'wrist', 'fifth', 'ceremony', 'colors', 'autocorrect', 'routine', 'cap', 'decline', 'funds', 'airline', 'depressed', 'rio', 'leo', 'mag', 'headline', 'madrid', 'picks', 'indonesia', 'lawyers', 'anyways', 'flame', 'turtle', 'andre', 'brazil', 'espn', 'independence', 'messi', 'saves', 'mexico', 'nov', 'scotland', 'includes', 'kit', 'versus', 'invest', 'innovation', 'fifa', 'dropbox', 'pakistani', 'solutions', 'ibm', 'cheat', 'apply', 'locker', 'unfortunately', 'passengers', 'opens', 'nexus', 'developers', 'travis', 'volume', 'asia', 'december', 'emojis', 'hundred', 'cooler', 'indians', 'mere', 'answering', 'washington', 'brick', 'promises', 'ranked', 'ch', 'channels', 'modi', 'bhai',
            'ebay', 'ranking', 'yang', 'ministry', 'hahahah', 'charged', 'myspace', '200'], axis=1)


class KNN:
    def __init__(self, X, y, n_neighbors=5, test_size=0.2, random_state=42):
        """
        Initialize the KNN classifier.

        Parameters:
        -----------
            X (pandas.DataFrame): The feature matrix of shape (n_samples, n_features).
            y (pandas.Series): The target vector of shape (n_samples,).
            n_neighbors (int, optional): The number of nearest neighbors to use in classification. Defaults to 5.
            test_size (float, optional): The proportion of samples to use for testing. Defaults to 0.2.
            random_state (int, optional): The random state to use for splitting the data. Defaults to 42.
        """
        self.X = X
        self.y = y
        self.n_neighbors = n_neighbors
        self.test_size = test_size
        self.random_state = random_state
        self.model = None
        self.best_params = None
        self.best_score = None
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=self.test_size, random_state=self.random_state, stratify=self.y)

    def predict(self):
        """
        Predict the target values for the test data.

        Parameters:
        -----------
        None

        Returns:
        -----------
            numpy.ndarray: The predicted target values of shape (n_samples,)
        """
        
        self.model.fit(self.X_train, self.y_train)
        return self.model.predict(self.X_test)

    def score(self):
        """
        Calculate the accuracy score and classification report for the KNN model.

        Parameters:
        -----------
        None

        Returns:
        -----------
        accuracy score: The ratio of the correctly predicted observations to the total observations.
        
        classification report: A text report of the main classification metrics such as precision, recall, f1-score and support for each class.
        """

        y_pred = self.predict()
        return accuracy_score(self.y_test, y_pred), classification_report(self.y_test, y_pred)

    def grid_search(self, param_grid):
        """
        Perform a grid search to find the best hyperparameters for the KNN model.

        Parameters:
        -----------
            param_grid (dict): A dictionary of hyperparameters to test.

        Returns:
        -----------
            None
        """
        # 5 fold cross validation + accuracy evaluation metric
        grid_search = GridSearchCV(
            KNeighborsClassifier(), param_grid, cv=5, scoring='accuracy')
        self.best_params = grid_search.fit(self.X, self.y).best_params_

        # metric -> euclidean, manhattan, minkowski, chebyshev
        self.model = KNeighborsClassifier(
            n_neighbors=self.best_params['n_neighbors'], metric=self.best_params['metric'])
        # holds the highest accuracy achieved during the grid search
        self.best_score = grid_search.fit(self.X, self.y).best_score_


class RandomForest:
    def __init__(self, X, y, random_state=RANDOM_STATE):
        """Initializes the RandomForest class with the input features X and target variable y,
        and the random state used for reproducibility.

        Parameters:
        -----------
        X : pandas.DataFrame
            The input feature matrix of shape (n_samples, n_features).
        y : pandas.Series
            The target variable of shape (n_samples,).
        random_state : int, default=RANDOM_STATE, which is 42
            The seed value for random number generator used to split the data.

        Returns:
        --------
        None"""

        self.X = X
        self.y = y
        self.random_state = random_state
        self.model = None
        self.best_params = {}
        self.best_score = 0
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)

    def predict(self):
        """Predicts the target variable of the test data using the trained random forest model.
        Returns the predicted target variable values.

        Parameters:
        -----------
        None

        Returns:
        --------
        numpy.ndarray: The predicted target variable values
        
        """
        # initalize self.model
        self.model.fit(self.X_train, self.y_train)
        return self.model.predict(self.X_test)

    def score(self):
        """Computes the accuracy score and classification report for the predicted target variable and the actual test target variable.

        Parameters:
        -----------
        None

        Returns:
        --------
        accuracy score: The ratio of the correctly predicted observations to the total observations.
        
        classification report: A text report of the main classification metrics such as precision, recall, f1-score and support for each class."""

        y_pred = self.predict()
        return accuracy_score(self.y_test, y_pred), classification_report(self.y_test, y_pred)

    def grid_search(self, param_grid):
        """Performs a grid search to find the best hyperparameters for the random forest model.

        Parameters:
        -----------
            param_grid (dict): A dictionary of hyperparameters to test.

        Returns:
        -----------
            None
        """
        # 5 fold cross validation + accuracy evaluation metric
        grid_search = GridSearchCV(RandomForestClassifier(
            random_state=self.random_state), param_grid, cv=5, scoring='accuracy')
        self.best_params = grid_search.fit(self.X, self.y).best_params_

        # , max_depth=self.best_params['max_depth'], random_state=self.random_state)
        self.model = RandomForestClassifier(
            n_estimators=self.best_params['n_estimators'])
        self.best_score = grid_search.fit(self.X, self.y).best_score_
    
    def plot_feature_importance(self, top=24):
        """Plots a horizontal bar chart of the top (by default 10) important features in the random forest model.

        Parameters:
        -----------
        top: The number of top important features to display. Default is 10.

        Returns: 
        -----------
        None"""
        plt.figure(figsize=(10, 6))
        importances = self.model.feature_importances_
        indices = np.argsort(importances)
        plt.title("Feature Importance")
        sns.barplot(x=[importances[i] for i in indices[-top:]], y=[self.X.columns[i] for i in indices[-top:]], orient='h')
        plt.yticks(range(top), [self.X.columns[i] for i in indices[-top:]])
        plt.xlabel("Relative Importance")
        #top 24 features list
        self.top_features = [self.X.columns[i] for i in indices[-top:]]
        plt.show()
        return self.top_features


class NeuralNetworkModel:
    def __init__(self, X, y, random_state=RANDOM_STATE):
        """Initializes the NeuralNetworkModel class with the input features X and target variable y,
        and the random state used for reproducibility.

        Parameters:
        -----------
        X : pandas.DataFrame
            The input feature matrix of shape (n_samples, n_features).
        y : pandas.Series
            The target variable of shape (n_samples,).
        random_state : int, default=RANDOM_STATE, which is 42
            The seed value for random number generator used to split the data.

        Returns:
        --------
        None"""

        self.X = X
        self.y = y
        self.random_state = random_state
        self.model = None
        self.history = None
        self.epochs = 100
        self.batch_size = 32
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)

    def build_model(self):
        """Builds and compiles the neural network model using the input features X.

        Parameters:
        -----------
        None

        Returns:
        --------
        None"""

        # Define the model
        self.model = keras.Sequential([
            layers.Dense(64, activation='relu', input_shape=[self.X.shape[1]]),
            layers.Dropout(0.5),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(1, activation='sigmoid')
        ])

        # Compile the model
        self.model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

    def fit(self):
        """Trains the neural network model on the input features X and target variable y.

        Parameters:
        -----------
        X_train : pandas.DataFrame
            The input feature matrix of shape (n_samples, n_features) for training.
        y_train : pandas.Series
            The target variable of shape (n_samples,) for training.
        X_val : pandas.DataFrame
            The input feature matrix of shape (n_samples, n_features) for validation.
        y_val : pandas.Series
            The target variable of shape (n_samples,) for validation.

        Returns:
        --------
        float: The accuracy score of the trained model on the validation data.
        """

        # Fit the model
        self.model.fit(
            self.X_train, self.y_train,
            epochs=self.epochs, batch_size=self.batch_size,
            validation_data=(self.X_test, self.y_test),
            verbose=0
        )

        # Compute the accuracy score
        y_pred = (self.model.predict(self.X_test)).astype("int32")
        score = accuracy_score(self.y_test, y_pred)

        return score

    def predict(self):
        """Predicts the target variable of the test data using the trained neural network model.
        Returns the predicted target variable values.

        Parameters:
        -----------
        None

        Returns:
        --------
        numpy.ndarray: The predicted target variable values
        """

        return (self.model.predict(self.X_test)).astype("int32")

    def score(self):
        """Computes the accuracy score and classification report for the predicted target variable and the actual test target variable.

        Parameters:
        X_test: array-like, shape (n_samples, n_features)
            The test input samples.
        y_test: array-like, shape (n_samples,)
            The true target values for X_test.

        Returns:
        accuracy: float
            The accuracy score of the model on the test data.
        classification_report: str
            A text report of the main classification metrics such as precision, recall, f1-score and support for each class.
        """
        y_pred = self.predict()
        accuracy = accuracy_score(self.y_test, y_pred)
        classificationReport = classification_report(self.y_test, y_pred)
        return accuracy, classificationReport


def train():
    fit_knn = KNN(X, y, n_neighbors=5, test_size=0.2, random_state=42)
    fit_knn.grid_search({'n_neighbors': [20], 'metric': ['manhattan']})

    fit_rf = RandomForest(X, y, random_state=42)
    fit_rf.grid_search({'n_estimators': [40]})

    fit_nn = NeuralNetworkModel(X, y, random_state=42)
    fit_nn.build_model()
    fit_nn.fit()

    return fit_knn, fit_rf, fit_nn
