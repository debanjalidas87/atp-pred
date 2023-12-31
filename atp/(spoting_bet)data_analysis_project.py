# -*- coding: utf-8 -*-
"""(Spoting Bet)Data Analysis Project.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1UBXSKvqIhLNbtqx_ag_5Dh-5THVZMEiA
"""

# Commented out IPython magic to ensure Python compatibility.
## Importation of different libraries and packages for data analysis

import pandas as pd
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 5000)
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
# %matplotlib inline
from tabulate import tabulate

# Read the ATP matches dataset CSV file
"""Here, the file related to the project sporting bet is loaded for the analysis"""
df_atp = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/atp_data.csv')

from google.colab import drive
drive.mount('/content/drive')

"""Data Exploration of the project sporting bet starts here
first o all the data structure of the original dataset is inspected followed by the data Preview, Descriptive statistics, handeling missing values as well as datatype
for the purpose of data exploration, data vizualization and preprocessing

# **1. Data Exploration:**

## **1. Inspect the data structure**
"""

# Check the number of rows and columns
print(df_atp.shape)

# Get a concise summary of the DataFrame
#Displaying the Data type and the count for non-null value

print(df_atp.info())

# View the column names
print(df_atp.columns)

"""## **2. Preview of data**"""

# Display the first few rows
df_atp.head()

# Display the last few rows
df_atp.tail()

"""## **3. Discriptive Statistics**"""

# Statistic description of quantitative data
df_atp.describe()

# Calculate the correlation between columns
df_atp.corr()

"""## **4. Checking and Handeling Missing Values**"""

# Displaying the total number of missing values
display(df_atp.isna())
display(df_atp.sum())
display(df_atp.isna().sum().sum())

# Calculating the percentage of missing values for each column
missing_percentages = (df_atp.isnull().sum() / len(df_atp)) * 100
# display(missing_percentages)

"""## **5. Handeling Datatype**"""

############################################################################################################
######## DataFrame with a column named 'date' of object datatype needed to be converted to datatype 'datetime'

import pandas as pd
df_atp['Date'] = pd.to_datetime(df_atp['Date'])
df_atp['LRank'] = pd.to_numeric(df_atp['LRank'], errors='coerce').astype('Int64')
# Verify the updated datatype
print(df_atp['Date'].dtype)
print(df_atp['LRank'].dtype)
df_atp.info()

"""Here in this part statistical analysis on the exploed dataset will be done followed by the data vizualization to understand the dataset
 more clearly for the further steps

# **2. Statistical Analysis and Data Vizualization**
"""

#############To assess whether a given dataset follows a specific probability distribution or not.##############################
#Importation of the statsmodels library

#Definition of the parameters
mu, sigma = 32, 18

# qqplot
ech = np.random.normal(loc = mu, scale= sigma, size = 4708)
sm.qqplot(ech, fit = True, line = '45')

############### Pearson Correlation test #######################################
#H0 : corr(x = df_atp["prob_elo"], y = df_atp["Wsets"])  = 0
#H1 : corr(x = df_atp["prob_elo"], y = df_atp["Wsets"]) != 0
from scipy.stats import pearsonr
# Calculate the Pearson correlation coefficient and p-value
correlation, p_value = pearsonr(x=df_atp["proba_elo"], y=df_atp["WRank"])

print("p-value:", p_value)
print("coefficient:", correlation)

# correlation between different variables vizulised through Heatmap

# Calculate the correlation matrix
correlation_matrix = df_atp.corr()

# Generate the heatmap
plt.figure(figsize=(15, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap')
# plt.show()

## Vizualization the matches are being distributed among the top players

# Group the data by player name and count the number of occurrences
player_counts = df_atp['Winner'].value_counts()

# Select the top N most competitive players
top_players = player_counts.head(25)

# Create a pie chart for the most competitive players
plt.figure(figsize=(15, 10))
plt.pie(top_players.values, labels=top_players.index, autopct='%1.1f%%')
plt.title('Distribution of Matches Among Top Players')

# Display the pie chart
plt.show()

# comparison between the elo winners and the probability of their winning

# Sort the data by Elo rating
sorted_data = df_atp.sort_values('elo_winner')

# Set color ranges
elo_ranges = [(1800, 2000), (2000, 2200), (2200, 2400), (2400, 2600), (2600, 2800)]

# Create a colormap with different colors for each range
colors = plt.cm.viridis(np.linspace(0, 1, len(elo_ranges)))

# Create a line plot with different colors for each range
plt.figure(figsize=(15, 8))
for i, (elo_min, elo_max) in enumerate(elo_ranges):
    range_data = sorted_data[(sorted_data['elo_winner'] >= elo_min) & (sorted_data['elo_winner'] < elo_max)]
    plt.scatter(range_data['elo_winner'], range_data['proba_elo'], color=colors[i], marker='o', label=f'Elo Range {elo_min}-{elo_max}')

# Set the plot title and labels
plt.title('Winning Probability vs. Elo Rating')
plt.xlabel('Elo Rating')
plt.ylabel('Winning Probability')
plt.legend()

# Display the plot
plt.show()

# Pie Chart vizualization for the Distribution of Surface

surface_distribution = df_atp['Surface'].value_counts()
plt.figure(figsize=(8, 8))
plt.pie(surface_distribution, labels=surface_distribution.index, autopct='%1.1f%%')
plt.title('Surface Distribution')
plt.show()

#Vizualization of top ten winnwers according to the ground/surface

# Filter for top 10 players
top_10_players = df_atp['Winner'].value_counts().nlargest(10).index

# Filter the dataframe for top 10 players and group by surface and winner
top_10_winners_by_surface = df_atp[df_atp['Winner'].isin(top_10_players)].groupby(['Surface', 'Winner']).size().reset_index(name='Wins')

# Sort by surface and number of wins
top_10_winners_by_surface = top_10_winners_by_surface.sort_values(['Surface', 'Wins'], ascending=[True, False])

# Create a separate bar chart for each surface
surfaces = top_10_winners_by_surface['Surface'].unique()
num_surfaces = len(surfaces)

colors = ['red', 'green', 'blue', 'orange']  # Add more colors if needed

plt.figure(figsize=(12, 8))

for i, surface in enumerate(surfaces):
    surface_data = top_10_winners_by_surface[top_10_winners_by_surface['Surface'] == surface][:10]  # Limit to top 10 winners per surface

    plt.subplot(num_surfaces, 1, i+1)
    bars = plt.bar(surface_data['Winner'], surface_data['Wins'], color=colors[i % len(colors)])
    plt.title(f'Top 10 Tournament Winners by Surface: {surface}')
    plt.xlabel('Player')
    plt.ylabel('Number of Wins')
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Display the number on top of each bar
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height, height, ha='center', va='bottom')

plt.show()

"""
# **3. Data cleaning and Preprocessing**

"""

df_atp.head()

# Drop rows with missing values in ['PSW', 'PSL', 'B365W', 'B365L']
df_atp_droped = df_atp.dropna(subset=['PSW', 'PSL', 'B365W', 'B365L'])

# Print the cleaned DataFrame
df_atp_droped.head()

print(df_atp_droped.shape)

"""Step-2. Removing irrelevant columns.

Here for the datset atp_data.csv, two irrelevent columns are found, these are Wsets and Lsets. These two will be removed otherwise the ML model can easily predict and the outcome of the model will be biased.
"""

# Create a list of irrelevant columns to remove
irrelevant_columns = ['Best of', 'Wsets', 'Lsets', 'Comment']

# Remove the irrelevant columns from the DataFrame
df_atp_droped_new = df_atp_droped.drop(irrelevant_columns, axis=1)

# Print the updated DataFrame to verify the changes
df_atp_droped_new.tail(10)

import pandas as pd
import numpy as np

# Rename the columns 'Loser' and 'Winner' to 'Player 1' and 'Player 2'
df_atp_droped_new.rename(columns={'Loser': 'Player 1', 'Winner': 'Player 2', 'WRank' : 'Rank_P2', 'LRank' : 'Rank_P1',
                                                  'PSW' : 'PS_P2', 'PSL' : 'PS_P1', 'B365W' : 'B365_P2', 'B365L' : 'B365_P1',
                                                  'elo_winner' : 'elo_P2', 'elo_loser' : 'elo_P1'}, inplace=True)

df_atp_droped_new.head()

# Target Variable Creation
import pandas as pd
import numpy as np
from prompt_toolkit.styles import named_colors

# Set the random seed for reproducibility
np.random.seed(42)

### conversion of text based values intu neumaric values
df_copy = df_atp_droped_new.copy(deep=True)

def create_target_variable(df):
  """
  """

  target = []

  for index, row in df.iterrows():
    player_1 = row['Player 1']
    player_2 = row['Player 2']
    rank_p2 = row['Rank_P2']
    rank_p1 = row['Rank_P1']
    ps_p2 = row['PS_P2']
    ps_p1 = row['PS_P1']
    b365_p2 = row['B365_P2']
    b365_p1 = row['B365_P1']
    elo_p2 = row['elo_P2']
    elo_p1 = row['elo_P1']

    # Simulate a coin toss to determine the target variable
    coin_toss = np.random.randint(0,2)  # Randomly generate either 0 or 1

    # Define your target variable conditions based on the coin toss
    if coin_toss == 0:
      target.append(0)  # Player 1 is the winner
    else:
      target.append(1)  # Player 2 is the winner
      # Swap 'Player 1' and 'Player 2' columns
      df.at[index, 'Player 1'], df.at[index, 'Player 2'] = player_2, player_1 #take other variables into account

      # Swap 'PSW' and 'PSL' columns
      df.at[index, 'Rank_P1'], df.at[index, 'Rank_P2'] = rank_p2, rank_p1

      # Swap 'PSW' and 'PSL' columns
      df.at[index, 'PS_P1'], df.at[index, 'PS_P2'] = ps_p2, ps_p1

      # Swap 'B365W' and 'B365L' columns
      df.at[index, 'B365_P1'], df.at[index, 'B365_P2'] = b365_p2, b365_p1

      # Swap 'PSW' and 'PSL' columns
      df.at[index, 'elo_P1'], df.at[index, 'elo_P1'] = elo_p2, elo_p1

  return target

# Create the target variable using the function
df_copy['Winning Player'] = create_target_variable(df_copy)

# Normalization of the features and Date

# iteration of changing string to numaric by column
def change_numaric_by_column(df,col_name, dict_for_df):
  dict_for_df[col_name] = {}
  temp_dict = dict_for_df[col_name]
  value = 0
  for ind, name_string in df[col_name].items():
    if name_string in temp_dict:
      old_val = temp_dict[name_string]
      df.at[ind, col_name] = old_val
    else:
      temp_dict[name_string] = value
      df.at[ind, col_name] = value
      value += 1

# Iteration of changing string to numeric
def change_string_to_numeric_in_df(df):
  df_encoded_dict = {}
  for column in df.columns:
    df_col = df[column]
    if df_col.dtype == 'O':
      change_numaric_by_column(df,column,df_encoded_dict)

change_string_to_numeric_in_df(df_copy)

# Sort the DataFrame by 'Date' in ascending order
df_copy_desc = df_copy.sort_values(by='Date', ascending=False)

# Extract date components into separate columns
df_copy_desc['Year'] = df_copy_desc['Date'].dt.year
df_copy_desc['Month'] = df_copy_desc['Date'].dt.month
df_copy_desc['Day'] = df_copy_desc['Date'].dt.day

df_copy_desc.head(32000)



"""# **Report -2: Modeling Report**"""

# Machine Learning Model Starts here


# All python module associated with MM go here
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Data Splitting

X = df_copy_desc.drop(columns=['Winning Player', 'Date'])
y = df_copy_desc['Winning Player']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42,)

#####Logistic Regression Model
model1_lr = LogisticRegression()
model1_lr.fit(X_train, y_train)

# Evaluate the model on the training data
train_score_lr = model1_lr.score(X_train, y_train)
print('train score : ', train_score_lr)

# Evaluate the model on the test data
test_score_lr = model1_lr.score(X_test, y_test)
print('test score : ', test_score_lr)

# Make predictions on the test set
y_pred_lr = model1_lr.predict(X_test)

# Generate classification report
classification_rep = classification_report(y_test, y_pred_lr)
print("Classification Report:\n", classification_rep)

# Generate confusion matrix
confusion_mat = confusion_matrix(y_test, y_pred_lr)
print("Confusion Matrix:\n", confusion_mat)

##### Decision Tree

# Decision Tree initialization
model2_dt = DecisionTreeClassifier()
# Fit transformation
model2_dt.fit(X_train, y_train)

# Evaluate the model on the training data
train_score_dt = model2_dt.score(X_train, y_train)
print('TRAIN SCORES : ', train_score_dt)

# Evaluate the model on the test data
test_score_dt = model2_dt.score(X_test, y_test)
print('TEST SCORES : ', test_score_dt)

# Make predictions on the test set
y_pred_dt = model2_dt.predict(X_test)

# Generate classification report
classification_rep = classification_report(y_test, y_pred_dt)
print("Classification Report:\n", classification_rep)

# Generate confusion matrix
confusion_mat = confusion_matrix(y_test, y_pred_dt)
print("Confusion Matrix:\n", confusion_mat)

### Random forest


# Random Forest Classifier
model3_rf = RandomForestClassifier()
model3_rf.fit(X_train, y_train)

# Evaluate the model on the training data
train_score_rf = model3_rf.score(X_train, y_train)
print('TRAIN SCORES : ', train_score_rf)

# Evaluate the model on the test data
test_score_rf = model3_rf.score(X_test, y_test)
print('TEST SCORES : ', test_score_rf)

# Make predictions on the test set
y_pred_rf = model3_rf.predict(X_test)

# Generate classification report
classification_rep = classification_report(y_test, y_pred_rf)
print("Classification Report:\n", classification_rep)

# Generate confusion matrix
confusion_mat = confusion_matrix(y_test, y_pred_rf)
print("Confusion Matrix:\n", confusion_mat)

######SVC
model4_svm = SVC(max_iter=10)
model4_svm.fit(X_train, y_train)

# Evaluate the model on the training data
train_score_svm = model4_svm.score(X_train, y_train)
print('TRAIN SCORES : ', train_score_svm)

# Evaluate the model on the test data
test_score_svm = model4_svm.score(X_test, y_test)
print('TEST SCORES : ', test_score_svm)

model4_svm = SVC()
model4_svm.fit(X_train, y_train)
# Make predictions on the test set
y_pred_svm = model4_svm.predict(X_test)

# Generate classification report
classification_rep = classification_report(y_test, y_pred_svm)
print("Classification Report:\n", classification_rep)

# Generate confusion matrix
confusion_mat = confusion_matrix(y_test, y_pred_svm)
print("Confusion Matrix:\n", confusion_mat)

#######  KNN
model5_knn = KNeighborsClassifier()
model5_knn.fit(X_train, y_train)

# Evaluate the model on the training data
train_score_knn = model5_knn.score(X_train, y_train)
print("Training Score:", train_score_knn)

# Evaluate the model on the test data
test_score_knn = model5_knn.score(X_test, y_test)
print("Test Score:", test_score_knn)

# Make predictions on the test set
y_pred_knn = model5_knn.predict(X_test)

# Generate classification report
classification_rep = classification_report(y_test, y_pred_knn)
print("Classification Report:\n", classification_rep)

# Generate confusion matrix
confusion_mat = confusion_matrix(y_test, y_pred_knn)
print("Confusion Matrix:\n", confusion_mat)

# Assuming df_atp is your DataFrame containing the ATP dataset
# Assuming target variable is 'Winning Player' with 1 representing Player 1 won and 2 representing Player 2 won

# Create a new column to store the bookmaker's prediction (1 or 2) based on the highest value between PSL and PSW
df_copy_desc['Bookmaker_Prediction1'] = df_copy_desc[['PS_P1', 'PS_P2']].idxmax(axis=1).apply(lambda x: 0 if 'P1' in x else 1)

# Compare the bookmaker's prediction with the target variable and create a new column for correctness
df_copy_desc['Bookmaker_Correct'] = df_copy_desc['Bookmaker_Prediction1'] == df_copy_desc['Winning Player']

# Calculate the accuracy of the bookmaker's predictions
bookmaker_accuracy = df_copy_desc['Bookmaker_Correct'].mean()

# Print the accuracy
print("PS Bookmaker Accuracy:", bookmaker_accuracy)

# Create a new column to store the bookmaker's prediction (1 or 2) based on the highest value between B365_P1 and B365_P2
df_copy_desc['Bookmaker_Prediction2'] = df_copy_desc[['B365_P1', 'B365_P2']].idxmax(axis=1).apply(lambda x: 0 if 'P1' in x else 1)

# Compare the bookmaker's prediction with the target variable and create a new column for correctness
df_copy_desc['Bookmaker_Correct_B365'] = df_copy_desc['Bookmaker_Prediction2'] == df_copy_desc['Winning Player']

# Calculate the accuracy of the bookmaker's predictions
bookmaker_accuracy_b365 = df_copy_desc['Bookmaker_Correct_B365'].mean()

# Print the accuracy
print("B365 Bookmaker Accuracy:", bookmaker_accuracy_b365)

# Calculate the accuracy of each model on the test data
model_accuracies = {
    'Logistic Regression': test_score_lr,
    'Decision Tree': test_score_dt,
    'Random Forest': test_score_rf,
    'SVM': test_score_svm,
    'KNN': test_score_knn
}

# Calculate the accuracy of the bookmakers' predictions based on PS (PSL and PSW)
bookmaker_accuracy_ps = df_copy_desc['Bookmaker_Correct'].mean()

# Calculate the accuracy of the bookmakers' predictions based on B365 (B365_P1 and B365_P2)
bookmaker_accuracy_b365 = df_copy_desc['Bookmaker_Correct_B365'].mean()

# Create a DataFrame to store the accuracy values
comparison_table = pd.DataFrame({
    'Model': list(model_accuracies.keys()) + ['Bookmakers PS', 'Bookmakers B365'],
    'Accuracy': list(model_accuracies.values()) + [bookmaker_accuracy_ps, bookmaker_accuracy_b365]
})

# Sort the DataFrame by accuracy in descending order
comparison_table = comparison_table.sort_values(by='Accuracy', ascending=False)

# Display the comparison table
print(comparison_table)

import matplotlib.pyplot as plt

# Data
models = list(model_accuracies.keys())
accuracies = list(model_accuracies.values())
bookmaker_ps_accuracy = bookmaker_accuracy_ps
bookmaker_b365_accuracy = bookmaker_accuracy_b365

# Add bookmakers' accuracies to the list of accuracies
accuracies.extend([bookmaker_ps_accuracy, bookmaker_b365_accuracy])

# Plot
plt.figure(figsize=(10, 6))
plt.bar(models + ['Bookmakers PS', 'Bookmakers B365'], accuracies, color=['blue', 'green', 'orange', 'red', 'purple', 'grey', 'brown'])
plt.xlabel('Models')
plt.ylabel('Accuracy')
plt.title('Model vs. Bookmakers Accuracy')
plt.ylim(0, 1)
plt.xticks(rotation=45)

# Display the values on top of the bars
for i, v in enumerate(accuracies):
    plt.text(i, v, f"{v:.3f}", ha='center', va='bottom')

plt.tight_layout()
plt.show()