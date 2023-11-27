# ================================================
# HEADER: Importing Necessary Libraries
# DESCRIPTION: This segment imports all the 
# necessary libraries for data analysis, 
# visualization, and machine learning.
# ================================================
import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

warnings.filterwarnings('ignore')

# ================================================
# HEADER: Data Loading and Initial Preprocessing
# DESCRIPTION: This segment loads the training 
# data, splits some columns into multiple columns, 
# and normalizes certain statistics.
# ================================================
train_df = pd.read_csv('Data/test.csv')
for i in ['DFO','OFO','FG','2PT','3PT','4PT','RA','LAYUP','SLAM']:
    train_df[[i+'_mk',i+'_att']] = train_df[i].str.split(' - ',expand=True).astype(int)
    train_df[i+'_pct'] = train_df[i+'_mk'].astype(float)/train_df[i+'_att'].astype(float)
    train_df = train_df.drop([i], axis=1)

train_df['PPFG'] = train_df.PTS.astype(float)/train_df.FG_att.astype(float)
for i in ['PTS','AST','LBR','TO','STPS','HITS']:
    train_df[i] = train_df[i].astype(float)/train_df['MIN'].astype(float)/(80*53*2/(train_df['MIN'].sum().astype(float)))

gunners = train_df[train_df['Pos'] == 'G']
handlers = train_df[train_df['Pos'] == 'H']
stoppers = train_df[train_df['Pos'] == 'S']

train_df = train_df.fillna(0)

# ================================================
# HEADER: Box Scores Data Processing
# DESCRIPTION: This segment processes individual 
# game box scores, adjusts for missing minutes, 
# and calculates win/loss for each team.
# ================================================
boxscores = pd.DataFrame()
for i in range(106):
    sample = pd.read_csv('Data/boxscores/bs'+str(i)+'.csv')
    sample['GTID'] = i
    if (sample['Min'].sum() < 80):
        missing_minutes_players = sample[(sample['Min'] == 0) & (sample['PTS'] + sample['Ast'] + sample['LBR'] + sample['TO'] + sample['STPS'] + sample['Hits'] > 0)].copy()

        if (missing_minutes_players.shape[0] > 0):
            # Distribute the minutes to the players with no minutes proportionally to their cumulative minutes per game

            # Calculate a "contribution score" for each player based on their stats
            missing_minutes_players['score'] = missing_minutes_players['PTS'] + missing_minutes_players['Ast'] + missing_minutes_players['LBR'] + missing_minutes_players['TO'] + missing_minutes_players['STPS'] + missing_minutes_players['Hits']
            
            # Calculate the proportion of the total score for each player
            missing_minutes_players['proportion'] = missing_minutes_players['score'] / missing_minutes_players['score'].sum()
            
            # Distribute the missing minutes based on this proportion
            missing_minutes_players['Min'] = (missing_minutes_players['proportion'] * (80-sample['Min'].sum()))
            
            # Update the main dataframe with the new minutes for these players
            sample.update(missing_minutes_players)
        else:
            sample['Min'] = sample['Min']/sample['Min'].sum()*80
    elif (sample['Min'].sum() > 80):
        sample['Min'] = sample['Min']/sample['Min'].sum()*80
    boxscores = pd.concat([boxscores,sample])

boxscores2 = pd.DataFrame()

for i in range(53):
    teamA = boxscores[boxscores['GTID'] == i*2]
    teamB = boxscores[boxscores['GTID'] == i*2+1]
    aPts = teamA['PTS'].sum() + teamA['PPA'].sum()
    bPts = teamB['PTS'].sum() + teamB['PPA'].sum()
    if aPts > bPts:
        teamA['WIN'] = 1
        teamB['WIN'] = 0
    else:
        teamA['WIN'] = 0
        teamB['WIN'] = 1
    boxscores2 = pd.concat([boxscores2,teamA,teamB])

boxscores = boxscores2

# ================================================
# HEADER: Further Data Preprocessing
# DESCRIPTION: This segment further processes the 
# box scores data, merges it with player data, 
# and normalizes certain statistics.
# ================================================
for i in ['DFO','OFO','FG','2PT','3PT','4PT','RA','Layup','Slam']:
    boxscores[[i+'_mk',i+'_att']] = boxscores[i].str.split(' - ',expand=True).astype(int)
    boxscores[i+'_pct'] = boxscores[i+'_mk'].astype(float)/boxscores[i+'_att'].astype(float)
    boxscores = boxscores.drop([i], axis=1)

boxscores = pd.merge(boxscores,train_df[["Player","Team","Pos"]],on="Player",how="inner")

boxscores['AST'] = boxscores['Ast']
boxscores['HITS'] = boxscores['Hits']
boxscores = boxscores.drop("Hits", axis=1)
boxscores = boxscores.drop("Ast", axis=1)
boxscores['PPFG'] = boxscores.PTS.astype(float)/boxscores.FG_att.astype(float)
for i in ['PTS','AST','LBR','TO','STPS','HITS']:
    boxscores[i] = boxscores[i].astype(float)/boxscores['Min'].astype(float)

bsg = boxscores[boxscores['Pos'] == 'G']
bsh = boxscores[boxscores['Pos'] == 'H']
bss = boxscores[boxscores['Pos'] == 'S']
g2 = train_df[train_df['Pos'] == 'G']
h2 = train_df[train_df['Pos'] == 'H']
s2 = train_df[train_df['Pos'] == 'S']

for i in ['PTS','AST','LBR','TO','STPS','HITS', 'PPFG']:
    bsg[i] = (bsg[i]-gunners[i].mean())*bsg['Min'].astype(float)
    bsh[i] = (bsh[i]-handlers[i].mean())*bsh['Min'].astype(float)
    bss[i] = (bss[i]-stoppers[i].mean())*bss['Min'].astype(float)
    g2[i] = (g2[i]-gunners[i].mean())*bsg['Min'].astype(float)
    h2[i] = (h2[i]-handlers[i].mean())*bsh['Min'].astype(float)
    s2[i] = (s2[i]-stoppers[i].mean())*bss['Min'].astype(float)

boxscores = pd.concat([bsg,bsh,bss])
boxscores = boxscores.fillna(0)
train_df = pd.concat([g2,h2,s2])
train_df = train_df.fillna(0)
teambox = pd.DataFrame()

for i in range(53):
    teamA = boxscores[boxscores['GTID'] == i*2]
    teamB = boxscores[boxscores['GTID'] == i*2+1]
    aRec = teamA.sum()
    aRec.name = i*2
    # Assign sum of all rows of DataFrame as a new Row
    teambox = pd.concat([teambox, aRec.transpose()])
    bRec = teamB.sum()
    bRec.name = i*2+1
    # Assign sum of all rows of DataFrame as a new Row
    teambox = pd.concat([teambox, bRec.transpose()])

boxscoresfull = boxscores

# ================================================
# HEADER: Define Training Data
# DESCRIPTION: This segment defines the training 
# data for the machine learning models.
# ================================================
X_train = boxscores.drop("WIN", axis=1)
X_train = X_train.drop("Player", axis=1)
X_train = X_train.drop("Team", axis=1)
X_train = X_train.drop("Pos", axis=1)
X_train = X_train.drop("GTID", axis=1)
Y_train = boxscores["WIN"]

# ================================================
# HEADER: Five-Fold Cross-Validation on all models
# DESCRIPTION: This segment performs five-fold 
# cross-validation on all models.
# ================================================
models = {
    "SGD": SGDClassifier(max_iter=5, tol=None),
    "SVC": SVC(),
    "GaussianNB": GaussianNB(),
    "DecisionTree": DecisionTreeClassifier(),
    "RandomForest": RandomForestClassifier(),
    "Perceptron": Perceptron(),
    "GradientBoostedTree": GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42),
    "NeuralNetworks": MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, alpha=0.0001, solver='adam', random_state=42)
}

# Evaluate each model using cross_val_score
print("Model Accuracies after Cross Validation:")
for name, model in models.items():
    scores = cross_val_score(model, X_train, Y_train, cv=5)
    print(f" - {name}: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

# ================================================
# HEADER: Five-Fold Cross-Validation on Model (SGD)
# DESCRIPTION: This segment performs five-fold 
# cross-validation on the model used (SGD).
# ================================================
kf = KFold(n_splits=5, shuffle=True, random_state=42)
ensemble_accuracies = []

# Initialize an ensemble of SGD classifiers
sgdArmy = [SGDClassifier(max_iter=5, tol=None) for _ in range(10)]

for train_index, test_index in kf.split(boxscores):
    X_train_fold, X_test_fold = X_train.iloc[train_index], X_train.iloc[test_index]
    Y_train_fold, Y_test_fold = Y_train.iloc[train_index], Y_train.iloc[test_index]

    fold_accuracies = []
    
    # Train and validate each SGD in the ensemble
    for sgd in sgdArmy:
        sgd.fit(X_train_fold, Y_train_fold)
        Y_pred_fold = sgd.predict(X_test_fold)
        accuracy = accuracy_score(Y_test_fold, Y_pred_fold)
        fold_accuracies.append(accuracy)
    
    # Calculate the average accuracy for this fold and append to ensemble_accuracies
    ensemble_accuracies.append(sum(fold_accuracies) / len(fold_accuracies))

# Calculate the overall average accuracy for the ensemble
average_ensemble_accuracy = sum(ensemble_accuracies) / len(ensemble_accuracies)
print(f"\nAverage Accuracy from Cross-Validation (SGD Ensemble): {average_ensemble_accuracy:.2f}")

# ================================================
# HEADER: Machine Learning Model Training
# DESCRIPTION: This segment trains a logistic 
# regression model and multiple SGD classifiers 
# using the processed data.
# ================================================
logreg = LogisticRegression()
logreg.fit(X_train, Y_train)

coeff_df = pd.DataFrame(boxscores.columns.delete([boxscores.columns.get_loc("WIN"),boxscores.columns.get_loc("Player"),boxscores.columns.get_loc("Team"),boxscores.columns.get_loc("Pos")]))
coeff_df.columns = ['Feature']
coeff_df["Correlation"] = pd.Series(logreg.coef_[0])

sgdArmy = []
for i in range(10):
    sgdArmy.append(SGDClassifier(max_iter=5, tol=None))
    sgdArmy[i].fit(X_train, Y_train)

train_dfBt = train_df.drop("GP", axis = 1)
train_dfBt = train_dfBt.drop("Player", axis = 1)
train_dfBt = train_dfBt.drop("Pos", axis = 1)
train_dfBt = train_dfBt.drop("Team", axis = 1)
train_dfBt["Layup_att"] = train_dfBt["LAYUP_att"]
train_dfBt = train_dfBt.drop("LAYUP_att", axis = 1)
train_dfBt["Layup_mk"] = train_dfBt["LAYUP_mk"]
train_dfBt = train_dfBt.drop("LAYUP_mk", axis = 1)
train_dfBt["Layup_pct"] = train_dfBt["LAYUP_pct"]
train_dfBt = train_dfBt.drop("LAYUP_pct", axis = 1)
train_dfBt["Slam_att"] = train_dfBt["SLAM_att"]
train_dfBt = train_dfBt.drop("SLAM_att", axis = 1)
train_dfBt["Slam_mk"] = train_dfBt["SLAM_mk"]
train_dfBt = train_dfBt.drop("SLAM_mk", axis = 1)
train_dfBt["Slam_pct"] = train_dfBt["SLAM_pct"]
train_dfBt = train_dfBt.drop("SLAM_pct", axis = 1)
train_dfBt["Min"] = train_dfBt["MIN"]
train_dfBt = train_dfBt.drop("MIN", axis = 1)
train_dfBt["Stl"] = train_dfBt["STL"]
train_dfBt = train_dfBt.drop("STL", axis = 1)
train_dfBt = train_dfBt[X_train.columns]
train_df["Prediction"] = 0
for i in range(10):
    train_df["Prediction"]  += sgdArmy[i].predict(train_dfBt)

# ================================================
# HEADER: Game Outcome Prediction
# DESCRIPTION: This segment uses the trained 
# models to predict game outcomes based on player 
# statistics and calculates the accuracy of the 
# predictions.
# ================================================
correct = 0

for j in range(53):
    teamA = boxscores[boxscores['GTID'] == j*2]
    teamB = boxscores[boxscores['GTID'] == j*2+1]
    # Assign sum of all rows of DataFrame as a new Row
    A_wins = 0
    B_wins = 0
    Awon = 0
    Bwon = 0
    for dummy, i in teamA.iterrows():
        Aname = i["Team"]
        Awon = i["WIN"]
        for dummy, j in train_df.iterrows():
            if j["Player"]==i["Player"]:
                A_wins += j["Prediction"]*i["Min"]
    for dummy, i in teamB.iterrows():
        Bname = i["Team"]
        Bwon = i["WIN"]
        for dummy, j in train_df.iterrows():
            if j["Player"]==i["Player"]:
                B_wins += j["Prediction"]*i["Min"]
    if A_wins > B_wins:
        correct += Awon
    else:
        correct += Bwon
print(f"\nGame Outcome Prediction Accuracy: {correct/53:.2f}\n")

X_train = boxscores[['STPS', 'HITS', 'VIO']].copy()
logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
acc_log = round(logreg.score(X_train, Y_train) * 100, 2)

coeff_df = pd.DataFrame(X_train.columns)
coeff_df.columns = ['Feature']
coeff_df["Correlation"] = pd.Series(logreg.coef_[0])
print(coeff_df.sort_values(by='Correlation', ascending=False))

# ================================================
# HEADER: Saving Processed Data
# DESCRIPTION: This segment saves the processed 
# box scores and cumulative stats data to CSV files.
# ================================================
boxscores.to_csv("Data/boxscores.csv")
train_df.to_csv("Data/cumulativestats.csv")
