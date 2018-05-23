import numpy as np
import pandas as pd
train = pd.read_csv("data/train.csv")
print(train.shape)
train
train = pd.read_csv("data/train.csv", parse_dates=["datetime"])
print(train.shape)

## Data processing
train["year"] = train["datetime"].dt.year
train["month"] = train["datetime"].dt.month
train["day"] = train["datetime"].dt.day
train["hour"] = train["datetime"].dt.hour
train["minute"] = train["datetime"].dt.minute
print(train.shape)
train[["datetime","year","month","day"]].head()
train["season_spring"] = train["season"]==1
train["season_summer"] = train["season"]==2
train["season_fall"] = train["season"]==3
train["season_winter"] = train["season"]==4
train[["season","season_spring","season_summer","season_fall","season_winter"]].head()
train["windspeed_fillin"] = train["windspeed"]
train.loc[train["windspeed"]==0,"windspeed_fillin"] = train["windspeed"].mean()
train.loc[:, ["windspeed","windspeed_fillin"]]
train["weekend"]=train["workingday"]+train["holiday"]==0
train["season_spring"] = train["season"]==1
train["season_summer"] = train["season"]==2
train["season_fall"] = train["season"]==3
train["season_winter"] = train["season"]==4
train.head(2)
train[["season","season_spring","season_summer","season_fall","season_winter"]].head()
test = pd.read_csv("data/test.csv", parse_dates=["datetime"])

test["year"] = test["datetime"].dt.year
test["month"] = test["datetime"].dt.month
test["day"] = test["datetime"].dt.day
test["hour"] = test["datetime"].dt.hour
test["minute"] = test["datetime"].dt.minute
test["weekend"]=test["workingday"]+test["holiday"]==0
test["windspeed_fillin"] = test["windspeed"]

## Train
feature_names = ["year","month","hour","holiday","season","weather","weekend","workingday","temp","atemp","humidity","windspeed","windspeed_fillin"]
feature_names
X_train = train[feature_names]
print(X_train.shape)
X_train.head()
X_test = test[feature_names]
print(X_test.shape)
X_test.head()
label_name = "count"
y_train = train[label_name]
print(y_train.shape)
y_train.head()

## Use Decision Tree
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(random_state=777)
#model

## model verification
#X_train
from sklearn.model_selection import cross_val_score
score = cross_val_score(model, X_train, y_train, cv=20, scoring="neg_mean_absolute_error").mean()
score = (-1)*score
print("Score = {0:.5f}".format(score))
model.fit(X_train, y_train)
predictions = model.predict(X_test)
print(predictions.shape)
predictions
submission = pd.read_csv("data/sampleSubmission.csv")
print(submission.shape)
submission.head()
submission["count"] = predictions
print(submission.shape)
submission.head()
submission.to_csv("data/baseline-script.csv", index=False)