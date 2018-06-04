import numpy as np
import pandas as pd
train = pd.read_csv("data/train.csv")
print(train.shape)
train
train = pd.read_csv("data/train.csv", parse_dates=["datetime"])
print(train.shape)

test = pd.read_csv("data/test.csv", parse_dates=["datetime"])

## Data processing
train["year"] = train["datetime"].dt.year
train["month"] = train["datetime"].dt.month
train["day"] = train["datetime"].dt.day
train["hour"] = train["datetime"].dt.hour
train["minute"] = train["datetime"].dt.minute
train["second"] = train["datetime"].dt.second
train["dayofweek"] = train["datetime"].dt.dayofweek

test["year"] = test["datetime"].dt.year
test["month"] = test["datetime"].dt.month
test["day"] = test["datetime"].dt.day
test["hour"] = test["datetime"].dt.hour
test["minute"] = test["datetime"].dt.minute
test["second"] = test["datetime"].dt.second
test["dayofweek"] = test["datetime"].dt.dayofweek


#train["windspeed_fillin"] = train["windspeed"]
#train.loc[train["windspeed"]==0,"windspeed_fillin"] = train["windspeed"].mean()
#train.loc[:, ["windspeed","windspeed_fillin"]]
trainWind0 = train.loc[train['windspeed'] == 0]
trainWindNot0 = train.loc[train['windspeed'] != 0]

from sklearn.ensemble import RandomForestClassifier

def predict_windspeed(data):
    
    # 풍속이 0인것과 아닌 것을 나누어 준다.
    dataWind0 = data.loc[data['windspeed'] == 0]
    dataWindNot0 = data.loc[data['windspeed'] != 0]
    
    # 풍속을 예측할 피처를 선택한다.
    wCol = ["season", "weather", "humidity", "month", "temp", "year", "atemp"]

    # 풍속이 0이 아닌 데이터들의 타입을 스트링으로 바꿔준다.
    dataWindNot0["windspeed"] = dataWindNot0["windspeed"].astype("str")

    # 랜덤포레스트 분류기를 사용한다.
    rfModel_wind = RandomForestClassifier()

    # wCol에 있는 피처의 값을 바탕으로 풍속을 학습시킨다.
    rfModel_wind.fit(dataWindNot0[wCol], dataWindNot0["windspeed"])

    # 학습한 값을 바탕으로 풍속이 0으로 기록 된 데이터의 풍속을 예측한다.
    wind0Values = rfModel_wind.predict(X = dataWind0[wCol])

    # 값을 다 예측 후 비교해 보기 위해
    # 예측한 값을 넣어 줄 데이터 프레임을 새로 만든다.
    predictWind0 = dataWind0
    predictWindNot0 = dataWindNot0

    # 값이 0으로 기록 된 풍속에 대해 예측한 값을 넣어준다.
    predictWind0["windspeed"] = wind0Values

    # dataWindNot0 0이 아닌 풍속이 있는 데이터프레임에 예측한 값이 있는 데이터프레임을 합쳐준다.
    data = predictWindNot0.append(predictWind0)

    # 풍속의 데이터타입을 float으로 지정해 준다.
    data["windspeed"] = data["windspeed"].astype("float")

    data.reset_index(inplace=True)
    data.drop('index', inplace=True, axis=1)
    
    return data

train = predict_windspeed(train)    




#print(test.head())

## Train
feature_names = ["season", "weather", "temp", "atemp", "humidity", "windspeed",
                 "year", "hour", "dayofweek", "holiday", "workingday"]

#feature_names = ["year","month","hour","weekday","holiday","weather","workingday","temp","atemp","humidity","season"]
#["year","month","hour","holiday","season","weather","weekend","workingday","temp","atemp","humidity","windspeed","windspeed_fillin"]
#,"Mon","Tue","Wen","Thu","Fri","Sat","Sun","season_spring","season_summer","season_fall","season_winter"
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

from sklearn.metrics import make_scorer

def rmsle(predicted_values, actual_values, convertExp=True):

    if convertExp:
        predicted_values = np.exp(predicted_values),
        actual_values = np.exp(actual_values)
        
    # 넘파이로 배열 형태로 바꿔준다.
    predicted_values = np.array(predicted_values)
    actual_values = np.array(actual_values)
    
    # 예측값과 실제 값에 1을 더하고 로그를 씌워준다.
    log_predict = np.log(predicted_values + 1)
    log_actual = np.log(actual_values + 1)
    
    # 위에서 계산한 예측값에서 실제값을 빼주고 제곱을 해준다.
    difference = log_predict - log_actual
    difference = np.square(difference)
    
    # 평균을 낸다.
    mean_difference = difference.mean()
    
    # 다시 루트를 씌운다.
    score = np.sqrt(mean_difference)
    
    return score

rmsle_scorer = make_scorer(rmsle)
rmsle_scorer

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

k_fold = KFold(n_splits=10, shuffle=True, random_state=0)



## Use Decision Tree
from sklearn.ensemble import RandomForestRegressor

max_depth_list = []

model = RandomForestRegressor(n_estimators=100,
                              n_jobs=-1,
                              random_state=0)

y_train_log = np.log1p(y_train)
model.fit(X_train, y_train_log)

preds = model.predict(X_train)
score = rmsle(np.exp(y_train_log),np.exp(preds),False)

print ("RMSLE Value For Random Forest: ",score)

predsTest = model.predict(X_test)

## model verification
#X_train
#score = cross_val_score(model, X_train, y_train, cv=k_fold, scoring=rmsle_scorer)
#score = score.mean()
# 0에 근접할수록 좋은 데이터
#print("Score= {0:.5f}".format(score))



#predictions = model.predict(X_test)
#print(predictions.shape)
#predictions
submission = pd.read_csv("data/sampleSubmission.csv")
print(submission.shape)
submission.head()
submission["count"] = np.exp(predsTest)
print(submission.shape)
submission.head()
submission.to_csv("data/baseline-script.csv", index=False)
