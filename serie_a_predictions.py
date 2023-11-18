import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score

matches = pd.read_csv("serie_a_matches.csv", index_col=0)
# Converting object to datatime so it's numerical
matches["Date"] = pd.to_datetime(matches["Date"])
# Converting from string into category into numbers
matches["Venue_Code"] = matches["Venue"].astype("category").cat.codes
matches["Opp_Code"] = matches["Opponent"].astype("category").cat.codes
# Adding a new column that takes the time and just says the hour 
matches["Hour"] = matches["Time"].str.replace(":.+", "", regex=True).astype("int")
matches["Day_Code"] = matches["Date"].dt.dayofweek
matches["Target"] = (matches["Result"] == "W").astype("int")

rf = RandomForestClassifier(n_estimators=50, min_samples_split=10, random_state=1)
train = matches[matches["Date"] < '2023-01-01']
test = matches[matches["Date"] > '2023-01-01']
predictors = ["Venue_Code", "Opp_Code", "Hour", "Day_Code"]
rf.fit(train[predictors], train["Target"])
preds = rf.predict(test[predictors])
acc = accuracy_score(test["Target"], preds)

combined = pd.DataFrame(dict(actual=test["Target"], prediction=preds))
pd.crosstab(index=combined["actual"], columns=combined["prediction"])
precision_score(test["Target"], preds)

grouped_matches = matches.groupby("Team")
group = grouped_matches.get_group("Roma")


def rolling_averages(group, cols, new_cols):
    group = group.sort_values("Date")
    rolling_stats = group[cols].rolling(3, closed='left').mean()
    group[new_cols] = rolling_stats
    group = group.dropna(subset=new_cols)
    return group


cols = ["GF", "GA", "Sh", "SoT", "Dist", "FK", "PK", "PKatt"]
new_cols = [f"{c}_rolling" for c in cols]

rolling_averages(group, cols, new_cols)

matches_rolling = matches.groupby("Team").apply(lambda x: rolling_averages(x, cols, new_cols))
matches_rolling = matches_rolling.droplevel("Team")
matches_rolling.index = range(matches_rolling.shape[0])


def make_predictions(data, predictors):
    train = data[data["Date"] < '2023-01-01']
    test = data[data["Date"] > '2023-01-01']
    rf.fit(train[predictors], train["Target"])
    preds = rf.predict(test[predictors])
    combined = pd.DataFrame(dict(actual=test["Target"], predicted=preds), index=test.index)
    precision = precision_score(test["Target"], preds)
    return combined, precision


combined, precision = make_predictions(matches_rolling, predictors + new_cols)
combined = combined.merge(matches_rolling[["Date", "Team", "Opponent", "Result"]], left_index=True, right_index=True)
merged = combined.merge(combined, left_on=["Date", "Team"], right_on=["Date", "Opponent"])
merged[(merged["predicted_x"] == 1) & (merged["predicted_y"] == 0)]["actual_x"].value_counts()
merged.to_csv("serie_a_predictions.csv")

