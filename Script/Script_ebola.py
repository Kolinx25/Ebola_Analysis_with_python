"""
This script analyzes 200 confirmed Ebola cases recorded in Sierra Leone
during the early months of the 2014 West Africa outbreak. The analysis
covers case demographics, geographic distribution, temporal trends,
and a basic estimation of the reproduction number (R0) during the
exponential growth phase.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from scipy.stats import linregress
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# consistent look across all plots
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["figure.dpi"] = 130


#LOAD & FIRST LOOK AT THE DATA

df = pd.read_csv("C:/Users/Collins Amoo/Desktop/ebola_sierra_leone.csv")

print("Shape:", df.shape)
print("\nColumn types:\n", df.dtypes)
print("\nMissing values:\n", df.isnull().sum())
print("\nFirst few rows:\n", df.head())

# parse dates — they come in as strings
df["date_of_onset"]  = pd.to_datetime(df["date_of_onset"])
df["date_of_sample"] = pd.to_datetime(df["date_of_sample"])

# how long did sample collection take after symptom onset?
df["days_to_sample"] = (df["date_of_sample"] - df["date_of_onset"]).dt.days

print("\nOutbreak window:")
print("  First case onset :", df["date_of_onset"].min().date())
print("  Last case onset  :", df["date_of_onset"].max().date())
print("  Duration         :", (df["date_of_onset"].max() - df["date_of_onset"].min()).days, "days")


#DEMOGRAPHICS 
# age has 4 missing values — fill with median, which is reasonable for skewed health data
median_age = df["age"].median()
df["age"] = df["age"].fillna(median_age)

print(f"\nAge summary (after imputation, median = {median_age}):")
print(df["age"].describe().round(1))

# sex breakdown
sex_counts = df["sex"].value_counts()
print("\nSex distribution:\n", sex_counts)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Case Demographics — Sierra Leone Ebola Outbreak (2014)", fontsize=13, fontweight="bold")

# age distribution
axes[0].hist(df["age"], bins=20, color="#2c6e9e", edgecolor="white", linewidth=0.6)
axes[0].axvline(median_age, color="#e05c2c", linestyle="--", linewidth=1.4, label=f"Median age: {median_age:.0f}")
axes[0].set_xlabel("Age (years)")
axes[0].set_ylabel("Number of cases")
axes[0].set_title("Age Distribution of Cases")
axes[0].legend()

# sex breakdown
colors = ["#2c6e9e", "#e05c2c"]
axes[1].bar(sex_counts.index, sex_counts.values, color=colors, width=0.5, edgecolor="white")
axes[1].set_xlabel("Sex")
axes[1].set_ylabel("Number of cases")
axes[1].set_title("Cases by Sex")
for i, v in enumerate(sex_counts.values):
    axes[1].text(i, v + 0.5, str(v), ha="center", fontsize=11, fontweight="bold")

plt.tight_layout()
plt.savefig("demographics.png", bbox_inches="tight")
plt.show()


#GEOGRAPHIC DISTRIBUTION 

district_counts = df["district"].value_counts().reset_index()
district_counts.columns = ["district", "cases"]

print("\nCases by district:\n", district_counts)

fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.barh(district_counts["district"], district_counts["cases"],
               color="#2c6e9e", edgecolor="white")
ax.set_xlabel("Number of confirmed cases")
ax.set_title("Geographic Distribution of Cases by District", fontweight="bold")

# label bars
for bar, val in zip(bars, district_counts["cases"]):
    ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
            str(val), va="center", fontsize=10)

ax.invert_yaxis()
plt.tight_layout()
plt.savefig("district_distribution.png", bbox_inches="tight")
plt.show()

# also look at sex breakdown within each district
district_sex = df.groupby(["district", "sex"]).size().unstack(fill_value=0)
print("\nDistrict x Sex breakdown:\n", district_sex)


#TEMPORAL TREND 

# daily case counts by onset date
daily_cases = (
    df.groupby("date_of_onset")
    .size()
    .reset_index(name="cases")
    .sort_values("date_of_onset")
)

# 7-day rolling average smooths out day-of-week noise
daily_cases["rolling_7d"] = daily_cases["cases"].rolling(window=7, min_periods=1).mean()

# cumulative cases — useful for spotting acceleration
daily_cases["cumulative"] = daily_cases["cases"].cumsum()

peak_day = daily_cases.loc[daily_cases["cases"].idxmax()]
print(f"\nPeak single-day incidence: {peak_day['cases']} cases on {peak_day['date_of_onset'].date()}")

fig, ax = plt.subplots(figsize=(13, 5))
ax.bar(daily_cases["date_of_onset"], daily_cases["cases"],
       color="#2c6e9e", alpha=0.5, label="Daily cases", width=0.9)
ax.plot(daily_cases["date_of_onset"], daily_cases["rolling_7d"],
        color="#e05c2c", linewidth=2.2, label="7-day rolling average")
ax.axvline(peak_day["date_of_onset"], color="#333333", linestyle="--",
           linewidth=1.2, label=f"Peak: {peak_day['date_of_onset'].date()}")

ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
plt.xticks(rotation=45)
ax.set_xlabel("Date of symptom onset")
ax.set_ylabel("Number of cases")
ax.set_title("Daily Case Incidence — Sierra Leone Ebola Outbreak (2014)", fontweight="bold")
ax.legend()
plt.tight_layout()
plt.savefig("temporal_trend.png", bbox_inches="tight")
plt.show()


# WAS THE OUTBREAK GROWING OR SLOWING? 

# look at the last 2 weeks of data to answer this
cutoff = df["date_of_onset"].max() - pd.Timedelta(days=14)
recent = daily_cases[daily_cases["date_of_onset"] >= cutoff].copy()
earlier = daily_cases[daily_cases["date_of_onset"] < cutoff].copy()

avg_recent  = recent["cases"].mean()
avg_earlier = earlier["cases"].mean()

print(f"\nAverage daily cases (before last 2 weeks): {avg_earlier:.1f}")
print(f"Average daily cases (last 2 weeks)       : {avg_recent:.1f}")

if avg_recent > avg_earlier:
    print(">> Outbreak was GROWING by end of June 2014")
else:
    print(">> Outbreak appeared to be SLOWING by end of June 2014")


# R0 ESTIMATION 
"""
R0 (the basic reproduction number) is the average number of secondary
cases one infected person generates in a fully susceptible population.
R0 > 1 means the outbreak is expanding; R0 < 1 means it is declining.

Method: During the early exponential growth phase, case counts grow as:
    C(t) = C(0) * exp(r * t)
So log(C(t)) grows linearly with slope r (the growth rate).

We then convert r to R0 using:
    R0 = 1 + r * T_serial
where T_serial is the serial interval — the average time between symptom
onset in a primary case and symptom onset in the person they infect.

For Ebola, the serial interval is approximately 15 days
(Camacho et al., 2014; WHO Ebola Response Team, 2014).
"""

SERIAL_INTERVAL = 15  # days — standard Ebola estimate from literature

# daily counts are noisy — cumulative cases give a much cleaner exponential signal
# use the first 30 days as the exponential growth phase
daily_cases["cumulative"] = daily_cases["cases"].cumsum()

exp_start = daily_cases["date_of_onset"].min()
exp_end   = exp_start + pd.Timedelta(days=30)
exp_phase = daily_cases[daily_cases["date_of_onset"] <= exp_end].copy()

exp_phase["day_number"] = (exp_phase["date_of_onset"] - exp_phase["date_of_onset"].min()).dt.days
exp_phase["log_cases"]  = np.log(exp_phase["cumulative"])

slope, intercept, r_value, p_value, std_err = linregress(
    exp_phase["day_number"], exp_phase["log_cases"]
)

r0 = 1 + slope * SERIAL_INTERVAL


print(f"  Exponential growth rate (r) : {slope:.4f} per day")
print(f"  Serial interval (T)         : {SERIAL_INTERVAL} days (Ebola literature)")
print(f"  Estimated R0                : {r0:.2f}")
print(f"  R² of exponential fit       : {r_value**2:.3f}")
print(f"  p-value                     : {p_value:.4f}")


if r0 > 1:
    print(f"  >> R0 = {r0:.2f} > 1: outbreak was expanding during this phase")
else:
    print(f"  >> R0 = {r0:.2f} < 1: outbreak was self-limiting during this phase")

# visualize the exponential fit
fig, ax = plt.subplots(figsize=(10, 5))
ax.scatter(exp_phase["day_number"], exp_phase["cumulative"],
           color="#2c6e9e", zorder=3, label="Cumulative cases (exponential phase)")

# fitted exponential curve
x_fit = np.linspace(0, exp_phase["day_number"].max(), 100)
y_fit = np.exp(intercept + slope * x_fit)
ax.plot(x_fit, y_fit, color="#e05c2c", linewidth=2,
        label=f"Exponential fit (R² = {r_value**2:.2f})")

ax.set_xlabel("Days since first case")
ax.set_ylabel("Cumulative confirmed cases")
ax.set_title(f"Exponential Growth Phase — Estimated R0 = {r0:.2f}", fontweight="bold")
ax.legend()
plt.tight_layout()
plt.savefig("r0_estimation.png", bbox_inches="tight")
plt.show()


#SAMPLE COLLECTION DELAY 
# how quickly were cases being identified and sampled after onset?
# shorter delay = better surveillance response

print("\nDays from symptom onset to sample collection:")
print(df["days_to_sample"].describe().round(1))

fig, ax = plt.subplots(figsize=(9, 4))
ax.hist(df["days_to_sample"], bins=15, color="#2c6e9e", edgecolor="white", linewidth=0.6)
ax.axvline(df["days_to_sample"].median(), color="#e05c2c", linestyle="--", linewidth=1.5,
           label=f"Median: {df['days_to_sample'].median():.0f} days")
ax.set_xlabel("Days from onset to sample")
ax.set_ylabel("Number of cases")
ax.set_title("Surveillance Response — Time from Onset to Sample Collection", fontweight="bold")
ax.legend()
plt.tight_layout()
plt.savefig("sample_delay.png", bbox_inches="tight")
plt.show()



#RANDOM FOREST — PREDICTING DETECTION SPEED 
"""
Question: what patient and geographic factors predict whether a case
was detected quickly (within 5 days of symptom onset) or late?

This matters practically — late detection means longer time in the
community while infectious, which drives onward transmission.

Target variable: early_detection (1 = sampled within 5 days, 0 = later)
Features: age, sex, district
"""

# build the target — 5 days is the median, so this splits the data roughly 50/50
df["early_detection"] = (df["days_to_sample"] <= 5).astype(int)

print(f"\nDetection split:")
print(f"  Early (<=5 days) : {df['early_detection'].sum()} cases")
print(f"  Late  (>5 days)  : {(df['early_detection'] == 0).sum()} cases")

# encode categoricals — RF needs numbers
le_sex      = LabelEncoder()
le_district = LabelEncoder()

df["sex_enc"]      = le_sex.fit_transform(df["sex"])
df["district_enc"] = le_district.fit_transform(df["district"])

features = ["age", "sex_enc", "district_enc"]
X = df[features]
y = df["early_detection"]

# 80/20 split — stratified to keep class balance in both sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTrain size: {len(X_train)}  |  Test size: {len(X_test)}")

# fit the model — 200 trees, balanced class weights since split isn't perfect
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=4,
    class_weight="balanced",
    random_state=42
)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Late detection", "Early detection"]))

# confusion matrix
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Random Forest — Predicting Ebola Case Detection Speed", fontsize=13, fontweight="bold")

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=["Late (>5 days)", "Early (<=5 days)"])
disp.plot(ax=axes[0], colorbar=False, cmap="Blues")
axes[0].set_title("Confusion Matrix")

# feature importance — which factor mattered most?
importances = pd.Series(rf.feature_importances_,
                        index=["Age", "Sex", "District"]).sort_values()

colors = ["#2c6e9e" if v < importances.max() else "#e05c2c" for v in importances.values]
axes[1].barh(importances.index, importances.values, color=colors, edgecolor="white")
axes[1].set_xlabel("Feature importance (mean decrease in impurity)")
axes[1].set_title("What Predicts Detection Speed?")

for i, v in enumerate(importances.values):
    axes[1].text(v + 0.001, i, f"{v:.3f}", va="center", fontsize=10)

plt.tight_layout()
plt.savefig("random_forest_results.png", bbox_inches="tight")
plt.show()

print("\nFeature importances:")
for feat, imp in zip(["Age", "Sex", "District"], rf.feature_importances_):
    print(f"  {feat:<12}: {imp:.4f}")
