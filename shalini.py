import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# ==========================================
# LOAD DATA
# ==========================================

df = pd.read_csv("teen_phone_addiction_dataset.csv")
print("data shape:", df.shape)
print(df.head())

# ==========================================
# ADD STUDY HOURS IF NOT PRESENT
# ==========================================

if "Study_Hours" not in df.columns:
    # realistic synthetic study hours
    df["Study_Hours"] = np.clip(
        6 - df["Daily_Usage_Hours"] + np.random.normal(0, 0.8, len(df)),
        0.5, 8
    )

# ==========================================
# PERFORMANCE LEVEL
# ==========================================

def performance_label(row):
    score = (
        row["Academic_Performance"]
        + row["Study_Hours"] * 4
        - row["Daily_Usage_Hours"] * 3
    )
    if score < 60:
        return 0
    elif score <= 80:
        return 1
    else:
        return 2

df["Performance_Level"] = df.apply(performance_label, axis=1)

# ==========================================
# STRESS LEVEL
# ==========================================

def stress_label(row):
    if (
        row["Daily_Usage_Hours"] > 7
        and row["Sleep_Hours"] < 6
        and row["Study_Hours"] < 2
        and row["Social_Interactions"] < 3
    ):
        return 2
    elif row["Daily_Usage_Hours"] > 5 or row["Sleep_Hours"] < 7:
        return 1
    else:
        return 0

df["Stress_Level"] = df.apply(stress_label, axis=1)

# ==========================================
# ANXIETY LEVEL
# ==========================================

def anxiety_label(row):
    if row["Stress_Level"] == 2 and row["Study_Hours"] < 2:
        return 2
    elif row["Stress_Level"] == 1 or row["Daily_Usage_Hours"] > 5:
        return 1
    else:
        return 0

df["Anxiety_Level"] = df.apply(anxiety_label, axis=1)

# ==========================================
# DEPRESSION LEVEL
# ==========================================

def depression_label(row):
    if (
        row["Social_Interactions"] < 3
        and row["Sleep_Hours"] < 6
        and row["Study_Hours"] < 2
    ):
        return 2
    elif row["Stress_Level"] == 1 or row["Study_Hours"] < 3:
        return 1
    else:
        return 0

df["Depression_Level"] = df.apply(depression_label, axis=1)

# ==========================================
# FEATURES & TARGETS
# ==========================================

features = [
    "Age",
    "Gender",
    "School_Grade",
    "Daily_Usage_Hours",
    "Study_Hours",
    "Sleep_Hours",
    "Social_Interactions"
]

X = df[features].copy()

y_perf = df["Performance_Level"]
y_stress = df["Stress_Level"]
y_anxiety = df["Anxiety_Level"]
y_depression = df["Depression_Level"]

# ==========================================
# ENCODING
# ==========================================

encoders = {}
for col in ["Gender", "School_Grade"]:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    encoders[col] = le

# ==========================================
# TRAIN TEST SPLIT
# ==========================================

X_train, X_test, y_train_perf, y_test_perf = train_test_split(
    X, y_perf, test_size=0.2, random_state=42, stratify=y_perf
)

_, _, y_train_stress, y_test_stress = train_test_split(
    X, y_stress, test_size=0.2, random_state=42, stratify=y_stress
)

_, _, y_train_anxiety, y_test_anxiety = train_test_split(
    X, y_anxiety, test_size=0.2, random_state=42, stratify=y_anxiety
)

_, _, y_train_depression, y_test_depression = train_test_split(
    X, y_depression, test_size=0.2, random_state=42, stratify=y_depression
)

# ==========================================
# MODELS
# ==========================================

performance_model = RandomForestClassifier(n_estimators=400, max_depth=12, random_state=42)
stress_model = RandomForestClassifier(n_estimators=400, max_depth=10, random_state=42)
anxiety_model = RandomForestClassifier(n_estimators=300, max_depth=8, random_state=42)
depression_model = RandomForestClassifier(n_estimators=300, max_depth=8, random_state=42)

performance_model.fit(X_train, y_train_perf)
stress_model.fit(X_train, y_train_stress)
anxiety_model.fit(X_train, y_train_anxiety)
depression_model.fit(X_train, y_train_depression)

# ==========================================
# ACCURACY
# ==========================================

print("\n📊 Model Accuracy")
print("----------------------------")
print(f"Academic Performance : {accuracy_score(y_test_perf, performance_model.predict(X_test))*100:.2f}%")
print(f"Stress Level         : {accuracy_score(y_test_stress, stress_model.predict(X_test))*100:.2f}%")
print(f"Anxiety Level        : {accuracy_score(y_test_anxiety, anxiety_model.predict(X_test))*100:.2f}%")
print(f"Depression Level     : {accuracy_score(y_test_depression, depression_model.predict(X_test))*100:.2f}%")

print("\n🧑‍🎓 Enter Student Details")

age = int(input("Age: "))

# Gender input loop
while True:
    gender_input = input("Gender (Male/Female): ").strip().lower()
    gender_map = {"male": "Male", "m": "Male", "female": "Female", "f": "Female"}
    if gender_input in gender_map:
        gender = gender_map[gender_input]
        break
    print("❌ Invalid gender. Try again.")

# Grade input loop
while True:
    grade_input = input("School Grade (9, 10, 11, 12): ").strip()
    grade_map = {
        "9": "9th", "10": "10th", "11": "11th", "12": "12th",
        "9th": "9th", "10th": "10th", "11th": "11th", "12th": "12th"
    }
    if grade_input in grade_map:
        grade = grade_map[grade_input]
        break
    print("❌ Invalid grade. Enter 9–12 only.")

daily_usage = float(input("Daily Phone Usage (hours): "))
study_hours = float(input("Study Hours per Day: "))
sleep_hours = float(input("Sleep Hours per Day: "))
social_interactions = int(input("Social Interactions (1–5): "))


new_student = {
    "Age": age,
    "Gender": gender,
    "School_Grade": grade,
    "Daily_Usage_Hours": daily_usage,
    "Study_Hours": study_hours,
    "Sleep_Hours": sleep_hours,
    "Social_Interactions": social_interactions
}


new_df = pd.DataFrame([new_student])

for col in ["Gender", "School_Grade"]:
    new_df[col] = encoders[col].transform(new_df[col])

perf_pred = performance_model.predict(new_df)[0]
stress_pred = stress_model.predict(new_df)[0]
anxiety_pred = anxiety_model.predict(new_df)[0]
depression_pred = depression_model.predict(new_df)[0]

perf_map = {0: "Low", 1: "Medium", 2: "High"}
mental_map = {0: "Low", 1: "Moderate", 2: "High"}

def give_suggestions(phone_hours, sleep_hours, stress, anxiety, depression):
    suggestions = []

    # Phone usage suggestion
    if phone_hours > 6:
        suggestions.append("📱 Reduce phone usage to 3–4 hours per day.")
    else:
        suggestions.append("📱 Maintain phone usage under 4 hours per day.")

    # Sleep suggestion
    if sleep_hours < 7:
        suggestions.append("😴 Increase sleep to at least 7–8 hours per night.")
    else:
        suggestions.append("😴 Good maintain healthy sleep of 7–8 hours.")

    # Mental health based suggestions
    if stress == 2 or anxiety == 2 or depression == 2:
        suggestions.append("🧠 Take regular breaks, exercise daily, and talk to friends/family.")
    elif stress == 1 or anxiety == 1 or depression == 1:
        suggestions.append("🧘 Practice relaxation techniques and maintain a routine.")
    else:
        suggestions.append("✅ Mental health indicators are good. Keep following healthy habits.")

    return suggestions


def risk_warning(stress, anxiety, depression):
    if stress == 2 and anxiety == 2 and depression == 2:
        return "🚨 HIGH RISK: Student may be experiencing serious mental health issues. Immediate support is recommended."
    
    elif stress == 2 or anxiety == 2 or depression == 2:
        return "⚠️ MODERATE–HIGH RISK: Student shows signs of mental health stress. Lifestyle changes and guidance are advised."
    
    elif stress == 1 or anxiety == 1 or depression == 1:
        return "⚠️ MODERATE RISK: Student should monitor habits and maintain balance."
    
    else:
        return "✅ LOW RISK: Student shows healthy mental and academic indicators."

def predict_with_percentage(model, X_row, label_map):
    probs = model.predict_proba(X_row)[0]
    pred_class = np.argmax(probs)
    confidence = probs[pred_class] * 100
    return label_map[pred_class], confidence


print("\n🔮 Student Prediction Result")
print("----------------------------")
print(f"Age                    : {new_student['Age']} years")
print(f"Study Hours / Day      : {new_student['Study_Hours']} hours")
print(f"Phone Usage / Day      : {new_student['Daily_Usage_Hours']} hours")
print(f"Sleep Hours / Day      : {new_student['Sleep_Hours']} hours")
print(f"Social Interactions    : {new_student['Social_Interactions']}")

print("\n📊 Predicted Outcomes")
print("----------------------------")
print("Academic Performance  :", perf_map[perf_pred])
print("Stress Level          :", mental_map[stress_pred])
print("Anxiety Level         :", mental_map[anxiety_pred])
print("Depression Level      :", mental_map[depression_pred])

suggestions = give_suggestions(
    phone_hours=new_student["Daily_Usage_Hours"],
    sleep_hours=new_student["Sleep_Hours"],
    stress=stress_pred,
    anxiety=anxiety_pred,
    depression=depression_pred
)

print("\n💡 Personalized Suggestions")
print("----------------------------")
for s in suggestions:
    print("-", s)

warning = risk_warning(stress_pred, anxiety_pred, depression_pred)
print("\n🚨 Risk Assessment")
print("----------------------------")
print(warning)



df["Performance_Level"].value_counts().sort_index().plot(kind="bar")
plt.xticks([0, 1, 2], ["Low", "Medium", "High"])
plt.xlabel("Academic Performance Level")
plt.ylabel("Number of Students")
plt.title("Distribution of Academic Performance")
plt.show()