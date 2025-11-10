import streamlit as st
import pandas as pd
import numpy as np
from ast import literal_eval
from collections import Counter
from pulp import LpProblem, LpVariable, LpMinimize, LpBinary, lpSum, PULP_CBC_CMD
import matplotlib.pyplot as plt
import re


st.set_page_config(page_title="NutriBridge — Smart Meal Planner", layout="wide")
st.title("🍽️ NutriBridge — Smart Nutrition Planner with Pantry Mode")


# --------------------------------------------------------------------------
# LOAD DATA FROM STREAMLIT ARTIFACT STORAGE
# --------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading dataset from cloud (Artifacts)...")
def load_artifact_data():
    csv_path = st.secrets["datasets"]["recipes"]
    return pd.read_csv(csv_path, engine="python", on_bad_lines="skip")


# Fallback: manual upload if Artifacts not found
def load_dataset():
    try:
        df = load_artifact_data()
        st.success("✅ Loaded dataset from Streamlit Cloud Artifacts")
        return df
    except:
        uploaded = st.file_uploader("Upload RAW_recipes_small.csv manually", type=["csv"])
        if uploaded:
            st.info("✅ Loaded uploaded dataset.")
            return pd.read_csv(uploaded, engine="python", on_bad_lines="skip")
        return None


# --------------------------------------------------------------------------
# DATA PREPROCESS
# --------------------------------------------------------------------------
df = load_dataset()
if df is None:
    st.stop()


def normalize_ing(x):
    """removes plural variations onion/onions + punctuation"""
    if not isinstance(x, str):
        return ""
    x = re.sub(r"[^a-zA-Z ]+", "", x.lower().strip())  # keep letters only
    x = x.replace("s ", " ") if x.endswith("s") else x  # onions → onion
    return "_".join(x.split())


def safe_list(x):
    try:
        v = literal_eval(x)
        return list(v) if isinstance(v, (list, tuple)) else []
    except:
        return []


# extract important columns only
df["ingredients"] = df["ingredients"].apply(safe_list)
df["steps"] = df["steps"].apply(safe_list)
df["ing_list"] = df["ingredients"].apply(lambda lst: [normalize_ing(i) for i in lst])

df["calories"] = df["nutrition"].apply(lambda x: literal_eval(x)[0] if isinstance(x,str) else np.nan)
df["protein_g"] = df["nutrition"].apply(lambda x: literal_eval(x)[1] if isinstance(x,str) else np.nan)


# --------------------------------------------------------------------------
# ✅ SIDEBAR — USER INPUT
# --------------------------------------------------------------------------
st.sidebar.header("👤 User Profile")

age = st.sidebar.number_input("Age", min_value=12, max_value=90, value=25)
gender = st.sidebar.selectbox("Gender", ["male", "female"])
height = st.sidebar.number_input("Height (cm)", value=170)
weight = st.sidebar.number_input("Weight (kg)", value=70)

activity = st.sidebar.selectbox(
    "Activity Level",
    ["sedentary","light","moderate","active","very_active"]
)

preferred_cuisines = [c.strip().lower() for c in st.sidebar.text_input(
    "Preferred cuisines (comma separated)", value="indian,italian"
).split(",")]

restrictions = [normalize_ing(r) for r in st.sidebar.text_input(
    "Restricted ingredients (comma separated)", value="peanut,sugar"
).split(",")]

pantry_items = [normalize_ing(p) for p in st.sidebar.text_input(
    "🧺 Pantry items (comma separated)", value="rice, egg, milk"
).split(",")]

mode = st.sidebar.radio("Cuisine Selection Mode", ["Flexible Mode (Recommended)", "Strict"])
pantry_mode = st.sidebar.checkbox("Enable Pantry Mode (use only available ingredients)")


# --------------------------------------------------------------------------
# NUTRITION FORMULAS
# --------------------------------------------------------------------------
act_factor = {"sedentary":1.2,"light":1.375,"moderate":1.55,"active":1.725,"very_active":1.9}

def calorie_target():
    base = 10*weight + 6.25*height - 5*age + (5 if gender=="male" else -161)
    return int(base * act_factor[activity])

def protein_target():
    return int(weight * 0.8)


# --------------------------------------------------------------------------
# MEAL PLANNING FUNCTION (ILP Optimization)
# --------------------------------------------------------------------------
def violates(ingredients):
    """true if ingredient matches restricted list"""
    return any(r in ingredients for r in restrictions)


def match_pantry(ingredients):
    """true if recipe uses at least one item from pantry"""
    return any(p in ingredients for p in pantry_items)


def plan_day(df, last_used_ids):
    CAL_TGT = calorie_target()
    PRO_TGT = protein_target()

    # base filtering: restrict ingredients
    df = df[~df["ing_list"].apply(violates)]

    # cuisine filtering
    if mode == "Strict":
        df = df[df["cuisine"].isin(preferred_cuisines)]

    # pantry mode filtering:
    if pantry_mode:
        df = df[df["ing_list"].apply(match_pantry)]

    if df.empty:
        return None

    # ILP setup
    pool = df.sample(min(500, len(df)))  # sample subset
    rec_map = {idx: r for idx, r in pool.iterrows()}

    prob = LpProblem("Daily_Meal_Plan", LpMinimize)
    x = {i: LpVariable(f"x_{i}", 0, 1, LpBinary) for i in rec_map.keys()}

    # exactly 3 meals (breakfast, lunch, dinner)
    meal_types = ["breakfast","lunch","dinner"]
    df["meal_type"] = df["tags"].apply(lambda t:str(t).lower())
    for m in meal_types:
        prob += lpSum(x[i] for i in x if m in rec_map[i]["meal_type"]) == 1

    total_cal = lpSum([x[i] * rec_map[i]["calories"] for i in x])
    total_pro = lpSum([x[i] * rec_map[i]["protein_g"] for i in x])

    prob += total_cal >= CAL_TGT * 0.9
    prob += total_cal <= CAL_TGT * 1.1
    prob += total_pro >= PRO_TGT * 0.8

    prob.solve(PULP_CBC_CMD(msg=0))

    chosen = []
    for i, var in x.items():
        if var.value() == 1:
            last_used_ids.add(rec_map[i]["id"])
            chosen.append(rec_map[i])

    return chosen


# --------------------------------------------------------------------------
# GENERATE WEEKLY PLAN
# --------------------------------------------------------------------------
st.subheader("📅 Generate Your 7-Day Weekly Meal Plan")

if st.button("Generate Plan"):
    week = []
    shopping = Counter()
    summary = []
    used_ids = set()

    for day in range(1, 8):
        meals = plan_day(df, used_ids)
        if not meals:
            st.error("No valid recipes found with given filters!")
            st.stop()

        total_cal = total_pro = 0

        for r in meals:
            total_cal += r["calories"]
            total_pro += r["protein_g"]

            week.append([
                day, r["name"], ", ".join(r["ingredients"]),
                " → ".join(r["steps"]),
                r["calories"], r["protein_g"]
            ])

            # shopping list
            for ing in r["ing_list"]:
                shopping[ing] += 1

        summary.append([day, total_cal, total_pro])

    week_df = pd.DataFrame(week, columns=["day","recipe","ingredients","steps","calories","protein"])
    week_df["calories"] = week_df["calories"].astype(str) + " cal"
    week_df["protein"] = week_df["protein"].astype(str) + " g"

    summary_df = pd.DataFrame(summary, columns=["day","calories","protein (g)"])

    st.success("✅ Weekly Meal Plan Generated!")

    st.dataframe(week_df)
    st.download_button("⬇ Download Weekly Plan CSV", week_df.to_csv(index=False), "meal_plan.csv")


    # ----------------------------------------------------------------------
    # SHOPPING LIST (remove universal ingredients & plural duplicates)
    # ----------------------------------------------------------------------
    universal = ["salt","water","oil","pepper"]
    shopping = {ing:qty for ing,qty in shopping.items() if ing not in universal}

    shopping_df = pd.DataFrame(shopping.items(), columns=["Ingredient","Qty"])
    st.subheader("🛒 Shopping List")
    st.dataframe(shopping_df)
    st.download_button("⬇ Download Shopping List CSV", shopping_df.to_csv(index=False), "shopping_list.csv")


    # ----------------------------------------------------------------------
    # GRAPHS
    # ----------------------------------------------------------------------
    st.subheader("📊 Daily Trend")

    fig, ax = plt.subplots()
    ax.plot(summary_df["day"], summary_df["calories"], marker="o")
    ax.set_xlabel("Day")
    ax.set_ylabel("Calories")
    st.pyplot(fig)

