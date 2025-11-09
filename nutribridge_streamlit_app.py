import streamlit as st
import pandas as pd
import numpy as np
from ast import literal_eval
from pulp import LpProblem, LpVariable, LpMinimize, LpBinary, lpSum, PULP_CBC_CMD
from collections import Counter
import matplotlib.pyplot as plt

st.set_page_config(page_title="NutriBridge", layout="wide")

# ===========================================================
# Helper Functions
# ===========================================================

def normalize_ing(x):
    if not isinstance(x, str):
        return ""
    s = x.lower().strip()
    return "_".join(s.replace(",", "").replace(".", "").replace("-", " ").split())

def safe_list(x):
    try:
        v = literal_eval(x)
        return list(v) if isinstance(v, (list, tuple)) else []
    except:
        return []

def detect_meal(name, tags):
    t = (str(name) + str(tags)).lower()
    if "breakfast" in t: return "breakfast"
    if "lunch" in t: return "lunch"
    if "dinner" in t: return "dinner"
    return "lunch"  # fallback ensures 3 meals/day exist

def detect_cuisine(name, tags):
    cuisines = ["indian","italian","mexican","chinese","thai","american","japanese","korean"]
    t = (str(name) + str(tags)).lower()
    for c in cuisines:
        if c in t:
            return c
    return "general"

# ===========================================================
# Load Data
# ===========================================================

uploaded = st.file_uploader("Upload RAW_recipes_small.csv", type=["csv"])
if uploaded:
    df = pd.read_csv(uploaded, engine="python", on_bad_lines="skip")

    df["ingredients"] = df["ingredients"].apply(safe_list)
    df["ing_list"] = df["ingredients"].apply(lambda lst: [normalize_ing(i) for i in lst])

    df["calories"] = df["nutrition"].apply(lambda x: literal_eval(x)[0] if isinstance(x,str) else np.nan)
    df["protein_g"] = df["nutrition"].apply(lambda x: literal_eval(x)[1] if isinstance(x,str) else np.nan)
    df["meal_type"] = df.apply(lambda r: detect_meal(r["name"], r["tags"]), axis=1)
    df["cuisine"] = df.apply(lambda r: detect_cuisine(r["name"], r["tags"]), axis=1)
    df["cost_est"] = 20

    df.dropna(subset=["calories","protein_g"], inplace=True)

else:
    df = pd.DataFrame()
    st.warning("⚠ Upload RAW_recipes_small.csv to continue.")

# ===========================================================
# User Inputs
# ===========================================================

st.sidebar.header("User Profile")

age = st.sidebar.number_input("Age", min_value=12, max_value=90, value=25)
gender = st.sidebar.selectbox("Gender", ["male", "female"])
height = st.sidebar.number_input("Height (cm)", value=170)
weight = st.sidebar.number_input("Weight (kg)", value=70)
activity = st.sidebar.selectbox("Activity Level", ["sedentary","light","moderate","active","very_active"])

preferred_cuisines = [c.strip().lower() for c in st.sidebar.text_input(
    "Preferred cuisines (comma separated)", value="indian,italian"
).split(",")]

restrictions = [normalize_ing(r) for r in st.sidebar.text_input(
    "Dietary restrictions", value="sugar,peanut"
).split(",")]

cal_tol = st.sidebar.slider("Calorie tolerance (± %)", 0.05, 0.30, 0.12)
pro_tol = st.sidebar.slider("Protein tolerance (± %)", 0.05, 0.30, 0.12)

act_factor = {"sedentary":1.2,"light":1.375,"moderate":1.55,"active":1.725,"very_active":1.9}

def calorie_target():
    base = 10*weight + 6.25*height - 5*age + (5 if gender=="male" else -161)
    return int(base * act_factor[activity])

def protein_target():
    return int(weight * 0.8)

# ===========================================================
# ILP Daily Planner
# ===========================================================

def violates(ingredients):
    return any(r in ingredients for r in restrictions)

def plan_day(day, df, last_cuis, used):
    df_f = df[df["cuisine"].isin(preferred_cuisines)]
    df_f = df_f[~df_f["ing_list"].apply(violates)]
    if df_f.empty: df_f = df

    CAL_TGT = calorie_target()
    PRO_TGT = protein_target()

    pool = df_f.sample(min(300, len(df_f)))
    rec_map = {(r["meal_type"], int(r["id"])): r for _, r in pool.iterrows()}

    prob = LpProblem(f"Day_{day}", LpMinimize)
    x = {k: LpVariable(f"x_{k}", 0, 1, LpBinary) for k in rec_map.keys()}

    for meal in ["breakfast","lunch","dinner"]:
        prob += lpSum(x[k] for k in x if k[0]==meal) == 1

    total_cal = lpSum([x[k] * rec_map[k]["calories"] for k in x])
    total_pro = lpSum([x[k] * rec_map[k]["protein_g"] for k in x])

    prob += total_cal >= CAL_TGT*(1-cal_tol)
    prob += total_cal <= CAL_TGT*(1+cal_tol)
    prob += total_pro >= PRO_TGT*(1-pro_tol)

    prob.solve(PULP_CBC_CMD(msg=0))

    chosen, cuisines, tot_cal, tot_pro = [], [], 0, 0

    for k, v in x.items():
        if v.value() == 1:
            r = rec_map[k]
            chosen.append(r)
            cuisines.append(r["cuisine"])
            used.add(int(r["id"]))
            tot_cal += r["calories"]
            tot_pro += r["protein_g"]

    return chosen, cuisines, tot_cal, tot_pro, used


# Estimated cost per ingredient category (you can adjust)
COST_MAP = {
    "Produce": 10,     
    "Dairy": 20,
    "Meat": 30,
    "Pantry": 10,
    "Other": 20
}


# ===========================================================
# Generate Weekly Plan
# ===========================================================

if st.button("Generate 7-Day Plan"):

    week = []
    summary = []
    shopping = Counter()
    used_recipes = set()
    last_cuis = []

    for d in range(1, 8):
        meals, last_cuis, tcal, tpro, used_recipes = plan_day(d, df, last_cuis, used_recipes)

        for r in meals:
            week.append([d, r["meal_type"], r["cuisine"], r["name"], r["calories"], r["protein_g"], r["cost_est"]])
            for ing in r["ing_list"]:
                shopping[ing] += 1

        summary.append([d, round(tcal,1), round(tpro,1)])

    week_df = pd.DataFrame(week, columns=["day","meal","cuisine","recipe","calories","protein","cost"])
    summary_df = pd.DataFrame(summary, columns=["day","total_calories","total_protein"])

    st.subheader("Weekly Meal Plan")
    st.dataframe(week_df)

    # Download CSV
    st.download_button("⬇ Download Weekly Plan CSV", week_df.to_csv(index=False), "nutribridge_week.csv")

    st.subheader("Daily Nutrition Summary")
    st.dataframe(summary_df)
    st.download_button("⬇ Download Nutrition Summary CSV", summary_df.to_csv(index=False), "nutribridge_summary.csv")

    # Categorized shopping list
    CATS = {"Produce": [], "Dairy": [], "Meat": [], "Pantry": [], "Other": []}
    def classify(ing):
        ing = ing.lower()
        if any(x in ing for x in ["onion","garlic","tomato","pepper","lettuce","cilantro","ginger"]): return "Produce"
        if any(x in ing for x in ["milk","cheese","butter","yogurt","paneer","ricotta"]): return "Dairy"
        if any(x in ing for x in ["chicken","beef","fish","pork","egg"]): return "Meat"
        if any(x in ing for x in ["rice","flour","salt","oil","bread","pasta"]): return "Pantry"
        return "Other"


    # Build categorized & cost list
    rows = []
    total_cost = 0

    for ing, qty in shopping.items():
        category = classify(ing)
        est_cost = COST_MAP[category] * qty  # qty × category price
        total_cost += est_cost

        rows.append([category, ing, qty, est_cost])

    shopping_df = pd.DataFrame(rows, columns=["Category", "Ingredient", "Quantity", "Estimated Cost"])

    st.subheader("Categorized Shopping List with Cost")
    st.dataframe(shopping_df)

    # Show total cost
    st.success(f"Estimated Total Weekly Grocery Cost: **₹ {round(total_cost, 2)}**")

    # CSV download
    st.download_button(
        "⬇ Download Categorized Shopping List (CSV)",
        shopping_df.to_csv(index=False).encode("utf-8"),
        "nutribridge_shopping_list_with_cost.csv",
        mime="text/csv"
    )

    # ===========================================================
    # Graph 1 — Calories & Protein Trend
    # ===========================================================

    st.subheader("Daily Calories & Protein Trend")

    fig, ax1 = plt.subplots()
    ax1.plot(summary_df.day, summary_df.total_calories, marker='o', color='blue', linewidth=3)
    ax1.set_xlabel("Day")
    ax1.set_ylabel("Calories", color='blue')

    ax2 = ax1.twinx()
    ax2.plot(summary_df.day, summary_df.total_protein, marker='s', color='green', linestyle="--", linewidth=3)
    ax2.set_ylabel("Protein (g)", color='green')

    st.pyplot(fig)

    # ===========================================================
    # Graph 2 — Ingredient Frequency
    # ===========================================================

    st.subheader("Top Ingredients Used (Frequency)")

    freq = pd.DataFrame(shopping.items(), columns=["Ingredient","Qty"])
    freq = freq.sort_values("Qty", ascending=False).head(15)

    plt.figure(figsize=(8,6))
    plt.barh(freq["Ingredient"], freq["Qty"])
    plt.title("Most Used Ingredients")
    plt.xlabel("Count")
    st.pyplot(plt)
