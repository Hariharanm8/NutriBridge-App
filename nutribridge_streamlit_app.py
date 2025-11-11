##############################
# NutriBridge Streamlit App
##############################

import streamlit as st
import pandas as pd
import numpy as np
from ast import literal_eval
from pulp import LpProblem, LpVariable, LpMinimize, LpBinary, lpSum, PULP_CBC_CMD
from collections import Counter
import matplotlib.pyplot as plt
import re

# NLTK for ingredient normalization
import nltk
from nltk.stem import PorterStemmer

# Download required NLTK data (runs once)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

# Initialize stemmer
stemmer = PorterStemmer()

st.set_page_config(page_title="Nutri-Bridge: Weekly Nutrition Planner", layout="wide")
st.title("Nutri-Bridge — AI Weekly Nutrition Planner")


# ===========================================================
# Helper Functions
# ===========================================================

def normalize_ing(x):
    """Normalize ingredient string with stemming for singular/plural handling"""
    if not isinstance(x, str):
        return ""
    # Remove special characters, lowercase
    cleaned = re.sub(r'[^\w\s]', '', x.lower())
    words = cleaned.split()
    # Stem each word to handle singular/plural (onion/onions -> onion)
    stemmed = [stemmer.stem(w) for w in words if len(w) > 2]
    return "_".join(stemmed) if stemmed else ""


def is_in_pantry(recipe_ing, pantry_set):
    """Check if recipe ingredient matches pantry with partial matching"""
    if not recipe_ing:
        return False
    
    recipe_norm = normalize_ing(recipe_ing)
    if not recipe_norm:
        return False
    
    # Exact match
    if recipe_norm in pantry_set:
        return True
    
    # Partial match: check if any pantry item is in recipe ingredient
    # (e.g., "salt" matches "salt_and_pepper")
    for pantry_item in pantry_set:
        if pantry_item and (pantry_item in recipe_norm or recipe_norm in pantry_item):
            return True
    
    return False


def violates_restrictions(ingredients, restrictions):
    """Check restrictions with improved matching"""
    for ing in ingredients:
        ing_norm = normalize_ing(ing)
        for restriction in restrictions:
            if restriction and ing_norm and (restriction in ing_norm or ing_norm in restriction):
                return True
    return False


def safe_list(x):
    """Convert string representation to list safely"""
    try:
        v = literal_eval(x)
        return list(v) if isinstance(v, (list, tuple)) else []
    except:
        return []


def detect_meal(name, tags):
    """Guess breakfast/lunch/dinner based on tags"""
    t = (str(name) + str(tags)).lower()
    if "breakfast" in t: return "breakfast"
    if "dinner" in t: return "dinner"
    return "lunch"


def detect_cuisine(name, tags):
    cuisines = ["indian","italian","mexican","chinese","thai","american","japanese","korean"]
    text = (str(name) + str(tags)).lower()
    for c in cuisines:
        if c in text:
            return c
    return "general"


# ===========================================================
# UPLOAD DATASET
# ===========================================================

uploaded = st.file_uploader("📤 Upload RAW_recipes_small.csv", type=["csv"])

if uploaded:
    df = pd.read_csv(uploaded, engine="python", on_bad_lines="skip")

    df["ingredients"] = df["ingredients"].apply(safe_list)
    df["steps"] = df["steps"].apply(safe_list)

    df["ing_list"] = df["ingredients"].apply(lambda lst: [normalize_ing(i) for i in lst])
    df["calories"] = df["nutrition"].apply(lambda x: literal_eval(x)[0] if isinstance(x,str) else np.nan)
    df["protein_g"] = df["nutrition"].apply(lambda x: literal_eval(x)[1] if isinstance(x,str) else np.nan)

    df["meal_type"] = df.apply(lambda r: detect_meal(r["name"], r["tags"]), axis=1)
    df["cuisine"] = df.apply(lambda r: detect_cuisine(r["name"], r["tags"]), axis=1)

    df.dropna(subset=["calories","protein_g"], inplace=True)

else:
    df = pd.DataFrame()
    st.warning("⚠ Upload RAW_recipes_small.csv to begin.")


# ===========================================================
# USER INPUTS
# ===========================================================

st.sidebar.header("User Profile")

age = st.sidebar.number_input("Age", 12, 90, 25)
gender = st.sidebar.selectbox("Gender", ["male","female"])
height = st.sidebar.number_input("Height (cm)", value=170)
weight = st.sidebar.number_input("Weight (kg)", value=70)
activity = st.sidebar.selectbox("Activity Level", ["sedentary","light","moderate","active","very_active"])

preferred_cuisines = [c.strip().lower() for c in st.sidebar.text_input(
    "Preferred cuisines (comma separated)", "indian,italian"
).split(",")]

restrictions_input = st.sidebar.text_input(
    "Dietary restrictions (comma separated)", "sugar"
)
restrictions = [normalize_ing(r.strip()) for r in restrictions_input.split(",") if r.strip()]

# Pantry Mode
st.sidebar.markdown("---")
st.sidebar.subheader("Pantry Mode")
pantry_items = st.sidebar.text_area("Items already available:", placeholder="e.g., onion, rice, egg")
PANTRY = {normalize_ing(i.strip()) for i in pantry_items.split(",") if i.strip()}

cal_tol = st.sidebar.slider("Calorie tolerance (± %)", 0.05, 0.30, 0.12)
pro_tol = st.sidebar.slider("Protein tolerance (± %)", 0.05, 0.30, 0.12)

# BMR function
ACTIVITY_FACTORS = {"sedentary":1.2,"light":1.375,"moderate":1.55,"active":1.725,"very_active":1.9}

def calorie_target():
    base = 10*weight + 6.25*height - 5*age + (5 if gender=="male" else -161)
    return int(base * ACTIVITY_FACTORS[activity])

def protein_target():
    return int(weight * 0.8)


# ===========================================================
# ILP DAILY MEAL PLANNER
# ===========================================================

def plan_day(day, df, last_cuis, used_recipes):
    df_f = df[df["cuisine"].isin(preferred_cuisines)]
    df_f = df_f[~df_f["ingredients"].apply(lambda ings: violates_restrictions(ings, restrictions))]
    if df_f.empty:
        df_f = df

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

    for k,v in x.items():
        if v.value() == 1:
            r = rec_map[k]
            chosen.append(r)
            cuisines.append(r["cuisine"])
            used_recipes.add(int(r["id"]))
            tot_cal += r["calories"]
            tot_pro += r["protein_g"]

    return chosen, cuisines, tot_cal, tot_pro, used_recipes


# Estimated cost per category
CATEGORY_COST = {"Produce":5, "Dairy":12, "Meat":35, "Pantry":8, "Other":10}


def classify(ing):
    """Assign ingredient category"""
    ing = ing.lower()
    if any(x in ing for x in ["onion","garlic","tomato","pepper","lettuce","cilantro","ginger"]): return "Produce"
    if any(x in ing for x in ["milk","cheese","butter","paneer","yogurt"]): return "Dairy"
    if any(x in ing for x in ["chicken","beef","fish","pork","egg"]): return "Meat"
    if any(x in ing for x in ["rice","flour","salt","oil","bread","pasta"]): return "Pantry"
    return "Other"


# ===========================================================
# GENERATE WEEKLY PLAN
# ===========================================================

if st.button("Generate 7-Day Meal Plan"):

    if df.empty:
        st.error("Upload a dataset first!")
        st.stop()

    week, summary = [], []
    SHOPPING = Counter()
    used_recipes = set()
    last_cuis = []

    for d in range(1,8):
        meals, last_cuis, tcal, tpro, used_recipes = plan_day(d, df, last_cuis, used_recipes)

        for r in meals:
            week.append([d, r["meal_type"], r["cuisine"], r["name"], r["ingredients"], r["steps"],
                         f"{r['calories']} cal", f"{r['protein_g']} g"])

            # Use improved pantry matching
            for ing in r["ingredients"]:
                if not is_in_pantry(ing, PANTRY):
                    normalized = normalize_ing(ing)
                    if normalized:  # Only add if normalization succeeded
                        SHOPPING[normalized] += 1

        summary.append([d, round(tcal,1), round(tpro,1)])


    # WEEK PLAN TABLE (with steps + ingredients)
    week_df = pd.DataFrame(
        week,
        columns=["Day","Meal","Cuisine","Recipe","Ingredients","Steps","Calories","Protein"]
    )
    st.subheader("Weekly Meal Plan")
    st.dataframe(week_df)


    # SUMMARY TABLE
    summary_df = pd.DataFrame(summary, columns=["Day","Calories","Protein (g)"])
    st.subheader("Daily Nutrition Summary")
    st.dataframe(summary_df)


    # SHOPPING LIST — Categorized + Cost
    rows = []
    total_cost = 0

    for ing, qty in SHOPPING.items():
        category = classify(ing)
        est_cost = CATEGORY_COST[category] * qty
        total_cost += est_cost
        rows.append([category, ing, qty, est_cost])

    shopping_df = pd.DataFrame(rows, columns=["Category","Ingredient","Quantity","Est. Cost (₹)"])
    shopping_df = shopping_df.sort_values("Category")

    st.subheader("Categorized Shopping List (Pantry Adjusted)")
    st.dataframe(shopping_df)
    st.success(f"Estimated Weekly Grocery Cost: **₹ {round(total_cost,2)}**")


    # DOWNLOAD BUTTONS
    st.download_button("⬇ Download Weekly Plan CSV", week_df.to_csv(index=False), "nutribridge_week.csv")
    st.download_button("⬇ Download Nutrition Summary CSV", summary_df.to_csv(index=False), "nutribridge_summary.csv")
    st.download_button("⬇ Download Shopping List CSV", shopping_df.to_csv(index=False), "nutribridge_shopping_list.csv")


    # Graph 1: Calories + Protein
    st.subheader("Nutrition Trend")
    fig, ax1 = plt.subplots()
    ax1.plot(summary_df.Day, summary_df.Calories, "o-", color="blue", label="Calories")
    ax1.set_ylabel("Calories", color="blue")
    ax2 = ax1.twinx()
    ax2.plot(summary_df.Day, summary_df["Protein (g)"], "s--", color="green", label="Protein")
    ax2.set_ylabel("Protein (g)", color="green")
    st.pyplot(fig)


    # Graph 2: Ingredient frequency
    st.subheader("Top Ingredients Used")
    if SHOPPING:
        freq = pd.DataFrame(SHOPPING.items(), columns=["Ingredient","Qty"]).sort_values("Qty", ascending=False).head(15)
        plt.figure(figsize=(8,6))
        plt.barh(freq["Ingredient"], freq["Qty"])
        st.pyplot(plt)


    if PANTRY:
        pantry_count = len(PANTRY)
        st.info(f"✅ Pantry Mode enabled — {pantry_count} ingredient type(s) excluded from shopping list.")
