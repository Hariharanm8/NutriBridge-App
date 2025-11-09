#  FINAL VERSION — NutriBridge Streamlit App
# -------------------------------------------------------
import streamlit as st
import pandas as pd
import numpy as np
import json
from ast import literal_eval
from collections import Counter
from pulp import LpProblem, LpVariable, LpMinimize, LpBinary, lpSum, PULP_CBC_CMD
import matplotlib.pyplot as plt


st.set_page_config(page_title="NutriBridge", layout="wide")

# -------------------------------------------------------
#  Helper Functions
# -------------------------------------------------------
def normalize_ing(x):
    if x is None: return ""
    s = str(x).lower().strip()
    for c in [".", ",", "(", ")", "/", "&"]:
        s = s.replace(c, " ")
    return "_".join(s.split())


def safe_eval(x):
    try:
        val = literal_eval(x)
        return list(val) if isinstance(val, (list, tuple)) else []
    except:
        return []


# -------------------------------------------------------
#  Load Recipes CSV (supports RAW_recipes_small.csv)
# -------------------------------------------------------
def load_recipes_from_csv(path_or_buffer):
    df = pd.read_csv(path_or_buffer, engine="python", on_bad_lines="skip", quoting=3)

    # Parse ingredient list
    df["ing_list_raw"] = df["ingredients"].apply(lambda x: safe_eval(x) if isinstance(x, str) else [])
    df["ing_list"] = df["ing_list_raw"].apply(lambda lst: [normalize_ing(i) for i in lst])

    # Parse calories & protein from "nutrition"
    def parse_nut(v):
        try:
            p = literal_eval(v)
            return float(p[0]), float(p[1])
        except:
            return np.nan, np.nan

    df["calories"], df["protein_g"] = zip(*df["nutrition"].apply(parse_nut))
    df["calories"] = df["calories"].fillna(200)
    df["protein_g"] = df["protein_g"].fillna(10)
    df["cost_est"] = df.get("cost_est", 20.0)

    #  Smart meal-type detection
    def detect_meal(row):
        name = str(row["name"]).lower()
        tags = str(row.get("tags", "")).lower()

        breakfast_kw = ["smoothie", "pancake", "toast", "omelette", "shake", "breakfast"]
        lunch_kw = ["wrap", "rice", "salad", "sandwich", "thali", "bowl"]
        dinner_kw = ["pasta", "noodle", "biryani", "stew", "pizza", "dinner", "curry"]

        if any(k in name for k in breakfast_kw): return "breakfast"
        if any(k in name for k in lunch_kw): return "lunch"
        if any(k in name for k in dinner_kw): return "dinner"

        return "general"

    df["meal_type"] = df.apply(detect_meal, axis=1)

    #  Cuisine detection
    cuisines = ["indian","mexican","italian","chinese","thai","japanese",
                "portuguese","american","korean","french"]
    df["cuisine"] = df["tags"].apply(lambda t: next((c for c in cuisines if c in str(t).lower()), "general"))

    return df


# -------------------------------------------------------
#  ILP DAILY PLANNER
# -------------------------------------------------------
def build_day_plan(day_idx, recipes_df, user, last_cuisines, used_recipes, cal_tgt, pro_tgt):
    df = recipes_df[recipes_df["cuisine"].isin(user["preferred_cuisines"])].copy()

    # Filter based on restrictions
    df = df[~df["ing_list"].apply(lambda lst: any(r in ing for r in user["restrictions"] for ing in lst))]

    # Ranking and pool size (helps avoid repetition)
    df["rank_score"] = (2 * (df["protein_g"] / df["calories"])) - (0.01 * df["cost_est"])
    pool = df.sort_values("rank_score", ascending=False).head(60)

    pools = {m: pool[pool["meal_type"] == m].head(20) for m in ["breakfast","lunch","dinner"]}
    for m in pools:
        if pools[m].empty: pools[m] = pool.head(20)

    # ILP Variables
    rec_map = {(m, int(r["id"])): r for m in pools for _, r in pools[m].iterrows()}
    prob = LpProblem(f"NutriBridge_Day_{day_idx}", LpMinimize)
    x = {k: LpVariable(f"x_{k[0]}_{k[1]}", cat=LpBinary) for k in rec_map}

    # Exactly one meal breakfast/lunch/dinner
    for m in ["breakfast","lunch","dinner"]:
        prob += lpSum(x[k] for k in x if k[0] == m) == 1

    total_cal = lpSum([x[k] * rec_map[k]["calories"] for k in x])
    total_pro = lpSum([x[k] * rec_map[k]["protein_g"] for k in x])

    prob += total_cal >= 0.8 * cal_tgt
    prob += total_cal <= 1.2 * cal_tgt
    prob += total_pro >= 0.8 * pro_tgt

    #  Strong penalties to avoid repetition
    penalty = lpSum([
        x[k] * (
            (25 if rec_map[k]["cuisine"] in last_cuisines else 0) +
            (60 if rec_map[k]["id"] in used_recipes else 0)
        )
        for k in x
    ])
    prob += penalty

    prob.solve(PULP_CBC_CMD(msg=0))

    # Collect output
    chosen = []
    cuisines = []
    tot_cal = tot_pro = 0

    for k, var in x.items():
        if var.value() == 1:
            rec = rec_map[k]
            used_recipes.add(rec["id"])
            cuisines.append(rec["cuisine"])
            chosen.append({
                "day": day_idx,
                "meal_type": k[0],
                "recipe_id": rec["id"],
                "name": rec["name"],
                "cuisine": rec["cuisine"],
                "calories": rec["calories"],
                "protein_g": rec["protein_g"],
                "cost_est": rec["cost_est"],
                "ingredients": rec["ing_list"]
            })
            tot_cal += rec["calories"]
            tot_pro += rec["protein_g"]

    return chosen, cuisines, tot_cal, tot_pro, used_recipes


# -------------------------------------------------------
#  Shopping Categorization
# -------------------------------------------------------
CATEGORIES = {
    "Produce": ["onion","tomato","garlic","spinach","pepper","ginger","lettuce"],
    "Dairy": ["milk","cheese","butter","paneer","cream","yogurt"],
    "Meat": ["chicken","egg","beef","fish","pork"],
    "Pantry": ["rice","flour","oil","salt","sugar","spice","pasta","noodle"]
}

def classify_ing(ing):
    for cat, items in CATEGORIES.items():
        if any(i in ing for i in items):
            return cat
    return "Other"


# -------------------------------------------------------
#  UI & APP
# -------------------------------------------------------
st.title(" NutriBridge — Personalized Nutrition Planner (ILP)")
uploaded = st.file_uploader("Upload RAW_recipes_small.csv", type=["csv"])

if uploaded:
    recipes = load_recipes_from_csv(uploaded)

    st.sidebar.header("User Profile")
    age = st.sidebar.number_input("Age", 5, 100, 30)
    gender = st.sidebar.selectbox("Gender", ["male", "female"])
    height = st.sidebar.number_input("Height (cm)", 120, 220, 170)
    weight = st.sidebar.number_input("Weight (kg)", 30, 200, 70)
    activity = st.sidebar.selectbox("Activity", ["sedentary","light","moderate","active","very_active"])
    preferred_cuisines = st.sidebar.text_input("Preferred cuisines", "indian,italian").lower().split(",")
    restrictions = st.sidebar.text_input("Dietary restrictions", "").lower().split(",")

    # Daily targets
    cal_tgt = int((10*weight + 6.25*height - 5*age + (5 if gender=="male" else -161)) *
                  {"sedentary":1.2,"light":1.375,"moderate":1.55,"active":1.725,"very_active":1.9}[activity])
    pro_tgt = int(weight * 0.8)

    if st.button("Generate Plan"):
        week_rows = []
        summary_rows = []
        shopping = Counter()
        used_recipes = set()
        last_cuis = []

        for day in range(1, 8):
            meals, last_cuis, tot_cal, tot_pro, used_recipes = build_day_plan(
                day, recipes,
                {"preferred_cuisines": preferred_cuisines, "restrictions": restrictions},
                last_cuis, used_recipes, cal_tgt, pro_tgt
            )

            week_rows.extend(meals)
            summary_rows.append({"day": day, "total_cal": tot_cal, "total_protein": tot_pro})
            for m in meals:
                for ing in m["ingredients"]:
                    shopping[ing] += 1

        week_df = pd.DataFrame(week_rows)
        summary_df = pd.DataFrame(summary_rows)

        st.subheader(" Weekly Plan")
        st.dataframe(week_df)

        st.subheader(" Nutrition Trend")
        fig, ax = plt.subplots()
        ax.plot(summary_df["day"], summary_df["total_cal"], label="Calories", color="blue")
        ax.plot(summary_df["day"], summary_df["total_protein"], label="Protein (g)", color="green")
        ax.legend()
        st.pyplot(fig)

        # Categorized shopping CSV
        shop_df = pd.DataFrame([(classify_ing(ing), ing, qty)
                                for ing, qty in shopping.items()],
                               columns=["Category","Ingredient","Quantity"])
        st.subheader(" Categorized Shopping List")
        st.dataframe(shop_df)

        st.download_button("Download Weekly CSV", week_df.to_csv(index=False), "nutribridge_week.csv")
        st.download_button("Download Shopping CSV", shop_df.to_csv(index=False), "nutribridge_shopping.csv")

