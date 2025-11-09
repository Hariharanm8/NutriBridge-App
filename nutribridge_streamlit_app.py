import streamlit as st
import pandas as pd
import numpy as np
from ast import literal_eval
from pulp import LpProblem, LpVariable, LpMinimize, LpBinary, lpSum, PULP_CBC_CMD
from collections import Counter
import matplotlib.pyplot as plt

st.title("Nutri-Bridge — Personalized Nutrition Planner")

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
        # fallback: if it's a comma-separated string
        if isinstance(x, str) and "," in x:
            return [s.strip() for s in x.split(",") if s.strip()]
        return []

def detect_meal(name, tags):
    t = (str(name) + " " + str(tags)).lower()
    if "breakfast" in t: return "breakfast"
    if "lunch" in t: return "lunch"
    if "dinner" in t: return "dinner"
    return "lunch"  # fallback ensures 3 meals/day exist

def detect_cuisine(name, tags):
    cuisines = ["indian","italian","mexican","chinese","thai","american","japanese","korean"]
    t = (str(name) + " " + str(tags)).lower()
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

    # ensure id exists
    if "id" not in df.columns:
        df["id"] = range(len(df))
    else:
        try:
            df["id"] = df["id"].astype(int)
        except:
            df["id"] = range(len(df))

    # ensure tags column exists
    if "tags" not in df.columns:
        df["tags"] = ""

    # ingredients parsing
    if "ingredients" in df.columns:
        df["ingredients"] = df["ingredients"].apply(safe_list)
    else:
        df["ingredients"] = [[] for _ in range(len(df))]

    df["ing_list"] = df["ingredients"].apply(lambda lst: [normalize_ing(i) for i in lst])

    # nutrition parse (safe)
    if "nutrition" in df.columns:
        def parse_nut(x):
            try:
                if isinstance(x, str):
                    arr = literal_eval(x)
                else:
                    arr = x
                if isinstance(arr, (list,tuple)) and len(arr) >= 2:
                    return float(arr[0]), float(arr[1])
            except:
                pass
            return np.nan, np.nan
        parsed = df["nutrition"].apply(parse_nut)
        df["calories"] = [p[0] for p in parsed]
        df["protein_g"] = [p[1] for p in parsed]
    else:
        # try common columns
        df["calories"] = pd.to_numeric(df.get("calories", np.nan), errors="coerce")
        df["protein_g"] = pd.to_numeric(df.get("protein_g", df.get("protein", np.nan)), errors="coerce")

    # drop any rows without nutrition and fill safe defaults
    df["calories"] = df["calories"].fillna(250.0)
    df["protein_g"] = df["protein_g"].fillna(12.0)

    # meal_type and cuisine detection (safe access to tags/name)
    df["meal_type"] = df.apply(lambda r: detect_meal(r.get("name", ""), r.get("tags", "")), axis=1)
    df["cuisine"] = df.apply(lambda r: detect_cuisine(r.get("name", ""), r.get("tags", "")), axis=1)

    # cost_est fallback
    df["cost_est"] = pd.to_numeric(df.get("cost_est", 20), errors="coerce").fillna(20.0)

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
    # ingredients: list of normalized strings
    if not restrictions or restrictions == [""]:
        return False
    return any(r in ing for r in restrictions for ing in ingredients)

def plan_day(day, df, last_cuis, used):
    df_f = df.copy()
    # prefer cuisines if provided
    if preferred_cuisines and preferred_cuisines != ['']:
        df_f = df_f[df_f["cuisine"].isin(preferred_cuisines)]
    # apply restrictions
    if restrictions and restrictions != ['']:
        df_f = df_f[~df_f["ing_list"].apply(violates)]
    if df_f.empty:
        df_f = df.copy()  # fallback to full set

    CAL_TGT = calorie_target()
    PRO_TGT = protein_target()

    pool = df_f.sample(min(300, len(df_f)))
    rec_map = {(r["meal_type"], int(r["id"])): r for _, r in pool.iterrows()}

    prob = LpProblem(f"Day_{day}", LpMinimize)
    x = {k: LpVariable(f"x_{k[0]}_{k[1]}", 0, 1, LpBinary) for k in rec_map.keys()}

    # one per meal
    for meal in ["breakfast","lunch","dinner"]:
        prob += lpSum(x[k] for k in x if k[0]==meal) == 1

    total_cal = lpSum([x[k] * rec_map[k]["calories"] for k in x])
    total_pro = lpSum([x[k] * rec_map[k]["protein_g"] for k in x])

    prob += total_cal >= CAL_TGT*(1-cal_tol)
    prob += total_cal <= CAL_TGT*(1+cal_tol)
    prob += total_pro >= PRO_TGT*(1-pro_tol)

    # solve
    prob.solve(PULP_CBC_CMD(msg=0))

    chosen, cuisines, tot_cal, tot_pro = [], [], 0, 0

    for k, v in x.items():
        try:
            val = v.value()
        except:
            val = None
        if val is not None and val >= 0.99:
            r = rec_map[k]
            chosen.append(r)
            cuisines.append(r["cuisine"])
            used.add(int(r["id"]))
            tot_cal += r["calories"]
            tot_pro += r["protein_g"]

    # fallback: if solver didn't pick 3 meals, pick top per meal type
    if len(chosen) < 3:
        chosen = []
        cuisines = []
        tot_cal = tot_pro = 0
        for meal in ["breakfast","lunch","dinner"]:
            cand = pool[pool["meal_type"]==meal].sort_values(by="calories", ascending=False)
            if cand.empty:
                cand = pool
            r = cand.iloc[0]
            chosen.append(r)
            cuisines.append(r["cuisine"])
            used.add(int(r["id"]))
            tot_cal += r["calories"]
            tot_pro += r["protein_g"]

    return chosen, cuisines, tot_cal, tot_pro, used

# ===========================================================
# Estimated cost map (per item)
# ===========================================================
COST_MAP = {
    "Produce": 5,     # unit cost (currency)
    "Dairy": 10,
    "Meat": 30,
    "Pantry": 10,
    "Other": 8
}

def classify(ing):
    ing = ing.lower()
    if any(x in ing for x in ["onion","garlic","tomato","pepper","lettuce","cilantro","ginger","spinach"]): return "Produce"
    if any(x in ing for x in ["milk","cheese","butter","yogurt","paneer","ricotta"]): return "Dairy"
    if any(x in ing for x in ["chicken","beef","fish","pork","egg","mutton","shrimp"]): return "Meat"
    if any(x in ing for x in ["rice","flour","salt","oil","bread","pasta","beans","sugar"]): return "Pantry"
    return "Other"

# ===========================================================
# Generate Weekly Plan
# ===========================================================
if st.button("Generate 7-Day Plan"):

    if df.empty:
        st.error("No dataset loaded. Upload RAW_recipes_small.csv.")
    else:
        week = []
        summary = []
        shopping = Counter()
        used_recipes = set()
        last_cuis = []

        for d in range(1, 8):
            meals, last_cuis, tcal, tpro, used_recipes = plan_day(d, df, last_cuis, used_recipes)

            for r in meals:
                # keep numeric values for CSV/graphs, add display later
                week.append({
                    "day": d,
                    "meal": r["meal_type"],
                    "cuisine": r["cuisine"],
                    "recipe": r.get("name",""),
                    "calories": float(r["calories"]),
                    "protein": float(r["protein_g"]),
                    "cost_est": float(r.get("cost_est", 20.0))
                })
                for ing in r.get("ing_list", []):
                    shopping[ing] += 1

            summary.append({"day": d, "total_calories": round(tcal,1), "total_protein": round(tpro,1)})

        week_df = pd.DataFrame(week)
        summary_df = pd.DataFrame(summary)

        # Prepare display copies with units (but keep originals for CSV/graphs)
        week_display = week_df.copy()
        week_display["calories"] = week_display["calories"].apply(lambda x: f"{round(x,1)} cal")
        week_display["protein"] = week_display["protein"].apply(lambda x: f"{round(x,1)} g")
        week_display["cost_est"] = week_display["cost_est"].apply(lambda x: f"₹ {round(x,2)}")

        summary_display = summary_df.copy()
        summary_display["total_calories"] = summary_display["total_calories"].apply(lambda x: f"{x} cal")
        summary_display["total_protein"] = summary_display["total_protein"].apply(lambda x: f"{x} g")

        # Display tables (with units)
        st.subheader("Weekly Meal Plan")
        st.dataframe(week_display)

        # Download original numeric CSV (no units)
        st.download_button("⬇ Download Weekly Plan CSV", week_df.to_csv(index=False).encode('utf-8'), "nutribridge_week.csv", "text/csv")

        st.subheader("Daily Nutrition Summary")
        st.dataframe(summary_display)
        st.download_button("⬇ Download Nutrition Summary CSV", summary_df.to_csv(index=False).encode('utf-8'), "nutribridge_summary.csv", "text/csv")

        # Categorized shopping list with cost (display)
        rows = []
        total_cost = 0.0
        for ing, qty in shopping.items():
            cat = classify(ing)
            est_cost = COST_MAP.get(cat, COST_MAP["Other"]) * qty
            total_cost += est_cost
            rows.append([cat, ing, int(qty), est_cost])

        shopping_df = pd.DataFrame(rows, columns=["Category","Ingredient","Quantity","Estimated Cost"])
        # display formatting: show currency
        shopping_display = shopping_df.copy()
        shopping_display["Estimated Cost"] = shopping_display["Estimated Cost"].apply(lambda x: f"₹ {round(x,2)}")

        st.subheader("Categorized Shopping List with Cost")
        st.dataframe(shopping_display)

        st.success(f"Estimated Total Weekly Grocery Cost: ₹ {round(total_cost,2)}")

        # download shopping csv (numeric)
        st.download_button("⬇ Download Categorized Shopping List (CSV)", shopping_df.to_csv(index=False).encode('utf-8'), "nutribridge_shopping_list_with_cost.csv", "text/csv")

        # ===========================================================
        # Graph 1 — Calories & Protein Trend (numeric)
        # ===========================================================
        fig, ax1 = plt.subplots(figsize=(9,4))
        ax1.plot(summary_df["day"], summary_df["total_calories"], marker='o', color='blue', linewidth=3, label="Calories (kcal)")
        ax1.set_xlabel("Day")
        ax1.set_ylabel("Calories (kcal)", color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')

        ax2 = ax1.twinx()
        ax2.plot(summary_df["day"], summary_df["total_protein"], marker='s', color='green', linestyle='--', linewidth=3, label="Protein (g)")
        ax2.set_ylabel("Protein (g)", color='green')
        ax2.tick_params(axis='y', labelcolor='green')

        # combined legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

        st.subheader("Daily Calories & Protein Trend")
        st.pyplot(fig)

        # ===========================================================
        # Graph 2 — Ingredient Frequency
        # ===========================================================
        freq = pd.DataFrame(list(shopping.items()), columns=["Ingredient","Qty"]).sort_values("Qty", ascending=False).head(15)
        if not freq.empty:
            fig2, ax3 = plt.subplots(figsize=(9, max(3, 0.25*len(freq))))
            ax3.barh(freq["Ingredient"][::-1], freq["Qty"][::-1])
            ax3.set_xlabel("Times required this week")
            ax3.set_title("Top Ingredients (frequency)")
            for i, v in enumerate(freq["Qty"][::-1]):
                ax3.text(v + 0.1, i, str(int(v)), va='center')
            st.subheader("Top Ingredients Used (Frequency)")
            st.pyplot(fig2)
        else:
            st.info("No ingredients found in shopping list (shopping list empty).")
