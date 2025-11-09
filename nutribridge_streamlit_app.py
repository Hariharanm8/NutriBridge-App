# nutribridge_streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import json
from ast import literal_eval
from collections import Counter
from pulp import LpProblem, LpVariable, LpMinimize, LpBinary, lpSum, PULP_CBC_CMD
import matplotlib.pyplot as plt
import io

st.set_page_config(page_title="Nutri-Bridge", layout="wide")

# ---------------------- Helpers ----------------------
def normalize_ing(x):
    if x is None:
        return ""
    s = str(x).lower().strip()
    s = s.replace("’", "'").replace("`", "'")
    s = s.replace(".", "").replace(",", "").replace("(", "").replace(")", "")
    s = s.replace("/", " ").replace("-", " ").replace("&", " and ")
    s = "_".join(s.split())  # flatten spaces to underscore tokens for consistency
    return s

def safe_eval(x):
    try:
        val = literal_eval(x)
        if isinstance(val, (list, tuple)):
            return list(val)
        else:
            return []
    except Exception:
        return []

def load_recipes_from_csv(path_or_buffer):
    # use pandas python engine and on_bad_lines skip for robustness
    try:
        df = pd.read_csv(path_or_buffer, engine="python", on_bad_lines="skip", quoting=3)
    except Exception as e:
        st.error(f"CSV load failed: {e}")
        return pd.DataFrame()
    # Ensure necessary columns exist, try to extract/rename common ones
    # Common dataset columns: id, name, ingredients, nutrition, calories/protein fields, tags
    if "id" not in df.columns and "recipe_id" in df.columns:
        df = df.rename(columns={"recipe_id": "id"})
    # Parse ingredient list
    if "ingredients" in df.columns:
        df["ing_list_raw"] = df["ingredients"].apply(lambda x: safe_eval(x) if isinstance(x, str) else [])
    else:
        df["ing_list_raw"] = [[] for _ in range(len(df))]
    df["ing_list"] = df["ing_list_raw"].apply(lambda lst: [normalize_ing(i) for i in lst if isinstance(i, str)])
    # Ensure numeric nutrition fields exist (try to parse 'nutrition' column)
    if "nutrition" in df.columns:
        def parse_nut(v):
            try:
                arr = literal_eval(v)
                # dataset nutrition vector ordering may vary; assume [calories, protein, ...]
                if isinstance(arr, (list, tuple)) and len(arr) >= 2:
                    return float(arr[0]), float(arr[1])
            except:
                pass
            return np.nan, np.nan
        parsed = df["nutrition"].apply(lambda x: parse_nut(x) if isinstance(x, str) else (np.nan, np.nan))
        df["calories"] = [p[0] for p in parsed]
        df["protein_g"] = [p[1] for p in parsed]
    # If calories/protein not present, try existing columns
    if "calories" not in df.columns:
        df["calories"] = df.get("calories", np.nan)
    if "protein_g" not in df.columns:
        # try "protein" or "protein_g"
        df["protein_g"] = df.get("protein_g", df.get("protein", np.nan))
    # cost_est fallback
    if "cost_est" not in df.columns:
        df["cost_est"] = 0.0
    # meal_type heuristic: check tags column or default to 'general'
    if "meal_type" not in df.columns:
        if "tags" in df.columns:
            def guess_meal(tags):
                try:
                    arr = literal_eval(tags) if isinstance(tags, str) else []
                    if any("breakfast" in str(t).lower() for t in arr): return "breakfast"
                    if any("lunch" in str(t).lower() for t in arr): return "lunch"
                    if any("dinner" in str(t).lower() for t in arr): return "dinner"
                except:
                    pass
                return "general"
            df["meal_type"] = df["tags"].apply(guess_meal)
        else:
            df["meal_type"] = "general"
    # cuisine detection from tags or name
    if "cuisine" not in df.columns:
        def detect_cuisine(row):
            keys = ["indian","mexican","italian","chinese","thai","japanese","portuguese","french","american","korean"]
            text = ""
            if "tags" in row and isinstance(row["tags"], str):
                text += " " + row["tags"].lower()
            if "name" in row and isinstance(row["name"], str):
                text += " " + row["name"].lower()
            for k in keys:
                if k in text:
                    return k
            return "general"
        df["cuisine"] = df.apply(detect_cuisine, axis=1)
    # ensure id is int-like
    if df["id"].dtype != np.int64 and df["id"].dtype != np.int32:
        try:
            df["id"] = df["id"].astype(int)
        except:
            df["id"] = range(len(df))
    # Fill missing calories/protein with safe defaults to avoid division by zero
    df["calories"] = pd.to_numeric(df["calories"], errors="coerce").fillna(200.0)
    df["protein_g"] = pd.to_numeric(df["protein_g"], errors="coerce").fillna(10.0)
    df["cost_est"] = pd.to_numeric(df["cost_est"], errors="coerce").fillna(20.0)
    # normalize ingredient lists (ensure list of strings)
    df["ing_list"] = df["ing_list"].apply(lambda lst: [normalize_ing(x) for x in lst if isinstance(x, str)])
    return df

# ---------------------- ILP Planner (daily) ----------------------
def violates_restrictions(ingredients, restrictions):
    if not restrictions:
        return False
    for r in restrictions:
        rn = normalize_ing(r)
        for ing in ingredients:
            ingn = normalize_ing(ing)
            if rn == ingn or rn in ingn or ingn in rn:
                return True
    return False

def build_day_plan(day_idx: int, recipes_df, user, last_cuisines: list, used_recipes: set, cal_tgt, pro_tgt):
    df = recipes_df[recipes_df["cuisine"].isin(user["preferred_cuisines"])].copy()
    df = df[~df["ing_list"].apply(lambda ings: violates_restrictions(ings, user["restrictions"]))]
    if df.shape[0] < 5:
        df = recipes_df.copy()  # fallback to entire dataset

    # ranking and pool
    df["rank_score"] = 2.0 * (df["protein_g"] / df["calories"].replace(0,1)) - 0.01 * df["cost_est"]
    pool = df.sort_values("rank_score", ascending=False).head(300)

    # pools per meal
    pools = {m: pool[pool["meal_type"] == m].head(60) for m in ["breakfast","lunch","dinner"]}
    for m in pools:
        if pools[m].empty:
            pools[m] = pool.head(60)

    # rec_map
    rec_map = {}
    for m in pools:
        for _, r in pools[m].iterrows():
            rec_map[(m, int(r["id"]))] = r

    # ILP
    prob = LpProblem(f"NutriBridge_Day_{day_idx}", LpMinimize)
    x = {k: LpVariable(f"x_{k[0]}_{k[1]}", cat=LpBinary) for k in rec_map.keys()}

    # exactly 1 per meal type
    for m in ["breakfast","lunch","dinner"]:
        prob += lpSum([x[k] for k in x if k[0]==m]) == 1

    total_cal = lpSum([x[k] * float(rec_map[k]["calories"]) for k in x])
    total_pro = lpSum([x[k] * float(rec_map[k]["protein_g"]) for k in x])

    prob += total_cal >= (1 - user["cal_tol"]) * cal_tgt
    prob += total_cal <= (1 + user["cal_tol"]) * cal_tgt
    prob += total_pro >= (1 - user["pro_tol"]) * pro_tgt

    # variety penalties
    PENALTY_REPEAT_CUISINE = 6.0
    PENALTY_REPEAT_RECIPE = 12.0

    penalty_terms = []
    for k in x:
        rec = rec_map[k]
        cuisine_pen = PENALTY_REPEAT_CUISINE if rec["cuisine"] in last_cuisines else 0.0
        recipe_pen = PENALTY_REPEAT_RECIPE if int(rec["id"]) in used_recipes else 0.0
        penalty_terms.append(x[k] * (cuisine_pen + recipe_pen))

    prob += lpSum(penalty_terms)

    # solve
    prob.solve(PULP_CBC_CMD(msg=0))

    # parse results
    chosen, cuisines_used, tot_cal, tot_pro = [], [], 0.0, 0.0
    for k, var in x.items():
        val = var.value()
        if val is not None and val >= 0.99:
            rec = rec_map[k]
            chosen.append({
                "day": day_idx,
                "meal_type": k[0],
                "recipe_id": int(rec["id"]),
                "name": rec["name"],
                "cuisine": rec["cuisine"],
                "calories": float(rec["calories"]),
                "protein_g": float(rec["protein_g"]),
                "cost_est": float(rec["cost_est"]),
                "ingredients": [normalize_ing(i) for i in rec["ing_list"]]
            })
            cuisines_used.append(rec["cuisine"])
            used_recipes.add(int(rec["id"]))
            tot_cal += float(rec["calories"])
            tot_pro += float(rec["protein_g"])

    return {"meals": chosen, "nutrition": (tot_cal, tot_pro), "cuisines": cuisines_used, "used_recipes": used_recipes}

# ---------------------- Categorize ingredient ----------------------
CATEGORIES = {
    "Produce": ["onion","garlic","tomato","lettuce","cilantro","ginger","pepper","spinach",
                "capsicum","bell pepper","zucchini","parsley","lemon","apple","banana"],
    "Dairy": ["milk","cheese","mozzarella","curd","butter","paneer","ricotta","cream","yogurt"],
    "Meat": ["chicken","pork","beef","tuna","fish","steak","mutton","egg"],
    "Pantry": ["flour","rice","pasta","oil","sugar","salt","vinegar","mustard","sauce",
               "seasoning","spice","lasagna","noodle","bread","corn","beans"]
}

def classify_ingredient(ing):
    il = ing.lower()
    for cat, keys in CATEGORIES.items():
        if any(k in il for k in keys):
            return cat
    return "Other"

# ---------------------- Streamlit UI ----------------------
st.title("Nutri-Bridge — Personalized Nutrition Planner")
st.markdown("Generate a 7-day nutrition plan with ILP, categorized shopping list and analytics.")

# Upload or use path
uploaded_file = st.file_uploader("Upload RAW_recipes.csv (or leave blank to use local /content/RAW_recipes.csv)", type=["csv"])
if uploaded_file is not None:
    recipes = load_recipes_from_csv(uploaded_file)
else:
    try:
        recipes = load_recipes_from_csv("/content/RAW_recipes.csv")
    except:
        st.warning("No dataset uploaded and /content/RAW_recipes.csv not found — upload a CSV.")
        recipes = pd.DataFrame()

st.info(f"Recipes loaded: {len(recipes)}")

# User inputs
st.sidebar.header("User Profile")
age = st.sidebar.number_input("Age", min_value=5, max_value=120, value=30)
gender = st.sidebar.selectbox("Gender", options=["male","female"], index=0)
height = st.sidebar.number_input("Height (cm)", min_value=50.0, max_value=250.0, value=170.0)
weight = st.sidebar.number_input("Weight (kg)", min_value=20.0, max_value=300.0, value=70.0)
activity = st.sidebar.selectbox("Activity", options=["sedentary","light","moderate","active","very_active"], index=2)
preferred_cuisines_raw = st.sidebar.text_input("Preferred cuisines (comma separated)", value="indian,italian").strip()
preferred_cuisines = [c.strip().lower() for c in preferred_cuisines_raw.split(",") if c.strip()]
restrictions_raw = st.sidebar.text_input("Dietary restrictions (comma separated)", value="").strip()
restrictions = [r.strip().lower() for r in restrictions_raw.split(",") if r.strip()]

cal_tol = st.sidebar.slider("Calorie tolerance (±%)", 0.0, 0.5, 0.12, step=0.01)
pro_tol = st.sidebar.slider("Protein tolerance (±%)", 0.0, 0.5, 0.12, step=0.01)

user = {
    "age": age, "gender": gender, "height": height, "weight": weight,
    "activity": activity, "preferred_cuisines": preferred_cuisines,
    "restrictions": restrictions, "cal_tol": cal_tol, "pro_tol": pro_tol
}

# Mifflin-St Jeor BMR and target functions
ACTIVITY_FACTORS = {"sedentary":1.2,"light":1.375,"moderate":1.55,"active":1.725,"very_active":1.9}
def mifflin(gender, age, h, w):
    return (10*w + 6.25*h - 5*age + (5 if gender.lower().startswith("m") else -161))
def calorie_target(user):
    return int(round(mifflin(user["gender"], user["age"], user["height"], user["weight"]) * ACTIVITY_FACTORS.get(user["activity"], 1.375)))
def protein_target(user):
    return int(round(0.8 * user["weight"]))

if st.button("Generate 7-day Plan"):
    if recipes.empty:
        st.error("No recipes available. Upload RAW_recipes.csv first.")
    else:
        # compute targets
        CAL_TGT = calorie_target(user)
        PRO_TGT = protein_target(user)
        st.success(f"Targets → Calories: {CAL_TGT} kcal/day, Protein: {PRO_TGT} g/day")

        # Weekly planner
        week_plans = []
        last_cuis = []
        used_recipes = set()
        summary_rows = []
        week_rows = []
        shopping = Counter()
        total_cost = 0.0

        st.info("Running ILP planner for 7 days — this may take a few seconds...")

        for day in range(1, 8):
            st.write(f"Generating Day {day} ...")
            plan = build_day_plan(day, recipes, user, last_cuis, used_recipes, CAL_TGT, PRO_TGT)
            last_cuis = plan["cuisines"]
            used_recipes = plan["used_recipes"]

            tot_cal, tot_pro = plan["nutrition"]
            summary_rows.append({"day": day, "total_calories": round(tot_cal,1), "total_protein_g": round(tot_pro,1)})

            for meal in plan["meals"]:
                week_rows.append({
                    "day": day,
                    "meal_type": meal["meal_type"],
                    "cuisine": meal["cuisine"],
                    "recipe_id": meal["recipe_id"],
                    "name": meal["name"],
                    "calories": round(meal["calories"],1),
                    "protein_g": round(meal["protein_g"],1),
                    "cost_est": round(meal["cost_est"],2)
                })
                total_cost += meal["cost_est"]
                for ing in meal["ingredients"]:
                    shopping[ing] += 1

        # Save outputs to memory buffers and display
        out_prefix = "/content/nutribridge"
        week_df = pd.DataFrame(week_rows)
        summary_df = pd.DataFrame(summary_rows)

        st.subheader("Weekly Meal Plan")
        st.dataframe(week_df)

        st.subheader("Daily Nutrition Summary")
        st.dataframe(summary_df)

        # Categorize shopping list
        categorized = {"Produce": {}, "Dairy": {}, "Meat": {}, "Pantry": {}, "Other": {}}
        for ing, qty in shopping.items():
            cat = classify_ingredient(ing)
            categorized[cat][ing] = qty

        # Show categorized as table
        rows = []
        for cat, items in categorized.items():
            for ing, qty in items.items():
                rows.append([cat, ing, qty])
        shopping_df = pd.DataFrame(rows, columns=["Category", "Ingredient", "Quantity"])

        st.subheader("Categorized Shopping List")
        st.dataframe(shopping_df)

        # Provide download buttons
        to_csv = week_df.to_csv(index=False).encode("utf-8")
        st.download_button("Download weekly plan CSV", data=to_csv, file_name="nutribridge_week.csv", mime="text/csv")

        to_csv2 = summary_df.to_csv(index=False).encode("utf-8")
        st.download_button("Download summary CSV", data=to_csv2, file_name="nutribridge_summary.csv", mime="text/csv")

        to_csv3 = shopping_df.to_csv(index=False).encode("utf-8")
        st.download_button("Download categorized shopping CSV", data=to_csv3, file_name="nutribridge_categorized_shopping_list.csv", mime="text/csv")

        # Graphs: Calories & Protein (two colored lines)
        st.subheader("Nutrition Trend (Calories & Protein)")
        fig, ax1 = plt.subplots(figsize=(9,4))
        ax1.plot(summary_df["day"], summary_df["total_calories"], marker="o", linewidth=3, color="blue", label="Calories (kcal)")
        ax1.set_xlabel("Day")
        ax1.set_ylabel("Calories (kcal)", color="blue")
        ax1.tick_params(axis="y", labelcolor="blue")
        ax2 = ax1.twinx()
        ax2.plot(summary_df["day"], summary_df["total_protein_g"], marker="s", linewidth=3, linestyle="--", color="green", label="Protein (g)")
        ax2.set_ylabel("Protein (g)", color="green")
        ax2.tick_params(axis="y", labelcolor="green")
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
        st.pyplot(fig)

        # Ingredient Frequency Graph
        st.subheader("Ingredient Frequency (Top 15)")
        freq_df = shopping_df.groupby("Ingredient")["Quantity"].sum().reset_index().sort_values("Quantity", ascending=False)
        top = freq_df.head(15).sort_values("Quantity")
        fig2, ax = plt.subplots(figsize=(9, max(3, 0.35*len(top))))
        ax.barh(top["Ingredient"], top["Quantity"])
        ax.set_xlabel("Times required this week")
        ax.set_title("Top Ingredients")
        for i, qty in enumerate(top["Quantity"]):
            ax.text(qty + 0.1, i, str(qty), va="center")
        st.pyplot(fig2)

        # Optionally show categorized JSON
        st.subheader("Categorized Shopping JSON (preview)")
        st.json(categorized)

        st.success(f"Plan generated. Estimated weekly recipe-cost sum (approx): {round(total_cost,2)} units")
        st.info("You can download CSVs using the buttons above.")
