# nutribridge_streamlit_app.py
# Final Nutri-Bridge Streamlit app — upload small CSV and generate 7-day plan
import streamlit as st
import pandas as pd
import numpy as np
from ast import literal_eval
from collections import Counter
from pulp import LpProblem, LpVariable, LpMinimize, LpBinary, lpSum, PULP_CBC_CMD
import matplotlib.pyplot as plt

st.set_page_config(page_title="Nutri-Bridge", layout="wide")

# -------------------- Helpers --------------------
def normalize_ing(x):
    """Normalize ingredient string to a compact token."""
    if x is None:
        return ""
    s = str(x).lower().strip()
    for ch in ['’','`','"',"'","(",")",",",".","/","\\","-","&",":",";"]:
        s = s.replace(ch, " ")
    return "_".join(s.split())

def safe_eval_list(x):
    """Try to literal_eval a list representation; return [] on error."""
    if pd.isna(x):
        return []
    if isinstance(x, (list, tuple)):
        return list(x)
    try:
        v = literal_eval(x)
        if isinstance(v, (list, tuple)):
            return list(v)
    except Exception:
        pass
    # fallback: if string with commas, split
    if isinstance(x, str) and ("," in x):
        return [s.strip() for s in x.split(",") if s.strip()]
    return []

# -------------------- CSV loader & preprocess --------------------
def load_recipes(path_or_buffer):
    # robust read (python engine tolerates odd rows)
    try:
        df = pd.read_csv(path_or_buffer, engine="python", on_bad_lines="skip", quoting=3)
    except Exception as e:
        st.error(f"Failed reading CSV: {e}")
        return pd.DataFrame()

    # ensure id/name exist
    if "id" not in df.columns and "recipe_id" in df.columns:
        df = df.rename(columns={"recipe_id":"id"})
    if "id" not in df.columns:
        df["id"] = range(len(df))

    if "name" not in df.columns:
        df["name"] = df["id"].astype(str)

    # ingredients: parse into list then normalize
    if "ingredients" in df.columns:
        df["ing_list_raw"] = df["ingredients"].apply(safe_eval_list)
    else:
        df["ing_list_raw"] = [[] for _ in range(len(df))]
    df["ing_list"] = df["ing_list_raw"].apply(lambda lst: [normalize_ing(i) for i in lst if i])

    # nutrition: try to parse "nutrition" or look for calories/protein columns
    df["calories"] = np.nan
    df["protein_g"] = np.nan
    if "nutrition" in df.columns:
        def parse_nut(v):
            try:
                arr = literal_eval(v) if isinstance(v, str) else v
                if isinstance(arr, (list,tuple)) and len(arr) >= 2:
                    return float(arr[0]), float(arr[1])
            except:
                pass
            return np.nan, np.nan
        parsed = df["nutrition"].apply(parse_nut)
        df["calories"] = [p[0] for p in parsed]
        df["protein_g"] = [p[1] for p in parsed]

    # fallback columns
    if "calories" in df.columns and df["calories"].isnull().all():
        try:
            df["calories"] = pd.to_numeric(df["calories"], errors="coerce")
        except:
            pass
    if "protein" in df.columns and ("protein_g" not in df.columns or df["protein_g"].isnull().all()):
        df["protein_g"] = pd.to_numeric(df["protein"], errors="coerce")

    # fill defaults for missing nutrition to avoid division by zero
    df["calories"] = pd.to_numeric(df["calories"], errors="coerce").fillna(250.0)
    df["protein_g"] = pd.to_numeric(df["protein_g"], errors="coerce").fillna(12.0)

    # cost estimate
    if "cost_est" not in df.columns:
        df["cost_est"] = 20.0
    df["cost_est"] = pd.to_numeric(df["cost_est"], errors="coerce").fillna(20.0)

    # meal_type detection (smart heuristic)
    def detect_meal(row):
        name = str(row.get("name","")).lower()
        tags = str(row.get("tags","")).lower()
        if any(w in name for w in ["omelet","omelette","pancake","cereal","breakfast","latte","smoothie","shake","waffle"]):
            return "breakfast"
        if any(w in name for w in ["sandwich","salad","bowl","wrap","burger","rice","lunch"]):
            return "lunch"
        if any(w in name for w in ["pasta","steak","curry","biryani","pizza","dinner","stew","roast"]):
            return "dinner"
        if "breakfast" in tags: return "breakfast"
        if "lunch" in tags: return "lunch"
        if "dinner" in tags: return "dinner"
        return "general"
    df["meal_type"] = df.apply(detect_meal, axis=1)

    # cuisine detection from tags/name
    cuisines_list = ["indian","mexican","italian","chinese","thai","japanese","portuguese","american","korean","french"]
    def detect_cuisine(row):
        text = (str(row.get("tags","")) + " " + str(row.get("name",""))).lower()
        for c in cuisines_list:
            if c in text:
                return c
        return "general"
    df["cuisine"] = df.apply(detect_cuisine, axis=1)

    # ensure id is int
    try:
        df["id"] = df["id"].astype(int)
    except:
        df["id"] = range(len(df))

    return df

# -------------------- ILP Daily Planner --------------------
def build_day_plan(day_idx: int, recipes_df: pd.DataFrame, user: dict,
                   last_cuisines: list, used_recipes: set, cal_tgt: float, pro_tgt: float):
    """
    Returns: dict with keys: meals (list of 3 meal dicts), nutrition (cal, protein), cuisines (list), used_recipes (set)
    """
    # filter by preferred cuisines (but fallback if too small)
    pref = [c.strip().lower() for c in user.get("preferred_cuisines", []) if c.strip()]
    if pref:
        df = recipes_df[recipes_df["cuisine"].isin(pref)].copy()
    else:
        df = recipes_df.copy()
    # respect restrictions
    restrictions = [normalize_ing(r) for r in user.get("restrictions", []) if r and r.strip()]
    if restrictions:
        df = df[~df["ing_list"].apply(lambda ings: any(r in ing for r in restrictions for ing in ings))]
    if df.shape[0] < 10:
        df = recipes_df.copy()

    # rank and limit pool
    df["rank_score"] = (2.0 * (df["protein_g"] / df["calories"].replace(0,1))) - (0.01 * df["cost_est"])
    pool = df.sort_values("rank_score", ascending=False).head(120)

    # build meal-specific pools, fallback to pool if none
    pools = {}
    for m in ["breakfast","lunch","dinner"]:
        sub = pool[pool["meal_type"] == m]
        pools[m] = sub.head(40) if not sub.empty else pool.head(40)

    # create rec_map and ILP variables
    rec_map = {}
    for m, sub in pools.items():
        for _, r in sub.iterrows():
            rec_map[(m, int(r["id"]))] = r

    # define ILP
    prob = LpProblem(f"NutriBridge_Day_{day_idx}", LpMinimize)
    x = {k: LpVariable(f"x_{k[0]}_{k[1]}", cat=LpBinary) for k in rec_map.keys()}

    # exactly one selection per meal
    for m in ["breakfast","lunch","dinner"]:
        prob += lpSum([x[k] for k in x if k[0] == m]) == 1

    # nutrition sums
    total_cal = lpSum([x[k] * float(rec_map[k]["calories"]) for k in x])
    total_pro = lpSum([x[k] * float(rec_map[k]["protein_g"]) for k in x])

    # constraints with tolerance
    prob += total_cal >= (1 - user.get("cal_tol", 0.12)) * cal_tgt
    prob += total_cal <= (1 + user.get("cal_tol", 0.12)) * cal_tgt
    prob += total_pro >= (1 - user.get("pro_tol", 0.12)) * pro_tgt

    # penalties to encourage variety
    PEN_CUISINE = 20.0
    PEN_RECIPE = 40.0
    penalty = lpSum([
        x[k] * ((PEN_CUISINE if rec_map[k]["cuisine"] in last_cuisines else 0.0) +
                (PEN_RECIPE if int(rec_map[k]["id"]) in used_recipes else 0.0))
        for k in x
    ])
    prob += penalty

    # solve quietly
    prob.solve(PULP_CBC_CMD(msg=0))

    # parse results
    chosen = []
    cuisines_used = []
    tot_cal = 0.0
    tot_pro = 0.0

    for k, var in x.items():
        try:
            val = var.value()
        except Exception:
            val = None
        if val is not None and val >= 0.99:
            rec = rec_map[k]
            meal = {
                "day": day_idx,
                "meal_type": k[0],
                "recipe_id": int(rec["id"]),
                "name": rec.get("name",""),
                "cuisine": rec.get("cuisine","general"),
                "calories": float(rec.get("calories", 0.0)),
                "protein_g": float(rec.get("protein_g", 0.0)),
                "cost_est": float(rec.get("cost_est", 0.0)),
                "ingredients": rec.get("ing_list", [])
            }
            chosen.append(meal)
            cuisines_used.append(meal["cuisine"])
            used_recipes.add(int(rec["id"]))
            tot_cal += meal["calories"]
            tot_pro += meal["protein_g"]

    # fallback: if solver failed to pick 3 meals, pick top-3 from pool safely
    if len(chosen) < 3:
        # choose top per meal type heuristically
        chosen = []
        cuisines_used = []
        for m in ["breakfast","lunch","dinner"]:
            cand = pools[m].sort_values("rank_score", ascending=False)
            if cand.shape[0] > 0:
                r = cand.iloc[0]
                meal = {
                    "day": day_idx,
                    "meal_type": m,
                    "recipe_id": int(r["id"]),
                    "name": r.get("name",""),
                    "cuisine": r.get("cuisine","general"),
                    "calories": float(r.get("calories", 0.0)),
                    "protein_g": float(r.get("protein_g", 0.0)),
                    "cost_est": float(r.get("cost_est", 0.0)),
                    "ingredients": r.get("ing_list", [])
                }
                chosen.append(meal)
                cuisines_used.append(meal["cuisine"])
                used_recipes.add(int(r["id"]))
                tot_cal += meal["calories"]
                tot_pro += meal["protein_g"]

    return {"meals": chosen, "nutrition": (tot_cal, tot_pro), "cuisines": cuisines_used, "used_recipes": used_recipes}

# -------------------- categorize shopping --------------------
CATEGORIES = {
    "Produce": ["onion","garlic","tomato","lettuce","cilantro","ginger","pepper","spinach","zucchini","parsley","lemon","apple","banana"],
    "Dairy": ["milk","cheese","mozzarella","paneer","ricotta","butter","yogurt","cream"],
    "Meat": ["chicken","pork","beef","tuna","fish","egg","mutton","shrimp"],
    "Pantry": ["flour","rice","pasta","oil","sugar","salt","vinegar","mustard","sauce","beans","bread"]
}
def classify_ingredient(ing):
    ing_l = ing.lower()
    for cat, keys in CATEGORIES.items():
        if any(k in ing_l for k in keys):
            return cat
    return "Other"

# -------------------- Streamlit UI --------------------
st.title("Nutri-Bridge — Personalized Nutrition Planner")
st.markdown("Upload `RAW_recipes_small.csv` (sampled small dataset) or use the file uploader to supply your CSV.")

uploaded = st.file_uploader("Upload RAW_recipes_small.csv (or leave blank to use local file)", type=["csv"])
if uploaded is not None:
    recipes = load_recipes(uploaded)
else:
    # attempt to use local default file
    try:
        recipes = load_recipes("RAW_recipes_small.csv")
    except Exception:
        recipes = pd.DataFrame()

st.info(f"Recipes available: {len(recipes)}")

# Sidebar user profile
st.sidebar.header("User Profile")
age = st.sidebar.number_input("Age", min_value=5, max_value=120, value=30)
gender = st.sidebar.selectbox("Gender", options=["male","female"], index=0)
height = st.sidebar.number_input("Height (cm)", min_value=100.0, max_value=230.0, value=170.0)
weight = st.sidebar.number_input("Weight (kg)", min_value=20.0, max_value=250.0, value=70.0)
activity = st.sidebar.selectbox("Activity", options=["sedentary","light","moderate","active","very_active"], index=2)
preferred_cuisines_raw = st.sidebar.text_input("Preferred cuisines (comma separated)", value="indian,italian").strip()
preferred_cuisines = [c.strip().lower() for c in preferred_cuisines_raw.split(",") if c.strip()]
restrictions_raw = st.sidebar.text_input("Dietary restrictions (comma separated)", value="").strip()
restrictions = [r.strip() for r in restrictions_raw.split(",") if r.strip()]
cal_tol = st.sidebar.slider("Calorie tolerance (±%)", 0.0, 0.5, 0.12, step=0.01)
pro_tol = st.sidebar.slider("Protein tolerance (±%)", 0.0, 0.5, 0.12, step=0.01)

user = {
    "age": age, "gender": gender, "height": height, "weight": weight,
    "activity": activity, "preferred_cuisines": preferred_cuisines,
    "restrictions": restrictions, "cal_tol": cal_tol, "pro_tol": pro_tol
}

# BMR / targets
ACTIVITY_FACTORS = {"sedentary":1.2,"light":1.375,"moderate":1.55,"active":1.725,"very_active":1.9}
def mifflin(gender, age, h, w):
    return (10*w + 6.25*h - 5*age + (5 if gender.lower().startswith("m") else -161))
def calorie_target(user):
    return int(round(mifflin(user["gender"], user["age"], user["height"], user["weight"]) * ACTIVITY_FACTORS.get(user["activity"], 1.375)))
def protein_target(user):
    return int(round(0.8 * user["weight"]))

# Generate button
if st.button("Generate 7-day Plan"):
    if recipes.empty:
        st.error("No recipes loaded. Upload RAW_recipes_small.csv or place it beside the app.")
    else:
        CAL_TGT = calorie_target(user)
        PRO_TGT = protein_target(user)
        st.success(f"Targets → Calories: {CAL_TGT} kcal/day, Protein: {PRO_TGT} g/day")

        week_rows = []
        summary_rows = []
        shopping = Counter()
        used_recipes = set()
        last_cuis = []

        st.info("Running planner for 7 days...")

        for day in range(1, 8):
            st.write(f"Generating Day {day}...")
            res = build_day_plan(day, recipes, user, last_cuis, used_recipes, CAL_TGT, PRO_TGT)
            day_plan = res["meals"]
            last_cuis = res["cuisines"]
            used_recipes = res["used_recipes"]
            tot_cal, tot_pro = res["nutrition"]

            summary_rows.append({"day": day, "total_calories": round(tot_cal,1), "total_protein_g": round(tot_pro,1)})
            for m in day_plan:
                week_rows.append({
                    "day": day,
                    "meal_type": m["meal_type"],
                    "cuisine": m["cuisine"],
                    "recipe_id": m["recipe_id"],
                    "name": m["name"],
                    "calories": round(m["calories"],1),
                    "protein_g": round(m["protein_g"],1),
                    "cost_est": round(m["cost_est"],2)
                })
                for ing in m.get("ingredients", []):
                    shopping[ing] += 1

        # DataFrames
        week_df = pd.DataFrame(week_rows)
        summary_df = pd.DataFrame(summary_rows)

        # Ensure 21 rows (3 meals per day) if some missing, fallback fill
        if week_df.shape[0] < 21:
            st.warning("Planner returned fewer than 21 meals; repeating top picks to fill week.")
            # repeat top rows until 21
            while week_df.shape[0] < 21 and not recipes.empty:
                top = recipes.sort_values("rank_score", ascending=False).iloc[0]
                week_df = week_df.append({
                    "day": week_df.shape[0]//3 + 1,
                    "meal_type": "dinner",
                    "cuisine": top.get("cuisine","general"),
                    "recipe_id": int(top["id"]),
                    "name": top.get("name",""),
                    "calories": float(top.get("calories",0)),
                    "protein_g": float(top.get("protein_g",0)),
                    "cost_est": float(top.get("cost_est",0))
                }, ignore_index=True)

        st.subheader("Weekly Meal Plan")
        st.dataframe(week_df)

        st.subheader("Daily Nutrition Summary")
        st.dataframe(summary_df)

        # Categorize shopping list -> DataFrame
        shop_rows = []
        for ing, qty in shopping.items():
            shop_rows.append([classify_ingredient(ing), ing, int(qty)])
        if shop_rows:
            shop_df = pd.DataFrame(shop_rows, columns=["Category","Ingredient","Quantity"])
        else:
            shop_df = pd.DataFrame(columns=["Category","Ingredient","Quantity"])

        st.subheader("Categorized Shopping List")
        st.dataframe(shop_df)

        # Downloads
        st.download_button("Download weekly plan CSV", week_df.to_csv(index=False).encode('utf-8'), "nutribridge_week.csv", "text/csv")
        st.download_button("Download nutrition summary CSV", summary_df.to_csv(index=False).encode('utf-8'), "nutribridge_summary.csv", "text/csv")
        st.download_button("Download shopping CSV", shop_df.to_csv(index=False).encode('utf-8'), "nutribridge_shopping.csv", "text/csv")

        # Nutrition trend graph (Calories & Protein different colors)
        st.subheader("Nutrition Trend (Calories & Protein)")
        fig, ax = plt.subplots(figsize=(9,4))
        ax.plot(summary_df["day"], summary_df["total_calories"], marker='o', linewidth=2, color='tab:blue', label='Calories (kcal)')
        ax.set_xlabel("Day")
        ax.set_ylabel("Calories (kcal)", color='tab:blue')
        ax2 = ax.twinx()
        ax2.plot(summary_df["day"], summary_df["total_protein_g"], marker='s', linewidth=2, color='tab:orange', label='Protein (g)')
        ax2.set_ylabel("Protein (g)", color='tab:orange')
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines + lines2, labels + labels2, loc='upper left')
        st.pyplot(fig)

        # Ingredient frequency bar chart (top 15)
        st.subheader("Ingredient Frequency (Top 15)")
        if not shop_df.empty:
            freq = shop_df.groupby("Ingredient")["Quantity"].sum().reset_index().sort_values("Quantity", ascending=False).head(15)
            fig2, ax3 = plt.subplots(figsize=(9, max(3, 0.25*len(freq))))
            ax3.barh(freq["Ingredient"][::-1], freq["Quantity"][::-1])
            ax3.set_xlabel("Times required this week")
            st.pyplot(fig2)
        else:
            st.info("No shopping items (shopping list is empty).")

        st.success("Plan generation complete.")
