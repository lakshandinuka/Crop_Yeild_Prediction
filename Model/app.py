import io
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from scipy.optimize import nnls
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------
st.set_page_config(
    page_title="PaddyRisk-SL | Rainfall Early Warning",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

TARGET = "Paddy_Yield_M.Tonnes"
REQUIRED = {
    "Year",
    "Season",
    "District",
    "Seasonal_Rainfall_mm",
    TARGET,
}
DEFAULT_DATA = Path(__file__).parent / "data" / "rainfall_agricultural_productivity_M_Tonnes.xlsx"


# ------------------------------------------------------------
# Data + feature engineering
# ------------------------------------------------------------
def load_data(source):
    if isinstance(source, (str, Path)):
        df = pd.read_excel(source)
    else:
        data = source.read()
        df = pd.read_excel(io.BytesIO(data))

    missing = REQUIRED - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = df.copy()
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    df["Seasonal_Rainfall_mm"] = pd.to_numeric(df["Seasonal_Rainfall_mm"], errors="coerce")
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df["Season"] = df["Season"].astype(str).str.strip()
    df["District"] = df["District"].astype(str).str.strip()

    df = df.dropna(subset=["Year", "Season", "District", "Seasonal_Rainfall_mm"])
    return df


def fit_rain_baselines(df):
    valid = df.dropna(subset=["Seasonal_Rainfall_mm"])
    ds = valid.groupby(["District", "Season"])["Seasonal_Rainfall_mm"].mean()
    season = valid.groupby("Season")["Seasonal_Rainfall_mm"].mean()
    overall = float(valid["Seasonal_Rainfall_mm"].mean())
    return ds.to_dict(), season.to_dict(), overall


def add_features(df, rainfall_baseline, season_baseline, overall_baseline, year_min):
    x = df.copy()
    keys = list(zip(x["District"], x["Season"]))
    baseline = np.array(
        [
            rainfall_baseline.get(k, season_baseline.get(k[1], overall_baseline))
            for k in keys
        ],
        dtype=float,
    )

    x["Rainfall_Baseline_mm"] = baseline
    x["Rainfall_Anomaly_mm"] = x["Seasonal_Rainfall_mm"] - x["Rainfall_Baseline_mm"]
    x["Rainfall_Ratio"] = x["Seasonal_Rainfall_mm"] / x["Rainfall_Baseline_mm"].replace(0, np.nan)
    x["Rainfall_Squared"] = (x["Seasonal_Rainfall_mm"] ** 2) / 1000.0
    x["Rainfall_Log1p"] = np.log1p(np.clip(x["Seasonal_Rainfall_mm"], a_min=0, a_max=None))
    x["Year_Index"] = x["Year"] - year_min
    x["District_Season"] = x["District"] + " | " + x["Season"]
    x["Rainfall_Stress"] = pd.cut(
        x["Rainfall_Ratio"],
        bins=[-np.inf, 0.70, 0.90, 1.10, 1.30, np.inf],
        labels=["Very Dry", "Below Normal", "Normal", "Above Normal", "Very Wet"],
    ).astype(str)
    return x


FEATURES = [
    "Seasonal_Rainfall_mm",
    "Rainfall_Anomaly_mm",
    "Rainfall_Ratio",
    "Rainfall_Squared",
    "Rainfall_Log1p",
    "Year_Index",
    "Season",
    "District",
    "District_Season",
    "Rainfall_Stress",
]
NUMERIC = [
    "Seasonal_Rainfall_mm",
    "Rainfall_Anomaly_mm",
    "Rainfall_Ratio",
    "Rainfall_Squared",
    "Rainfall_Log1p",
    "Year_Index",
]
CATEGORICAL = ["Season", "District", "District_Season", "Rainfall_Stress"]


def make_preprocessor():
    return ColumnTransformer(
        [
            ("num", StandardScaler(), NUMERIC),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ],
        remainder="drop",
    )


def fit_components(train_df):
    train_df = train_df.sort_values("Year").reset_index(drop=True)
    rb, sb, ob = fit_rain_baselines(train_df)
    x = add_features(train_df, rb, sb, ob, int(train_df["Year"].min()))
    y = train_df[TARGET].astype(float).values

    pre = make_preprocessor()
    z = pre.fit_transform(x[FEATURES])

    linear = Ridge(alpha=10.0, random_state=42)
    forest = RandomForestRegressor(
        n_estimators=500,
        min_samples_leaf=2,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1,
    )
    linear.fit(z, y)
    forest.fit(z, y)
    return {
        "preprocessor": pre,
        "linear": linear,
        "forest": forest,
        "rainfall_baseline": rb,
        "season_baseline": sb,
        "overall_rainfall_baseline": ob,
        "year_min": int(train_df["Year"].min()),
    }


def component_predict(bundle, raw_df):
    x = add_features(
        raw_df,
        bundle["rainfall_baseline"],
        bundle["season_baseline"],
        bundle["overall_rainfall_baseline"],
        bundle["year_min"],
    )
    z = bundle["preprocessor"].transform(x[FEATURES])
    return bundle["linear"].predict(z), bundle["forest"].predict(z)


def learn_ensemble_weights(train_df, n_splits=4):
    # Time-based CV keeps later years out of each fold's training data.
    work = train_df.sort_values("Year").reset_index(drop=True)
    y = work[TARGET].astype(float).values
    splitter = TimeSeriesSplit(n_splits=n_splits)
    oof_linear = np.full(len(work), np.nan)
    oof_forest = np.full(len(work), np.nan)

    for fold_train, fold_valid in splitter.split(work):
        tr = work.iloc[fold_train].copy()
        va = work.iloc[fold_valid].copy()
        bundle = fit_components(tr)
        p_lin, p_rf = component_predict(bundle, va)
        oof_linear[fold_valid] = p_lin
        oof_forest[fold_valid] = p_rf

    mask = np.isfinite(oof_linear) & np.isfinite(oof_forest)
    weights, _ = nnls(np.column_stack([oof_linear[mask], oof_forest[mask]]), y[mask])
    if weights.sum() == 0:
        weights = np.array([0.5, 0.5])
    else:
        weights = weights / weights.sum()
    return float(weights[0]), float(weights[1])


def fit_final_model(train_df):
    w_linear, w_forest = learn_ensemble_weights(train_df)
    bundle = fit_components(train_df)
    bundle["weight_linear"] = w_linear
    bundle["weight_forest"] = w_forest
    return bundle


def predict_ensemble(bundle, raw_df):
    p_linear, p_forest = component_predict(bundle, raw_df)
    pred = bundle["weight_linear"] * p_linear + bundle["weight_forest"] * p_forest
    return np.clip(pred, 0, None), p_linear, p_forest


def make_time_split(df):
    usable = df.dropna(subset=[TARGET]).sort_values("Year").reset_index(drop=True)
    years = sorted(usable["Year"].unique())
    n_test_years = max(2, int(np.ceil(len(years) * 0.20))) if len(years) >= 5 else 1
    test_years = years[-n_test_years:]
    train = usable[~usable["Year"].isin(test_years)].copy()
    test = usable[usable["Year"].isin(test_years)].copy()
    return train, test, test_years


@st.cache_data(show_spinner=False)
def prepare_cached(df):
    train, test, test_years = make_time_split(df)
    return train, test, test_years


@st.cache_resource(show_spinner=False)
def train_cached(df):
    train, test, test_years = make_time_split(df)
    bundle = fit_final_model(train)
    pred, p_lin, p_rf = predict_ensemble(bundle, test)

    metrics = {
        "R2": r2_score(test[TARGET], pred),
        "MAE": mean_absolute_error(test[TARGET], pred),
        "RMSE": np.sqrt(mean_squared_error(test[TARGET], pred)),
        "test_years": test_years,
        "train_n": len(train),
        "test_n": len(test),
    }
    validation = test[["Year", "Season", "District", "Seasonal_Rainfall_mm", TARGET]].copy()
    validation["Predicted"] = pred
    validation["Absolute_Error"] = np.abs(validation[TARGET] - validation["Predicted"])
    return bundle, metrics, validation


# ------------------------------------------------------------
# App helpers
# ------------------------------------------------------------
def money_like(v):
    return f"{v:,.2f}"


def risk_from_ratio(ratio):
    if ratio < 0.70:
        return "High Risk", "Prioritise contingency planning and closer monitoring."
    if ratio < 0.90:
        return "Moderate Risk", "Increase monitoring and prepare targeted support."
    return "Normal", "No production warning from this prototype threshold."


def get_ds_baseline(bundle, district, season):
    value = bundle["rainfall_baseline"].get((district, season))
    if value is None:
        value = bundle["season_baseline"].get(season, bundle["overall_rainfall_baseline"])
    return float(value)


def get_production_baselines(df):
    usable = df.dropna(subset=[TARGET])
    return (
        usable.groupby(["District", "Season"])[TARGET]
        .mean()
        .rename("Baseline_Production")
        .reset_index()
    )


def selected_context(bundle, district, season, rainfall):
    rain_base = get_ds_baseline(bundle, district, season)
    anomaly = rainfall - rain_base
    ratio = rainfall / rain_base if rain_base else np.nan
    if ratio < 0.70:
        stress = "Very Dry"
    elif ratio < 0.90:
        stress = "Below Normal"
    elif ratio < 1.10:
        stress = "Normal"
    elif ratio < 1.30:
        stress = "Above Normal"
    else:
        stress = "Very Wet"
    return rain_base, anomaly, ratio, stress


def prediction_table_output(row):
    return pd.DataFrame([row])


# ------------------------------------------------------------
# UI
# ------------------------------------------------------------
st.markdown(
    """
    <div style='padding: 0.6rem 0 0.2rem 0;'>
      <h1 style='margin-bottom:0;'>🌾 PaddyRisk-SL</h1>
      <p style='font-size:1.08rem; opacity:0.78; margin-top:0.2rem;'>Rainfall-Based Paddy Production Early Warning & Decision Support</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Data & Scenario")
    uploaded = st.file_uploader("Upload Excel dataset", type=["xlsx", "xls"])
    source = uploaded if uploaded is not None else DEFAULT_DATA

    try:
        df = load_data(source)
    except Exception as exc:
        st.error(f"Could not load dataset: {exc}")
        st.stop()

    train, test, test_years = prepare_cached(df)
    bundle, metrics, validation = train_cached(df)

    st.caption(
        f"Dataset: {len(df):,} rows | usable target rows: {df[TARGET].notna().sum():,} | missing target: {df[TARGET].isna().sum():,}"
    )

    districts = sorted(df["District"].dropna().unique().tolist())
    seasons = sorted(df["Season"].dropna().unique().tolist())
    years = sorted(df["Year"].dropna().astype(int).unique().tolist())

    district = st.selectbox("District", districts, index=0)
    season = st.selectbox("Season", seasons, index=0)
    future_year = st.number_input(
        "Scenario year",
        min_value=int(min(years)),
        max_value=int(max(years) + 10),
        value=int(max(years)),
        step=1,
    )
    district_rain_base = get_ds_baseline(bundle, district, season)
    rainfall = st.number_input(
        "Seasonal rainfall (mm)",
        min_value=0.0,
        max_value=5000.0,
        value=float(round(district_rain_base, 1)),
        step=10.0,
    )

    st.divider()
    st.subheader("Prototype thresholds")
    st.write("Risk is based on predicted production relative to the historical district-season baseline.")
    st.code("< 70%  → High Risk\n70–90% → Moderate Risk\n≥ 90%  → Normal", language="text")
    st.caption("These thresholds are decision-support prototypes and should be validated by an agricultural expert.")

# Core scenario
scenario = pd.DataFrame(
    [{"Year": future_year, "Season": season, "District": district, "Seasonal_Rainfall_mm": rainfall}]
)
scenario_pred, scenario_linear, scenario_forest = predict_ensemble(bundle, scenario)
predicted = float(scenario_pred[0])
pred_linear = float(scenario_linear[0])
pred_forest = float(scenario_forest[0])

production_base = get_production_baselines(df)
row_base = production_base[
    (production_base["District"] == district) & (production_base["Season"] == season)
]
if not row_base.empty:
    baseline_production = float(row_base["Baseline_Production"].iloc[0])
else:
    baseline_production = float(df[TARGET].dropna().mean())

production_ratio = predicted / baseline_production if baseline_production else np.nan
shortfall = max(baseline_production - predicted, 0)
risk_level, action = risk_from_ratio(production_ratio)
rain_base, rain_anomaly, rain_ratio, rain_stress = selected_context(bundle, district, season, rainfall)

# Header metrics
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Predicted Production", f"{predicted:,.1f}")
m2.metric("Historical Baseline", f"{baseline_production:,.1f}")
m3.metric("Production Ratio", f"{production_ratio*100:,.1f}%")
m4.metric("Expected Shortfall", f"{shortfall:,.1f}")
m5.metric("Rainfall vs Normal", f"{rain_ratio*100:,.1f}%")

st.info(
    f"**{risk_level}:** {action}  |  Rainfall context: **{rain_stress}** ({rain_anomaly:+.1f} mm from the historical district-season rainfall baseline)."
)

# Model metrics
st.subheader("Model validation")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Time-holdout R²", f"{metrics['R2']:.3f}")
c2.metric("MAE", f"{metrics['MAE']:,.1f}")
c3.metric("RMSE", f"{metrics['RMSE']:,.1f}")
c4.metric("Holdout years", f"{min(test_years)}–{max(test_years)}")
st.caption(
    f"The holdout contains {metrics['test_n']} observations. The model is a time-aware ensemble: "
    f"Ridge weight {bundle['weight_linear']:.2f} + Random Forest weight {bundle['weight_forest']:.2f}."
)

left, right = st.columns([1.15, 1])
with left:
    st.subheader("🌧️ Rainfall sensitivity")
    scenario_grid = np.linspace(max(0, rain_base * 0.50), rain_base * 1.50, 41)
    curve = []
    for r in scenario_grid:
        s = pd.DataFrame([{"Year": future_year, "Season": season, "District": district, "Seasonal_Rainfall_mm": float(r)}])
        p, _, _ = predict_ensemble(bundle, s)
        curve.append(float(p[0]))
    curve_df = pd.DataFrame({"Rainfall_mm": scenario_grid, "Predicted_Production": curve})
    fig = px.line(curve_df, x="Rainfall_mm", y="Predicted_Production", markers=True)
    fig.add_vline(x=rain_base, line_dash="dash", annotation_text="historical normal")
    fig.add_vline(x=rainfall, line_dash="dot", annotation_text="scenario")
    fig.update_layout(height=360, margin=dict(l=10, r=10, t=20, b=10))
    st.plotly_chart(fig, use_container_width=True)

    # Local sensitivity around selected rainfall
    eps = max(10.0, rain_base * 0.02)
    low = pd.DataFrame([{"Year": future_year, "Season": season, "District": district, "Seasonal_Rainfall_mm": max(0.0, rainfall - eps)}])
    high = pd.DataFrame([{"Year": future_year, "Season": season, "District": district, "Seasonal_Rainfall_mm": rainfall + eps}])
    p_low, _, _ = predict_ensemble(bundle, low)
    p_high, _, _ = predict_ensemble(bundle, high)
    mm_per_100 = (float(p_high[0]) - float(p_low[0])) / (2 * eps) * 100
    st.metric("Local rainfall response", f"{mm_per_100:,.1f} production-units / 100 mm")
    st.caption("This is a local model sensitivity, not a causal estimate.")

with right:
    st.subheader("🔎 Prediction decomposition")
    decomp = pd.DataFrame(
        {"Component": ["Context model (Ridge)", "Nonlinear model (Random Forest)"], "Prediction": [pred_linear, pred_forest]}
    )
    fig2 = px.bar(decomp, x="Prediction", y="Component", orientation="h", text_auto=".1f")
    fig2.update_layout(height=220, margin=dict(l=10, r=10, t=20, b=10))
    st.plotly_chart(fig2, use_container_width=True)
    st.markdown(
        f"**Interpretation:** the final prediction combines the contextual estimate and nonlinear estimate using learned weights of "
        f"**{bundle['weight_linear']:.0%}** and **{bundle['weight_forest']:.0%}**."
    )

    st.subheader("📌 Decision signals")
    signals = [
        ("Rainfall anomaly", f"{rain_anomaly:+.1f} mm", "departure from district-season normal"),
        ("Rainfall stress", rain_stress, "prototype climate context"),
        ("Production ratio", f"{production_ratio:.2f}×", "predicted / historical baseline"),
        ("Expected shortfall", f"{shortfall:,.1f}", "baseline minus prediction, floored at zero"),
    ]
    for label, value, note in signals:
        st.write(f"**{label}:** {value}\n\n_{note}_")

# Explainability
st.subheader("🧠 What is driving the model?")
usable_for_importance = test.copy()
if len(usable_for_importance) > 5:
    # Feature importance is computed on raw, human-readable variables.
    def raw_predict(frame):
        return predict_ensemble(bundle, frame)[0]

    try:
        perm = permutation_importance(
            estimator=type("ScenarioEstimator", (), {"predict": raw_predict})(),
            X=usable_for_importance[["Year", "Season", "District", "Seasonal_Rainfall_mm"]],
            y=usable_for_importance[TARGET],
            n_repeats=5,
            random_state=42,
            scoring="neg_mean_absolute_error",
        )
        imp = pd.DataFrame(
            {
                "Feature": ["Year", "Season", "District", "Seasonal_Rainfall_mm"],
                "Importance": perm.importances_mean,
            }
        ).sort_values("Importance", ascending=False)
        fig3 = px.bar(imp, x="Importance", y="Feature", orientation="h", title="Permutation importance (MAE impact)")
        fig3.update_layout(height=300, margin=dict(l=10, r=10, t=45, b=10))
        st.plotly_chart(fig3, use_container_width=True)
    except Exception as exc:
        st.warning(f"Importance calculation skipped: {exc}")

# Historical benchmark
st.subheader("📊 Selected district-season history")
hist = df[(df["District"] == district) & (df["Season"] == season)].dropna(subset=[TARGET]).sort_values("Year")
if len(hist):
    hist_fig = go.Figure()
    hist_fig.add_trace(go.Scatter(x=hist["Year"], y=hist[TARGET], mode="lines+markers", name="Historical production"))
    hist_fig.add_hline(y=baseline_production, line_dash="dash", annotation_text="historical baseline")
    hist_fig.add_trace(go.Scatter(x=[future_year], y=[predicted], mode="markers", marker=dict(size=12), name="Scenario prediction"))
    hist_fig.update_layout(height=360, margin=dict(l=10, r=10, t=20, b=10), xaxis_title="Year", yaxis_title="Production (dataset units)")
    st.plotly_chart(hist_fig, use_container_width=True)
else:
    st.warning("No historical observations are available for this district-season combination.")

# Validation errors
st.subheader("📐 Holdout validation")
val_fig = px.scatter(
    validation,
    x=TARGET,
    y="Predicted",
    color="Season",
    hover_data=["Year", "District", "Seasonal_Rainfall_mm"],
    trendline="ols",
)
min_v = min(validation[TARGET].min(), validation["Predicted"].min())
max_v = max(validation[TARGET].max(), validation["Predicted"].max())
val_fig.add_trace(go.Scatter(x=[min_v, max_v], y=[min_v, max_v], mode="lines", name="Perfect prediction"))
val_fig.update_layout(height=430, margin=dict(l=10, r=10, t=20, b=10))
st.plotly_chart(val_fig, use_container_width=True)

# Download
st.subheader("⬇️ Export scenario")
output_row = {
    "Year": future_year,
    "Season": season,
    "District": district,
    "Seasonal_Rainfall_mm": rainfall,
    "Rainfall_Baseline_mm": rain_base,
    "Rainfall_Anomaly_mm": rain_anomaly,
    "Rainfall_Ratio": rain_ratio,
    "Predicted_Production": predicted,
    "Historical_Baseline_Production": baseline_production,
    "Production_Ratio": production_ratio,
    "Expected_Shortfall": shortfall,
    "Risk_Level": risk_level,
}
out = prediction_table_output(output_row)
st.download_button(
    "Download scenario as CSV",
    out.to_csv(index=False).encode("utf-8"),
    file_name="paddyrisk_scenario.csv",
    mime="text/csv",
)

st.caption(
    "Important: this prototype predicts conditional on the supplied rainfall scenario. It does not forecast future rainfall itself, "
    "and statistical association should not be interpreted as causation."
)
