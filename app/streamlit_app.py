"""
Predictive maintenance early-warning dashboard.

Shows the test-period (Sep–Dec 2015) predictions of a gradient boosting model
that warns 24 hours before a machine failure. All heavy computation happens in
the notebook; this app only reads the exported files in app/data/.
"""
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

DATA = Path(__file__).parent / "data"
IMAGES = Path(__file__).parent.parent / "images"
H = 24                      # warning horizon in hours (must match the notebook)
DEFAULT_THR = 0.5           # chosen on validation in the notebook
SENSORS = ["volt", "rotate", "pressure", "vibration"]
AMBER, DARK, RED, STEEL, GREEN = "#C98E00", "#1D2A33", "#A8261D", "#3E6A85", "#2E7D4F"

st.set_page_config(page_title="Predictive maintenance dashboard", page_icon="🛠️", layout="wide")


# ---------------------------------------------------------------- data
@st.cache_data
def load():
    d = pd.read_parquet(DATA / "app_data.parquet")
    d["datetime"] = pd.to_datetime(d["datetime"])
    f = pd.read_csv(DATA / "failures_test.csv", parse_dates=["datetime"])
    imp = pd.read_csv(DATA / "feature_importance.csv", index_col=0)
    return d, f, imp


@st.cache_data
def prepare():
    """Arrange risks so any threshold can be evaluated instantly."""
    d, f, _ = load()
    start, end = d.datetime.min(), d.datetime.max()
    n_hours = int((end - start) / pd.Timedelta(hours=1)) + 1
    machines = np.sort(d.machineID.unique())
    row = {m: i for i, m in enumerate(machines)}

    # risk[machine, hour]: one row per machine, one column per test-period hour
    risk = np.full((len(machines), n_hours), np.nan, dtype="float32")
    hour_idx = ((d.datetime - start) / pd.Timedelta(hours=1)).astype(int).to_numpy()
    risk[d.machineID.map(row).to_numpy(), hour_idx] = d.risk.to_numpy()

    # failure events with a full 24 h window inside the test period
    ev = f[["machineID", "datetime"]].drop_duplicates()
    ev = ev[(ev.datetime >= start + pd.Timedelta(hours=H)) & (ev.datetime <= end)]
    ev_hour = ((ev.datetime - start) / pd.Timedelta(hours=1)).astype(int).to_numpy()
    # windows[e] = risk in the 24 hours before event e, oldest hour first
    windows = np.stack([risk[row[m], h - H:h] for m, h in zip(ev.machineID, ev_hour)])

    neg_risk = d.risk.to_numpy()[d.y.to_numpy() == 0]
    weeks = (end - start).days / 7
    return windows, neg_risk, weeks


def evaluate(thr):
    """Same logic as event_report() in the notebook."""
    windows, neg_risk, weeks = prepare()
    alarm = windows >= thr
    caught = alarm.any(axis=1)
    first = alarm.argmax(axis=1)                       # first alarm hour in the window
    leads = H - first[caught]
    false_hrs = (neg_risk >= thr).sum() / weeks
    return int(caught.sum()), len(windows), (float(leads.mean()) if caught.any() else 0.0), float(false_hrs)


data, failures, importance = load()

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Alarm threshold")
    thr = st.slider("Raise an alarm when risk is at least", 0.05, 0.95, DEFAULT_THR, 0.05)
    st.caption(f"The notebook chose **{DEFAULT_THR}** on the validation months (Jul–Aug): "
               "the lowest threshold with zero false alarms. Move the slider to see the trade-off.")
    st.divider()
    st.caption("Data: Microsoft Azure predictive maintenance sample (synthetic). "
               "Test period: Sep–Dec 2015, never seen during training.")

# ---------------------------------------------------------------- header
st.title("🛠️ Equipment failure early warning")
st.write("A model that flags machines likely to fail **within the next 24 hours**, so a crew and spare parts "
         "can be ready before the machine stops. Everything below is from the unseen test period.")

caught, n_events, lead, false_hrs = evaluate(thr)
k1, k2, k3, k4 = st.columns(4)
k1.metric("Failures caught", f"{caught} / {n_events}")
k2.metric("Average warning", f"{lead:.0f} h")
k3.metric("False-alarm hours / week", f"{false_hrs:.1f}", help="Alarm hours with no failure coming, across all 100 machines")
k4.metric("Test PR-AUC", "0.998", help="Threshold-free ranking score; random guessing = 0.018")

tab_fleet, tab_machine, tab_thr, tab_how = st.tabs(["Fleet risk", "Machine explorer", "Threshold trade-off", "How it works"])

# ---------------------------------------------------------------- fleet
with tab_fleet:
    st.subheader("Which machines need attention right now?")
    c1, c2 = st.columns([2, 1])
    days = pd.Series(data.datetime.dt.normalize().unique()).sort_values()
    day = c1.select_slider("Day", options=list(days), value=days.iloc[len(days) // 2],
                           format_func=lambda x: x.strftime("%d %b %Y"))
    hour = c2.slider("Hour", 0, 23, 12)
    now = pd.Timestamp(day) + pd.Timedelta(hours=hour)
    snap = data[data.datetime == now].copy()
    if snap.empty:
        st.info("No readings at this hour (the test period starts on 2 Sep 2015). Pick another time.")
    else:
        snap["alarm"] = snap.risk >= thr
        snap["fails within 24 h?"] = np.where(snap.y == 1, "yes", "no")
        n_alarm = int(snap.alarm.sum())
        st.write(f"**{now:%d %b %Y, %H:00}** · {n_alarm} machine{'s' if n_alarm != 1 else ''} above the threshold.")
        top = snap.sort_values("risk", ascending=False).head(10)
        st.dataframe(
            top[["machineID", "risk", "alarm", "fails within 24 h?", *SENSORS]],
            hide_index=True, width="stretch",
            column_config={
                "machineID": st.column_config.NumberColumn("Machine", format="%d"),
                "risk": st.column_config.ProgressColumn("Risk", min_value=0.0, max_value=1.0, format="%.2f"),
                "alarm": st.column_config.CheckboxColumn("Alarm"),
                "volt": st.column_config.NumberColumn(format="%.0f"),
                "rotate": st.column_config.NumberColumn(format="%.0f"),
                "pressure": st.column_config.NumberColumn(format="%.0f"),
                "vibration": st.column_config.NumberColumn(format="%.1f"),
            })
        st.caption("The 10 riskiest machines at this hour. \u201cFails within 24 h?\u201d is what actually happened, "
                   "which the model could not see.")

# ---------------------------------------------------------------- machine
with tab_machine:
    st.subheader("One machine through the test period")
    counts = failures.groupby("machineID").size()
    machines = sorted(data.machineID.unique())
    default_m = int(counts.idxmax()) if len(counts) else machines[0]
    c1, c2 = st.columns([1, 3])
    m = c1.selectbox("Machine", machines, index=machines.index(default_m))
    sensor = c1.radio("Sensor", SENSORS, index=0)
    md = data[data.machineID == m]
    lo, hi = md.datetime.min().to_pydatetime(), md.datetime.max().to_pydatetime()
    rng = c2.slider("Date range", min_value=lo, max_value=hi, value=(lo, hi), format="DD MMM")
    md = md[(md.datetime >= rng[0]) & (md.datetime <= rng[1])]
    mf = failures[(failures.machineID == m) & (failures.datetime >= rng[0]) & (failures.datetime <= rng[1])]
    c1.metric("Failures in range", len(mf))

    base = alt.Chart(md).encode(x=alt.X("datetime:T", title=None))
    fail_rules = alt.Chart(mf).mark_rule(color=RED, strokeWidth=2).encode(
        x="datetime:T", tooltip=[alt.Tooltip("datetime:T", title="Failure"), alt.Tooltip("comp:N", title="Part")])
    risk_chart = (base.mark_line(color=DARK, strokeWidth=1.2).encode(
                      y=alt.Y("risk:Q", title="Risk", scale=alt.Scale(domain=[0, 1])),
                      tooltip=[alt.Tooltip("datetime:T"), alt.Tooltip("risk:Q", format=".2f")])
                  + alt.Chart(pd.DataFrame({"t": [thr]})).mark_rule(color=AMBER, strokeDash=[6, 4], strokeWidth=2).encode(y="t:Q")
                  + fail_rules).properties(height=220, title="Model risk (amber = alarm threshold, red = failures)")
    sensor_chart = (base.mark_line(color=STEEL, strokeWidth=1).encode(
                        y=alt.Y(f"{sensor}:Q", title=sensor, scale=alt.Scale(zero=False)),
                        tooltip=[alt.Tooltip("datetime:T"), alt.Tooltip(f"{sensor}:Q", format=".1f")])
                    + fail_rules).properties(height=220, title=f"{sensor.capitalize()} readings")
    c2.altair_chart(alt.vconcat(risk_chart, sensor_chart).resolve_scale(x="shared"), width="stretch")
    if len(mf):
        c2.caption("Zoom in on a failure with the date range slider: the risk climbs about a day before each red line, "
                   "and one sensor jumps around two days before. Which sensor depends on the part (comp1 → volt, "
                   "comp2 → rotate, comp3 → pressure, comp4 → vibration).")

# ---------------------------------------------------------------- threshold
with tab_thr:
    st.subheader("Earlier warnings vs. fewer false alarms")
    grid = np.round(np.arange(0.05, 0.96, 0.05), 2)
    rows = [dict(zip(["threshold", "caught", "events", "warning_h", "false_hrs"], (t, *evaluate(t)))) for t in grid]
    tdf = pd.DataFrame(rows)
    tdf["caught_pct"] = 100 * tdf.caught / tdf.events
    sel = alt.Chart(pd.DataFrame({"threshold": [thr]})).mark_rule(color=AMBER, strokeWidth=2).encode(x="threshold:Q")
    c1, c2 = st.columns(2)
    c1.altair_chart((alt.Chart(tdf).mark_line(point=True, color=RED).encode(
        x=alt.X("threshold:Q", title="Threshold"), y=alt.Y("false_hrs:Q", title="False-alarm hours / week"),
        tooltip=["threshold", alt.Tooltip("false_hrs:Q", format=".1f")]) + sel).properties(height=280, title="False alarms"),
        width="stretch")
    c2.altair_chart((alt.Chart(tdf).mark_line(point=True, color=GREEN).encode(
        x=alt.X("threshold:Q", title="Threshold"), y=alt.Y("caught_pct:Q", title="Failures caught (%)", scale=alt.Scale(domain=[0, 100])),
        tooltip=["threshold", "caught", alt.Tooltip("warning_h:Q", format=".1f", title="avg warning (h)")]) + sel)
        .properties(height=280, title="Failures caught"), width="stretch")
    st.write("On this dataset the warning signs are strong, so almost every failure is caught at any threshold, and the "
             "real choice is how many false alarms to accept. On real mine data, where wear is gradual, raising the "
             "threshold would also shorten the warning time.")
    st.caption("Honesty note: the threshold used in the reported results (0.5) was fixed on validation data before the test "
               "period was scored. Exploring here is fine, but picking a new threshold from this test data would make the "
               "result optimistic.")

# ---------------------------------------------------------------- how it works
with tab_how:
    st.subheader("How the model works")
    img = IMAGES / "warning_sensors.png"
    if img.exists():
        st.image(str(img), caption="Average readings in the week before all 761 failures, aligned on the failure hour. "
                                   "Each part has its own warning sensor, which jumps about 48 hours before failure.")
    st.markdown(
        """
1. **Label:** for every machine and hour, *will this machine fail in the next 24 hours?* (about 2% of hours are "yes").
2. **Features (past data only, per machine):** 3 h and 24 h rolling means and spreads of each sensor, error counts in the last 24 h,
   hours since each part was replaced, machine age and model.
3. **Split by time:** train Jan–Aug, test Sep–Dec, with a 1-day gap so labels can't peek across the boundary.
4. **Model:** gradient boosting (`HistGradientBoostingClassifier`) with balanced class weights.
        """)
    c1, c2 = st.columns(2)
    c1.markdown("**Baselines vs. model** (test PR-AUC)")
    c1.dataframe(pd.DataFrame({"Method": ["Random guessing", '"Oldest part" rule', '"Recent errors" rule', "Model"],
                               "PR-AUC": [0.018, 0.021, 0.459, 0.998]}), hide_index=True, width="stretch")
    c2.markdown("**Leak test:** retrain with parts removed (validation PR-AUC)")
    c2.dataframe(pd.DataFrame({"Features": ["All", "Error counts only", "Sensor features only", "Raw hourly sensors only", "All, 24–48 h ahead"],
                               "PR-AUC": [0.9997, 0.67, 0.25, 0.07, 0.83]}), hide_index=True, width="stretch")
    st.caption("No single feature group gives the answer away, so the high score comes from combining error bursts with sensor shifts, not from a leak.")

    st.markdown("**What the model relies on** (drop in PR-AUC when each feature is shuffled)")
    imp = importance.reset_index().rename(columns={"index": "feature"}).head(10)
    imp.columns = ["feature", "importance"]
    st.altair_chart(alt.Chart(imp).mark_bar(color=STEEL).encode(
        x=alt.X("importance:Q", title="Importance"), y=alt.Y("feature:N", sort="-x", title=None),
        tooltip=["feature", alt.Tooltip("importance:Q", format=".3f")]).properties(height=300), width="stretch")

    st.markdown("**Limitations:** the data is synthetic, so warning signs are sudden jumps rather than gradual wear, labels are "
                "clean, and there is no operating context (payload, road grade, shift). Real mine data would need resampling, "
                "label cleaning and longer trend features.")
