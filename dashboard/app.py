import time
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Robot PM Dashboard", layout="wide")

st.title("🏭 Robot Predictive Maintenance Dashboard (Workshop)")

csv_path = st.sidebar.text_input("Processed CSV path", "data/processed/processed_robot_data.csv")
robot = st.sidebar.text_input("Robot ID (robot_1 / robot_2 / robot_3)", "robot_1")
refresh_seconds = st.sidebar.slider("Refresh interval (seconds)", 1, 5, 2)

st.sidebar.markdown("---")
alert_days = st.sidebar.number_input("Alert threshold (days)", value=14.0, step=1.0)

placeholder = st.empty()

while True:
    try:
        df = pd.read_csv(csv_path)
        df_robot = df[df["robot_id"] == robot].copy()
        if df_robot.empty:
            st.warning("No rows for this robot_id yet.")
            time.sleep(refresh_seconds)
            continue

        # latest rows
        df_robot["Time"] = pd.to_datetime(df_robot["Time"], utc=True, errors="coerce")
        df_robot = df_robot.sort_values("Time")

        latest = df_robot.tail(1).iloc[0]
        ttf = float(latest["time_to_failure_days"])

        col1, col2, col3 = st.columns(3)
        col1.metric("Robot", str(latest["robot_id"]))
        col2.metric("Latest Time", str(latest["Time"]))
        col3.metric("Predicted Time-to-Failure (days)", f"{ttf:.2f}")

        if ttf <= alert_days:
            st.error(f"🚨 ALERT: Predicted failure within {alert_days} days (TTF={ttf:.2f})")
        else:
            st.success(f"✅ Healthy runway: TTF={ttf:.2f} days")

        st.subheader("Recent Stream")
        placeholder.dataframe(df_robot.tail(50), use_container_width=True)

    except Exception as e:
        st.error(f"Dashboard error: {e}")

    time.sleep(refresh_seconds)
    st.rerun()
