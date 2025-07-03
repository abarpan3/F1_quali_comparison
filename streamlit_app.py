import streamlit as st
import fastf1
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import io
import os
from dotenv import load_dotenv
import google.generativeai as genai
from PIL import Image
import tempfile
from datetime import datetime, timezone
import matplotlib.pyplot as plt
from matplotlib import gridspec


# Ensure cache directory exists
os.makedirs('cache', exist_ok=True)

fastf1.Cache.enable_cache('cache')  # Enable FastF1 cache for speed

st.set_page_config(page_title="F1 Lap Comparison", layout="wide")
st.title("F1 Qualifying Fastest Lap Comparison (2025)")

# 1. Select race
year = 2025
@st.cache_data(show_spinner=True)
def get_event_schedule(year):
    return fastf1.get_event_schedule(year)

event_schedule = get_event_schedule(year)
event_schedule['Session5DateUtc'] = pd.to_datetime(event_schedule['Session5DateUtc'], utc=True)
now = pd.Timestamp.now(tz='UTC')
# Only include events whose 'Session5DateUtc' (the race) is in the past
past_events = event_schedule[event_schedule['Session5DateUtc'] < now]
event_names = past_events['EventName'].tolist()
event_name = st.selectbox("Select Race", event_names)

# 2. Load session and get drivers
@st.cache_data(show_spinner=True)
def load_session(year, event_name):
    session = fastf1.get_session(year, event_name, 'Q')
    session.load()
    return session

session = None
drivers = []
driver_abbr_to_name = {}

st.sidebar.title("Gemini API Key")
user_api_key = st.sidebar.text_input("Enter your Gemini API key", type="password")


if event_name:
    session = load_session(year, event_name)
    drivers = session.results['Abbreviation'].tolist()
    driver_abbr_to_name = dict(zip(session.results['Abbreviation'], session.results['FullName']))

    col1, col2 = st.columns(2)
    with col1:
        driver1_code = st.selectbox("Select Driver 1", drivers, key="d1")
    with col2:
        driver2_code = st.selectbox("Select Driver 2", drivers, key="d2")

    # Helper to get fastest lap in Q3/Q2/Q1
    def get_fastest_lap(laps):
        for q in [2, 1, 0]:
            try:
                lap = laps.split_qualifying_sessions()[q].pick_fastest()
                if lap is not None:
                    return lap
            except Exception:
                continue
        return None

    # Gemini analysis functions (from test.ipynb)
    def analyze_lap_data_with_gemini(tmp_path, track_name, api_key, short=False):
        if not api_key:
            return "No Gemini API key provided."
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.5-flash")

        img = Image.open(tmp_path)
        if short:
            analysis_prompt = """
Based on the provided lap data graphs comparing the drivers at [Insert Specific Race Track Name], generate a concise summary (approximately 3 lines in bullet style) providing specific insights into their performance differences. The Lap data graphs show the speed as well as the time difference between the two drivers. Additionally, the graphs show the vertical lines for the corner numbers.

Your analysis should break down their performance into distinct phases of the lap, correlating the 'Speed in km/h' graph with the 'Time Difference' graph. Specifically, identify and discuss:

1.  **Overall Performance:** Who is consistently faster and by how much at the end of the lap?
2.  **Major Gain Zones (Driver A):** Where does Driver A make their most significant time gains (i.e., where does the time difference line steepen most dramatically in their favor)? For each zone, infer the contributing factors based on the speed trace (e.g., better braking, higher minimum speed through corners, stronger traction on exit). Categorize these gains by corner type (low-speed, medium-speed, high-speed) or straight-line performance.
3.  **Major Gain Zones (Driver B):** Where does Driver B make their most significant time gains, or where do they best minimize their losses to Driver A? (If Driver B never gains, focus on areas where the time difference flattens or increases less rapidly). Infer contributing factors as above.
4.  **Cornering Analysis:**
    * **Low-Speed Corners:** Analyze performance in the slowest sections. Who demonstrates superior braking, corner entry, mid-corner speed, and exit traction?
    * **Medium-Speed Corners:** How do they compare through the more flowing, technical medium-speed sections? Who carries more speed and maintains better rhythm?
    * **High-Speed Corners:** Who shows better confidence and car stability through the fastest corners, indicated by higher minimum speeds and consistent acceleration?
5.  **Straight-Line Speed & Braking:**
    * Who appears to have superior straight-line speed (after corner exit and before braking)?
    * Who demonstrates stronger braking performance into key corners (later braking points, better deceleration)?
6.  **Car Characteristics/Driving Style Inferences:** Based on the analysis, what can you infer about the characteristics of each car (e.g., high downforce, low drag, good mechanical grip) or the driving style of each driver (e.g., aggressive on entry, smooth through corners, strong on traction)?

Focus on actionable, data-driven observations within the length constraint, identifying performance characteristics from the graphs. Present your findings clearly, referencing specific corners from the graphs where relevant. Assume a basic knowledge of the track layout and its typical corner speeds for a more in-depth analysis.
"""
        else:
            analysis_prompt = """
Analyze the provided lap data graphs, comparing the drivers over a single qualifying lap at the [Insert Specific Race Track Name].  The Lap data graphs show the speed as well as the time difference between the two drivers. Additionally, the graphs show the vertical lines for the corner numbers.

Your analysis should break down their performance into distinct phases of the lap, correlating the 'Speed in km/h' graph with the 'Time Difference' graph. Specifically, identify and discuss:

1.  **Overall Performance:** Who is consistently faster and by how much at the end of the lap?
2.  **Major Gain Zones (Driver A):** Where does Driver A make their most significant time gains (i.e., where does the time difference line steepen most dramatically in their favor)? For each zone, infer the contributing factors based on the speed trace (e.g., better braking, higher minimum speed through corners, stronger traction on exit). Categorize these gains by corner type (low-speed, medium-speed, high-speed) or straight-line performance.
3.  **Major Gain Zones (Driver B):** Where does Driver B make their most significant time gains, or where do they best minimize their losses to Driver A? (If Driver B never gains, focus on areas where the time difference flattens or increases less rapidly). Infer contributing factors as above.
4.  **Cornering Analysis:**
    * **Low-Speed Corners:** Analyze performance in the slowest sections. Who demonstrates superior braking, corner entry, mid-corner speed, and exit traction?
    * **Medium-Speed Corners:** How do they compare through the more flowing, technical medium-speed sections? Who carries more speed and maintains better rhythm?
    * **High-Speed Corners:** Who shows better confidence and car stability through the fastest corners, indicated by higher minimum speeds and consistent acceleration?
5.  **Straight-Line Speed & Braking:**
    * Who appears to have superior straight-line speed (after corner exit and before braking)?
    * Who demonstrates stronger braking performance into key corners (later braking points, better deceleration)?
6.  **Car Characteristics/Driving Style Inferences:** Based on the analysis, what can you infer about the characteristics of each car (e.g., high downforce, low drag, good mechanical grip) or the driving style of each driver (e.g., aggressive on entry, smooth through corners, strong on traction)?

Present your findings clearly, referencing specific corners from the graphs where relevant. Assume a basic knowledge of the track layout and its typical corner speeds for a more in-depth analysis.
"""
        customized_prompt = analysis_prompt.replace("[Insert Specific Race Track Name]", track_name)
        response = model.generate_content([customized_prompt, img])
        return response.text

    if driver1_code and driver2_code and driver1_code != driver2_code:
        d1_lap = get_fastest_lap(session.laps.pick_drivers(driver1_code))
        d2_lap = get_fastest_lap(session.laps.pick_drivers(driver2_code))

        if d1_lap is not None and d2_lap is not None:
            d1_tel = d1_lap.get_car_data().add_distance()
            d2_tel = d2_lap.get_car_data().add_distance()

            # Prepare a common distance axis
            common_distance = np.linspace(
                max(d1_tel['Distance'].min(), d2_tel['Distance'].min()),
                min(d1_tel['Distance'].max(), d2_tel['Distance'].max()),
                2000
            )

            # Interpolate cumulative time (in seconds) for both drivers
            # Also get interpolated speeds for hover
            # Interpolate cumulative time (in seconds) for both drivers
            d1_time_interp = np.interp(common_distance, d1_tel['Distance'], d1_tel['Time'].dt.total_seconds())
            d2_time_interp = np.interp(common_distance, d2_tel['Distance'], d2_tel['Time'].dt.total_seconds())
            d1_speed_interp = np.interp(common_distance, d1_tel['Distance'], d1_tel['Speed'])
            d2_speed_interp = np.interp(common_distance, d2_tel['Distance'], d2_tel['Speed'])

            # Determine which driver is faster at the end
            if d1_time_interp[-1] < d2_time_interp[-1]:
                faster_driver = driver1_code
                slower_driver = driver2_code
                time_diff = d2_time_interp - d1_time_interp
            else:
                faster_driver = driver2_code
                slower_driver = driver1_code
                time_diff = d1_time_interp - d2_time_interp

            # Now, time_diff[-1] will always be positive for the slower driver
            # Update the label accordingly:
            time_diff_label = f"Time Difference: {slower_driver} behind {faster_driver} (positive means {slower_driver} is behind)"

            # Get circuit info for corners
            circuit_info = session.get_circuit_info()

            # Create subplots
            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.08,
                row_heights=[0.7, 0.3],
                subplot_titles=(
                    f"Fastest Lap Comparison<br>{session.event['EventName']} {session.event.year} Qualifying",
                    time_diff_label
                )
            )

            # Speed traces (top) with custom hover
            fig.add_trace(
                go.Scatter(
                    x=common_distance, y=d1_speed_interp,
                    mode='lines', name=driver1_code, line=dict(color='#FCCB06'),
                    hovertemplate=f"%{{y:.1f}} km/h",
                    customdata=[f"{t:.3f}" for t in d1_time_interp]
                ),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=common_distance, y=d2_speed_interp,
                    mode='lines', name=driver2_code, line=dict(color='#B1DDF1', dash='dash'), opacity=0.7,
                    hovertemplate=f"%{{y:.1f}} km/h",
                    customdata=[f"{t:.3f}" for t in d2_time_interp]
                ),
                row=1, col=1
            )

            # Vertical lines and corner labels (top)
            for _, corner in circuit_info.corners.iterrows():
                fig.add_vline(
                    x=corner['Distance'],
                    line=dict(color='grey', width=1, dash='dot'),

                    row=1, col=1
                )
                fig.add_annotation(
                    x=corner['Distance'],
                    y=max(d1_tel['Speed'].max(), d2_tel['Speed'].max()) + 20,
                    text=f"{corner['Number']}{corner['Letter']}",
                    showarrow=False,
                    yanchor='top',
                    xanchor='center',
                    font=dict(size=10),
                    row=1, col=1
                )

            # Time difference trace (bottom) with custom hover
            fig.add_trace(
                go.Scatter(
                    x=common_distance, y=time_diff,
                    mode='lines', name=f"Time Diff ({slower_driver} - {faster_driver})",
                    line=dict(color='lime'),
                    hovertemplate="Time Diff: %{y:.3f} s<extra></extra>"
                ),
                row=2, col=1
            )

            # Vertical lines and corner labels (bottom)
            for _, corner in circuit_info.corners.iterrows():
                fig.add_vline(
                    x=corner['Distance'],
                    line=dict(color='grey', width=1, dash='dot'),
                    opacity=0.5,
                    row=2, col=1
                )


            # Update layout for dark theme
            fig.update_layout(
                width=1200, height=900,
                legend=dict(x=0.85, y=1.15, font=dict(color='white')),
                xaxis=dict(title='Distance in m', color='white', gridcolor='#444',showgrid=False),
                yaxis=dict(title='Speed in km/h', color='white', gridcolor='#444',showgrid=False),
                xaxis2=dict(title='Distance in m', color='white', gridcolor='#444'),
                yaxis2=dict(title='Time Difference (s)', color='white', gridcolor='#444'),
                hovermode='x unified',
                template='plotly_dark',
                font=dict(color='white')
            )

            st.plotly_chart(fig, use_container_width=True)

            # Download as PNG
            fig = plt.figure(figsize=(12, 9), facecolor='black')
            gs = gridspec.GridSpec(2, 1, height_ratios=[0.7, 0.3])
            ax1 = fig.add_subplot(gs[0])
            ax2 = fig.add_subplot(gs[1], sharex=ax1)

            # Top: Speed traces
            ax1.plot(common_distance, d1_speed_interp, label=driver1_code, color='orange')
            ax1.plot(common_distance, d2_speed_interp, label=driver2_code, color='white', linestyle='--', alpha=0.7)

            # Corner markers
            max_speed = max(max(d1_speed_interp), max(d2_speed_interp))
            for _, corner in circuit_info.corners.iterrows():
                ax1.axvline(x=corner['Distance'], color='gray', linestyle=':', linewidth=1)
                ax1.text(corner['Distance'], max_speed + 20, f"{corner['Number']}{corner['Letter']}",
                        color='white', fontsize=8, ha='center', va='top')

            # Bottom: Time difference trace
            ax2.plot(common_distance, time_diff, color='lime', label='Time Diff')
            for _, corner in circuit_info.corners.iterrows():
                ax2.axvline(x=corner['Distance'], color='gray', linestyle=':', linewidth=1, alpha=0.4)

            # Dark styling
            for ax in [ax1, ax2]:
                ax.set_facecolor('black')
                ax.tick_params(colors='white')
                # ax.grid(False)

            ax1.set_ylabel("Speed in km/h", color='white')
            ax2.set_ylabel("Time Difference (s)", color='white')
            ax2.set_xlabel("Distance in m", color='white')

            # Central figure title instead of subplot titles
            fig.suptitle(f"Fastest Lap Comparison — {session.event['EventName']} {session.event.year} Qualifying",
                        color='white', fontsize=14)

            # Adjust spacing
            fig.subplots_adjust(top=0.92, hspace=0.25)

            # Legend
            ax1.legend(loc='lower right', facecolor='black', edgecolor='white', labelcolor='white')
            ax2.set_title(time_diff_label,
                        color='white', fontsize=11)
            # Save without tight_layout (prevents title clipping)
            plt.savefig("final_clean_plot.png", dpi=150, facecolor=fig.get_facecolor())
            plt.show()

            # Gemini analysis buttons
            colA, colB = st.columns(2)
            with colA:
                if st.button("Gemini Short Analysis"):
                    with st.spinner("Gemini is analyzing (short form)..."):
                        analysis = analyze_lap_data_with_gemini("final_clean_plot.png", session.event['EventName'], user_api_key, short=True)
                        st.subheader("Gemini Short Analysis")
                        st.write(analysis)
            with colB:
                if st.button("Gemini Detailed Analysis"):
                    with st.spinner("Gemini is analyzing (detailed)..."):
                        analysis = analyze_lap_data_with_gemini("final_clean_plot.png", session.event['EventName'], user_api_key, short=False)
                        st.subheader("Gemini Detailed Analysis")
                        st.write(analysis)
        else:
            st.warning("Could not find valid laps for both drivers in qualifying.")
    else:
        st.info("Please select two different drivers.")
else:
    st.info("Please select a race.") 