import streamlit as st
import pandas as pd
import re

st.set_page_config(page_title="OEE Analysis", layout="wide")
st.title("💬 OEE Drivers and Draggers Chatbot")

# --- Load your data as before ---
@st.cache_data
def load_oee_data():
    df = pd.read_excel("Data/OEE.xlsm", engine="openpyxl")
    df.columns = df.columns.str.strip()
    df['P_YearPeriod'] = df['P_YearPeriod'].astype(str).str.strip()
    return df

df = load_oee_data()
@st.cache_data
def load_aggregated_oee():
    file_path = "Data/aggregated_OEE.xlsx"
    try:
        df = pd.read_excel(file_path, engine="openpyxl")
        df.columns = df.columns.str.strip()
        df['P_YearPeriod'] = df['P_YearPeriod'].astype(str).str.strip()

        # Ensure OEE% numeric
        if df['OEE%'].dtype == 'object':
            df['OEE%'] = df['OEE%'].astype(str).str.rstrip('%').astype(float)
        return df
    except Exception as e:
        st.error(f"Error reading aggregated OEE file: {e}")
        return None

# --- Simple function to extract two periods from user text ---
def extract_periods(query, available_periods):
    found = []
    for p in available_periods:
        if str(p).lower() in query.lower():
            found.append(p)
    return found[:2]  # take first two matches

# --- Chatbot interface ---
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Display previous chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User input
if user_input := st.chat_input("Ask me about OEE periods (e.g., Compare P1 2023 vs P2 2023):"):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Extract periods
    if any(word in user_input.lower() for word in ["chart", "plot", "trend", "graph","OEE data"]):
        aggregated_df = load_aggregated_oee()

        if aggregated_df is not None:
            reply = "📊 Here’s the OEE trend over time:"
            st.session_state.messages.append({"role": "assistant", "content": reply})
            with st.chat_message("assistant"):
                st.markdown(reply)

                chart_data = aggregated_df[['P_YearPeriod', 'OEE%']].copy()
                chart_data['OEE%'] = chart_data['OEE%'].astype(float).round(2)
                st.bar_chart(data=chart_data, x='P_YearPeriod', y='OEE%')
        else:
            reply = "⚠️ Aggregated OEE data file not found."
            st.session_state.messages.append({"role": "assistant", "content": reply})
            with st.chat_message("assistant"):
                st.markdown(reply)
    else:
        periods = df["P_YearPeriod"].unique().tolist()
        selected_periods = extract_periods(user_input, periods)

        if len(selected_periods) < 2:
            reply = "❌ Couldn't identify two valid periods. Please try like 'P1 2023 vs P2 2023'."
        else:
            first_period, second_period = selected_periods
            reply = f"✅ Comparing **{first_period}** vs **{second_period}**..."

            # --- Your existing analysis logic ---
            p1 = df[df['P_YearPeriod'] == first_period].copy()
            p2 = df[df['P_YearPeriod'] == second_period].copy()

            # Ensure OEE% is numeric
            for d in [p1, p2]:
                if d['OEE%'].dtype == 'object':
                    d['OEE%'] = d['OEE%'].astype(str).str.rstrip('%').astype(float) / 100
                elif d['OEE%'].max() > 1:
                    d['OEE%'] = d['OEE%'] / 100

            merged = pd.merge(
                p1[['Site Name', 'Area', 'OEE%']],
                p2[['Site Name', 'Area', 'OEE%']],
                on=['Site Name', 'Area'],
                suffixes=(f'_{first_period}', f'_{second_period}')
            )
            merged['OEE_Difference'] = merged[f'OEE%_{second_period}'] - merged[f'OEE%_{first_period}']

            # Drivers and draggers
            drivers = merged.nlargest(5, 'OEE_Difference')[
                ['Site Name', 'Area', f'OEE%_{first_period}', f'OEE%_{second_period}', 'OEE_Difference']
            ]
            draggers = merged.nsmallest(5, 'OEE_Difference')[
                ['Site Name', 'Area', f'OEE%_{first_period}', f'OEE%_{second_period}', 'OEE_Difference']
            ]

            # Convert to percentages for readability
            for df_ in [drivers, draggers]:
                df_[f'OEE%_{first_period}'] = (df_[f'OEE%_{first_period}'] * 100).round(1)
                df_[f'OEE%_{second_period}'] = (df_[f'OEE%_{second_period}'] * 100).round(1)
                df_['OEE_Difference'] = (df_['OEE_Difference'] * 100).round(1)

            # Rename columns nicely
            drivers.columns = ['Site', 'Area', f'OEE% {first_period}', f'OEE% {second_period}', 'Change (%)']
            draggers.columns = ['Site', 'Area', f'OEE% {first_period}', f'OEE% {second_period}', 'Change (%)']

            reply += "\n\n**Top Drivers (Improvements):**\n" + drivers.to_markdown(index=False)
            reply += "\n\n**Top Draggers (Declines):**\n" + draggers.to_markdown(index=False)

        # Save assistant reply
        st.session_state.messages.append({"role": "assistant", "content": reply})
        with st.chat_message("assistant"):
            st.markdown(reply)
