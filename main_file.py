import streamlit as st
import pandas as pd
import re
from openai import OpenAI
from typing import Dict, List, Tuple, Optional
import json

st.set_page_config(page_title="OEE and Net Sales Analysis", layout="wide")
st.title("💬 Analysis Chatbot")

# --- Configuration ---
# Add your OpenAI API key here or use Streamlit secrets
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", "")
client = None
if OPENAI_API_KEY:
    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
    except Exception as e:
        st.error(f"OpenAI initialization error: {e}")
        client = None


# --- Load OEE data ---
@st.cache_data
def load_oee_data():
    df = pd.read_excel("Data/OEE.xlsm", engine="openpyxl")
    df.columns = df.columns.str.strip()
    df['P_YearPeriod'] = df['P_YearPeriod'].astype(str).str.strip()
    return df


# --- Load aggregated OEE data ---
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


# --- Load Net Sales data ---
@st.cache_data
def load_net_sales_data():
    try:
        df = pd.read_excel("Data/Net_Sales.xlsm", engine="openpyxl")
        df.columns = df.columns.str.strip()
        df['P_YearPeriod'] = df['P_YearPeriod'].astype(str).str.strip()
        return df
    except Exception as e:
        st.error(f"Error reading Net Sales file: {e}")
        return None


# --- Load aggregated Net Sales data ---
@st.cache_data
def load_aggregated_net_sales():
    try:
        df = pd.read_excel("Data/Aggregated_Net_Sales.xlsx", engine="openpyxl")
        df.columns = df.columns.str.strip()
        df['P_YearPeriod'] = df['P_YearPeriod'].astype(str).str.strip()
        return df
    except Exception as e:
        st.error(f"Error reading aggregated Net Sales file: {e}")
        return None


# --- LLM Query Analysis ---
def analyze_query_with_llm(user_query: str, available_periods: List[str]) -> Dict:
    """
    Analyze user query using LLM to understand intent and extract parameters
    """
    if not client:
        # Fallback to simple analysis if no client available
        return analyze_query_simple(user_query, available_periods)

    prompt = f"""
    Analyze the following business analytics query and extract the key information:

    User Query: "{user_query}"
    Available Periods: {available_periods}

    Please return a JSON response with the following structure:
    {{
        "analysis_type": "oee" or "net_sales",
        "visualization_type": "comparison" or "chart" or "trend",
        "periods": ["period1", "period2"] or ["all"] for charts,
        "focus": "drivers" or "draggers" or "both" or "summary",
        "keywords": ["list", "of", "relevant", "keywords"],
        "intent_summary": "Brief summary of what user wants"
    }}

    Guidelines:
    - If query mentions "net sales", "sales", "revenue" → analysis_type: "net_sales"
    - If query mentions "oee", "equipment effectiveness" or neither specified → analysis_type: "oee"
    - If query asks for "chart", "plot", "trends", "graph" → visualization_type: "chart"
    - If query asks about specific periods comparison → visualization_type: "comparison"
    - If query asks for "drivers", "improvements", "increases", "better", "positive" → focus: "drivers"
    - If query asks for "draggers", "declines", "decreases", "worse", "negative" → focus: "draggers"
    - If query asks for "both" or general comparison → focus: "both"
    - Extract periods from available_periods that match the query

    Return only valid JSON, no other text.
    """

    try:
        response = client.chat.completions.create(
            model=st.secrets.get("OPENAI_MODEL", "gpt-4o"),
            messages=[{"role": "user", "content": prompt}],
            temperature=float(st.secrets.get("OPENAI_TEMPERATURE", 0.1)),
            max_tokens=int(st.secrets.get("OPENAI_MAX_TOKENS", 500))
        )

        # Parse the JSON response
        content = response.choices[0].message.content.strip()

        # Try to extract JSON if it's wrapped in markdown code blocks
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].strip()

        result = json.loads(content)
        return result

    except json.JSONDecodeError as e:
        st.warning(f"Could not parse LLM response as JSON: {e}. Using fallback analysis.")
        return analyze_query_simple(user_query, available_periods)
    except Exception as e:
        st.warning(f"LLM Analysis Error: {e}. Using fallback analysis.")
        return analyze_query_simple(user_query, available_periods)


def analyze_query_simple(user_query: str, available_periods: List[str]) -> Dict:
    """
    Simple fallback analysis without LLM
    """
    query_lower = user_query.lower()

    # Determine analysis type
    is_net_sales = any(term in query_lower for term in ["net sales", "sales", "revenue"])
    analysis_type = "net_sales" if is_net_sales else "oee"

    # Determine visualization type
    if any(word in query_lower for word in ["chart", "plot", "trend", "graph"]):
        visualization_type = "chart"
    else:
        visualization_type = "comparison"

    # Determine focus
    if any(word in query_lower for word in ["driver", "improvement", "increase", "better", "positive"]):
        focus = "drivers"
    elif any(word in query_lower for word in ["dragger", "decline", "decrease", "worse", "negative"]):
        focus = "draggers"
    else:
        focus = "both"

    # Extract periods
    periods = extract_periods(user_query, available_periods)

    return {
        "analysis_type": analysis_type,
        "visualization_type": visualization_type,
        "periods": periods if periods else [],
        "focus": focus,
        "keywords": [],
        "intent_summary": f"User wants {focus} for {analysis_type} analysis"
    }


def generate_smart_response(query_analysis: Dict, drivers_data: pd.DataFrame,
                            draggers_data: pd.DataFrame, periods: List[str]) -> str:
    """
    Generate contextual response based on query analysis and data
    """
    if not client:
        return generate_simple_response(query_analysis, drivers_data, draggers_data, periods)

    # Prepare data summary for LLM
    data_summary = {
        "periods_compared": periods,
        "analysis_type": query_analysis["analysis_type"],
        "top_drivers": drivers_data.head(3).to_dict('records') if not drivers_data.empty else [],
        "top_draggers": draggers_data.head(3).to_dict('records') if not draggers_data.empty else [],
        "user_focus": query_analysis["focus"]
    }

    prompt = f"""
    Based on the business analytics query and data analysis results, generate a professional response.

    Query Analysis: {query_analysis}
    Data Summary: {json.dumps(data_summary, indent=2)}

    Instructions:
    1. Start with a brief explanation of what was analyzed. Do not start with thank you instead give Based on analysis of the query.
    2. Focus on what the user specifically asked for:
       - If focus is "drivers": Only discuss improvements/positive changes
       - If focus is "draggers": Only discuss declines/negative changes  
       - If focus is "both": Discuss both but clearly separate them
    3. Provide business insights and context
    4. Keep response professional and concise
    5. End with what data tables will be shown below

    Generate a response that directly answers the user's question without showing data they didn't ask for.
    """

    try:
        response = client.chat.completions.create(
            model=st.secrets.get("OPENAI_MODEL", "gpt-3.5-turbo"),
            messages=[{"role": "user", "content": prompt}],
            temperature=float(st.secrets.get("OPENAI_TEMPERATURE", 0.3)),
            max_tokens=int(st.secrets.get("OPENAI_MAX_TOKENS", 400))
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        st.warning(f"Response Generation Error: {e}. Using simple response.")
        return generate_simple_response(query_analysis, drivers_data, draggers_data, periods)


def generate_simple_response(query_analysis: Dict, drivers_data: pd.DataFrame,
                             draggers_data: pd.DataFrame, periods: List[str]) -> str:
    """
    Simple response generation without LLM
    """
    analysis_type = query_analysis["analysis_type"].replace("_", " ").title()
    focus = query_analysis["focus"]

    if len(periods) >= 2:
        period_text = f"**{periods[0]}** vs **{periods[1]}**"
    else:
        period_text = "the specified periods"

    base_response = f"✅ Analyzing {analysis_type} for {period_text}..."

    if focus == "drivers":
        base_response += f"\n\nFocusing on **improvements** in {analysis_type} performance:"
    elif focus == "draggers":
        base_response += f"\n\nFocusing on **declines** in {analysis_type} performance:"
    else:
        base_response += f"\n\nAnalyzing both improvements and declines in {analysis_type} performance:"

    return base_response


# --- Simple function to extract two periods from user text ---
def extract_periods(query, available_periods):
    found = []
    for p in available_periods:
        if str(p).lower() in query.lower():
            found.append(p)
    return found[:2]  # take first two matches


def display_data_table(data: pd.DataFrame, title: str):
    """Display data table with proper formatting"""
    if not data.empty:
        st.markdown(f"**{title}:**")
        st.dataframe(data)
        st.markdown("")  # Add spacing


# --- Main Chatbot Interface ---
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Display previous chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User input
if user_input := st.chat_input(
        "Ask me about OEE or Net Sales periods (e.g., Show me only draggers for P1 2024 vs P2 2024):"):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Determine available periods based on data type
    oee_df = load_oee_data()
    net_sales_df = load_net_sales_data()

    # Initial analysis to determine data type
    initial_analysis = analyze_query_simple(user_input, [])

    if initial_analysis["analysis_type"] == "net_sales" and net_sales_df is not None:
        available_periods = net_sales_df["P_YearPeriod"].unique().tolist()
    else:
        available_periods = oee_df["P_YearPeriod"].unique().tolist()

    # Perform detailed LLM analysis
    query_analysis = analyze_query_with_llm(user_input, available_periods)

    # Handle chart requests
    if query_analysis["visualization_type"] == "chart":
        if query_analysis["analysis_type"] == "net_sales":
            aggregated_df = load_aggregated_net_sales()
            if aggregated_df is not None:
                reply = "📊 Here's the Net Sales trend over time:"
                st.session_state.messages.append({"role": "assistant", "content": reply})
                with st.chat_message("assistant"):
                    st.markdown(reply)
                    chart_data = aggregated_df[['P_YearPeriod', 'SUM of Net Sales Actual']].copy()
                    st.bar_chart(data=chart_data, x='P_YearPeriod', y='SUM of Net Sales Actual')
            else:
                reply = "⚠️ Aggregated Net Sales data file not found."
                st.session_state.messages.append({"role": "assistant", "content": reply})
                with st.chat_message("assistant"):
                    st.markdown(reply)
        else:  # OEE
            aggregated_df = load_aggregated_oee()
            if aggregated_df is not None:
                reply = "📊 Here's the OEE trend over time:"
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

    # Handle comparison requests
    else:
        if query_analysis["analysis_type"] == "net_sales":
            df = load_net_sales_data()
            if df is None:
                reply = "⚠️ Net Sales data file not found."
                st.session_state.messages.append({"role": "assistant", "content": reply})
                with st.chat_message("assistant"):
                    st.markdown(reply)
            else:
                selected_periods = query_analysis["periods"]

                if len(selected_periods) < 2:
                    reply = "❌ Couldn't identify two valid periods. Please try like 'Show me draggers for P1 2023 vs P2 2023 for Net Sales'."
                    st.session_state.messages.append({"role": "assistant", "content": reply})
                    with st.chat_message("assistant"):
                        st.markdown(reply)
                else:
                    first_period, second_period = selected_periods

                    # Filter data for the two periods
                    p1 = df[df['P_YearPeriod'] == first_period].copy()
                    p2 = df[df['P_YearPeriod'] == second_period].copy()

                    # Merge on Cluster
                    merged = pd.merge(
                        p1[['FP&A Cluster', 'Net Sales Actual']],
                        p2[['FP&A Cluster', 'Net Sales Actual']],
                        on=['FP&A Cluster'],
                        suffixes=(f'_{first_period}', f'_{second_period}')
                    )
                    merged['Net_Sales_Difference'] = merged[f'Net Sales Actual_{second_period}'] - merged[
                        f'Net Sales Actual_{first_period}']

                    # Get drivers and draggers
                    drivers = merged.nlargest(5, 'Net_Sales_Difference')[
                        ['FP&A Cluster', f'Net Sales Actual_{first_period}', f'Net Sales Actual_{second_period}',
                         'Net_Sales_Difference']
                    ].copy()
                    draggers = merged.nsmallest(5, 'Net_Sales_Difference')[
                        ['FP&A Cluster', f'Net Sales Actual_{first_period}', f'Net Sales Actual_{second_period}',
                         'Net_Sales_Difference']
                    ].copy()

                    # Format numbers with commas
                    for df_temp in [drivers, draggers]:
                        for col in df_temp.columns[1:]:
                            df_temp[col] = df_temp[col].apply(lambda x: f"{x:,.0f}")

                    # Rename columns nicely
                    drivers.columns = ['Cluster', f'Net Sales {first_period}', f'Net Sales {second_period}', 'Change']
                    draggers.columns = ['Cluster', f'Net Sales {first_period}', f'Net Sales {second_period}', 'Change']

                    # Generate smart response
                    reply = generate_smart_response(query_analysis, drivers, draggers, selected_periods)

                    # Save and display response
                    st.session_state.messages.append({"role": "assistant", "content": reply})
                    with st.chat_message("assistant"):
                        st.markdown(reply)

                        # Display only requested data
                        if query_analysis["focus"] == "drivers":
                            display_data_table(drivers, "Top Drivers (Improvements)")
                        elif query_analysis["focus"] == "draggers":
                            display_data_table(draggers, "Top Draggers (Declines)")
                        else:  # both
                            display_data_table(drivers, "Top Drivers (Improvements)")
                            display_data_table(draggers, "Top Draggers (Declines)")

        else:  # OEE analysis
            df = load_oee_data()
            selected_periods = query_analysis["periods"]

            if len(selected_periods) < 2:
                reply = "❌ Couldn't identify two valid periods. Please try like 'Show me only drivers for P1 2023 vs P2 2023'."
                st.session_state.messages.append({"role": "assistant", "content": reply})
                with st.chat_message("assistant"):
                    st.markdown(reply)
            else:
                first_period, second_period = selected_periods

                # Filter data for the two periods
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

                # Get drivers and draggers
                drivers = merged.nlargest(5, 'OEE_Difference')[
                    ['Site Name', 'Area', f'OEE%_{first_period}', f'OEE%_{second_period}', 'OEE_Difference']
                ].copy()
                draggers = merged.nsmallest(5, 'OEE_Difference')[
                    ['Site Name', 'Area', f'OEE%_{first_period}', f'OEE%_{second_period}', 'OEE_Difference']
                ].copy()

                # Convert to percentages for readability
                for df_ in [drivers, draggers]:
                    df_[f'OEE%_{first_period}'] = (df_[f'OEE%_{first_period}'] * 100).round(1)
                    df_[f'OEE%_{second_period}'] = (df_[f'OEE%_{second_period}'] * 100).round(1)
                    df_['OEE_Difference'] = (df_['OEE_Difference'] * 100).round(1)

                # Rename columns nicely
                drivers.columns = ['Site', 'Area', f'OEE% {first_period}', f'OEE% {second_period}', 'Change (%)']
                draggers.columns = ['Site', 'Area', f'OEE% {first_period}', f'OEE% {second_period}', 'Change (%)']

                # Generate smart response
                reply = generate_smart_response(query_analysis, drivers, draggers, selected_periods)

                # Save and display response
                st.session_state.messages.append({"role": "assistant", "content": reply})
                with st.chat_message("assistant"):
                    st.markdown(reply)

                    # Display only requested data
                    if query_analysis["focus"] == "drivers":
                        display_data_table(drivers, "Top Drivers (Improvements)")
                    elif query_analysis["focus"] == "draggers":
                        display_data_table(draggers, "Top Draggers (Declines)")
                    else:  # both
                        display_data_table(drivers, "Top Drivers (Improvements)")
                        display_data_table(draggers, "Top Draggers (Declines)")

# --- Sidebar for Configuration ---
st.sidebar.title("Configuration")
st.sidebar.info("💡 **Tips:**\n"
                "- Ask for 'drivers' to see only improvements\n"
                "- Ask for 'draggers' to see only declines\n"
                "- Use specific periods like 'P1 2024 vs P2 2024'\n"
                "- Specify 'Net Sales' or 'OEE' for clarity")

if not client:
    st.sidebar.warning("⚠️ No OpenAI API key found. Using simple analysis mode.")
    st.sidebar.info("Add your OpenAI API key to Streamlit secrets for enhanced LLM analysis.")
else:
    st.sidebar.success("✅ LLM integration active")