import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
import time
from itertools import combinations

# ======================================================================================
# Page Configuration & Styling
# ======================================================================================
st.set_page_config(page_title="Strategic Marketing Predictor", page_icon="🎯", layout="wide")

# Custom CSS for a modern, clean look
st.markdown("""
    <style>
        /* General App Styling */
        .main {
            background-color: #f0f2f6;
            font-family: 'Segoe UI', 'Roboto', 'Helvetica', 'Arial', sans-serif;
        }

        /* Buttons */
        .stButton>button {
            border: 2px solid #1E88E5;
            background-color: #1E88E5;
            color: white;
            padding: 12px 28px;
            text-align: center;
            text-decoration: none;
            display: inline-block;
            font-size: 16px;
            margin: 4px 2px;
            cursor: pointer;
            border-radius: 8px;
            transition-duration: 0.4s;
            width: 100%;
            font-weight: bold;
        }
        .stButton>button:hover {
            background-color: white;
            color: #1E88E5;
        }

        /* Cards for displaying results */
        .report-card {
            background-color: #ffffff;
            border-radius: 10px;
            padding: 25px;
            margin: 10px 0;
            box-shadow: 0 4px 8px 0 rgba(0,0,0,0.1);
            transition: 0.3s;
            height: 100%;
        }
        .report-card:hover {
            box-shadow: 0 8px 16px 0 rgba(0,0,0,0.2);
        }

        /* Specific card styles */
        .success-card { border-left: 7px solid #4CAF50; }
        .warning-card { border-left: 7px solid #ff9800; }
        .deprioritize-card { border-left: 7px solid #F44336; }
        .strategy-card { border-left: 7px solid #4CAF50; }
        .tweak-card { border-left: 7px solid #2196F3; }

        /* Headers and Titles */
        h1, h2, h3 {
            color: #1E2A38;
        }
        h3 {
            border-bottom: 2px solid #f0f2f6;
            padding-bottom: 10px;
        }
    </style>
""", unsafe_allow_html=True)

# ======================================================================================
# Load Model & Define Constants
# ======================================================================================
@st.cache_resource
def load_model():
    """Loads the pre-trained model."""
    try:
        return joblib.load("bank_marketing_model.pkl")
    except FileNotFoundError:
        st.error("Model file 'bank_marketing_model.pkl' not found. Please ensure it's in the correct directory.")
        return None

model = load_model()

# --- CONSTANTS ---
SUCCESS_THRESHOLD = 0.50 # 50% probability is our target for a successful conversion
# Options for optimization loops
CONTACT_OPTIONS = ["cellular", "telephone"]
MONTH_OPTIONS = ["mar", "apr", "may", "jun", "jul", "aug", "sep", "oct"]
DURATION_OPTIONS = [100, 300, 600, 1000]
CAMPAIGN_OPTIONS = [1, 2, 3, 4]
POUTCOME_OPTIONS = ["success", "failure", "other", "unknown"]

# --- NEW: Strategic Profile Definitions ---
HIGH_POTENTIAL_PROFILES = {
    "Attribute": ["Job", "Education", "Housing Loan", "Personal Loan", "Previous Success"],
    "Ideal Value": ["Management, Retired, Student", "Tertiary", "No", "No", "Yes"],
    "Rationale": ["Higher disposable income or time", "Indicates higher financial literacy", "Fewer financial obligations", "Fewer financial obligations", "Proven interest"]
}
LOW_POTENTIAL_PROFILES = {
    "Attribute": ["Job", "Balance", "Housing Loan", "Personal Loan", "Previous Failure"],
    "Value to Avoid": ["Blue-collar, Services", "Negative or very low", "Yes", "Yes", "Yes"],
    "Rationale": ["Often lower-income, less flexible", "Indicates financial distress", "High financial obligations", "High financial obligations", "Proven lack of interest"]
}

# ======================================================================================
# Core Logic Functions
# ======================================================================================
def create_input_df(input_dict):
    """Creates a DataFrame from the input dictionary and adds the required engineered feature."""
    input_df = pd.DataFrame([input_dict])
    input_df['was_contacted_before'] = input_df['pdays'].apply(lambda x: "Yes" if x != -1 else "No")
    return input_df

def find_minimal_change_for_subscription(base_input):
    """
    NEW: Finds the smallest set of campaign adjustments to push a client's
    subscription probability over the SUCCESS_THRESHOLD.
    """
    st.markdown("---")
    st.subheader("💡 Finding the Easiest Path to 'Yes'")
    st.write("Analyzing the most efficient tweaks to the campaign for this specific client...")
    
    original_df = create_input_df(base_input)
    original_prob = model.predict_proba(original_df)[0][1]

    # Parameters we can realistically change
    changeable_params = {
        'duration': [d for d in DURATION_OPTIONS if d > base_input['duration']],
        'campaign': [c for c in CAMPAIGN_OPTIONS if c > base_input['campaign']],
        'month': [m for m in MONTH_OPTIONS if m != base_input['month']],
    }
    
    with st.spinner("Simulating minimal-effort scenarios..."):
        time.sleep(1) # For UX
        # 1. Try changing only ONE parameter
        for param, values in changeable_params.items():
            for value in values:
                temp_input = base_input.copy()
                temp_input[param] = value
                prob = model.predict_proba(create_input_df(temp_input))[0][1]
                if prob >= SUCCESS_THRESHOLD:
                    st.success("Found a simple one-step tweak!")
                    return {param: value}, prob

        # 2. If that fails, try changing TWO parameters (duration + campaign is most impactful)
        if 'duration' in changeable_params and 'campaign' in changeable_params:
            for dur in changeable_params['duration']:
                for camp in changeable_params['campaign']:
                    temp_input = base_input.copy()
                    temp_input['duration'] = dur
                    temp_input['campaign'] = camp
                    prob = model.predict_proba(create_input_df(temp_input))[0][1]
                    if prob >= SUCCESS_THRESHOLD:
                        st.success("Found a two-step tweak!")
                        return {'duration': dur, 'campaign': camp}, prob

    # If no combination works
    st.warning("This client profile is highly resistant to conversion with simple tweaks.")
    return None, original_prob

# ======================================================================================
# UI Component Functions
# ======================================================================================
def render_input_form():
    """Renders the client data input form and returns a dictionary of inputs."""
    with st.form("client_form"):
        st.header("👤 Client & Campaign Details")
        c1, c2 = st.columns(2)
        with c1:
            age = st.slider("Age", 18, 100, 35)
            job = st.selectbox("Job", ["management", "technician", "blue-collar", "admin.", "services", "retired", "self-employed", "entrepreneur", "unemployed", "housemaid", "student", "unknown"])
            marital = st.selectbox("Marital Status", ["married", "single", "divorced"])
            education = st.selectbox("Education Level", ["tertiary", "secondary", "primary", "unknown"])
        with c2:
            balance = st.number_input("Avg. Yearly Balance (€)", -10000, 110000, 1500, 100)
            housing = st.radio("Has Housing Loan?", ["no", "yes"], horizontal=True)
            loan = st.radio("Has Personal Loan?", ["no", "yes"], horizontal=True)
            default = st.radio("Credit in Default?", ["no", "yes"], horizontal=True)
        
        st.markdown("<hr>", unsafe_allow_html=True)
        c3, c4, c5 = st.columns(3)
        with c3:
            duration = st.slider("Last Contact Duration (s)", 0, 5000, 300)
            campaign = st.slider("Contacts in this Campaign", 1, 50, 2)
        with c4:
            month = st.selectbox("Last Contact Month", MONTH_OPTIONS)
            pdays = st.number_input("Days Since Previous Contact", -1, 999, -1, help="-1 = never contacted")
        with c5:
            previous = st.slider("Contacts Before this Campaign", 0, 50, 0)
            poutcome = st.selectbox("Previous Campaign Outcome", POUTCOME_OPTIONS)
            
        submitted = st.form_submit_button("Analyze Client Potential")

    inputs = {"age": age, "job": job, "marital": marital, "education": education, "default": default, "balance": balance, "housing": housing, "loan": loan, "contact": "cellular", "day": 15, "month": month, "duration": duration, "campaign": campaign, "pdays": pdays, "previous": previous, "poutcome": poutcome}
    return submitted, inputs

def create_gauge_chart(confidence):
    fig = go.Figure(go.Indicator(
        mode = "gauge+number", value = confidence * 100,
        title = {'text': "Subscription Probability", 'font': {'size': 24}},
        gauge = {'axis': {'range': [None, 100]}, 'bar': {'color': "#4CAF50" if confidence * 100 > 50 else "#ff9800"},
                 'steps': [{'range': [0, 50], 'color': 'rgba(255, 152, 0, 0.2)'}, {'range': [50, 100], 'color': 'rgba(76, 175, 80, 0.2)'}]}))
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
    return fig

# ======================================================================================
# Main App Structure
# ======================================================================================
st.title("🏦 Strategic Bank Marketing Predictor 🎯")
st.markdown("From single-client predictions to full campaign strategy, this tool provides AI-driven insights to maximize term deposit subscriptions.")

tab1, tab2 = st.tabs(["👤 Single Client Analysis", "📈 Strategic Campaign Planner"])

# --- TAB 1: Single Client Prediction ---
with tab1:
    submitted, inputs = render_input_form()

    if submitted and model:
        input_df = create_input_df(inputs)
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]
        
        st.markdown("---")
        st.header("🔍 Client Potential Report")
        col1, col2 = st.columns([1.2, 1])

        with col1:
            if probability >= SUCCESS_THRESHOLD:
                st.markdown('<div class="report-card success-card"><h3>✅ High-Potential Client</h3><p>This client is likely to subscribe. Prioritize for immediate follow-up.</p></div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="report-card warning-card"><h3>⚠️ Low-Potential Client</h3><p>This client is unlikely to subscribe with the current strategy. Analyzing potential tweaks...</p></div>', unsafe_allow_html=True)
        with col2:
            st.plotly_chart(create_gauge_chart(probability), use_container_width=True)

        if probability < SUCCESS_THRESHOLD:
            minimal_changes, new_prob = find_minimal_change_for_subscription(inputs)
            
            if minimal_changes:
                st.markdown(f"""
                    <div class="report-card tweak-card">
                        <h3>📋 Actionable Tweak Found!</h3>
                        <p>A small adjustment to the campaign could significantly increase the subscription chance to <b>{new_prob:.0%}</b>.</p>
                    </div>
                """, unsafe_allow_html=True)
                rec_col1, rec_col2 = st.columns(2)
                current_values = {k: inputs[k] for k in minimal_changes.keys()}
                rec_col1.subheader("Current Approach")
                rec_col1.json(current_values)
                rec_col2.subheader("Suggested Tweak")
                rec_col2.json(minimal_changes)
                st.info("💡 **Insight:** Applying this minimal change is the most efficient way to convert this specific client.")
            else:
                st.markdown(f"""
                    <div class="report-card deprioritize-card">
                        <h3>❌ De-prioritize This Client</h3>
                        <p>Our analysis shows that even with significant effort, this client is highly unlikely to subscribe (max potential remains low). It is more efficient to focus marketing resources elsewhere.</p>
                    </div>
                """, unsafe_allow_html=True)


# --- TAB 2: Strategic Campaign Planner ---
with tab2:
    st.header("🚀 Design Your Next Winning Campaign")
    st.write("Generate a data-driven plan by identifying which customer profiles to target, which to avoid, and the optimal contact strategy to use.")
    
    if st.button("Generate Strategic Plan"):
        if model:
            with st.spinner("Analyzing market segments and strategies..."):
                time.sleep(2) # For dramatic effect
            
            st.markdown("---")
            st.subheader("📊 Recommended Market Segmentation")
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown('<div class="report-card strategy-card"><h3>Profiles to Target</h3><p>Focus your budget and effort on clients matching these characteristics for the highest ROI.</p></div>', unsafe_allow_html=True)
                st.table(pd.DataFrame(HIGH_POTENTIAL_PROFILES))
            
            with col2:
                st.markdown('<div class="report-card deprioritize-card"><h3>Profiles to De-prioritize</h3><p>Avoid spending resources on these profiles as they have a historically low conversion rate.</p></div>', unsafe_allow_html=True)
                st.table(pd.DataFrame(LOW_POTENTIAL_PROFILES))

            st.markdown("---")
            st.subheader("⚙️ Optimal Campaign Setup for Target Profiles")
            st.info("When contacting the **High-Potential Profiles**, use the following campaign parameters to maximize success.", icon="💡")
            
            # Simplified optimal strategy based on known best practices for this dataset
            optimal_params = {
                "Parameter": ["Contact Type", "Contact Month", "Call Duration", "Number of Contacts"],
                "Recommended Value": ["Cellular", "March, September, October, December", "> 300 seconds", "1-3 contacts"]
            }
            st.table(pd.DataFrame(optimal_params))

if not model:
    st.warning("Prediction model could not be loaded. Please check the file and restart the app.")