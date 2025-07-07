import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
import plotly.express as px
import time
import io
from datetime import datetime
import numpy as np

# --- Dependency Checks ---
def check_library(library_name, feature_name, install_command):
    try:
        __import__(library_name)
        return True
    except ImportError:
        st.warning(
            f"The '{library_name}' library is not installed. {feature_name} will be disabled. "
            f"Install it using: `pip install {install_command}`"
        )
        return False

reportlab_available = check_library("reportlab", "PDF report generation", "reportlab")
shap_available = check_library("shap", "Feature importance visualization (SHAP)", "shap")

# ======================================================================================
# Page Configuration & Professional Styling
# ======================================================================================
st.set_page_config(
    page_title="Strategic Marketing Predictor",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
        .main { background-color: #F0F2F6; font-family: 'Segoe UI', 'Roboto', sans-serif; }
        .sidebar .sidebar-content { background-color: #FFFFFF; }
        h1, h2, h3 { color: #1A202C; font-weight: 600; }
        .stButton > button {
            border: 2px solid #2C5282; background-color: #2C5282; color: white;
            padding: 0.5rem 1rem; border-radius: 0.5rem; font-weight: 600;
            transition: all 0.3s ease-in-out;
        }
        .stButton > button:hover {
            background-color: white; color: #2C5282; transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }
        .card {
            background-color: white; border-radius: 0.75rem; padding: 1.5rem;
            margin-bottom: 1rem; box-shadow: 0 4px 12px rgba(0,0,0,0.08);
            border-left: 6px solid;
        }
        .success-card { border-color: #48BB78; }
        .warning-card { border-color: #ED8936; }
        .deprioritize-card { border-color: #F56565; }
        .strategy-card { border-color: #4299E1; }
        .tweak-card { border-color: #9F7AEA; }
        [data-testid="stTooltip"] { cursor: help; }
    </style>
""", unsafe_allow_html=True)

# ======================================================================================
# Load Model & Define Constants
# ======================================================================================
@st.cache_resource
def load_model():
    """Loads the pre-trained PyCaret pipeline from disk."""
    try:
        model = joblib.load("bank_marketing_model.pkl")
        return model
    except FileNotFoundError:
        st.error("🚨 Critical Error: Model file 'bank_marketing_model.pkl' not found. The application cannot proceed.")
        return None
    except Exception as e:
        st.error(f"An error occurred while loading the model: {e}")
        return None

model = load_model()

# --- Constants for UI and Logic ---
RAW_FEATURE_COLUMNS = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan', 'contact', 'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome']
SUCCESS_THRESHOLD = 0.50
MONTH_OPTIONS = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
POUTCOME_OPTIONS = ["success", "failure", "nonexistent", "other"]

HIGH_POTENTIAL_PROFILES = {
    "Attribute": ["Job", "Education", "Previous Success", "Housing Loan"],
    "Ideal Value": ["Management, Retired, Student", "Tertiary", "Yes", "No"],
    "Rationale": ["Higher disposable income or time", "Higher financial literacy", "Proven interest in products", "Fewer financial obligations"]
}
OPTIMAL_CAMPAIGN_PARAMS = {
    "Parameter": ["Contact Type", "Contact Month", "Call Duration", "Number of Contacts"],
    "Recommended Value": ["Cellular", "Mar, Sep, Oct, Dec", "> 319 seconds", "1-3 contacts"],
    "Reasoning": ["Higher success rate", "Months with highest conversion", "Correlates with interest", "Avoids customer fatigue"]
}

# ======================================================================================
# Core Logic Functions
# ======================================================================================
def create_input_df(input_dict):
    """Creates a Pandas DataFrame and adds the required engineered feature."""
    df = pd.DataFrame([input_dict])
    df['was_contacted_before'] = np.where(df['pdays'] != -1, "Yes", "No")
    return df

def find_minimal_change_for_success(base_input):
    """Iteratively searches for the simplest campaign adjustment."""
    original_prob = model.predict_proba(create_input_df(base_input))[0][1]
    changeable_params = {
        'duration': [d for d in [100, 300, 600, 1000] if d > base_input['duration']],
        'campaign': [c for c in [1, 2, 3] if c < base_input['campaign']],
        'month': ['mar', 'sep', 'oct', 'dec']
    }
    for param, values in changeable_params.items():
        for value in values:
            temp_input = base_input.copy()
            temp_input[param] = value
            prob = model.predict_proba(create_input_df(temp_input))[0][1]
            if prob >= SUCCESS_THRESHOLD: return {param: value}, prob
    return None, original_prob

def perform_sensitivity_analysis(base_input):
    results = []
    for dur in np.linspace(max(0, base_input['duration']-200), base_input['duration']+400, 7):
        temp_input = base_input.copy()
        temp_input['duration'] = int(dur)
        prob = model.predict_proba(create_input_df(temp_input))[0][1]
        results.append({'Parameter': 'Duration (s)', 'Value': int(dur), 'Probability': prob})
    for camp in range(1, 8):
        temp_input = base_input.copy()
        temp_input['campaign'] = camp
        prob = model.predict_proba(create_input_df(temp_input))[0][1]
        results.append({'Parameter': 'Campaign Contacts', 'Value': camp, 'Probability': prob})
    return pd.DataFrame(results)

@st.cache_data
def get_feature_importance(_model_pipeline, input_df):
    if not shap_available: return None
    try:
        explainer = shap.KernelExplainer(_model_pipeline.predict_proba, input_df)
        shap_values = explainer.shap_values(input_df, nsamples=100)
        shap_df = pd.DataFrame({'feature': input_df.columns, 'importance_value': shap_values[1][0]})
        shap_df['color'] = np.where(shap_df['importance_value'] < 0, 'red', 'green')
        shap_df = shap_df.reindex(shap_df.importance_value.abs().sort_values(ascending=False).index).head(10)
        return shap_df.sort_values(by='importance_value', ascending=True)
    except Exception as e:
        st.warning(f"Could not compute SHAP feature importance: {e}")
        return None

# <<< FIX: Added robust validation for uploaded CSV files.
def process_bulk_upload(file):
    """Validates, processes, and predicts on an uploaded CSV file."""
    try:
        df = pd.read_csv(file)

        # Step 1: Validate that all required columns exist.
        missing_cols = [col for col in RAW_FEATURE_COLUMNS if col not in df.columns]
        if missing_cols:
            st.error(f"**Upload Failed:** Your CSV file is missing the following required columns: **{', '.join(missing_cols)}**")
            return None

        # Step 2: Create the required engineered feature.
        df['was_contacted_before'] = np.where(df['pdays'] != -1, "Yes", "No")

        # Step 3: Predict using the model.
        probabilities = model.predict_proba(df)[:, 1]
        
        # Step 4: Append results to the original dataframe.
        output_df = pd.read_csv(file) # Re-read original file to avoid returning the engineered col
        output_df['Subscription Probability'] = probabilities
        output_df['Recommendation'] = np.where(output_df['Subscription Probability'] >= SUCCESS_THRESHOLD, 'Prioritize', 'De-prioritize')
        
        return output_df
    except Exception as e:
        st.error(f"An unexpected error occurred during bulk processing: {e}")
        return None

# ======================================================================================
# UI Component Functions
# ======================================================================================
def create_gauge_chart(probability):
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=probability * 100,
        title={'text': "Subscription Propensity", 'font': {'size': 20, 'color': '#1A202C'}},
        number={'suffix': "%", 'font': {'size': 28}},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "#48BB78" if probability >= SUCCESS_THRESHOLD else "#ED8936"},
            'steps': [
                {'range': [0, SUCCESS_THRESHOLD * 100], 'color': 'rgba(237, 137, 54, 0.2)'},
                {'range': [SUCCESS_THRESHOLD * 100, 100], 'color': 'rgba(72, 187, 120, 0.2)'},
            ],
            'threshold': {'line': {'color': "#F56565", 'width': 4}, 'thickness': 0.8, 'value': SUCCESS_THRESHOLD * 100}
        }))
    fig.update_layout(height=250, margin=dict(l=30, r=30, t=50, b=20), paper_bgcolor="white", font_color="#1A202C")
    return fig

def render_input_form():
    with st.form("client_form"):
        st.write("Enter client and campaign details to predict subscription likelihood.")
        col1, col2 = st.columns(2, gap="large")
        with col1:
            st.subheader("👤 Client Profile")
            age = st.slider("Age", 18, 100, 41)
            job = st.selectbox("Job", ["management", "technician", "entrepreneur", "blue-collar", "unknown", "retired", "admin.", "services", "self-employed", "unemployed", "housemaid", "student"], 0)
            education = st.selectbox("Education", ["tertiary", "secondary", "unknown", "primary"], 0)
            balance = st.number_input("Avg Yearly Balance (€)", -8019, 102127, 1500)
        with col2:
            st.subheader("💳 Financial Status")
            marital = st.selectbox("Marital Status", ["married", "single", "divorced"], 0)
            housing = st.radio("Housing Loan?", ["no", "yes"], 0, horizontal=True)
            loan = st.radio("Personal Loan?", ["no", "yes"], 0, horizontal=True)
            default = st.radio("Credit in Default?", ["no", "yes"], 0, horizontal=True)

        st.markdown("<hr>", unsafe_allow_html=True)
        st.subheader("📞 Campaign Interaction")
        c1, c2, c3 = st.columns(3)
        duration = c1.slider("Last Contact Duration (s)", 0, 900, 260)
        campaign = c2.slider("Contacts This Campaign", 1, 63, 2)
        month = c3.selectbox("Last Contact Month", MONTH_OPTIONS, 4)
        
        # These are fixed for single-client prediction for simplicity
        contact = "cellular"
        day = 15

        st.subheader("🗓️ Previous Campaign History")
        c4, c5, c6 = st.columns(3)
        pdays = c4.number_input("Days Since Previous Contact", -1, 871, -1, help="-1 = never contacted")
        previous = c5.slider("Contacts Before This Campaign", 0, 275, 0)
        poutcome = c6.selectbox("Previous Outcome", POUTCOME_OPTIONS, 1)

        submitted = st.form_submit_button("Analyze Client Potential")

    inputs = {
        "age": age, "job": job, "marital": marital, "education": education,
        "default": default, "balance": balance, "housing": housing, "loan": loan,
        "contact": contact, "day": day, "month": month, "duration": duration,
        "campaign": campaign, "pdays": pdays, "previous": previous, "poutcome": poutcome
    }
    return submitted, inputs

# ======================================================================================
# Main App Structure
# ======================================================================================
st.title("🎯 Strategic Marketing Predictor")
st.markdown("An AI-powered dashboard to maximize term deposit subscriptions by providing actionable client insights and campaign strategies.")

if not model: st.stop()

tab1, tab2, tab3 = st.tabs(["👤 Single Client Analysis", "📈 Strategic Campaign Planner", "📂 Bulk Client Analysis"])

with tab1:
    submitted, inputs = render_input_form()
    if submitted:
        with st.spinner("Analyzing..."):
            input_df = create_input_df(inputs)
            probability = model.predict_proba(input_df)[0][1]

            st.markdown("---")
            st.header("🔍 Client Potential Report")
            col1, col2 = st.columns([1, 1])
            with col1:
                st.plotly_chart(create_gauge_chart(probability), use_container_width=True)
            with col2:
                if probability >= SUCCESS_THRESHOLD:
                    st.markdown('<div class="card success-card"><h3>✅ High-Potential Client</h3><p>Prioritize for immediate follow-up.</p></div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="card warning-card"><h3>⚠️ Low-Potential Client</h3><p>See below for actionable tweaks.</p></div>', unsafe_allow_html=True)
            
            st.markdown("---")
            exp_col1, exp_col2 = st.columns(2, gap="large")
            with exp_col1:
                st.subheader("💡 Key Influencers")
                shap_df = get_feature_importance(model, input_df)
                if shap_df is not None:
                    fig = px.bar(shap_df, x='importance_value', y='feature', orientation='h',
                                 color='color', color_discrete_map={'red':'#F56565', 'green':'#48BB78'},
                                 labels={'importance_value': 'Impact on Probability', 'feature': 'Feature'})
                    fig.update_layout(showlegend=False, yaxis_title=None, paper_bgcolor='white', plot_bgcolor='white')
                    st.plotly_chart(fig, use_container_width=True)

            with exp_col2:
                if probability < SUCCESS_THRESHOLD:
                    st.subheader("🛠️ Actionable Tweaks")
                    minimal_changes, new_prob = find_minimal_change_for_success(inputs)
                    if minimal_changes:
                        st.markdown(f"""
                        <div class="card tweak-card"><h3>🚀 Tweak Found!</h3>
                        <p>An adjustment could increase the chance to <strong>{new_prob:.0%}</strong>.</p>
                        <ul>{''.join([f"<li><strong>{k.title()}:</strong> {v}</li>" for k,v in minimal_changes.items()])}</ul>
                        </div>""", unsafe_allow_html=True)
                    else:
                        st.markdown("""<div class="card deprioritize-card"><h3>❌ De-prioritize</h3>
                        <p>Focus resources on higher-potential leads.</p></div>""", unsafe_allow_html=True)
                else:
                     st.subheader("🏆 Winning Strategy")
                     st.markdown("""<div class="card success-card"><p>The current plan is effective. Proceed to secure the subscription.</p></div>""", unsafe_allow_html=True)
            
            st.markdown("---")
            st.subheader("🔬 Sensitivity Analysis")
            sensitivity_df = perform_sensitivity_analysis(inputs)
            fig = px.line(sensitivity_df, x='Value', y='Probability', color='Parameter',
                          title="Impact of Campaign Changes", markers=True)
            fig.update_layout(paper_bgcolor='white', plot_bgcolor='white')
            st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.header("🚀 Design Your Next Winning Campaign")
    st.markdown("Generate a data-driven plan by identifying target profiles and optimal contact strategies.")
    if st.button("Generate Strategic Plan"):
        with st.spinner("Analyzing..."):
            time.sleep(1.5)
            st.markdown("---")
            st.subheader("📊 Recommended Market Segmentation")
            col1, col2 = st.columns(2, gap="large")
            with col1:
                st.markdown('<div class="card strategy-card"><h3>🎯 Profiles to Target</h3><p>Focus your budget on clients matching these characteristics for the highest ROI.</p></div>', unsafe_allow_html=True)
                st.table(pd.DataFrame(HIGH_POTENTIAL_PROFILES))
            with col2:
                st.subheader("⚙️ Optimal Campaign Setup")
                st.info("When contacting target profiles, use these parameters to maximize success.", icon="💡")
                st.table(pd.DataFrame(OPTIMAL_CAMPAIGN_PARAMS))
            st.markdown("---")
            st.subheader("💰 Quick Cost-Benefit Analysis")
            c1, c2, c3 = st.columns(3)
            cost = c1.number_input("Cost per Contact (€)", 0.0, 100.0, 5.0, 0.1, key="cpc")
            value = c2.number_input("Value per Conversion (€)", 0, 10000, 800, 10, key="vpc")
            clients = c3.number_input("Number of Target Clients", 10, 10000, 500, key="tc")
            rate, contacts = 0.45, 2
            total_cost, total_rev = clients * contacts * cost, clients * rate * value
            roi = ((total_rev - total_cost) / total_cost) * 100 if total_cost > 0 else 0
            st.markdown(f"""<div class="card"><h4>ROI Estimate</h4>
                <ul><li>Est. Total Cost: <strong>€{total_cost:,.2f}</strong></li>
                <li>Est. Total Revenue: <strong>€{total_rev:,.2f}</strong></li></ul>
                <h3 style="text-align:center; color: {'#48BB78' if roi >= 0 else '#F56565'};">Est. ROI: {roi:.2f}%</h3>
            </div>""", unsafe_allow_html=True)

with tab3:
    st.header("📂 Bulk Client Analysis")
    st.markdown("Upload a CSV file to get predictions for your entire client list.")
    st.info(f"Your CSV must contain the following columns: {', '.join(RAW_FEATURE_COLUMNS)}")
    
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"], help="The CSV must contain all the raw feature columns.")
    
    if uploaded_file:
        with st.spinner("Processing..."):
            result_df = process_bulk_upload(uploaded_file)
        if result_df is not None:
            st.subheader("✅ Bulk Analysis Complete")
            st.dataframe(result_df)
            csv_data = result_df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Results", csv_data, f"bulk_analysis_results_{datetime.now().strftime('%Y%m%d')}.csv", "text/csv")