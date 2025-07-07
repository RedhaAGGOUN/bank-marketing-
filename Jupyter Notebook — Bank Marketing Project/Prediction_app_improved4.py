import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import time

# --- Dependency Checks & User Guidance ---
# Check for optional, but powerful, libraries. Guide the user on how to install them if missing.
def check_library(library_name: str, feature_name: str, install_command: str) -> bool:
    """Checks if a library is installed and displays a warning if not."""
    try:
        __import__(library_name)
        return True
    except ImportError:
        st.warning(
            f"The '{library_name}' library is not installed. {feature_name} will be disabled. "
            f"To enable this feature, please install it using: `pip install {install_command}`"
        )
        return False

# Perform checks at the start
shap_available = check_library("shap", "Feature Importance (SHAP) visualizations", "shap")

# ======================================================================================
# Page Configuration & Professional Styling
# ======================================================================================
st.set_page_config(
    page_title="Strategic Marketing Predictor",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.example.com/help',
        'Report a bug': "https://www.example.com/bug",
        'About': "# This is a *Strategic Marketing Predictor* app!"
    }
)

# Custom CSS for a professional and modern look
st.markdown("""
    <style>
        /* General styling */
        .main {
            background-color: #F0F2F6; /* Light grey background */
            font-family: 'Segoe UI', 'Roboto', 'Helvetica Neue', sans-serif;
        }
        /* Sidebar styling */
        .sidebar .sidebar-content {
            background-color: #FFFFFF;
        }
        /* Headers */
        h1, h2, h3 {
            color: #1A202C; /* Darker text for contrast */
            font-weight: 600;
        }
        /* Custom styled buttons */
        .stButton > button {
            border: 2px solid #2C5282; /* Primary blue */
            background-color: #2C5282;
            color: white;
            padding: 0.5rem 1rem;
            border-radius: 0.5rem;
            font-weight: 600;
            transition: all 0.3s ease-in-out;
        }
        .stButton > button:hover {
            background-color: white;
            color: #2C5282;
            transform: translateY(-2px); /* Subtle lift effect */
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }
        /* Info cards for highlighting key information */
        .card {
            background-color: white;
            border-radius: 0.75rem;
            padding: 1.5rem;
            margin-bottom: 1rem;
            box-shadow: 0 4px 12px rgba(0,0,0,0.08);
            border-left: 6px solid;
        }
        .success-card { border-color: #48BB78; } /* Green for success */
        .warning-card { border-color: #ED8936; } /* Orange for warning */
        .deprioritize-card { border-color: #F56565; } /* Red for deprioritization */
        .strategy-card { border-color: #4299E1; } /* Blue for strategy */
        .tweak-card { border-color: #9F7AEA; } /* Purple for tweaks */
        /* Tooltip styling */
        [data-testid="stTooltip"] {
            cursor: help;
        }
    </style>
""", unsafe_allow_html=True)


# ======================================================================================
# Load Model, Constants, & Cached Resources
# ======================================================================================
@st.cache_resource
def load_model_pipeline():
    """
    Loads the pre-trained model pipeline from disk.
    Uses @st.cache_resource for efficient memory management.
    """
    try:
        model = joblib.load("bank_marketing_model.pkl")
        return model
    except FileNotFoundError:
        st.error("🚨 Critical Error: Model file 'bank_marketing_model.pkl' not found. "
               "Please ensure the model file is in the same directory as the app. "
               "The application cannot proceed without the model.", icon="🔥")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred while loading the model: {e}")
        return None

model = load_model_pipeline()

# --- Constants for UI and Logic ---
# Defined here for easy modification and consistency
RAW_FEATURE_COLUMNS = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan', 'contact', 'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome']
SUCCESS_THRESHOLD = 0.50
MONTH_OPTIONS = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
POUTCOME_OPTIONS = ["success", "failure", "nonexistent", "other"]

# Static data for the "Strategic Planner" tab, making it easily updatable
HIGH_POTENTIAL_PROFILES = {
    "Attribute": ["Job", "Education", "Previous Success", "Housing Loan"],
    "Ideal Value": ["Management, Retired, Student", "Tertiary", "Yes", "No"],
    "Rationale": ["Higher disposable income or more available time", "Higher financial literacy", "Proven interest in financial products", "Fewer financial obligations"]
}
OPTIMAL_CAMPAIGN_PARAMS = {
    "Parameter": ["Contact Type", "Contact Month", "Call Duration", "Number of Contacts"],
    "Recommended Value": ["Cellular", "Mar, Sep, Oct, Dec", "> 319 seconds", "1-3 contacts"],
    "Reasoning": ["Historically higher success rate", "Months with highest conversion rates", "Strongly correlates with client interest", "Avoids customer fatigue and negative perception"]
}


# ======================================================================================
# Core Logic Functions
# ======================================================================================
def create_input_df(input_dict: dict) -> pd.DataFrame:
    """
    Creates a Pandas DataFrame from user input and adds the required engineered feature.
    This feature is crucial for the model's performance.
    """
    df = pd.DataFrame([input_dict])
    # Feature Engineering: The model was trained with this feature.
    df['was_contacted_before'] = np.where(df['pdays'] != -1, "Yes", "No")
    return df

@st.cache_data
def find_minimal_change_for_success(base_input: dict) -> tuple:
    """
    Iteratively searches for the simplest campaign adjustment to push a client's
    probability above the success threshold. Caches results for identical inputs.

    Returns:
        A tuple containing (dictionary of changes, new probability) or (None, original probability).
    """
    original_prob = model.predict_proba(create_input_df(base_input))[0][1]

    # Define realistic, actionable changes a marketer could make
    changeable_params = {
        'duration': [d for d in [100, 300, 600, 1000] if d > base_input['duration']],
        'campaign': [c for c in [1, 2, 3] if c < base_input['campaign']],
        'month': ['mar', 'sep', 'oct', 'dec'] # Prime months
    }

    for param, values in changeable_params.items():
        for value in values:
            temp_input = base_input.copy()
            temp_input[param] = value
            prob = model.predict_proba(create_input_df(temp_input))[0][1]
            if prob >= SUCCESS_THRESHOLD:
                return {param: value}, prob

    return None, original_prob

@st.cache_data
def perform_sensitivity_analysis(base_input: dict) -> pd.DataFrame:
    """
    Calculates how the prediction probability changes as key campaign parameters are varied.
    This helps visualize the impact of marketer's actions.
    """
    results = []
    # Analyze Call Duration
    for dur in np.linspace(max(0, base_input['duration'] - 200), base_input['duration'] + 400, 7):
        temp_input = base_input.copy()
        temp_input['duration'] = int(dur)
        prob = model.predict_proba(create_input_df(temp_input))[0][1]
        results.append({'Parameter': 'Duration (s)', 'Value': int(dur), 'Probability': prob})

    # Analyze Number of Contacts
    for camp in range(1, 8):
        temp_input = base_input.copy()
        temp_input['campaign'] = camp
        prob = model.predict_proba(create_input_df(temp_input))[0][1]
        results.append({'Parameter': 'Campaign Contacts', 'Value': camp, 'Probability': prob})

    return pd.DataFrame(results)

@st.cache_data
def get_feature_importance(_model_pipeline, input_df: pd.DataFrame) -> pd.DataFrame | None:
    """
    Calculates SHAP values to explain the model's prediction for a single client.
    Returns a sorted DataFrame of feature importances.
    """
    if not shap_available:
        return None
    try:
        # Using KernelExplainer as it's model-agnostic and works with pipelines
        explainer = shap.KernelExplainer(_model_pipeline.predict_proba, input_df)
        shap_values = explainer.shap_values(input_df, nsamples=100)

        # Format results for visualization
        shap_df = pd.DataFrame({
            'feature': input_df.columns,
            'importance_value': shap_values[1][0] # Probability of "success" class
        })
        shap_df['color'] = np.where(shap_df['importance_value'] < 0, '#F56565', '#48BB78') # Red for negative, Green for positive
        # Sort by absolute importance and take top 10 for clarity
        shap_df = shap_df.reindex(shap_df.importance_value.abs().sort_values(ascending=False).index).head(10)
        return shap_df.sort_values(by='importance_value', ascending=True)
    except Exception as e:
        st.warning(f"Could not compute SHAP feature importance: {e}", icon="⚠️")
        return None

def process_bulk_upload(file) -> pd.DataFrame | None:
    """
    Validates, processes, and predicts on an uploaded CSV file.
    Includes robust validation to prevent errors from malformed files.
    """
    try:
        df = pd.read_csv(file)

        # --- Robust Validation ---
        # 1. Check for missing columns
        missing_cols = [col for col in RAW_FEATURE_COLUMNS if col not in df.columns]
        if missing_cols:
            st.error(f"**Upload Failed:** Your CSV file is missing the following required columns: **{', '.join(missing_cols)}**")
            return None

        # 2. Add the necessary engineered feature
        df['was_contacted_before'] = np.where(df['pdays'] != -1, "Yes", "No")

        # --- Prediction ---
        probabilities = model.predict_proba(df)[:, 1]

        # --- Format Output ---
        # Re-read original file to return a clean output without engineered features
        output_df = pd.read_csv(file)
        output_df['Subscription Probability'] = probabilities
        output_df['Recommendation'] = np.where(output_df['Subscription Probability'] >= SUCCESS_THRESHOLD, 'Prioritize', 'De-prioritize')

        return output_df

    except pd.errors.ParserError:
        st.error("Upload Failed: The uploaded file could not be parsed as a CSV. Please check the file format.")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred during bulk processing: {e}")
        return None


# ======================================================================================
# UI Component Functions
# ======================================================================================
def create_gauge_chart(probability: float) -> go.Figure:
    """Creates a visually appealing gauge chart using Plotly."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        title={'text': "Subscription Propensity", 'font': {'size': 20, 'color': '#1A202C'}},
        number={'suffix': "%", 'font': {'size': 28, 'color': '#1A202C'}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "#48BB78" if probability >= SUCCESS_THRESHOLD else "#ED8936"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "#E2E8F0",
            'steps': [
                {'range': [0, SUCCESS_THRESHOLD * 100], 'color': 'rgba(237, 137, 54, 0.2)'},
                {'range': [SUCCESS_THRESHOLD * 100, 100], 'color': 'rgba(72, 187, 120, 0.2)'},
            ],
            'threshold': {
                'line': {'color': "#F56565", 'width': 4},
                'thickness': 0.8,
                'value': SUCCESS_THRESHOLD * 100
            }
        }))
    fig.update_layout(
        height=250,
        margin=dict(l=30, r=30, t=50, b=20),
        paper_bgcolor="white",
        font_color="#1A202C"
    )
    return fig

def render_input_sidebar() -> tuple[bool, dict]:
    """Renders the input form in the sidebar for a cleaner main view."""
    with st.sidebar:
        st.header("👤 Client & Campaign Details")
        with st.form("client_form"):
            st.subheader("Client Profile")
            age = st.slider("Age", 18, 100, 41, help="Client's age in years.")
            job = st.selectbox("Job", ["management", "technician", "entrepreneur", "blue-collar", "unknown", "retired", "admin.", "services", "self-employed", "unemployed", "housemaid", "student"], index=0)
            education = st.selectbox("Education Level", ["tertiary", "secondary", "unknown", "primary"], index=0)
            balance = st.number_input("Average Yearly Balance (€)", -8019, 102127, 1500, help="Average bank balance over the year.")

            st.subheader("Financial Status")
            marital = st.selectbox("Marital Status", ["married", "single", "divorced"], index=0)
            housing = st.radio("Has Housing Loan?", ["no", "yes"], index=0, horizontal=True)
            loan = st.radio("Has Personal Loan?", ["no", "yes"], index=0, horizontal=True)
            default = st.radio("Has Credit in Default?", ["no", "yes"], index=0, horizontal=True)

            st.subheader("Campaign Interaction")
            duration = st.slider("Last Contact Duration (s)", 0, 900, 260, help="Duration of the last call in seconds.")
            campaign = st.slider("Contacts This Campaign", 1, 63, 2, help="Number of times the client was contacted during this campaign.")
            month = st.selectbox("Last Contact Month", MONTH_OPTIONS, index=4)

            st.subheader("Previous Campaign History")
            pdays = st.number_input("Days Since Previous Contact", -1, 871, -1, help="-1 indicates the client was not previously contacted.")
            previous = st.slider("Contacts Before This Campaign", 0, 275, 0)
            poutcome = st.selectbox("Previous Outcome", POUTCOME_OPTIONS, index=2)

            # These features have low variance or are less controllable, so we fix them for simplicity
            contact = "cellular"
            day = 15

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

if not model:
    st.stop() # Stop execution if the model failed to load

submitted, inputs = render_input_sidebar()

# Create a placeholder for the main content area
main_content = st.container()

if submitted:
    with main_content:
        with st.spinner("🧠 Analyzing client profile and crunching the numbers..."):
            input_df = create_input_df(inputs)
            probability = model.predict_proba(input_df)[0][1]

            st.markdown("---")
            st.header("🔍 Client Potential Report")

            col1, col2 = st.columns([1, 1])
            with col1:
                st.plotly_chart(create_gauge_chart(probability), use_container_width=True)
            with col2:
                if probability >= SUCCESS_THRESHOLD:
                    st.markdown('<div class="card success-card"><h3>✅ High-Potential Client</h3><p>This client has a high likelihood of subscribing. Prioritize for immediate and personalized follow-up.</p></div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="card warning-card"><h3>⚠️ Low-Potential Client</h3><p>This client is unlikely to subscribe with the current approach. See below for actionable tweaks to improve the outcome.</p></div>', unsafe_allow_html=True)

            st.markdown("---")
            st.header("💡 Why this Prediction? (The Key Drivers)")
            exp_col1, exp_col2 = st.columns(2, gap="large")

            with exp_col1:
                st.subheader("Feature Influence")
                if shap_available:
                    shap_df = get_feature_importance(model, input_df)
                    if shap_df is not None:
                        fig = px.bar(shap_df, x='importance_value', y='feature', orientation='h',
                                     color='color', color_discrete_map={'#F56565':'#F56565', '#48BB78':'#48BB78'},
                                     labels={'importance_value': 'Impact on Probability (SHAP Value)', 'feature': 'Feature'},
                                     title="Top Factors Influencing the Prediction")
                        fig.update_layout(showlegend=False, yaxis_title=None, paper_bgcolor='white', plot_bgcolor='white', yaxis={'categoryorder':'total ascending'})
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("Could not generate the feature importance plot.")
                else:
                    st.info("Install the `shap` library to see a breakdown of which features influenced this prediction the most.")

            with exp_col2:
                if probability < SUCCESS_THRESHOLD:
                    st.subheader("🛠️ Actionable Tweaks")
                    minimal_changes, new_prob = find_minimal_change_for_success(inputs)
                    if minimal_changes:
                        st.markdown(f"""
                        <div class="card tweak-card"><h3>🚀 Tweak Found!</h3>
                        <p>A simple adjustment could potentially increase the subscription chance to <strong>{new_prob:.0%}</strong>.</p>
                        <p><strong>Recommendation:</strong></p>
                        <ul>{''.join([f"<li>Change <strong>{k.replace('_', ' ').title()}:</strong> from {inputs[k]} to <strong>{v}</strong></li>" for k,v in minimal_changes.items()])}</ul>
                        </div>""", unsafe_allow_html=True)
                    else:
                        st.markdown("""<div class="card deprioritize-card"><h3>❌ De-prioritize</h3>
                        <p>Significant changes are needed to make this client a viable lead. It's recommended to focus marketing resources on higher-potential clients.</p></div>""", unsafe_allow_html=True)
                else:
                     st.subheader("🏆 Winning Strategy")
                     st.markdown("""<div class="card success-card"><h4>Proceed with Confidence</h4><p>The current campaign parameters and client profile are highly favorable. Focus on closing the deal and providing excellent service.</p></div>""", unsafe_allow_html=True)

            st.markdown("---")
            st.header("🔬 What-If? Sensitivity Analysis")
            st.markdown("See how changing key campaign actions could affect the outcome for this specific client.")
            sensitivity_df = perform_sensitivity_analysis(inputs)
            fig_sens = px.line(sensitivity_df, x='Value', y='Probability', color='Parameter',
                          facet_col='Parameter', facet_col_wrap=2,
                          title="Impact of Campaign Changes on Subscription Probability",
                          markers=True, labels={"Probability": "Success Probability"})
            fig_sens.update_yaxes(matches=None, showticklabels=True) # Unlink y-axes
            fig_sens.update_layout(paper_bgcolor='white', plot_bgcolor='white')
            st.plotly_chart(fig_sens, use_container_width=True)

else:
    # --- Default view when no form is submitted ---
    with main_content:
        st.markdown("---")
        st.header("🚀 Welcome to the Strategic Marketing Predictor!")
        st.markdown("This tool is designed to help you make smarter, data-driven marketing decisions. Here's how to get started:")
        
        cols = st.columns(3)
        with cols[0]:
            st.markdown("""
            <div class="card strategy-card">
            <h4>1. Analyze a Single Client</h4>
            <p>Use the sidebar on the left to enter a client's details. The app will predict their likelihood to subscribe and provide actionable insights to improve your chances.</p>
            </div>
            """, unsafe_allow_html=True)
        with cols[1]:
            st.markdown("""
            <div class="card strategy-card">
            <h4>2. Plan a Campaign</h4>
            <p>Go to the <strong>Strategic Planner</strong> tab to see data-driven recommendations for target client profiles and optimal campaign settings for maximum ROI.</p>
            </div>
            """, unsafe_allow_html=True)
        with cols[2]:
            st.markdown("""
            <div class="card strategy-card">
            <h4>3. Process a Client List</h4>
            <p>Use the <strong>Bulk Analysis</strong> tab to upload a CSV of your clients and get predictions for the entire list in one go.</p>
            </div>
            """, unsafe_allow_html=True)

# These tabs are now separate from the main form logic
tab2, tab3 = st.tabs(["📈 Strategic Campaign Planner", "📂 Bulk Client Analysis"])

with tab2:
    st.header("🚀 Design Your Next Winning Campaign")
    st.markdown("Generate a data-driven plan by identifying high-value target profiles and optimal contact strategies based on historical data.")

    with st.expander("Show Recommended Strategic Plan", expanded=True):
        st.markdown("---")
        st.subheader("📊 Recommended Market Segmentation")
        col1, col2 = st.columns(2, gap="large")
        with col1:
            st.markdown('<div class="card strategy-card"><h3>🎯 Profiles to Target</h3><p>Focus your budget on clients matching these characteristics for the highest Return on Investment (ROI).</p></div>', unsafe_allow_html=True)
            st.table(pd.DataFrame(HIGH_POTENTIAL_PROFILES))
        with col2:
            st.markdown('<div class="card strategy-card"><h3>⚙️ Optimal Campaign Setup</h3><p>When contacting target profiles, use these parameters to maximize your success rate.</p></div>', unsafe_allow_html=True)
            st.table(pd.DataFrame(OPTIMAL_CAMPAIGN_PARAMS))

    st.markdown("---")
    st.subheader("💰 Quick Cost-Benefit & ROI Calculator")
    st.markdown("Estimate the potential return on investment for a campaign targeting high-potential clients.")
    c1, c2, c3 = st.columns(3)
    cost = c1.number_input("Avg. Cost per Contact (€)", 0.0, 100.0, 5.0, 0.1, key="cpc", help="The total cost associated with making one contact with a client.")
    value = c2.number_input("Avg. Value per Conversion (€)", 0, 10000, 800, 10, key="vpc", help="The average revenue or profit generated from one successful subscription.")
    clients = c3.number_input("Number of Target Clients", 10, 10000, 500, key="tc", help="The number of high-potential clients you plan to contact.")

    # Using assumptions from our optimal campaign data
    assumed_conversion_rate, avg_contacts_needed = 0.45, 2
    total_cost = clients * avg_contacts_needed * cost
    total_conversions = clients * assumed_conversion_rate
    total_revenue = total_conversions * value
    roi = ((total_revenue - total_cost) / total_cost) * 100 if total_cost > 0 else 0

    st.markdown(f"""
    <div class="card">
        <h4>ROI Estimate</h4>
        <p><em>Based on contacting <strong>{clients}</strong> clients with an assumed conversion rate of <strong>{assumed_conversion_rate:.0%}</strong> after <strong>{avg_contacts_needed}</strong> contacts.</em></p>
        <ul>
            <li>Estimated Total Campaign Cost: <strong>€{total_cost:,.2f}</strong></li>
            <li>Estimated Successful Conversions: <strong>{int(total_conversions)}</strong></li>
            <li>Estimated Total Revenue: <strong>€{total_revenue:,.2f}</strong></li>
        </ul>
        <h3 style="text-align:center; color: {'#48BB78' if roi >= 0 else '#F56565'};">Estimated ROI: {roi:.2f}%</h3>
    </div>
    """, unsafe_allow_html=True)

with tab3:
    st.header("📂 Bulk Client Analysis")
    st.markdown("Upload a CSV file containing client data to get subscription probability predictions for your entire list.")
    st.info(f"**Important:** Your CSV file must contain the following columns in any order: `{', '.join(RAW_FEATURE_COLUMNS)}`", icon="📋")

    uploaded_file = st.file_uploader("Upload Your Client List (CSV)", type=["csv"], help="The CSV must contain all the raw feature columns for accurate prediction.")

    if uploaded_file:
        with st.spinner("Processing file... This may take a moment for large files."):
            result_df = process_bulk_upload(uploaded_file)

        if result_df is not None:
            st.subheader("✅ Bulk Analysis Complete")
            st.markdown("Below are the prediction results for your uploaded list.")
            st.dataframe(result_df)

            # Prepare data for download
            csv_data = result_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Results as CSV",
                data=csv_data,
                file_name=f"bulk_analysis_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
            )