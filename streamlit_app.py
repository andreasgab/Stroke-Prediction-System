import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
import plotly.express as px
import shap
import matplotlib.pyplot as plt

# PAGE CONFIG
st.set_page_config(page_title="Stroke Risk Predictor", layout="wide")

# THEME-AWARE CSS
st.markdown("""
    <style>
    [data-testid="metric-container"] {
        background-color: rgba(125, 125, 125, 0.1);
        border: 1px solid rgba(125, 125, 125, 0.2);
        padding: 15px;
        border-radius: 12px;
    }
    </style>
    """, unsafe_allow_html=True)

# MODEL LOADING
@st.cache_resource
def load_model():
    return joblib.load('model/adaboost_stroke_prediction_model_f2.pkl')

try:
    pipe = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# SIDEBAR NAVIGATION
with st.sidebar:
    st.title("Settings & Navigation")
    
    # Navigation Radio
    page_selection = st.radio(
        "Select Mode:",
        ["Individual Patient Entry", "Batch Results & Analytics"],
        index=0  # Default selection
    )
    st.divider()
    st.header("Upload Patients' Data CSV")
    uploaded_file = st.sidebar.file_uploader("And Change Mode to 'Batch Results & Analytics'", type="csv")
    st.info("CSV should include: gender, age, hypertension, heart_disease, ever_married, work_type, Residence_type, avg_glucose_level, bmi, smoking_status")

# MAIN CONTENT LOGIC
if page_selection == "Individual Patient Entry":
    st.title("Individual Patient Assessment")
    st.markdown("Enter patient information below for an immediate risk scan.")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input("Age (0-120)", min_value=0, max_value=120, value=50)
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        hypertension = st.selectbox("Hypertension", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        heart_disease = st.selectbox("Heart Disease", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    with col2:
        avg_glucose = st.number_input("Avg. Glucose Level (50-300)", min_value=50.0, max_value=300.0, value=100.0)
        bmi = st.number_input("BMI (10-60)", min_value=10.0, max_value=60.0, value=25.0)
        smoking = st.selectbox("Smoking Status", ["Never Smoked", "Formerly Smoked", "Smokes", "Unknown"])
    with col3:
        work = st.selectbox("Work Type", ["Private Sector", "Self-employed", "Govt job", "Children", "Never worked"])
        residence = st.selectbox("Residence", ["Urban", "Rural"])
        married = st.selectbox("Ever Married", ["Yes", "No"])

    if st.button("Run Diagnostic Scan", type="primary"):
        input_data = pd.DataFrame({
            'gender': [gender], 'age': [age], 'hypertension': [hypertension],
            'heart_disease': [heart_disease], 'ever_married': [married],
            'work_type': [work], 'Residence_type': [residence],
            'avg_glucose_level': [avg_glucose], 'bmi': [bmi],
            'smoking_status': [smoking]
        })
        
        prob = pipe.predict_proba(input_data)[0][1]
        pred = pipe.predict(input_data)[0]
        
        if pred == 1:
            st.error(f"### High Risk Identified ({prob:.1%})")
        else:
            st.success(f"### Low Risk Identified ({prob:.1%})")

# CLINICAL INSIGHTS SECTION (EXPANDED)
        st.divider()
        st.subheader("🩺 Clinical Risk Factor Analysis")
        st.markdown("""
        This analysis shows specifically the most impactful patient metrics that moved the model's 
        prediction away from the baseline. This is intended for decision support only.
        """)
        with st.spinner("Analyzing risk drivers..."):
            model = pipe.named_steps['model'] 
            preprocessor = pipe.named_steps['preprocessor']
            
            # 1. Transform and clean names
            X_transformed = preprocessor.transform(input_data)
            if hasattr(X_transformed, "toarray"):
                X_transformed = X_transformed.toarray()
            
            raw_names = preprocessor.get_feature_names_out()
            clean_names = [n.replace('num__','').replace('cat__','').replace('_',' ').title() for n in raw_names]

            # 2. SHAP Calculation (Back-end only)
            def model_predict(data):
                return model.predict_proba(data)[:, 1]

            background = np.zeros((1, len(clean_names))) 
            explainer = shap.Explainer(model_predict, background, feature_names=clean_names)
            shap_values = explainer(X_transformed)
            vals = shap_values.values[0]

            # 3. Create Categorized Insights
            risk_escalators = []
            protective_factors = []

            for i in range(len(vals)):
                if vals[i] > 0.001: # Significant increase
                    risk_escalators.append((clean_names[i], vals[i]))
                elif vals[i] < -0.001: # Significant decrease
                    protective_factors.append((clean_names[i], vals[i]))

            # Sort by absolute impact
            risk_escalators.sort(key=lambda x: x[1], reverse=True)
            protective_factors.sort(key=lambda x: x[1])

            # 4. Display Results in a Clinical Format
            col_risk, col_prot = st.columns(2)

            with col_risk:
                st.markdown("#### 🚩 Risk Escalators")
                if risk_escalators:
                    for name, impact in risk_escalators:
                        # Create a custom progress-bar style indicator for impact
                        st.info(f"**{name}**")
                        st.markdown(f"Metric Impact: {'Critical' if impact > 0.25 else ('High' if impact > 0.1 else 'Moderate')} | Increased probability by {impact*100:.1f}%")
                else:
                    st.write("No major risk escalators identified.")
                    
            with col_prot:
                st.markdown("#### ✅ Protective Factors")
                if protective_factors:
                    for name, impact in protective_factors:
                        st.info(f"**{name}**")
                        st.markdown(f"Reduced probability by {abs(impact)*100:.1f}%, contributing to lower risk score")
                        # st.markdown("Contributing to lower risk score")
                else:
                    st.write("No major protective factors identified.")
elif page_selection == "Batch Results & Analytics":
    st.title("Batch Results & Analytics")
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        
        with st.spinner('Analyzing...'):
            probs = pipe.predict_proba(df)[:, 1]
            preds = pipe.predict(df)
            
            # Creating the display version of the table
            df['Risk Probability'] = (probs * 100).round(2).astype(str) + '%'
            df['Prediction'] = preds
            df['Prediction'] = df['Prediction'].map({1: "HIGH RISK", 0: "Safe"})

        # Metrics Card Row
        m1, m2 = st.columns(2)
        m1.metric("Patients Screened", len(df))
        m2.metric("High Risk Detected", int(preds.sum()), delta_color="inverse")

        # Sort the dataframe for display
        df_display = df.sort_values('Risk Probability', ascending=False)
        
        st.dataframe(df_display, use_container_width=True)

        # DOWNLOAD BUTTON SECTION
        csv_data = df_display.to_csv(index=False).encode('utf-8')
        
        file_name = os.path.splitext(uploaded_file.name)[0]
        
        st.download_button(
            label="Download Patients' Report With Predictions",
            data=csv_data,
            file_name=f"{file_name}_risk_predictions.csv",
            mime="text/csv",
            help="Click to download the table above with risk scores and predictions."
        )
        
        st.divider()
        
# ENHANCED PLOTLY CHART
        df['Risk Color'] = df['Risk Probability'].str.rstrip('%').astype(float) / 100
        df['Status'] = df['Risk Color'].apply(lambda x: 'High Risk' if x >= 0.5 else 'Safe')

        fig = px.histogram(
            df, 
            x='Risk Color', 
            color='Status',
            title="Distribution of Patient Risk Scores",
            labels={'Risk Color': 'Probability of Stroke', 'count': 'Number of Patients'},
            color_discrete_map={'High Risk': '#ef553b', 'Safe': '#636efa'},
            text_auto=True,
            nbins=20
        )

        # FORCE WHITE LABELS FOR HIGH RISK
        fig.update_traces(
            textfont_color="white", 
            textposition="inside" # Ensures labels stay inside the bars where they are readable
        )

        fig.update_layout(
            bargap=0.1,
            xaxis_tickformat='.0%',
            showlegend=True,
            legend_title_text='Clinical Status'
        )

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Please upload a CSV file via the sidebar to view Batch Analytics.")