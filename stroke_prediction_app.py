import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
import plotly.express as px
import shap
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# PAGE CONFIG
st.set_page_config(page_title="Stroke Risk Predictor", layout="wide")

# CSS
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

# Cache both models to avoid reloading on every interaction
@st.cache_resource
def load_core_model():
    return joblib.load(os.path.join('model', 'core_stroke_prediction_model.pkl'))

@st.cache_resource
def load_full_model():
    return joblib.load(os.path.join('model', 'comprehensive_stroke_prediction_model.pkl'))

# SIDEBAR
with st.sidebar:
    st.title("Settings & Navigation")
    
    # MODEL SELECTION
    model_choice = st.radio(
        "Select Model Version:",
        ("Core Model", "Comprehensive Model"),
        help='The "Core Model" is best for testing on the 1k-rows unseen dataset. The "Comprehensive Model" has been trained on the entire dataset.'
    )
    # Load chosen model into 'pipe'
    try:
        if model_choice == "Core Model":
            pipe = load_core_model()
        else:
            pipe = load_full_model()
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

    st.divider()
    
    page_selection = st.radio(
        "Select Mode:",
        ["Individual Patient Entry", "Batch Results & Analytics"],
        index=0
    )
    st.divider()
    st.header("Upload Patients' Data CSV")
    uploaded_file = st.sidebar.file_uploader("And Change Mode to 'Batch Results & Analytics'", type="csv")
    st.info("CSV should include information: gender, age, hypertension, heart_disease, ever_married, work_type, Residence_type, avg_glucose_level, bmi, smoking_status")

# MAIN LOGIC
if page_selection == "Individual Patient Entry":
    st.title("Individual Patient Assessment")
    st.markdown("Enter patient information below for an immediate risk scan.")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input("Age (0-120 Years)", min_value=0, max_value=120, value=50)
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        hypertension = st.selectbox("Hypertension", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        heart_disease = st.selectbox("Heart Disease", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    with col2:
        avg_glucose = st.number_input("Avg. Glucose Level (50-300 mg/dL)", min_value=50.0, max_value=300.0, value=100.0)
        bmi = st.number_input("BMI (10-60 kg/m²)", min_value=10.0, max_value=60.0, value=25.0)
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
        
        # RISK DISPLAY LOGIC
        if prob >= 0.50:
            st.error(f"### 🚩 High Risk Identified ({prob:.1%})")
            st.warning("**Priority:** High. Clinical intervention and further testing is highly recommended.")
        elif 0.40 <= prob < 0.50:
            st.warning(f"### ⚠️ Moderate Risk ({prob:.1%})")
            st.info("**Priority:** Patient is close to the risk threshold. Lifestyle review and active monitoring is recommended.")
        else:
            st.success(f"### ✅ Low Risk Identified ({prob:.1%})")

        # CLINICAL INSIGHTS
        st.divider()
        st.subheader("🩺 Clinical Risk Factor Analysis")
        st.markdown("Analysis of metrics influencing the model's decision relative to a neutral baseline.")
        
        with st.spinner("Analyzing risk drivers..."):
            model = pipe.named_steps['model'] 
            preprocessor = pipe.named_steps['preprocessor']
            
            X_transformed = preprocessor.transform(input_data)
            if hasattr(X_transformed, "toarray"):
                X_transformed = X_transformed.toarray()
            
            raw_names = preprocessor.get_feature_names_out()
            clean_names = [n.replace('num__','').replace('cat__','').replace('_',' ').title() for n in raw_names]

            def model_predict(data):
                return model.predict_proba(data)[:, 1]

            background = np.zeros((1, len(clean_names)))
            explainer = shap.Explainer(model_predict, background, feature_names=clean_names)
            shap_values = explainer(X_transformed)
            vals = shap_values.values[0]

            risk_escalators = []
            protective_factors = []

            for i in range(len(vals)):
                if vals[i] > 0.0001: 
                    risk_escalators.append((clean_names[i], vals[i]))
                elif vals[i] < -0.0001:
                    protective_factors.append((clean_names[i], vals[i]))

            risk_escalators.sort(key=lambda x: x[1], reverse=True)
            protective_factors.sort(key=lambda x: x[1])

            col_risk, col_prot = st.columns(2)

            with col_risk:
                st.markdown("#### 🚩 Risk Escalators")
                if risk_escalators:
                    for name, impact in risk_escalators:
                        priority = 'Critical' if impact > 0.25 else ('High' if impact > 0.1 else 'Moderate')
                        st.info(f"**{name}**")
                        st.markdown(f"Impact: **{priority}** | Increased probability by {impact*100:.1f}%")
                else:
                    st.write("No major risk escalators identified.")
                    
            with col_prot:
                st.markdown("#### ✅ Protective Factors")
                if protective_factors:
                    for name, impact in protective_factors:
                        st.success(f"**{name}**")
                        st.markdown(f"Reduced probability by {abs(impact)*100:.1f}%")
                else:
                    st.write("No major protective factors identified.")

elif page_selection == "Batch Results & Analytics":
    st.title("Batch Results & Analytics")
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        
        with st.spinner('Analyzing dataset...'):
            probs = pipe.predict_proba(df)[:, 1]
            preds = pipe.predict(df)  # Needed for the Confusion Matrix
            
            def risk_class(p):
                if p >= 0.50: return "HIGH RISK"
                if p >= 0.40: return "Moderate Risk"
                return "Safe"

            df['Risk Probability'] = (probs * 100).round(2).astype(str) + '%'
            df['Risk Status'] = [risk_class(p) for p in probs]

        m1, m2, m3 = st.columns(3)
        m1.metric("Patients Screened", len(df))
        m2.metric("High Risk", sum(probs >= 0.5))
        m3.metric("Moderate Risk", sum((probs >= 0.4) & (probs < 0.5)))

        df_display = df.sort_values('Risk Probability', ascending=False)
        st.dataframe(df_display, use_container_width=True)

        # DOWNLOAD BUTTON
        csv_data = df_display.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Risk Report",
            data=csv_data,
            file_name=f"{os.path.splitext(uploaded_file.name)[0]}_results.csv",
            mime="text/csv"
        )
        
        st.divider()
        
        # ENHANCED RISK GROUP HISTOGRAM
        df['Risk_Float'] = probs
        df['Group'] = df['Risk_Float'].apply(
            lambda x: 'High Risk' if x >= 0.5 else ('Moderate Risk' if x >= 0.4 else 'Safe')
        )

        fig_hist = px.histogram(
            df, 
            x='Risk_Float', 
            color='Group',
            title=f"Patient Risk Distribution Groups - {model_choice}",
            labels={'Risk_Float': 'Stroke Probability', 'count': 'Patient Count'},
            color_discrete_map={
                'High Risk': '#ef553b',   
                'Moderate Risk': '#fec032',  
                'Safe': '#636efa'         
            },
            text_auto=True,
            nbins=20
        )

        fig_hist.update_traces(textfont_color="white", textposition="inside")
        fig_hist.update_layout(bargap=0.1, xaxis_tickformat='.0%', legend_title_text='Risk Group')

        st.plotly_chart(fig_hist, use_container_width=True)

        # CONFUSION MATRIX SECTION
        st.divider()
        col1, col2, col3 = st.columns([1.1, 0.5, 2]) 

        with col1:
            if 'stroke' in df.columns:
                st.subheader("Performance")

                y_total = df['stroke']
                cm = confusion_matrix(y_total, preds)
                
                fig_cm, ax = plt.subplots(figsize=(4, 3)) 
                disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No', 'Yes'])
                disp.plot(cmap='Reds', ax=ax, values_format='d', colorbar=False)
                
                ax.set_title(f"Confusion Matrix", fontsize=10)
                plt.xticks(fontsize=8); plt.yticks(fontsize=8)
                ax.set_xlabel("Predicted", fontsize=8); ax.set_ylabel("Actual", fontsize=8)
                plt.grid(False)
                st.pyplot(fig_cm, use_container_width=False) 
            else:
                st.info("Upload 'stroke' data for accuracy.")

        with col2:
            st.subheader("Group Percentages")
            risk_counts = df['Group'].value_counts(normalize=True) * 100
            
            high_risk_pct = risk_counts.get('High Risk', 0)
            mod_risk_pct = risk_counts.get('Moderate Risk', 0)
            safe_pct = risk_counts.get('Safe', 0)

            def color_dot(hex_color):
                return f'<span style="color:{hex_color}; font-size: 20px; margin-right: 5px;">●</span>'

            st.markdown(f"{color_dot('#ef553b')} **High Risk:** {high_risk_pct:.1f}%", unsafe_allow_html=True)
            st.markdown(f"{color_dot('#fec032')} **Moderate Risk:** {mod_risk_pct:.1f}%", unsafe_allow_html=True)
            st.markdown(f"{color_dot('#636efa')} **Safe:** {safe_pct:.1f}%", unsafe_allow_html=True)
            
            # Screening Load Logic
            screening_load = high_risk_pct + mod_risk_pct
            st.metric(label="Patient Screening Load", value=f"{screening_load:.1f}%", 
                      help="Percentage of patients requiring clinical attention (High + Moderate Risk)")
            st.divider()
            st.subheader("Safety")
            if 'stroke' in df.columns:
                tp = cm[1,1]
                fn = cm[1,0]
                
                # Recall
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                st.metric(label="Strokes Caught", value=f"{recall:.1%}",
                          help="True Negative Rate: The percentage of actual strokes the model suceeded to predict.")                
                # Missed Stroke Rate (FNR)
                fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
                st.metric(label="Missed Stroke Rate", value=f"{fnr:.1%}", 
                          help="False Negative Rate: The percentage of actual strokes the model failed to catch.")
            else:
                st.write("N/A")
                st.caption("Ground truth 'stroke' column missing.")
                
            import plotly.express as px
            
        with col3:
            st.subheader("Age vs. Risk Analysis")
            fig_scatter = px.scatter(
                df, 
                x="age", 
                y="Risk_Float", 
                color="Group",
                color_discrete_map={'High Risk': '#ef553b', 'Moderate Risk': '#fec032', 'Safe': '#636efa'},
                labels={"age": "Patient Age", "Risk_Float": "Probability of Stroke"},
                template="plotly_white"
            )
            fig_scatter.update_layout(yaxis_tickformat='.0%')
            st.plotly_chart(fig_scatter, use_container_width=True)             
                
    else:
        st.info("**Note:** Upload a CSV containing a 'stroke' column to view the Confusion Matrix and accuracy metrics.")
else:
    st.warning("Please upload a CSV file via the sidebar to view Batch Analytics.")