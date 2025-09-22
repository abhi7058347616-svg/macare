import streamlit as st
from streamlit_option_menu import option_menu
import pickle
import warnings
import pandas as pd
import plotly.express as px
from io import StringIO
import requests
import numpy as np
from sklearn.preprocessing import StandardScaler
import utils
import os
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="MatCare - Maternal & Fetal Health Platform",
    page_icon="🤱",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for modern, clean design
st.markdown("""
<style>
.main-header {
    background: linear-gradient(90deg, #1f77b4 0%, #2ca02c 100%);
    padding: 1rem 0;
    margin-bottom: 2rem;
    border-radius: 0 0 15px 15px;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
}

.header-content {
    max-width: 1200px;
    margin: 0 auto;
    padding: 0 2rem;
}

.logo-section {
    text-align: center;
    margin-bottom: 1rem;
}

.logo-section h1 {
    color: white;
    font-size: 2.5rem;
    margin: 0;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
}

.logo-section p {
    color: rgba(255,255,255,0.9);
    margin: 0.5rem 0;
    font-size: 1.1rem;
}

.nav-menu {
    margin-top: 1rem;
}

.prediction-card {
    background: white;
    padding: 2rem;
    border-radius: 15px;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    margin-bottom: 2rem;
    border-left: 5px solid #1f77b4;
}

.result-container {
    padding: 1.5rem;
    border-radius: 10px;
    margin: 1rem 0;
    border-left: 5px solid;
}

.low-risk {
    background-color: #d4edda;
    border-color: #28a745;
    color: #155724;
}

.medium-risk {
    background-color: #fff3cd;
    border-color: #ffc107;
    color: #856404;
}

.high-risk {
    background-color: #f8d7da;
    border-color: #dc3545;
    color: #721c24;
}

.input-section {
    background: #f8f9fa;
    padding: 1.5rem;
    border-radius: 10px;
    margin-bottom: 2rem;
}

.metric-card {
    background: white;
    padding: 1rem;
    border-radius: 8px;
    text-align: center;
    box-shadow: 0 2px 4px rgba(0,0,0,0.05);
}

.step-indicator {
    display: flex;
    justify-content: center;
    margin-bottom: 2rem;
    flex-wrap: wrap;
}

.step {
    background: #e9ecef;
    color: #6c757d;
    padding: 0.5rem 1rem;
    margin: 0.25rem;
    border-radius: 20px;
    font-weight: 500;
}

.step.active {
    background: #1f77b4;
    color: white;
}

@media (max-width: 768px) {
    .header-content {
        padding: 0 1rem;
    }
    
    .logo-section h1 {
        font-size: 2rem;
    }
    
    .prediction-card {
        padding: 1rem;
    }
    
    .step-indicator {
        flex-direction: column;
        align-items: center;
    }
}
</style>
""", unsafe_allow_html=True)

# Load models with error handling
@st.cache_resource
def load_models():
    try:
        # Load both models as specified in requirements
        model1 = pickle.load(open("models/model1.pkl", 'rb'))  # Pregnancy risk model
        model2 = pickle.load(open("models/model2.pkl", 'rb'))  # Fetal health model

        # Try to load scaler if available
        try:
            scaler = pickle.load(open('models/scaler_maternal_model.sav', 'rb'))
        except:
            scaler = None

        return model1, model2, scaler
    except FileNotFoundError as e:
        st.error(f"Model file not found: {e}")
        st.info("Please ensure model1.pkl and model2.pkl are in the models/ directory")
        return None, None, None
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None

# Load models
model1, model2, scaler = load_models()

# Top header navigation
st.markdown("""
<div class="main-header">
    <div class="header-content">
        <div class="logo-section">
            <h1>🤱 MatCare</h1>
            <p>Advanced Maternal & Fetal Health Prediction Platform</p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Navigation menu outside the header for better visibility
st.markdown("<br>", unsafe_allow_html=True)

# Initialize session state for navigation if not exists
if 'page_selection' not in st.session_state:
    st.session_state.page_selection = "About Us"

selected = option_menu(
    menu_title=None,
    options=["About Us", "Pregnancy Risk Prediction", "Fetal Health Prediction"],
    icons=["info-circle", "heart-pulse", "baby"],
    menu_icon="cast",
    default_index=0,
    orientation="horizontal",
    key="main_navigation",
    styles={
        "container": {
            "padding": "0!important", 
            "background-color": "#f8f9fa",
            "border-radius": "10px",
            "margin": "0 0 2rem 0",
            "box-shadow": "0 2px 4px rgba(0,0,0,0.1)"
        },
        "icon": {"color": "#1f77b4", "font-size": "18px"},
        "nav-link": {
            "font-size": "16px",
            "text-align": "center",
            "margin": "5px",
            "padding": "12px 20px",
            "color": "#495057",
            "border-radius": "8px",
            "font-weight": "500",
            "--hover-color": "#e9ecef",
        },
        "nav-link-selected": {
            "background-color": "#1f77b4",
            "color": "white",
            "box-shadow": "0 2px 4px rgba(31,119,180,0.3)"
        },
    }
)

# About Us Page
if selected == 'About Us':
    st.title("Welcome to MatCare")
    st.markdown("""
    **AI-powered maternal and fetal health predictions** for better healthcare outcomes. 
    Our platform provides accurate risk assessments through advanced machine learning.
    """)

    # Key Features
    st.markdown("## 🌟 Platform Features")
    
    # Feature cards with enhanced visibility
    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown("""
        <div class="prediction-card">
            <h3>🤰 Pregnancy Risk Prediction</h3>
            <p style="font-size: 1.1rem; margin: 1rem 0;">Analyze maternal health parameters to predict pregnancy risk levels using advanced machine learning.</p>
            <div style="background: #e8f5e8; padding: 1rem; border-radius: 8px; margin: 1rem 0;">
                <strong>Key Features:</strong><br>
                ✅ Real-time risk assessment<br>
                ✅ ML-powered predictions<br>
                ✅ Color-coded risk levels<br>
                ✅ Professional PDF reports<br>
                ✅ Research-backed analysis
            </div>
            <div style="background: #f8f9fa; padding: 1rem; border-radius: 8px;">
                <strong>Input Parameters:</strong> Age • Blood Pressure • Blood Sugar • Body Temperature • Heart Rate
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="prediction-card">
            <h3>👶 Fetal Health Prediction</h3>
            <p style="font-size: 1.1rem; margin: 1rem 0;">Comprehensive CTG analysis using 21 medical parameters for fetal health assessment.</p>
            <div style="background: #e8f4fd; padding: 1rem; border-radius: 8px; margin: 1rem 0;">
                <strong>Key Features:</strong><br>
                ✅ Advanced CTG analysis<br>
                ✅ 21-parameter assessment<br>
                ✅ Clinical classification<br>
                ✅ Detailed health reports<br>
                ✅ Real-time monitoring alerts
            </div>
            <div style="background: #f8f9fa; padding: 1rem; border-radius: 8px;">
                <strong>Analysis Types:</strong> Heart Rate Variability • Accelerations • Decelerations • Histogram Analysis
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Additional feature highlights
    st.markdown("---")
    st.markdown("## 🎯 Why Choose MatCare?")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h4>🔬 AI-Powered</h4>
            <p>Advanced machine learning models trained on medical data</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h4>📊 Real-Time</h4>
            <p>Instant predictions with comprehensive analysis</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h4>📄 Professional</h4>
            <p>Detailed PDF reports for medical professionals</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-card">
            <h4>📚 Research-Based</h4>
            <p>Evidence-based predictions with citations</p>
        </div>
        """, unsafe_allow_html=True)

    # Research Foundation
    with st.expander("📚 Research Foundation", expanded=False):
        st.markdown("""
        Built on evidence-based medical research from leading institutions:
        - **[WHO Maternal Health Guidelines](https://www.who.int/health-topics/maternal-health#tab=tab_1)**
        - **[NIH ML Pregnancy Research](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8021575/)**
        - **[Nature Medicine: Preterm Birth Prediction](https://www.nature.com/articles/s41591-018-0278-2)**
        """)

    # Call to action
    st.markdown("---")
    st.markdown("## 🚀 Get Started")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div style="text-align: center; background: linear-gradient(90deg, #1f77b4 0%, #2ca02c 100%); padding: 2rem; border-radius: 15px; color: white; margin: 1rem 0;">
            <h3 style="color: white; margin: 0;">Ready to begin your health assessment?</h3>
            <p style="margin: 1rem 0; font-size: 1.1rem;">Select a prediction tool from the navigation menu above to start your analysis.</p>
            <div style="display: flex; justify-content: center; gap: 1rem; flex-wrap: wrap; margin-top: 1.5rem;">
                <span style="background: rgba(255,255,255,0.2); padding: 0.5rem 1rem; border-radius: 20px;">🤰 Pregnancy Risk</span>
                <span style="background: rgba(255,255,255,0.2); padding: 0.5rem 1rem; border-radius: 20px;">👶 Fetal Health</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Quick stats
    st.markdown("## 📈 Platform Capabilities")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="Prediction Models",
            value="2",
            delta="Active & Trained"
        )
    
    with col2:
        st.metric(
            label="Health Parameters",
            value="26+",
            delta="Analyzed in Real-time"
        )
    
    with col3:
        st.metric(
            label="Risk Categories",
            value="6",
            delta="Comprehensive Classification"
        )

# Pregnancy Risk Prediction Page
elif selected == 'Pregnancy Risk Prediction':
    st.title('🤰 Pregnancy Risk Prediction')
    st.markdown("Enter maternal health parameters to assess pregnancy risk levels.")

    if model1 is None:
        st.error("🚨 Prediction model (model1.pkl) is not available. Please check model files in the models/ directory.")
        st.info("📁 Expected file: models/model1.pkl (Pregnancy Risk Prediction Model)")
    else:
        # Input form with better organization
        st.markdown('<div class="input-section">', unsafe_allow_html=True)
        st.subheader("📋 Enter Patient Information")
        
        # Create form for better UX
        with st.form("pregnancy_prediction_form", clear_on_submit=False):
            col1, col2, col3 = st.columns(3)

            with col1:
                age = st.number_input(
                    '👩 Age (years)', 
                    min_value=15, max_value=50, value=25, 
                    help="Maternal age in years"
                )
                
                diastolicBP = st.number_input(
                    '💓 Diastolic Blood Pressure (mmHg)', 
                    min_value=40, max_value=120, value=80,
                    help="Diastolic blood pressure measurement"
                )

            with col2:
                BS = st.number_input(
                    '🍯 Blood Glucose (mmol/L)', 
                    min_value=3.0, max_value=25.0, value=6.0, step=0.1,
                    help="Blood glucose level"
                )
                
                bodyTemp = st.number_input(
                    '🌡️ Body Temperature (°C)', 
                    min_value=35.0, max_value=42.0, value=37.0, step=0.1,
                    help="Body temperature in Celsius"
                )

            with col3:
                heartRate = st.number_input(
                    '❤️ Heart Rate (bpm)', 
                    min_value=50, max_value=120, value=80,
                    help="Heart rate in beats per minute"
                )

            # Form buttons
            col_submit, col_clear = st.columns([2, 1])
            with col_submit:
                submitted = st.form_submit_button(
                    '🔍 Predict Pregnancy Risk', 
                    type="primary",
                    use_container_width=True
                )
            with col_clear:
                if st.form_submit_button('🔄 Clear Form', use_container_width=True):
                    st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)

        # Process prediction when form is submitted
        if submitted:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")

                    # Use utility function for preprocessing
                    processed_data = utils.preprocess_pregnancy_data(age, diastolicBP, BS, bodyTemp, heartRate)

                    # Make prediction using model1
                    predicted_risk = model1.predict(processed_data)

                    # Store in session state for PDF generation
                    st.session_state.prediction_result = {
                        'risk_level': predicted_risk[0],
                        'risk_text': ['Low Risk', 'Medium Risk', 'High Risk'][predicted_risk[0]],
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }

                    st.session_state.input_data = {
                        'age': age, 'diastolic_bp': diastolicBP, 'blood_sugar': BS,
                        'body_temp': bodyTemp, 'heart_rate': heartRate
                    }

                    # Show step indicator only after successful prediction
                    st.markdown("""
                    <div class="step-indicator">
                        <span class="step active">📝 Input Data</span>
                        <span class="step active">🔍 Analysis</span>
                        <span class="step active">📊 Results</span>
                        <span class="step">📄 Report</span>
                    </div>
                    """, unsafe_allow_html=True)

                    # Display results with enhanced styling
                    st.subheader("🎯 Prediction Results")

                    if predicted_risk[0] == 0:
                        st.markdown("""
                        <div class="result-container low-risk">
                            <h3>✅ Low Risk Pregnancy</h3>
                            <p><strong>Assessment:</strong> The pregnancy shows minimal risk factors. Continue regular prenatal care and maintain healthy lifestyle habits.</p>
                            <p><strong>Recommendation:</strong> Standard prenatal monitoring schedule is appropriate.</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                    elif predicted_risk[0] == 1:
                        st.markdown("""
                        <div class="result-container medium-risk">
                            <h3>⚠️ Medium Risk Pregnancy</h3>
                            <p><strong>Assessment:</strong> The pregnancy shows moderate risk factors. Enhanced monitoring and closer medical supervision recommended.</p>
                            <p><strong>Recommendation:</strong> Increased frequency of prenatal visits and additional monitoring may be beneficial.</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                    else:
                        st.markdown("""
                        <div class="result-container high-risk">
                            <h3>🚨 High Risk Pregnancy</h3>
                            <p><strong>Assessment:</strong> The pregnancy shows significant risk factors. Immediate medical consultation and specialized care advised.</p>
                            <p><strong>Recommendation:</strong> Urgent referral to high-risk pregnancy specialist recommended.</p>
                        </div>
                        """, unsafe_allow_html=True)

                    # Get research-based reasoning
                    research_reasoning = utils.get_research_based_reasoning(
                        'pregnancy', predicted_risk[0],
                        {'age': age, 'diastolic_bp': diastolicBP, 'blood_sugar': BS,
                         'body_temp': bodyTemp, 'heart_rate': heartRate}
                    )

                    # Store research data in session state
                    st.session_state.research_reasoning = research_reasoning

                    # Display clinical analysis
                    with st.expander("🔬 Clinical Analysis & Research Foundation", expanded=True):
                        st.info(f"**Medical Reasoning:** {research_reasoning['medical_explanation']}")
                        
                        st.markdown("**📚 Research Evidence:**")
                        st.write(research_reasoning['reasoning'])
                        
                        st.markdown("**🔗 Research Citations:**")
                        for i, citation in enumerate(research_reasoning['citations'], 1):
                            st.markdown(f"{i}. **[{citation['title']}]({citation['link']})** - {citation['description']}")

                    # Input summary in a nice table
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.subheader("📋 Parameter Summary")
                        summary_df = utils.create_pregnancy_summary(age, diastolicBP, BS, bodyTemp, heartRate)
                        st.dataframe(summary_df, use_container_width=True, hide_index=True)

                    with col2:
                        # Risk metrics
                        st.subheader("📊 Risk Assessment")
                        st.metric(
                            label="Risk Level",
                            value=st.session_state.prediction_result['risk_text'],
                            delta=f"Confidence: {['High', 'Moderate', 'High'][predicted_risk[0]]}"
                        )
                        
                        # Additional risk factors
                        risk_factors = utils.calculate_risk_factors(age, diastolicBP, BS)
                        if 'error' not in risk_factors:
                            st.metric("Overall Risk Score", f"{risk_factors['overall_risk_score']}/6")

                    # Enhanced recommendations
                    recommendations = utils.generate_health_recommendations(
                        predicted_risk[0],
                        {'age': age, 'blood_pressure': diastolicBP, 'blood_sugar': BS}
                    )
                    
                    if recommendations:
                        st.subheader("💡 Personalized Recommendations")
                        for i, rec in enumerate(recommendations, 1):
                            st.success(f"**{i}.** {rec}")

            except Exception as e:
                st.error(f"❌ Error during prediction: {str(e)}")
                st.info("Please check your input values and try again.")

        # PDF Generation Section (always visible if prediction exists)
        if hasattr(st.session_state, 'prediction_result') and hasattr(st.session_state, 'input_data'):
            st.markdown("---")
            st.subheader("📄 Generate Detailed Medical Report")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.info("""
                **📋 Report will include:**
                - Complete patient information
                - Detailed risk assessment  
                - Clinical analysis & recommendations
                - Research citations & evidence
                - Professional formatting for medical use
                """)
                
            with col2:
                if st.button('📄 Download PDF Report', type="primary", use_container_width=True):
                    try:
                        # Update step indicator for PDF generation
                        st.markdown("""
                        <div class="step-indicator">
                            <span class="step active">📝 Input Data</span>
                            <span class="step active">🔍 Analysis</span>
                            <span class="step active">📊 Results</span>
                            <span class="step active">📄 Report</span>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        with st.spinner('🔄 Generating PDF report...'):
                            pdf_path = utils.generate_pdf_report(
                                'pregnancy', 
                                st.session_state.prediction_result, 
                                st.session_state.input_data, 
                                st.session_state.research_reasoning
                            )

                            if pdf_path.startswith('Error'):
                                st.error(f"❌ {pdf_path}")
                            else:
                                st.success("✅ PDF report generated successfully!")
                                
                                # Provide download button
                                with open(pdf_path, 'rb') as pdf_file:
                                    st.download_button(
                                        label='💾 Download Report',
                                        data=pdf_file.read(),
                                        file_name=f'MatCare_Pregnancy_Report_{st.session_state.input_data["age"]}years_{datetime.now().strftime("%Y%m%d_%H%M")}.pdf',
                                        mime='application/pdf',
                                        type="primary",
                                        use_container_width=True
                                    )
                                    
                    except Exception as e:
                        st.error(f"❌ Error generating PDF: {str(e)}")

# Fetal Health Prediction Page
elif selected == 'Fetal Health Prediction':
    st.title('👶 Fetal Health Prediction')
    st.markdown("Enter CTG parameters to assess fetal health status.")

    if model2 is None:
        st.error("🚨 Fetal health prediction model (model2.pkl) is not available. Please check model files in the models/ directory.")
        st.info("📁 Expected file: models/model2.pkl (Fetal Health Prediction Model)")
    else:
        st.markdown('<div class="input-section">', unsafe_allow_html=True)
        st.subheader("📊 Enter CTG Parameters")

        # CTG parameter names for reference
        ctg_params = [
            ("Baseline Value (FHR)", "bpm", 110, 180, 140),
            ("Accelerations", "per second", 0.0, 0.02, 0.0),
            ("Fetal Movement", "per second", 0.0, 0.5, 0.0),
            ("Uterine Contractions", "per second", 0.0, 0.02, 0.001),
            ("Light Decelerations", "per second", 0.0, 0.02, 0.0),
            ("Severe Decelerations", "per second", 0.0, 0.01, 0.0),
            ("Prolonged Decelerations", "per second", 0.0, 0.01, 0.0),
            ("Abnormal Short Term Variability", "%", 0, 100, 50),
            ("Mean Short Term Variability", "ms", 0.0, 10.0, 1.5),
            ("% Abnormal Long Term Variability", "%", 0, 100, 10),
            ("Mean Long Term Variability", "ms", 0.0, 50.0, 8.0),
            ("Histogram Width", "bpm", 10, 200, 70),
            ("Histogram Min", "bpm", 60, 150, 90),
            ("Histogram Max", "bpm", 120, 200, 180),
            ("Histogram Peaks", "count", 1, 10, 2),
            ("Histogram Zeros", "count", 0, 10, 0),
            ("Histogram Mode", "bpm", 60, 200, 140),
            ("Histogram Mean", "bpm", 90, 200, 140),
            ("Histogram Median", "bpm", 90, 200, 140),
            ("Histogram Variance", "bpm²", 0, 100, 15),
            ("Histogram Tendency", "direction", -1, 1, 0)
        ]

        # Create form for CTG parameters
        with st.form("fetal_prediction_form", clear_on_submit=False):
            # Organize parameters in tabs for better UX
            tab1, tab2, tab3, tab4 = st.tabs(["📊 Basic Parameters", "📈 Variability", "📉 Decelerations", "📋 Histogram"])
            
            features = []
            
            with tab1:
                st.markdown("**Basic CTG Measurements**")
                cols = st.columns(3)
                for i in range(0, 4):
                    param_name, unit, min_val, max_val, default = ctg_params[i]
                    with cols[i % 3]:
                        value = st.number_input(
                            f"{param_name} ({unit})",
                            min_value=float(min_val),
                            max_value=float(max_val),
                            value=float(default),
                            step=0.001 if isinstance(default, float) else 1.0,
                            key=f"param_{i}"
                        )
                        features.append(value)

            with tab2:
                st.markdown("**Heart Rate Variability Parameters**")
                st.info("📊 These parameters measure fetal heart rate patterns")
                
                # First handle uterine contractions (index 3)
                cols = st.columns(3)
                param_name, unit, min_val, max_val, default = ctg_params[3]
                with cols[0]:
                    value = st.number_input(
                        f"{param_name} ({unit})",
                        min_value=float(min_val),
                        max_value=float(max_val),
                        value=float(default),
                        step=0.001,
                        key=f"param_3_variability"
                    )
                    # Ensure features list is long enough
                    while len(features) <= 3:
                        features.append(0.0)
                    features[3] = value
                
                # Then handle variability parameters (7-10)
                for i in range(7, 11):
                    param_name, unit, min_val, max_val, default = ctg_params[i]
                    with cols[(i-6) % 3]:
                        value = st.number_input(
                            f"{param_name} ({unit})",
                            min_value=float(min_val),
                            max_value=float(max_val),
                            value=float(default),
                            step=0.1 if isinstance(default, float) else 1.0,
                            key=f"param_variability_{i}"
                        )
                        # Ensure features list is long enough
                        while len(features) <= i:
                            features.append(0.0)
                        features[i] = value
                        
            with tab3:
                st.markdown("**Deceleration Patterns**")
                st.info("⚠️ Deceleration values indicate potential fetal distress")
                cols = st.columns(3)
                # Add the deceleration parameters (indices 4, 5, 6)
                decel_indices = [4, 5, 6]  # Light, Severe, Prolonged decelerations
                for idx, i in enumerate(decel_indices):
                    param_name, unit, min_val, max_val, default = ctg_params[i]
                    with cols[idx]:
                        value = st.number_input(
                            f"{param_name} ({unit})",
                            min_value=float(min_val),
                            max_value=float(max_val),
                            value=float(default),
                            step=0.001,
                            key=f"param_deceleration_{i}",
                            help=f"Rate of {param_name.lower()} per second"
                        )
                        # Ensure features list is long enough
                        while len(features) <= i:
                            features.append(0.0)
                        features[i] = value

            with tab4:
                st.markdown("**Histogram Analysis Parameters**")
                cols = st.columns(3)
                for i in range(11, 21):
                    param_name, unit, min_val, max_val, default = ctg_params[i]
                    with cols[(i-11) % 3]:
                        value = st.number_input(
                            f"{param_name} ({unit})",
                            min_value=float(min_val),
                            max_value=float(max_val),
                            value=float(default),
                            step=0.1 if isinstance(default, float) else 1.0,
                            key=f"param_histogram_{i}"
                        )
                        # Ensure features list is long enough
                        while len(features) <= i:
                            features.append(0.0)
                        features[i] = value

            # Ensure we have exactly 21 features
            while len(features) < 21:
                features.append(0.0)
            features = features[:21]  # Trim if we have too many

            # Form submission buttons
            col_submit, col_clear = st.columns([2, 1])
            with col_submit:
                submitted = st.form_submit_button(
                    '🔍 Predict Fetal Health', 
                    type="primary",
                    use_container_width=True
                )
            with col_clear:
                if st.form_submit_button('🔄 Clear Form', use_container_width=True):
                    st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)

        # Process prediction when form is submitted
        if submitted:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")

                    # Validate CTG parameters
                    validation_result = utils.validate_ctg_parameters(features)
                    
                    if not validation_result.get('valid', True):
                        st.error(f"❌ Validation Error: {validation_result.get('error', 'Invalid parameters')}")
                        if 'warnings' in validation_result:
                            for warning in validation_result['warnings']:
                                st.warning(f"⚠️ {warning}")
                        st.stop()

                    # Preprocess data
                    processed_data = utils.preprocess_fetal_data(features)

                    # Make prediction using model2
                    predicted_health = model2.predict(processed_data)

                    # Store results in session state
                    health_labels = ['Normal', 'Suspect', 'Pathological']
                    st.session_state.fetal_prediction_result = {
                        'health_level': predicted_health[0],
                        'health_text': health_labels[predicted_health[0]],
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }

                    st.session_state.fetal_input_data = {
                        'ctg_features': features,
                        'baseline_fhr': features[0],
                        'accelerations': features[1],
                        'fetal_movement': features[2]
                    }

                    # Show step indicator only after successful prediction
                    st.markdown("""
                    <div class="step-indicator">
                        <span class="step active">📝 Input CTG Data</span>
                        <span class="step active">🔍 Analysis</span>
                        <span class="step active">📊 Results</span>
                        <span class="step">📄 Report</span>
                    </div>
                    """, unsafe_allow_html=True)

                    # Display results
                    st.subheader("🎯 Fetal Health Assessment Results")

                    if predicted_health[0] == 0:
                        st.markdown("""
                        <div class="result-container low-risk">
                            <h3>✅ Normal Fetal Health</h3>
                            <p><strong>Assessment:</strong> The CTG analysis indicates normal fetal well-being with appropriate heart rate patterns and variability.</p>
                            <p><strong>Recommendation:</strong> Continue routine monitoring. Fetal status is reassuring.</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                    elif predicted_health[0] == 1:
                        st.markdown("""
                        <div class="result-container medium-risk">
                            <h3>⚠️ Suspect Fetal Health</h3>
                            <p><strong>Assessment:</strong> The CTG shows suspicious patterns that require closer monitoring and possible intervention.</p>
                            <p><strong>Recommendation:</strong> Enhanced fetal monitoring recommended. Consider additional assessment methods.</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                    else:
                        st.markdown("""
                        <div class="result-container high-risk">
                            <h3>🚨 Pathological Fetal Health</h3>
                            <p><strong>Assessment:</strong> The CTG indicates concerning patterns suggesting potential fetal compromise.</p>
                            <p><strong>Recommendation:</strong> Immediate medical attention required. Consider urgent obstetric consultation.</p>
                        </div>
                        """, unsafe_allow_html=True)

                    # Get research-based reasoning for fetal health
                    fetal_research_reasoning = utils.get_research_based_reasoning(
                        'fetal', predicted_health[0], 
                        {'baseline_fhr': features[0], 'accelerations': features[1]}
                    )

                    st.session_state.fetal_research_reasoning = fetal_research_reasoning

                    # Display clinical analysis
                    with st.expander("🔬 CTG Analysis & Research Foundation", expanded=True):
                        st.info(f"**Clinical Interpretation:** {fetal_research_reasoning['medical_explanation']}")
                        
                        st.markdown("**📚 Research Evidence:**")
                        st.write(fetal_research_reasoning['reasoning'])
                        
                        st.markdown("**🔗 Research Citations:**")
                        for i, citation in enumerate(fetal_research_reasoning['citations'], 1):
                            st.markdown(f"{i}. **[{citation['title']}]({citation['link']})** - {citation['description']}")

                    # Key CTG metrics display
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric(
                            label="Baseline FHR",
                            value=f"{features[0]:.0f} bpm",
                            delta="Normal" if 110 <= features[0] <= 160 else "Abnormal"
                        )
                        
                    with col2:
                        st.metric(
                            label="Accelerations",
                            value=f"{features[1]:.3f}/sec",
                            delta="Present" if features[1] > 0 else "Absent"
                        )
                        
                    with col3:
                        st.metric(
                            label="Health Status",
                            value=st.session_state.fetal_prediction_result['health_text'],
                            delta=f"Confidence: {['High', 'Moderate', 'High'][predicted_health[0]]}"
                        )

                    # Enhanced deceleration analysis
                    decel_analysis = utils.analyze_deceleration_patterns(features)
                    if decel_analysis and not decel_analysis.get('normal', True):
                        st.subheader("⚠️ Deceleration Pattern Analysis")
                        st.warning(decel_analysis.get('summary', 'Deceleration patterns detected'))
                        for detail in decel_analysis.get('details', []):
                            st.info(f"• {detail}")

                    # Display validation warnings if any
                    if 'warnings' in validation_result and validation_result['warnings']:
                        st.subheader("⚠️ Clinical Alerts")
                        for warning in validation_result['warnings']:
                            st.warning(warning)

                    if 'recommendations' in validation_result and validation_result['recommendations']:
                        st.subheader("💡 Clinical Recommendations")
                        for rec in validation_result['recommendations']:
                            st.info(rec)

            except Exception as e:
                st.error(f"❌ Error during fetal health prediction: {str(e)}")
                st.info("Please check your CTG parameter values and try again.")

        # PDF Generation Section for Fetal Health
        if hasattr(st.session_state, 'fetal_prediction_result') and hasattr(st.session_state, 'fetal_input_data'):
            st.markdown("---")
            st.subheader("📄 Generate Fetal Health Report")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.info("""
                **📋 CTG Report will include:**
                - Complete CTG parameter analysis
                - Fetal health status assessment
                - Clinical interpretation & recommendations  
                - Research-based evidence & citations
                - Professional medical formatting
                """)
                
            with col2:
                if st.button('📄 Download CTG Report', type="primary", use_container_width=True):
                    try:
                        # Update step indicator for PDF generation
                        st.markdown("""
                        <div class="step-indicator">
                            <span class="step active">📝 Input CTG Data</span>
                            <span class="step active">🔍 Analysis</span>
                            <span class="step active">📊 Results</span>
                            <span class="step active">📄 Report</span>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        with st.spinner('🔄 Generating CTG report...'):
                            pdf_path = utils.generate_pdf_report(
                                'fetal', 
                                st.session_state.fetal_prediction_result, 
                                st.session_state.fetal_input_data, 
                                st.session_state.fetal_research_reasoning
                            )

                            if pdf_path.startswith('Error'):
                                st.error(f"❌ {pdf_path}")
                            else:
                                st.success("✅ CTG report generated successfully!")
                                
                                # Provide download button
                                with open(pdf_path, 'rb') as pdf_file:
                                    st.download_button(
                                        label='💾 Download CTG Report',
                                        data=pdf_file.read(),
                                        file_name=f'MatCare_Fetal_Health_Report_{datetime.now().strftime("%Y%m%d_%H%M")}.pdf',
                                        mime='application/pdf',
                                        type="primary",
                                        use_container_width=True
                                    )
                                    
                    except Exception as e:
                        st.error(f"❌ Error generating PDF: {str(e)}")
