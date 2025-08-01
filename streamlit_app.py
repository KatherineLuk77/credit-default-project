import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# Create a simple demo model for the app
@st.cache_resource
def create_demo_model():
    """Create a simple demo model for the credit default prediction"""
    # Generate synthetic data for demo
    np.random.seed(42)
    n_samples = 1000
    
    # Create synthetic features
    age = np.random.normal(35, 10, n_samples)
    income = np.random.normal(60000, 20000, n_samples)
    debt_ratio = np.random.uniform(0.1, 0.8, n_samples)
    credit_lines = np.random.poisson(5, n_samples)
    late_payments = np.random.poisson(1, n_samples)
    
    # Create target variable (default risk)
    default_risk = (
        (age < 30) * 0.3 +
        (income < 40000) * 0.4 +
        (debt_ratio > 0.6) * 0.5 +
        (late_payments > 2) * 0.6 +
        np.random.normal(0, 0.1, n_samples)
    )
    default = (default_risk > 0.5).astype(int)
    
    # Create features dataframe
    X = pd.DataFrame({
        'age': age,
        'income': income,
        'debt_ratio': debt_ratio,
        'credit_lines': credit_lines,
        'late_payments': late_payments
    })
    
    # Train a simple model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, default)
    
    return model

# Load the demo model
model = create_demo_model()

# App interface
st.set_page_config(page_title="Credit Default Prediction", layout="wide")

st.title("🏦 Credit Default Risk Prediction")
st.markdown("---")

# Sidebar for input
st.sidebar.header("📊 Customer Information")
st.sidebar.markdown("Enter customer details to predict default risk:")

# Input fields
age = st.sidebar.number_input("Age", min_value=18, max_value=100, value=35, help="Customer's age")
income = st.sidebar.number_input("Annual Income ($)", min_value=10000, max_value=200000, value=60000, step=5000, help="Customer's annual income")
debt_ratio = st.sidebar.slider("Debt Ratio", min_value=0.0, max_value=1.0, value=0.3, step=0.05, help="Ratio of debt to income")
credit_lines = st.sidebar.number_input("Number of Open Credit Lines", min_value=0, max_value=20, value=5, help="Number of active credit accounts")
late_payments = st.sidebar.number_input("Number of Late Payments (6 months)", min_value=0, max_value=10, value=1, help="Late payments in the last 6 months")

# Prediction button
if st.sidebar.button("🔮 Predict Default Risk", type="primary"):
    # Create input dataframe
    input_data = pd.DataFrame({
        'age': [age],
        'income': [income],
        'debt_ratio': [debt_ratio],
        'credit_lines': [credit_lines],
        'late_payments': [late_payments]
    })
    
    # Make prediction
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]
    
    # Display results
    st.header("📈 Prediction Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric(
            label="Default Risk",
            value="HIGH RISK" if prediction == 1 else "LOW RISK",
            delta=f"{probability:.1%} probability"
        )
    
    with col2:
        risk_color = "🔴" if prediction == 1 else "🟢"
        st.metric(
            label="Risk Level",
            value=f"{risk_color} {'High' if prediction == 1 else 'Low'}",
            delta="Default Likely" if prediction == 1 else "Default Unlikely"
        )
    
    # Risk assessment
    st.subheader("📋 Risk Assessment")
    
    if prediction == 1:
        st.error("⚠️ This customer shows signs of potential default risk.")
        st.markdown("**Recommendations:**")
        st.markdown("- Consider reducing credit limit")
        st.markdown("- Implement stricter payment monitoring")
        st.markdown("- Offer financial counseling services")
    else:
        st.success("✅ This customer appears to be a low default risk.")
        st.markdown("**Recommendations:**")
        st.markdown("- Standard credit terms appropriate")
        st.markdown("- Regular monitoring sufficient")
        st.markdown("- Consider credit limit increases")
    
    # Feature importance
    st.subheader("📊 Feature Analysis")
    
    # Create a simple feature importance visualization
    feature_importance = pd.DataFrame({
        'Feature': ['Age', 'Income', 'Debt Ratio', 'Credit Lines', 'Late Payments'],
        'Value': [age, income, debt_ratio, credit_lines, late_payments],
        'Risk Factor': [
            'High' if age < 30 else 'Medium' if age < 50 else 'Low',
            'Low' if income > 80000 else 'Medium' if income > 50000 else 'High',
            'High' if debt_ratio > 0.6 else 'Medium' if debt_ratio > 0.4 else 'Low',
            'High' if credit_lines > 8 else 'Medium' if credit_lines > 4 else 'Low',
            'High' if late_payments > 2 else 'Medium' if late_payments > 0 else 'Low'
        ]
    })
    
    st.dataframe(feature_importance, use_container_width=True)

# App description
st.markdown("---")
st.markdown("""
### 🎯 About This Demo
This interactive demo showcases a **Machine Learning-based Credit Default Prediction System** built with:
- **Python & Streamlit** for the web interface
- **Scikit-learn** for the ML model
- **Random Forest** algorithm for predictions
- **Real-time risk assessment** capabilities

### 💼 Business Value
- **Proactive risk management** for financial institutions
- **Data-driven credit decisions** based on customer profiles
- **Automated risk scoring** to support loan officers
- **Portfolio optimization** through better risk assessment

### 🔧 Technical Features
- **Interactive input forms** for customer data
- **Real-time predictions** with probability scores
- **Risk level classification** (High/Medium/Low)
- **Feature importance analysis** for decision transparency
- **Professional UI/UX** suitable for business applications
""")

# Footer
st.markdown("---")
st.markdown("*Built with ❤️ using Streamlit and Machine Learning*")