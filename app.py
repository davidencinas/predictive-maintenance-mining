import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# ── Configuración de página
st.set_page_config(
    page_title="Predictive Maintenance Dashboard",
    page_icon="⚙️",
    layout="wide"
)

# ── Estilos
st.markdown("""
<style>
    .main-header {font-size: 2rem; font-weight: bold; color: #1D4E89;}
    .metric-card {background: #F0F4F8; padding: 1rem; border-radius: 8px; text-align: center;}
    .alert-high {background: #FDECEA; border-left: 4px solid #D32F2F; padding: 1rem; border-radius: 4px;}
    .alert-low {background: #E8F5E9; border-left: 4px solid #388E3C; padding: 1rem; border-radius: 4px;}
</style>
""", unsafe_allow_html=True)

# ── Cargar y entrenar modelo
@st.cache_resource
def load_model():
    df = pd.read_csv('ai4i2020.csv')
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    df['Type_encoded'] = le.fit_transform(df['Type'])
    df['temp_diff'] = df['Process temperature [K]'] - df['Air temperature [K]']
    df['power'] = df['Torque [Nm]'] * df['Rotational speed [rpm]'] * (2 * np.pi / 60)

    FEATURES = ['Type_encoded', 'Air_temp_K', 'Process_temp_K',
                'Rotational_speed_rpm', 'Torque_Nm', 'Tool_wear_min',
                'temp_diff', 'power']

    df = df.rename(columns={
        'Air temperature [K]': 'Air_temp_K',
        'Process temperature [K]': 'Process_temp_K',
        'Rotational speed [rpm]': 'Rotational_speed_rpm',
        'Torque [Nm]': 'Torque_Nm',
        'Tool wear [min]': 'Tool_wear_min'
    })

    X = df[FEATURES]
    y = df['Machine failure']

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=FEATURES)

    X_train, _, y_train, _ = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y)

    scale = (y_train == 0).sum() / (y_train == 1).sum()
    model = XGBClassifier(scale_pos_weight=scale, n_estimators=200,
                          max_depth=3, learning_rate=0.05,
                          subsample=0.8, random_state=42, verbosity=0)
    model.fit(X_train, y_train)
    return model, scaler, df

model, scaler, df = load_model()

# ── Header
st.markdown('<p class="main-header">⚙️ Predictive Maintenance Dashboard</p>', 
            unsafe_allow_html=True)
st.markdown("**Real-time equipment failure prediction for mining operations**")
st.markdown("---")

# ── Layout
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("🔧 Sensor Input")
    st.markdown("Adjust sensor readings to predict failure probability:")

    product_type = st.selectbox("Product Type", ["H — High Quality", "L — Low Quality", "M — Medium Quality"])
    type_map = {"H — High Quality": 0, "L — Low Quality": 1, "M — Medium Quality": 2}
    type_encoded = type_map[product_type]

    air_temp = st.slider("Air Temperature (K)", 295.0, 305.0, 300.0, 0.1)
    process_temp = st.slider("Process Temperature (K)", 305.0, 315.0, 310.0, 0.1)
    rot_speed = st.slider("Rotational Speed (rpm)", 1168, 2886, 1500)
    torque = st.slider("Torque (Nm)", 3.8, 76.6, 40.0, 0.1)
    tool_wear = st.slider("Tool Wear (min)", 0, 253, 100)

    # Calcular features derivadas
    temp_diff = process_temp - air_temp
    power = torque * rot_speed * (2 * np.pi / 60)

    st.markdown("---")
    st.markdown("**Derived Features:**")
    st.markdown(f"- Temp Differential: `{temp_diff:.2f} K`")
    st.markdown(f"- Power: `{power:,.0f} W`")

with col2:
    # Preparar input
    input_data = pd.DataFrame([[type_encoded, air_temp, process_temp,
                                 rot_speed, torque, tool_wear,
                                 temp_diff, power]],
        columns=['Type_encoded', 'Air_temp_K', 'Process_temp_K',
                 'Rotational_speed_rpm', 'Torque_Nm', 'Tool_wear_min',
                 'temp_diff', 'power'])

    input_scaled = pd.DataFrame(
        scaler.transform(input_data), columns=input_data.columns)

    prob = model.predict_proba(input_scaled)[0][1]
    prediction = model.predict(input_scaled)[0]

    # ── Gauge de riesgo
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=prob * 100,
        title={'text': "Failure Probability (%)", 'font': {'size': 18}},
        delta={'reference': 50, 'increasing': {'color': "#D32F2F"}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1},
            'bar': {'color': "#D32F2F" if prob > 0.5 else "#388E3C"},
            'steps': [
                {'range': [0, 30], 'color': '#E8F5E9'},
                {'range': [30, 60], 'color': '#FFF9C4'},
                {'range': [60, 100], 'color': '#FDECEA'}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 3},
                'thickness': 0.75,
                'value': 50
            }
        }
    ))
    fig.update_layout(height=280, margin=dict(t=40, b=0, l=20, r=20))
    st.plotly_chart(fig, use_container_width=True)

    # ── Alerta
    if prediction == 1:
        st.markdown("""
        <div class="alert-high">
        <strong>⚠️ HIGH FAILURE RISK DETECTED</strong><br>
        Immediate maintenance inspection recommended. Schedule equipment review before next operational cycle.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="alert-low">
        <strong>✅ EQUIPMENT OPERATING NORMALLY</strong><br>
        No immediate maintenance required. Continue monitoring sensor readings.
        </div>
        """, unsafe_allow_html=True)

    # ── Métricas
    st.markdown("---")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Failure Probability", f"{prob*100:.1f}%")
    m2.metric("Tool Wear", f"{tool_wear} min", 
              delta=f"{tool_wear-200} min vs threshold" if tool_wear > 150 else None,
              delta_color="inverse")
    m3.metric("Temp Differential", f"{temp_diff:.1f} K")
    m4.metric("Power Output", f"{power/1000:.1f} kW")

    # ── Variables fuera de rango
    st.markdown("---")
    st.subheader("📊 Sensor Status")
    
    warnings_list = []
    if tool_wear > 200:
        warnings_list.append(f"🔴 Tool wear ({tool_wear} min) exceeds critical threshold (200 min)")
    if torque > 60:
        warnings_list.append(f"🔴 Torque ({torque} Nm) in high-risk zone (> 60 Nm)")
    if temp_diff < 8.5:
        warnings_list.append(f"🟡 Temperature differential ({temp_diff:.1f} K) below optimal range")
    if rot_speed < 1300:
        warnings_list.append(f"🟡 Rotational speed ({rot_speed} rpm) below normal range")

    if warnings_list:
        for w in warnings_list:
            st.markdown(w)
    else:
        st.markdown("✅ All sensors within normal operating ranges")

# ── Footer
st.markdown("---")
st.markdown("**Model:** XGBoost Balanced | **ROC-AUC:** 98.1% | **Recall:** 80.9% | "
            "[View on GitHub](https://github.com/DavidEncinas/predictive-maintenance-mining)")
