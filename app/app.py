import streamlit as st
import pandas as pd
import numpy as np
import time
import plotly.graph_objects as go
import joblib
import os

# ----------------------------------------
# 1. ตั้งค่าหน้าเว็บ
# ----------------------------------------
st.set_page_config(page_title="SCADA AI Dashboard", layout="wide")

RULES = {
    'death_zone': {
        'temp_min': 91.02, 'temp_max': 99.96,
        'vib_min': 62.83, 'vib_max': 77.59,
        'hum_min': 49.53, 'hum_max': 74.36,
        'pres_min': 2.75, 'pres_max': 4.92,
        'eng_min': 1.67, 'eng_max': 3.63
    },
    'golden_zone': {
        'temp_min': 49.69, 'temp_max': 101.46,
        'vib_min': 17.07, 'vib_max': 90.28,
        'hum_min': 30.03, 'hum_max': 79.90,
        'pres_min': 1.00, 'pres_max': 5.00,
        'eng_min': 0.50, 'eng_max': 5.00
    }
}

# ----------------------------------------
# 2. ฟังก์ชันโหลดโมเดล AI (Cache ไว้เพื่อความเร็ว)
# ----------------------------------------
@st.cache_resource
def load_ai_models():
    models = {'anomaly': None, 'predictive': None, 'preprocessor': None}
    
    if os.path.exists('anomaly_model.pkl'):
        models['anomaly'] = joblib.load('anomaly_model.pkl')
        
    if os.path.exists('predictive_model.pkl') and os.path.exists('preprocessor_predictive.pkl'):
        models['predictive'] = joblib.load('predictive_model.pkl')
        models['preprocessor'] = joblib.load('preprocessor_predictive.pkl')
        
    return models

ai_models = load_ai_models()

# ----------------------------------------
# 3. ฟังก์ชันประเมินสถานะ Rule-based
# ----------------------------------------
def evaluate_machine_state(temp, vib):
    if (RULES['death_zone']['temp_min'] <= temp <= RULES['death_zone']['temp_max']) and \
       (RULES['death_zone']['vib_min'] <= vib <= RULES['death_zone']['vib_max']):
        return "CRITICAL (DEATH ZONE)", "🔴 อันตรายสูงสุด! สั่งหยุดเครื่องจักรเดี๋ยวนี้!", "red"
    elif temp >= RULES['death_zone']['temp_min'] or vib >= RULES['death_zone']['vib_min']:
        return "WARNING", "🟡 แจ้งเตือน! กำลังวิ่งเข้าหาโซนมรณะ", "orange"
    else:
        return "OPTIMAL (GOLDEN ZONE)", "🟢 เสถียรภาพเยี่ยม! เครื่องจักรทำงานในโซนที่ปลอดภัย", "green"

def render_custom_metric(title, value):
    return f"""
    <div style="background-color: #1e1e2e; padding: 15px; border-radius: 10px; text-align: center; border-bottom: 3px solid #3a3a4a;">
        <p style="margin: 0; font-size: 16px; color: #a6accd;">{title}</p>
        <h2 style="margin: 0; font-size: 38px; font-weight: bold; color: #ffffff;">{value:.2f}</h2>
    </div>
    """

# ----------------------------------------
# 4. โหลดข้อมูล CSV
# ----------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv('smart_manufacturing_data.csv')
    if 'timestamp' in df.columns:
        df = df.sort_values(by=['machine_id', 'timestamp']).reset_index(drop=True)
    return df

df_all = load_data()

# ----------------------------------------
# 5. โครงสร้าง UI หน้าเว็บ
# ----------------------------------------
st.title("Smart Manufacturing: AI Core Dashboard")

st.sidebar.header("Control Panel")
machine_list = sorted(df_all['machine_id'].unique())
selected_machine = st.sidebar.selectbox("🎯 เลือก Machine ID", machine_list)
run_simulator = st.sidebar.checkbox("▶️ Start Data Playback")
speed = st.sidebar.slider("ความเร็ว (วินาที/ข้อมูล)", 0.05, 2.0, 0.2)

df_machine = df_all[df_all['machine_id'] == selected_machine].reset_index(drop=True)

if not ai_models['anomaly']:
    st.sidebar.warning("⚠️ ไม่พบไฟล์ 'anomaly_model.pkl'")
if not ai_models['predictive'] or not ai_models['preprocessor']:
    st.sidebar.warning("⚠️ ไม่พบไฟล์ 'predictive_model.pkl' หรือ 'preprocessor.pkl'")

# สร้าง 3 Tabs
tab1, tab2, tab3 = st.tabs([
    "🔍 1. Anomaly Detection", 
    "🔮 2. Predictive Maintenance", 
    "⚙️ 3. Optimizer (Rule-based)"
])

with tab1:
    st.markdown("### 🔍 ระบบตรวจจับสิ่งผิดปกติ (Anomaly Detection)")
    ph_anomaly = st.empty()

with tab2:
    st.markdown("### 🔮 ระบบพยากรณ์ความเสี่ยงและอายุการใช้งาน (Predictive Maintenance)")
    ph_predictive = st.empty()

with tab3:
    st.markdown("### ⚙️ ระบบแนะนำการตั้งค่าเครื่องจักร (DBSCAN Golden Zone)")
    ph_optimize = st.empty()

# ----------------------------------------
# 6. ลูปจำลองการทำงาน
# ----------------------------------------
if run_simulator:
    history_time, history_temp, history_vib = [], [], []

    for index, row in df_machine.iterrows():
        current_temp, current_vib = row['temperature'], row['vibration']
        current_hum, current_pres, current_eng = row['humidity'], row['pressure'], row['energy_consumption']
        timestamp = row['timestamp'] if 'timestamp' in df_machine.columns else f"T+{index}"
        
        history_time.append(timestamp)
        history_temp.append(current_temp); history_vib.append(current_vib)
        history_time, history_temp, history_vib = history_time[-50:], history_temp[-50:], history_vib[-50:]
        
        current_row_df = pd.DataFrame([row])
        
        # ==================================================
        # อัปเดต Tab 1: Anomaly Detection
        # ==================================================
        with ph_anomaly.container():
            is_anomaly = False
            if ai_models['anomaly']:
                features_anomaly = ['temperature', 'vibration', 'humidity', 'pressure', 'energy_consumption', 'predicted_remaining_life', 'downtime_risk']
                X_anomaly = current_row_df[features_anomaly]
                pred_anomaly = ai_models['anomaly'].predict(X_anomaly)[0]
                is_anomaly = True if pred_anomaly == 1 else False
            else:
                is_anomaly = True if np.random.uniform(0, 1) > 0.9 else False
            
            if is_anomaly: st.error("🚨 **DETECTED ANOMALY!** พบความผิดปกติในพฤติกรรมของเซ็นเซอร์")
            else: st.success("✅ **NORMAL BEHAVIOR** พฤติกรรมเครื่องจักรปกติ")
                
            fig_anom = go.Figure(go.Scatter(x=history_time, y=history_temp, mode='lines+markers', line=dict(color='gray'), marker=dict(color=['red' if is_anomaly else 'blue'] * len(history_time), size=8)))
            fig_anom.update_layout(title="Live Temperature vs Anomaly State (Red = Anomaly)", height=300, margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig_anom, use_container_width=True)

        # ==================================================
        # อัปเดต Tab 2: Predictive Maintenance
        # ==================================================
        with ph_predictive.container():
            failure_prob = 0.0
            if ai_models['predictive'] and ai_models['preprocessor']:
                try:
                    X_pred = current_row_df.drop(columns=['maintenance_required'], errors='ignore')
                    X_pred_scaled = ai_models['preprocessor'].transform(X_pred)
                    failure_prob = ai_models['predictive'].predict_proba(X_pred_scaled)[0][1] * 100
                except Exception as e: st.error(f"Error in Prediction: {e}")
            else:
                failure_prob = min(100, max(0, ((current_temp - 50) / 50) * 100))
            
            c1, c2, c3 = st.columns(3)
            c1.metric("⚡ Failure Probability", f"{failure_prob:.1f} %", "โอกาสพังในรอบถัดไป")
            c2.metric("⏳ Estimated RUL", f"{row['predicted_remaining_life']:.1f} Cycles/Hrs")
            c3.metric("⚠️ Downtime Risk Score", f"{row['downtime_risk']:.2f}")
            
            fig_gauge_p = go.Figure(go.Indicator(
                mode = "gauge+number", value = failure_prob, title = {'text': "Failure Risk Level (%)"},
                gauge = {'axis': {'range': [0, 100]}, 'bar': {'color': "darkred"},
                         'steps': [{'range': [0, 50], 'color': "lightgreen"}, {'range': [50, 80], 'color': "yellow"}, {'range': [80, 100], 'color': "red"}]}
            ))
            fig_gauge_p.update_layout(height=300, margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig_gauge_p, use_container_width=True)

        # ==================================================
        # ⚙️ อัปเดต Tab 3: Optimizer (ปรับปรุงเกจไม่ให้เขียวเลยแดง)
        # ==================================================
        with ph_optimize.container():
            status, message, banner_color = evaluate_machine_state(current_temp, current_vib)
            
            if banner_color == "red": st.error(f"**STATUS: {status}** | {message}")
            elif banner_color == "orange": st.warning(f"**STATUS: {status}** | {message}")
            else: st.success(f"**STATUS: {status}** | {message}")
            
            c1, c2, c3, c4, c5 = st.columns(5)
            with c1: st.markdown(render_custom_metric("🌡️ Temp (°C)", current_temp), unsafe_allow_html=True)
            with c2: st.markdown(render_custom_metric("📳 Vibration", current_vib), unsafe_allow_html=True)
            with c3: st.markdown(render_custom_metric("💧 Humidity", current_hum), unsafe_allow_html=True)
            with c4: st.markdown(render_custom_metric("📊 Pressure", current_pres), unsafe_allow_html=True)
            with c5: st.markdown(render_custom_metric("⚡ Energy", current_eng), unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("####AI Process Control: Always-on Golden Zones")
            
            g_col1, g_col2 = st.columns(2)
            
            # เกจฝั่งซ้าย: Temperature (เขียวชนแดงพอดี)
            with g_col1:
                fig_g1 = go.Figure(go.Indicator(
                    mode = "gauge+number", value = current_temp,
                    title = {'text': "🌡️ Temperature Control Zone"},
                    gauge = {
                        'axis': {'range': [40, 110]},
                        'bar': {'color': "white"},
                        'steps' : [
                            {'range': [40, RULES['golden_zone']['temp_min']], 'color': "#1e1e2e"},
                            {'range': [RULES['golden_zone']['temp_min'], RULES['death_zone']['temp_min']], 'color': "#00FF00"}, # หยุดเขียวที่เส้นแดง
                            {'range': [RULES['death_zone']['temp_min'], 110], 'color': "#3d1313"} # พื้นหลังโซนอันตรายเป็นสีแดงเข้ม
                        ],
                        'threshold' : {'line': {'color': "red", 'width': 5}, 'thickness': 0.75, 'value': RULES['death_zone']['temp_min']}
                    }
                ))
                fig_g1.update_layout(height=380, margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig_g1, use_container_width=True)
                
            # เกจฝั่งขวา: Vibration (เขียวชนแดงพอดี)
            with g_col2:
                fig_g2 = go.Figure(go.Indicator(
                    mode = "gauge+number", value = current_vib,
                    title = {'text': "📳 Vibration Control Zone"},
                    gauge = {
                        'axis': {'range': [10, 100]},
                        'bar': {'color': "white"},
                        'steps' : [
                            {'range': [10, RULES['golden_zone']['vib_min']], 'color': "#1e1e2e"},
                            {'range': [RULES['golden_zone']['vib_min'], RULES['death_zone']['vib_min']], 'color': "#00FF00"}, # หยุดเขียวที่เส้นแดง
                            {'range': [RULES['death_zone']['vib_min'], 100], 'color': "#3d1313"} # พื้นหลังโซนอันตรายเป็นสีแดงเข้ม
                        ],
                        'threshold' : {'line': {'color': "red", 'width': 5}, 'thickness': 0.75, 'value': RULES['death_zone']['vib_min']}
                    }
                ))
                fig_g2.update_layout(height=380, margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig_g2, use_container_width=True)

        time.sleep(speed)