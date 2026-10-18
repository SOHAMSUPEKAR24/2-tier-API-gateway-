import streamlit as st
import sqlite3
import pandas as pd
import time
import plotly.express as px
import plotly.graph_objects as go
import subprocess

st.set_page_config(page_title="two-tier gateway", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,100..900;1,9..144,100..900&family=Inter:wght@300;400;500;600&display=swap');
        
        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif !important;
            background-color: #f7f5f0 !important;
            color: #4a4a4a !important;
        }
        
        .stApp {
            background-color: #f7f5f0;
            background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 200 200' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noiseFilter'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.8' numOctaves='3' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noiseFilter)' opacity='0.04'/%3E%3C/svg%3E") !important;
        }
        
        h1, h2, h3, h4, h5, h6 {
            font-family: 'Fraunces', serif !important;
            font-weight: 300 !important;
            color: #3b3b3b !important;
            text-transform: lowercase;
            letter-spacing: -0.02em;
        }
        
        .tactile-panel {
            background: #ffffff;
            border-radius: 12px;
            padding: 24px;
            margin-bottom: 24px;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.03);
            border: 1px solid rgba(0,0,0,0.02);
            transition: transform 0.4s cubic-bezier(0.2, 0.8, 0.2, 1), box-shadow 0.4s cubic-bezier(0.2, 0.8, 0.2, 1);
        }
        
        .tactile-panel:hover {
            transform: translateY(-2px);
            box-shadow: 0 12px 30px rgba(0, 0, 0, 0.06);
        }
        
        @keyframes breathe {
            0% { box-shadow: 0 0 0 0 rgba(204, 119, 102, 0.1); }
            50% { box-shadow: 0 0 20px 4px rgba(204, 119, 102, 0.12); }
            100% { box-shadow: 0 0 0 0 rgba(204, 119, 102, 0.1); }
        }
        
        .breathing-panel {
            background: #fdfbf9;
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 16px;
            border: 1px solid rgba(204, 119, 102, 0.15);
            animation: breathe 4s ease-in-out infinite;
            color: #5c4e48;
        }
        
        .stButton>button {
            font-family: 'Inter', sans-serif !important;
            background-color: #8c9a89 !important;
            color: white !important;
            border: none !important;
            border-radius: 8px !important;
            padding: 10px 24px !important;
            font-weight: 400 !important;
            text-transform: lowercase;
            transition: all 0.3s ease;
            box-shadow: 0 2px 10px rgba(140, 154, 137, 0.2);
        }
        
        .stButton>button:hover {
            background-color: #7b8878 !important;
            transform: translateY(-1px);
            box-shadow: 0 4px 15px rgba(140, 154, 137, 0.3);
        }

        .status-forwarded { color: #8c9a89; }
        .status-blocked { color: #cc7766; }
        .status-escalated { color: #d6b55e; }
        
    </style>
""", unsafe_allow_html=True)

DB_PATH = "dashboard_logs.db"

def fetch_live_traffic():
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query("SELECT timestamp, ip_source, fwd_packet_length_max, flow_bytes_s, tier1_confidence, status FROM live_traffic ORDER BY id DESC LIMIT 15", conn)
        conn.close()
        # Clean up statuses to remove emojis and lowercase them
        df['status'] = df['status'].str.replace('🟢 FORWARDED', 'forwarded')
        df['status'] = df['status'].str.replace('🔴 BLOCKED', 'blocked')
        df['status'] = df['status'].str.replace('🟡 ESCALATED', 'escalated')
        return df
    except:
        return pd.DataFrame()

def fetch_escalated_traffic():
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query("SELECT timestamp, ip_source, tier1_confidence FROM live_traffic WHERE escalated=1 ORDER BY id DESC LIMIT 4", conn)
        conn.close()
        return df
    except:
        return pd.DataFrame()

def fetch_metrics():
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query("SELECT timestamp, accuracy, escalation_rate, cycle FROM metrics ORDER BY id ASC", conn)
        conn.close()
        return df
    except:
        return pd.DataFrame()

# layout and typography
st.sidebar.markdown("## controls")
if st.sidebar.button("start traffic simulation"):
    subprocess.Popen(["python", "src/dashboard/simulator.py"])

st.sidebar.markdown("---")
st.sidebar.markdown("<p style='font-size: 0.9em; color: #888;'>adaptive two-tier malware detection.<br>monitoring live telemetry.</p>", unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center; margin-bottom: 5px;'>adaptive two-tier malware detection</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #888; font-family: Inter, sans-serif; font-size: 1.1em; margin-bottom: 40px;'>live telemetry and ambient feedback loop</p>", unsafe_allow_html=True)

col1, col2 = st.columns([2, 1])

placeholder_metrics = st.empty()
placeholder_traffic = st.empty()
placeholder_radar = st.empty()
placeholder_sandbox = st.sidebar.empty()

i = 0
while True:
    i += 1
    traffic_df = fetch_live_traffic()
    metrics_df = fetch_metrics()
    escalated_df = fetch_escalated_traffic()
    
    with placeholder_metrics.container():
        st.markdown("<div class='tactile-panel'>", unsafe_allow_html=True)
        st.markdown("<h3>system vitals</h3>", unsafe_allow_html=True)
        if not metrics_df.empty:
            m_col1, m_col2 = st.columns(2)
            
            with m_col1:
                fig_acc = px.line(metrics_df, x='timestamp', y='accuracy', 
                                  labels={'accuracy': 'accuracy (%)', 'timestamp': ''}, 
                                  color_discrete_sequence=['#8c9a89']) # muted sage
                fig_acc.update_traces(line_shape='spline', line_width=3)
                fig_acc.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', 
                    font=dict(family='Inter', color='#4a4a4a', size=12),
                    xaxis=dict(showgrid=False, zeroline=False, showline=False, showticklabels=False, title=''),
                    yaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.05)', zeroline=False, showline=False, range=[99.0, 100.0]),
                    margin=dict(l=0, r=0, t=20, b=0)
                )
                st.plotly_chart(fig_acc, use_container_width=True, key=f"acc_chart_{i}")
                
            with m_col2:
                fig_esc = px.line(metrics_df, x='timestamp', y='escalation_rate', 
                                  labels={'escalation_rate': 'escalation to tier 2 (%)', 'timestamp': ''},
                                  color_discrete_sequence=['#cc7766']) # terracotta
                fig_esc.update_traces(line_shape='spline', line_width=3)
                fig_esc.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', 
                    font=dict(family='Inter', color='#4a4a4a', size=12),
                    xaxis=dict(showgrid=False, zeroline=False, showline=False, showticklabels=False, title=''),
                    yaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.05)', zeroline=False, showline=False, range=[0.5, 1.5]),
                    margin=dict(l=0, r=0, t=20, b=0)
                )
                st.plotly_chart(fig_esc, use_container_width=True, key=f"esc_chart_{i}")
        else:
            st.markdown("<p style='color: #888; font-style: italic;'>awaiting telemetry data...</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
    with placeholder_traffic.container():
        st.markdown("<div class='tactile-panel'>", unsafe_allow_html=True)
        st.markdown("<h3>live traffic feed</h3>", unsafe_allow_html=True)
        if not traffic_df.empty:
            st.dataframe(traffic_df, use_container_width=True, hide_index=True)
        else:
            st.markdown("<p style='color: #888; font-style: italic;'>listening for traffic...</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
    with placeholder_radar.container():
        st.markdown("<div class='tactile-panel'>", unsafe_allow_html=True)
        st.markdown("<h3>decision drivers (feature importances)</h3>", unsafe_allow_html=True)
        
        try:
            import numpy as np
            bench_data = np.load("data/processed/benchmark_dataset.npz")
            from sklearn.ensemble import RandomForestClassifier
            rf_driver = RandomForestClassifier(n_estimators=30, max_depth=5, random_state=42)
            rf_driver.fit(bench_data["ftp_only_X_train"][:1000], bench_data["ftp_only_y_train"][:1000])
            from src.utils.feature_guard import TIER1_ALLOWLIST
            top_indices = np.argsort(-rf_driver.feature_importances_)[:6]
            categories = [TIER1_ALLOWLIST[idx].lower() for idx in top_indices]
            r_values = [float(rf_driver.feature_importances_[idx]) for idx in top_indices]
        except Exception:
            categories = ['fwd packet length mean', 'flow bytes/s', 'bwd packet length max', 'flow packets/s', 'fwd iat mean', 'ack flag count']
            r_values = [0.28, 0.22, 0.18, 0.14, 0.10, 0.08]

        fig_radar = go.Figure(data=go.Scatterpolar(
          r=r_values,
          theta=categories,
          fill='toself',
          fillcolor='rgba(214, 181, 94, 0.3)',
          line=dict(color='#d6b55e', width=2)
        ))

        fig_radar.update_layout(
          polar=dict(
            radialaxis=dict(visible=False, range=[0, max(r_values) * 1.2]),
            angularaxis=dict(tickfont=dict(family='Inter', color='#4a4a4a', size=11), linecolor='rgba(0,0,0,0)')
          ),
          showlegend=False,
          paper_bgcolor='rgba(0,0,0,0)',
          plot_bgcolor='rgba(0,0,0,0)',
          margin=dict(l=40, r=40, t=20, b=20)
        )
        st.plotly_chart(fig_radar, use_container_width=True, key=f"radar_chart_{i}")
        st.markdown("</div>", unsafe_allow_html=True)
        
    with placeholder_sandbox.container():
        st.markdown("<h3>deep inspection queue</h3>", unsafe_allow_html=True)
        if not escalated_df.empty:
            st.markdown(f"<p style='color: #cc7766; margin-bottom: 15px;'>{len(escalated_df)} payloads analyzing...</p>", unsafe_allow_html=True)
            for index, row in escalated_df.iterrows():
                st.markdown(f"""
                <div class='breathing-panel'>
                    <p style='margin: 0; font-size: 0.9em;'><strong>source:</strong> {row['ip_source']}</p>
                    <p style='margin: 0; font-size: 0.9em; color: #888;'>confidence: {row['tier1_confidence']}</p>
                    <p style='margin: 5px 0 0 0; font-size: 0.85em; font-style: italic;'>reading pe structure...</p>
                </div>
                """, unsafe_allow_html=True)
            st.toast("tier 2 verdict reached. feedback triggered.")
        else:
            st.markdown("<p style='color: #888; font-style: italic;'>queue clear</p>", unsafe_allow_html=True)

    time.sleep(1.5)
