import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import pickle
from datetime import datetime, timedelta
from tensorflow.keras.models import load_model
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Demand Forecasting Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    .main-header {
        font-family: 'Inter', sans-serif;
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.3);
        transition: transform 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-5px);
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    .metric-label {
        font-size: 1rem;
        opacity: 0.9;
    }
    .sidebar-header {
        font-family: 'Inter', sans-serif;
        font-size: 1.5rem;
        font-weight: 600;
        color: #2c3e50;
        margin-bottom: 1rem;
        border-bottom: 2px solid #667eea;
        padding-bottom: 0.5rem;
    }
    .insight-box {
        background-color: #e6f3ff;
        border: 1px solid #a8d5ff;
        border-left: 5px solid #667eea;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    .insight-box h3 {
        color: #4a517e;
        margin-top: 0;
        margin-bottom: 0.5rem;
    }
    .insight-box p {
        color: #333;
        line-height: 1.6;
    }
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.5rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'models_loaded' not in st.session_state:
    st.session_state.models_loaded = False
if 'xgb_model' not in st.session_state:
    st.session_state.xgb_model = None
if 'lstm_model' not in st.session_state:
    st.session_state.lstm_model = None
if 'scaler' not in st.session_state:
    st.session_state.scaler = None
if 'model_info' not in st.session_state:
    st.session_state.model_info = {}
if 'df' not in st.session_state:
    st.session_state.df = None
if 'insights' not in st.session_state:
    st.session_state.insights = []
if 'model_choice' not in st.session_state:
    st.session_state.model_choice = "XGBoost (Daily)"

# Utility Functions
@st.cache_data
def load_and_prepare_data():
    """Load and prepare data from Sample_data.xlsx"""
    try:
        df = pd.read_excel('Sample_data.xlsx')
        st.success("✅ Data loaded successfully from Sample_data.xlsx!")
    except FileNotFoundError:
        st.error("❌ Sample_data.xlsx not found. Please ensure the file is in the same directory as the script.")
        return None
    except Exception as e:
        st.error(f"❌ Error loading data: {e}")
        return None
    
    df['DOB'] = pd.to_datetime(df['DOB'])
    df['timestamp'] = df['DOB'] + pd.to_timedelta(df['Hour'], unit='h')
    df['Year'] = df['DOB'].dt.year
    df['Month'] = df['DOB'].dt.month
    df['Quarter'] = df['DOB'].dt.quarter
    df['weekday'] = df['DOB'].dt.day_name()
    df['AM_PM'] = df['Hour'].apply(lambda x: 'AM' if x < 12 else 'PM')
    df['Hour_12'] = df['Hour'].apply(lambda x: 12 if x == 12 or x == 0 else x % 12)
    df['Hour_AMPM'] = df['Hour_12'].astype(str) + ' ' + df['AM_PM']
    df['Quarter_Year'] = 'Q' + df['Quarter'].astype(str) + ' ' + df['Year'].astype(str)
    return df

def prepare_xgb_data(df):
    """Prepare data for XGBoost model, aligned with model.py"""
    df_daily = df.set_index('timestamp').resample('D')['Demand in Kgs'].sum()
    df_xgb = df_daily.to_frame()
    
    df_xgb['dayofweek'] = df_xgb.index.dayofweek
    df_xgb['is_weekend'] = df_xgb['dayofweek'].isin([5, 6]).astype(int)
    df_xgb['month'] = df_xgb.index.month
    df_xgb['quarter'] = df_xgb.index.quarter
    df_xgb['lag_1'] = df_xgb['Demand in Kgs'].shift(1)
    df_xgb['lag_2'] = df_xgb['Demand in Kgs'].shift(2)
    df_xgb['lag_7'] = df_xgb['Demand in Kgs'].shift(7)
    df_xgb['rolling_mean_3'] = df_xgb['Demand in Kgs'].shift(1).rolling(window=3).mean()
    df_xgb['rolling_mean_7'] = df_xgb['Demand in Kgs'].shift(1).rolling(window=7).mean()
    df_xgb['rolling_std_7'] = df_xgb['Demand in Kgs'].shift(1).rolling(window=7).std()
    
    df_xgb.dropna(inplace=True)
    return df_xgb

def prepare_lstm_data(df, scaler, window_size=24):
    """Prepare data for LSTM model, aligned with model.py"""
    df_hourly = df.set_index('timestamp').sort_index()['Demand in Kgs']
    
    if scaler is None:
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        scaler.fit(df_hourly.values.reshape(-1, 1))
        st.session_state.scaler = scaler
    
    scaled_data = scaler.transform(df_hourly.values.reshape(-1, 1))
    
    X = []
    if len(scaled_data) >= window_size:
        for i in range(len(scaled_data) - window_size):
            X.append(scaled_data[i:i+window_size])
    else:
        st.warning(f"Not enough data to create LSTM sequences. Need at least {window_size} hours of data.")
        return np.array([]), window_size
    
    X = np.array(X).reshape((len(X), window_size, 1))
    return X, window_size

def generate_xgb_forecast(model, df_xgb, feature_columns, days=7):
    """Generate XGBoost forecasts, aligned with model.py"""
    if df_xgb.empty:
        st.error("XGBoost data is empty. Cannot generate forecast.")
        return [], []
    
    last_row = df_xgb.iloc[-1].copy()
    forecasts = []
    forecast_dates = []
    
    for i in range(days):
        next_date = last_row.name + pd.Timedelta(days=1)
        current_features = {
            'dayofweek': next_date.dayofweek,
            'is_weekend': 1 if next_date.dayofweek in [5, 6] else 0,
            'month': next_date.month,
            'quarter': next_date.quarter,
            'lag_1': last_row['lag_1'],
            'lag_2': last_row['lag_2'],
            'lag_7': last_row['lag_7'] if i < 7 else forecasts[-7] if len(forecasts) >= 7 else last_row['lag_7'],
            'rolling_mean_3': last_row['rolling_mean_3'],
            'rolling_mean_7': last_row['rolling_mean_7'],
            'rolling_std_7': last_row['rolling_std_7']
        }
        
        if i > 0:
            current_features['lag_2'] = last_row['lag_1']
            current_features['lag_1'] = forecasts[-1]
            if len(forecasts) >= 2:
                current_features['rolling_mean_3'] = np.mean(forecasts[-2:] + [last_row['lag_1']])
            if len(forecasts) >= 6:
                current_features['rolling_mean_7'] = np.mean(forecasts[-6:] + [last_row['lag_1']])
        
        features_series = pd.Series(current_features)
        features_to_predict = features_series[feature_columns].values.reshape(1, -1)
        
        forecast = model.predict(features_to_predict)[0]
        forecasts.append(max(0, forecast))
        forecast_dates.append(next_date)
        
        last_row['Demand in Kgs'] = forecast
        last_row['lag_2'] = current_features['lag_2']
        last_row['lag_1'] = current_features['lag_1']
        last_row['lag_7'] = current_features['lag_7']
        last_row.name = next_date
    
    return forecast_dates, forecasts

def generate_lstm_forecast(model, scaler, last_sequence, window_size, hours=24, last_timestamp=None):
    """Generate LSTM forecasts, aligned with model.py"""
    if last_sequence.size == 0:
        st.error("LSTM sequence is empty. Cannot generate forecast.")
        return [], []
    
    forecasts = []
    forecast_timestamps = []
    current_sequence = last_sequence.copy()
    
    if last_timestamp is None:
        last_timestamp = pd.Timestamp.now()
    
    for i in range(hours):
        next_timestamp = last_timestamp + pd.Timedelta(hours=i+1)
        X_pred = current_sequence.reshape(1, window_size, 1)
        forecast_scaled = model.predict(X_pred, verbose=0)[0]
        forecast = scaler.inverse_transform(forecast_scaled.reshape(-1, 1))[0, 0]
        forecasts.append(max(0, forecast))
        forecast_timestamps.append(next_timestamp)
        
        current_sequence = np.roll(current_sequence, -1, axis=0)
        current_sequence[-1] = forecast_scaled
        
    return forecast_timestamps, forecasts

def detect_anomalies(df):
    """Detect demand anomalies using rolling mean and standard deviation"""
    df_daily = df.groupby('DOB')['Demand in Kgs'].sum().reset_index()
    if len(df_daily) >= 7:
        df_daily['rolling_mean'] = df_daily['Demand in Kgs'].rolling(window=7, center=True).mean()
        df_daily['rolling_std'] = df_daily['Demand in Kgs'].rolling(window=7, center=True).std()
        df_daily['z_score'] = df_daily.apply(lambda row: (row['Demand in Kgs'] - row['rolling_mean']) / row['rolling_std'] if row['rolling_std'] != 0 else 0, axis=1)
        df_daily['is_anomaly'] = df_daily['z_score'].abs() > 2.5
    else:
        df_daily['is_anomaly'] = False
    return df_daily

# Page Functions
def show_data_overview(df):
    """Enhanced data overview with comprehensive analytics"""
    st.header("📊 Data Overview & Analysis")
    
    col1, col2, col3, col4 = st.columns(4)
    metrics = [
        ("Total Demand", f"{df['Demand in Kgs'].sum():,.0f} Kg", "📊"),
        ("Average Daily", f"{df.groupby('DOB')['Demand in Kgs'].sum().mean():.1f} Kg", "📈"),
        ("Peak Hour Demand", f"{df['Demand in Kgs'].max():.1f} Kg", "⚡"),
        ("Total Records", f"{len(df):,}", "📝")
    ]
    for i, (label, value, icon) in enumerate(metrics):
        with [col1, col2, col3, col4][i]:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{icon}</div>
                <div class="metric-value">{value}</div>
                <div class="metric-label">{label}</div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")

    st.subheader("📈 Demand Trends")
    col1, col2 = st.columns(2)
    with col1:
        daily_demand = df.groupby('DOB')['Demand in Kgs'].sum().reset_index()
        fig_daily = px.line(daily_demand, x='DOB', y='Demand in Kgs', title="Daily Demand Trend", template="plotly_white")
        fig_daily.update_traces(line_color='#667eea', line_width=2)
        fig_daily.update_layout(title_font_size=16, showlegend=False, height=400)
        st.plotly_chart(fig_daily, use_container_width=True)
    with col2:
        hourly_avg = df.groupby('Hour')['Demand in Kgs'].mean().reset_index()
        fig_hourly = px.bar(hourly_avg, x='Hour', y='Demand in Kgs', title="Average Hourly Demand", template="plotly_white", color='Demand in Kgs', color_continuous_scale='Blues')
        fig_hourly.update_layout(title_font_size=16, showlegend=False, height=400)
        st.plotly_chart(fig_hourly, use_container_width=True)
    
    st.markdown("---")

    st.subheader("🔍 Advanced Analytics")
    col1, col2 = st.columns(2)
    with col1:
        weekly_avg = df.groupby('weekday')['Demand in Kgs'].mean().reset_index()
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        weekly_avg['weekday'] = pd.Categorical(weekly_avg['weekday'], categories=day_order, ordered=True)
        weekly_avg = weekly_avg.sort_values('weekday')
        fig_weekly = px.bar(weekly_avg, x='weekday', y='Demand in Kgs', title="Average Demand by Day of Week", template="plotly_white", color='Demand in Kgs', color_continuous_scale='Viridis')
        fig_weekly.update_layout(title_font_size=16, showlegend=False, height=400, xaxis_tickangle=-45)
        st.plotly_chart(fig_weekly, use_container_width=True)
    with col2:
        df_anomalies = detect_anomalies(df)
        fig_anomaly = go.Figure()
        fig_anomaly.add_trace(go.Scatter(x=df_anomalies['DOB'], y=df_anomalies['Demand in Kgs'], mode='lines', name='Demand', line=dict(color='#667eea')))
        fig_anomaly.add_trace(go.Scatter(x=df_anomalies[df_anomalies['is_anomaly']]['DOB'], y=df_anomalies[df_anomalies['is_anomaly']]['Demand in Kgs'], mode='markers', name='Anomalies', marker=dict(color='red', size=10)))
        fig_anomaly.update_layout(title="Demand Anomalies", template="plotly_white", height=400)
        st.plotly_chart(fig_anomaly, use_container_width=True)

    st.markdown("---")
    st.subheader("📊 Deeper Dive into Demand Patterns")
    col1, col2 = st.columns(2)
    with col1:
        hourly_daily_avg = df.groupby(['weekday', 'Hour'])['Demand in Kgs'].mean().unstack().fillna(0)
        hourly_daily_avg = hourly_daily_avg.reindex(day_order)
        fig_heatmap = px.imshow(
            hourly_daily_avg,
            labels=dict(x="Hour of Day", y="Day of Week", color="Average Demand (Kg)"),
            x=hourly_daily_avg.columns,
            y=hourly_daily_avg.index,
            title="Average Demand Heatmap (Hour vs. Day of Week)",
            aspect="auto",
            color_continuous_scale="Blues",
        )
        fig_heatmap.update_layout(
            xaxis_title="Hour of Day",
            yaxis_title="Day of Week",
            title_font_size=16,
            height=400,
            coloraxis_colorbar=dict(title="Avg Demand (Kg)")
        )
        st.plotly_chart(fig_heatmap, use_container_width=True)

    with col2:
        fig_boxplot = px.box(df, x='Month', y='Demand in Kgs', 
                             title="Demand Distribution by Month",
                             template="plotly_white",
                             color_discrete_sequence=px.colors.qualitative.Pastel)
        fig_boxplot.update_layout(xaxis_title="Month", yaxis_title="Demand (Kg)", title_font_size=16, height=400)
        st.plotly_chart(fig_boxplot, use_container_width=True)

    st.markdown("---")
    st.subheader("📈 Demand Volume per Hour & Month (By Year)")
    hourly_summary = df.groupby(['Year', 'Month', 'Hour_AMPM'])['Demand in Kgs'].sum().reset_index()
    hour_order = sorted(df['Hour_AMPM'].unique(), key=lambda x: int(x.split()[0]) + (0 if x.endswith('AM') else 12))
    hourly_summary['Hour_AMPM'] = pd.Categorical(hourly_summary['Hour_AMPM'], categories=hour_order, ordered=True)
    fig = px.density_heatmap(
        hourly_summary,
        x='Hour_AMPM',
        y='Month',
        z='Demand in Kgs',
        facet_col='Year',
        color_continuous_scale='Blues',
        title='📈 Demand Volume per Hour & Month (By Year)',
        labels={'Demand in Kgs': 'Total Demand (Kgs)', 'Hour_AMPM': 'Hour'}
    )
    fig.update_layout(height=500, width=1000)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("📊 Total Demand per Quarter (2019–2021)")
    quarterly_sum = df.groupby('Quarter_Year')['Demand in Kgs'].sum().reset_index()
    fig_quarterly = px.bar(
        quarterly_sum,
        x='Demand in Kgs',
        y='Quarter_Year',
        text='Demand in Kgs',
        title='📊 Total Demand per Quarter (2019–2021)',
        labels={'Quarter_Year': 'Quarter', 'Demand in Kgs': 'Total Demand (Kgs)'},
        template="plotly_white"
    )
    fig_quarterly.update_traces(
        texttemplate='%{text:.0f}',
        textposition='outside',
        marker_color='#667eea'
    )
    fig_quarterly.update_layout(
        xaxis_tickangle=-45,
        title_font_size=16,
        height=400,
        showlegend=False
    )
    st.plotly_chart(fig_quarterly, use_container_width=True)

def show_business_insights(df):
    """Display business insights"""
    st.header("💡 Business Insights")
    
    all_insights = []
    total_demand = df['Demand in Kgs'].sum()
    avg_daily_demand = df.groupby('DOB')['Demand in Kgs'].sum().mean()
    all_insights.append({
        'title': "Overall Demand Performance",
        'insight': f"The total recorded demand is **{total_demand:,.0f} Kg**, with an average daily demand of **{avg_daily_demand:.1f} Kg**. This provides a baseline for operational planning.",
        'action': "Use these figures to set high-level production targets and assess overall resource allocation."
    })

    weekly_avg = df.groupby('weekday')['Demand in Kgs'].mean().reset_index()
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    weekly_avg['weekday'] = pd.Categorical(weekly_avg['weekday'], categories=day_order, ordered=True)
    weekly_avg = weekly_avg.sort_values('weekday')
    peak_day = weekly_avg.loc[weekly_avg['Demand in Kgs'].idxmax(), 'weekday']
    peak_day_demand = weekly_avg['Demand in Kgs'].max()
    low_day = weekly_avg.loc[weekly_avg['Demand in Kgs'].idxmin(), 'weekday']
    low_day_demand = weekly_avg['Demand in Kgs'].min()
    all_insights.append({
        'title': "Weekly Demand Cycle",
        'insight': f"Demand consistently peaks on **{peak_day}s** with an average of **{peak_day_demand:.1f} kg**, and is lowest on **{low_day}s** with **{low_day_demand:.1f} kg**.",
        'action': f"Strategically increase staffing and inventory on **{peak_day}s** to meet peak demand, and consider optimizing resources or running promotions on **{low_day}s**."
    })

    hourly_avg = df.groupby('Hour')['Demand in Kgs'].mean().reset_index()
    peak_hour = hourly_avg.loc[hourly_avg['Demand in Kgs'].idxmax(), 'Hour']
    peak_hour_demand = hourly_avg['Demand in Kgs'].max()
    low_hour = hourly_avg.loc[hourly_avg['Demand in Kgs'].idxmin(), 'Hour']
    low_hour_demand = hourly_avg['Demand in Kgs'].min()
    all_insights.append({
        'title': "Intra-Day Demand Variations",
        'insight': f"The highest demand occurs typically around **{peak_hour}:00** with **{peak_hour_demand:.1f} kg**, while demand is lowest around **{low_hour}:00**.",
        'action': f"Adjust shift schedules and resource availability to match these hourly fluctuations. For example, ensure maximum staff presence around **{peak_hour}:00**."
    })

    monthly_avg = df.groupby('Month')['Demand in Kgs'].sum().reset_index()
    monthly_avg['Month_Name'] = monthly_avg['Month'].apply(lambda x: datetime(2000, x, 1).strftime('%B'))
    peak_month = monthly_avg.loc[monthly_avg['Demand in Kgs'].idxmax(), 'Month_Name']
    peak_month_demand = monthly_avg['Demand in Kgs'].max()
    low_month = monthly_avg.loc[monthly_avg['Demand in Kgs'].idxmin(), 'Month_Name']
    low_month_demand = monthly_avg['Demand in Kgs'].min()
    all_insights.append({
        'title': "Seasonal Demand Trends",
        'insight': f"Demand is highest in **{peak_month}** ({peak_month_demand:,.0f} kg) and lowest in **{low_month}** ({low_month_demand:,.0f} kg), indicating strong seasonality.",
        'action': f"Plan inventory levels and marketing campaigns according to these seasonal patterns. Stock up before **{peak_month}** and consider promotions during **{low_month}**."
    })

    df_anomalies = detect_anomalies(df)
    anomaly_count = df_anomalies['is_anomaly'].sum()
    if anomaly_count > 0:
        anomaly_dates = df_anomalies[df_anomalies['is_anomaly']]['DOB'].dt.strftime('%Y-%m-%d').tolist()
        insight_text = f"Detected **{anomaly_count}** unusual demand spikes. These occurred on dates such as: {', '.join(anomaly_dates[:5])}."
        action_text = "Investigate these dates for external factors (e.g., holidays, special events, marketing campaigns, or data entry errors) that might have caused the spikes."
    else:
        insight_text = "No significant demand anomalies detected within the analyzed period, indicating relatively stable demand patterns."
        action_text = "Continue monitoring for unusual demand shifts."
    all_insights.append({
        'title': "Demand Anomalies",
        'insight': insight_text,
        'action': action_text
    })

    for insight in all_insights:
        st.markdown(f"""
        <div class="insight-box">
            <h3>{insight['title']}</h3>
            <p><strong>Insight:</strong> {insight['insight']}</p>
            <p><strong>Action:</strong> {insight['action']}</p>
        </div>
        """, unsafe_allow_html=True)

def show_forecasting(df):
    """Advanced forecasting interface with custom single-point prediction display"""
    st.header("🔮 Demand Forecasting")
    
    if not st.session_state.models_loaded:
        st.warning("⚠️ Please load models first in the **Model Management** section to enable forecasting.")
        return
    
    model_choice = st.session_state.model_choice
    st.info(f"Using **{model_choice}** model for forecasting. Change the model in the sidebar if needed.")

    if st.session_state.model_info:
        st.subheader("Current Model Information")
        col_m1, col_m2, col_m3 = st.columns([2, 2, 1.5])  # Adjusted column widths: 2:2:1.5 for larger image
        with col_m1:
            if 'xgb' in st.session_state.model_info:
                st.markdown(f"**XGBoost Model Details:**")
                test_metrics = st.session_state.model_info['xgb'].get('test_metrics', {})
                st.write(f"Features Used: {len(st.session_state.model_info['xgb'].get('feature_columns', []))} features")
                st.write(f"MAE: {test_metrics.get('MAE', 'N/A'):.2f}")
                st.write(f"RMSE: {test_metrics.get('RMSE', 'N/A'):.2f}")
                st.write(f"MAPE: {test_metrics.get('MAPE', 'N/A'):.2f}%")
            else:
                st.info("XGBoost model info not available.")
        with col_m2:
            if 'lstm' in st.session_state.model_info:
                st.markdown(f"**LSTM Model Details:**")
                test_metrics = st.session_state.model_info['lstm'].get('test_metrics', {})
                st.write(f"Window Size: {st.session_state.model_info['lstm'].get('window_size', 'N/A')} hours")
                st.write(f"MAE: {test_metrics.get('MAE', 'N/A'):.2f}")
                st.write(f"RMSE: {test_metrics.get('RMSE', 'N/A'):.2f}")
                st.write(f"MAPE: {test_metrics.get('MAPE', 'N/A'):.2f}%")
            else:
                st.info("LSTM model info not available.")
        with col_m3:
            try:
                st.image('ChatGPT-Image-17-يونيو-2025،-12_12_17-ص.gif', use_container_width=True, width=250)
            except FileNotFoundError:
                st.error("❌ Image 'ChatGPT-Image-17-يونيو-2025،-12_12_17-ص.gif' not found. Please ensure the file is in the same directory as the script.")
    
    st.markdown("---")

    st.subheader("📍 Single-Point Demand Prediction")
    st.markdown("Enter a specific date and time to get a demand prediction using the selected model.")

    col_date, col_time = st.columns(2)
    with col_date:
        prediction_date = st.date_input(
            "Select Prediction Date",
            value=df['DOB'].max().date()
        )
    with col_time:
        if model_choice == "XGBoost (Daily)":
            st.info("XGBoost provides daily predictions. Time input is ignored.")
            prediction_hour = None
        else:
            prediction_hour = st.selectbox("Select Prediction Hour", options=list(range(0, 24)), index=0)

    if st.button("Predict Demand", key="btn_single_predict"):
        with st.spinner(f"Generating {model_choice} prediction..."):
            try:
                if model_choice == "XGBoost (Daily)":
                    if st.session_state.xgb_model is None:
                        st.error("XGBoost model not loaded.")
                    else:
                        df_xgb = prepare_xgb_data(df)
                        if df_xgb.empty:
                            st.error("Not enough historical data to generate XGBoost features.")
                        else:
                            mask = (df['DOB'].dt.date == prediction_date)
                            if mask.any():
                                day_sum = df.loc[mask, 'Demand in Kgs'].sum()
                                st.markdown(
                                    f"<b>✅ Total real demand on {prediction_date}: {day_sum:.2f} Kgs (from history)</b>",
                                    unsafe_allow_html=True
                                )
                            else:
                                last_date = df_xgb.index[-1].date()
                                if prediction_date > last_date:
                                    days_diff = (pd.to_datetime(prediction_date) - df_xgb.index[-1]).days
                                    if days_diff > 0:
                                        last_row = df_xgb.iloc[-1].copy()
                                        temp_row = last_row.copy()
                                        forecast = None
                                        for i in range(1, days_diff + 1):
                                            temp_date = df_xgb.index[-1] + pd.Timedelta(days=i)
                                            temp_features = {
                                                'dayofweek': temp_date.dayofweek,
                                                'is_weekend': 1 if temp_date.dayofweek in [5, 6] else 0,
                                                'month': temp_date.month,
                                                'quarter': temp_date.quarter,
                                                'lag_1': temp_row['lag_1'],
                                                'lag_2': temp_row['lag_2'],
                                                'lag_7': temp_row['lag_7'] if i < 7 else temp_row['lag_7'],
                                                'rolling_mean_3': temp_row['rolling_mean_3'],
                                                'rolling_mean_7': temp_row['rolling_mean_7'],
                                                'rolling_std_7': temp_row['rolling_std_7']
                                            }
                                            temp_series = pd.Series(temp_features)
                                            features_to_predict = temp_series[st.session_state.model_info['xgb']['feature_columns']].values.reshape(1, -1)
                                            forecast = st.session_state.xgb_model.predict(features_to_predict)[0]
                                            forecast = max(0, forecast)
                                            temp_row['lag_2'] = temp_row['lag_1']
                                            temp_row['lag_1'] = forecast
                                            temp_row['Demand in Kgs'] = forecast
                                            if i >= 2:
                                                temp_row['rolling_mean_3'] = np.mean([temp_row['lag_1'], temp_row['lag_2'], last_row['lag_1']])
                                            if i >= 6:
                                                temp_row['rolling_mean_7'] = np.mean([temp_row['Demand in Kgs']] + [last_row['lag_1']] * 6)
                                            temp_row['rolling_std_7'] = temp_row['rolling_std_7']
                                        st.markdown(
                                            f"<b>🔮 Demand on {prediction_date} is predicted to be: {forecast:.2f} Kgs</b>",
                                            unsafe_allow_html=True
                                        )
                                else:
                                    st.warning("Data not available for this date. Prediction only available for future dates.")
                else:
                    if st.session_state.lstm_model is None or st.session_state.scaler is None:
                        st.error("LSTM model or scaler not loaded.")
                    else:
                        prediction_datetime = pd.Timestamp.combine(prediction_date, pd.to_datetime(str(prediction_hour) + ":00").time())
                        mask = (df['timestamp'] == prediction_datetime)
                        if mask.any():
                            real_value = df.loc[mask, 'Demand in Kgs'].values[0]
                            st.markdown(
                                f"<b>✅ Real demand at {prediction_datetime}: {real_value:.2f} Kgs (from history)</b>",
                                unsafe_allow_html=True
                            )
                        else:
                            last_timestamp = df['timestamp'].max()
                            if prediction_datetime > last_timestamp:
                                df_hourly = df.set_index('timestamp').sort_index()['Demand in Kgs']
                                window_size = st.session_state.model_info['lstm']['window_size']
                                if len(df_hourly) < window_size:
                                    st.error(f"Not enough data to create LSTM sequences with window size {window_size}.")
                                else:
                                    hours_diff = int((prediction_datetime - last_timestamp).total_seconds() // 3600)
                                    if hours_diff > 0:
                                        last_sequence = df_hourly.values[-window_size:].reshape(-1, 1)
                                        last_sequence_scaled = st.session_state.scaler.transform(last_sequence)
                                        current_sequence = last_sequence_scaled.copy()
                                        for i in range(hours_diff):
                                            X_pred = current_sequence.reshape(1, window_size, 1)
                                            forecast_scaled = st.session_state.lstm_model.predict(X_pred, verbose=0)[0]
                                            forecast = st.session_state.scaler.inverse_transform(forecast_scaled.reshape(-1, 1))[0, 0]
                                            current_sequence = np.roll(current_sequence, -1, axis=0)
                                            current_sequence[-1] = forecast_scaled
                                        st.markdown(
                                            f"<b>🔮 Demand at {prediction_datetime} is predicted to be: {forecast:.2f} Kgs</b>",
                                            unsafe_allow_html=True
                                        )
                            else:
                                st.warning("Data not available for this datetime. Prediction only available for future datetimes.")
            except Exception as e:
                st.error(f"Error generating {model_choice} prediction: {e}")

    st.markdown("---")

    st.subheader("Generate Multi-Period Forecast")
    if model_choice == "XGBoost (Daily)":
        forecast_period = st.slider("Number of Days to Forecast", min_value=1, max_value=30, value=7, key="forecast_period")
        period_label = "Days"
    else:
        forecast_period = st.slider("Number of Hours to Forecast", min_value=1, max_value=168, value=24, key="forecast_period")
        period_label = "Hours"

    if st.button("Generate Forecast", key="btn_forecast"):
        with st.spinner(f"Generating {model_choice} forecast..."):
            try:
                if model_choice == "XGBoost (Daily)":
                    if st.session_state.xgb_model is None:
                        st.error("XGBoost model not loaded.")
                    else:
                        df_xgb = prepare_xgb_data(df)
                        if df_xgb.empty:
                            st.error("Not enough historical data to generate XGBoost features.")
                        else:
                            forecast_dates, forecasts = generate_xgb_forecast(
                                st.session_state.xgb_model,
                                df_xgb,
                                st.session_state.model_info['xgb']['feature_columns'],
                                days=forecast_period
                            )
                            forecast_df = pd.DataFrame({'Date': forecast_dates, 'Forecasted Demand (Kg)': forecasts})
                            st.success("XGBoost forecast generated!")
                            st.dataframe(forecast_df, use_container_width=True)
                            fig_forecast = px.line(
                                forecast_df,
                                x='Date',
                                y='Forecasted Demand (Kg)',
                                title=f"XGBoost Daily Demand Forecast ({forecast_period} {period_label})",
                                template="plotly_white"
                            )
                            fig_forecast.update_traces(line_color='#667eea', line_width=2)
                            st.plotly_chart(fig_forecast, use_container_width=True)
                else:
                    if st.session_state.lstm_model is None or st.session_state.scaler is None:
                        st.error("LSTM model or scaler not loaded.")
                    else:
                        df_hourly = df.set_index('timestamp').sort_index()['Demand in Kgs']
                        window_size = st.session_state.model_info['lstm']['window_size']
                        if len(df_hourly) < window_size:
                            st.error(f"Not enough data to create LSTM sequences with window size {window_size}.")
                        else:
                            last_sequence = df_hourly.values[-window_size:].reshape(-1, 1)
                            last_sequence_scaled = st.session_state.scaler.transform(last_sequence)
                            last_timestamp = df_hourly.index[-1]
                            forecast_timestamps, forecasts = generate_lstm_forecast(
                                st.session_state.lstm_model,
                                st.session_state.scaler,
                                last_sequence_scaled,
                                window_size,
                                hours=forecast_period,
                                last_timestamp=last_timestamp
                            )
                            forecast_df = pd.DataFrame({'Timestamp': forecast_timestamps, 'Forecasted Demand (Kg)': forecasts})
                            st.success("LSTM forecast generated!")
                            st.dataframe(forecast_df, use_container_width=True)
                            fig_forecast = px.line(
                                forecast_df,
                                x='Timestamp',
                                y='Forecasted Demand (Kg)',
                                title=f"LSTM Hourly Demand Forecast ({forecast_period} {period_label})",
                                template="plotly_white"
                            )
                            fig_forecast.update_traces(line_color='#764ba2', line_width=2)
                            st.plotly_chart(fig_forecast, use_container_width=True)
            except Exception as e:
                st.error(f"Error generating {model_choice} forecast: {e}")

def show_model_management(df):
    """Model management interface"""
    st.header("⚙️ Model Management")
    st.markdown("Here you can load pre-trained models or retrain them with the current dataset.")

    st.subheader("📥 Load Pre-trained Models")
    st.info("Loading pre-trained models is faster and suitable for immediate forecasting.")
    if st.button("Load Pre-trained Models"):
        try:
            st.session_state.xgb_model = joblib.load('models/xgb_model.pkl')
            st.session_state.lstm_model = load_model('models/lstm_model.h5')
            st.session_state.scaler = joblib.load('models/scaler.pkl')
            with open('models/model_info.pkl', 'rb') as f:
                st.session_state.model_info = pickle.load(f)
            st.session_state.models_loaded = True
            st.success("✅ Models and associated info loaded successfully!")
        except FileNotFoundError:
            st.error("❌ Model files not found. Ensure 'xgb_model.pkl', 'lstm_model.h5', 'scaler.pkl', and 'model_info.pkl' are in the 'models/' directory.")
        except Exception as e:
            st.error(f"❌ Error loading models: {e}")
    
    st.markdown("---")

    st.subheader("🚀 Retrain Models (Optional)")
    st.warning("Retraining models can take time depending on dataset size and model complexity.")
    st.info("Requires a `model.py` file with `DemandForecastingModels` class.")

    if st.button("Retrain Models"):
        try:
            from model import DemandForecastingModels
            trainer = DemandForecastingModels()
            with st.spinner("Training models..."):
                df_xgb_prepared = trainer.prepare_xgb_data(df)
                if df_xgb_prepared.empty:
                    st.error("Cannot train XGBoost model: not enough data.")
                else:
                    trainer.train_xgb_model(df_xgb_prepared)
                    st.session_state.xgb_model = trainer.xgb_model
                    st.session_state.model_info['xgb'] = trainer.model_info['xgb']
                    st.success("XGBoost model retrained successfully!")

                X_lstm_prepared, y_lstm_prepared, window_size = trainer.prepare_lstm_data(df)
                if X_lstm_prepared.size == 0 or y_lstm_prepared.size == 0:
                    st.error(f"Cannot train LSTM model: not enough data for sequence creation.")
                else:
                    trainer.train_lstm_model(X_lstm_prepared, y_lstm_prepared, window_size)
                    st.session_state.lstm_model = trainer.lstm_model
                    st.session_state.scaler = trainer.scaler
                    st.session_state.model_info['lstm'] = trainer.model_info['lstm']
                    st.success("LSTM model retrained successfully!")

                trainer.save_models()
                st.session_state.models_loaded = True
                st.success("✅ All models retrained and saved successfully!")
        except ImportError:
            st.error("❌ 'model.py' or 'DemandForecastingModels' not found. Please ensure the `model.py` file is in the same directory with the provided code.")
        except Exception as e:
            st.error(f"❌ Error during model retraining: {e}")

# Main Application
def main():
    st.markdown('<div class="main-header">📈 Advanced Demand Forecasting Dashboard</div>', unsafe_allow_html=True)
    st.sidebar.markdown('<div class="sidebar-header">🔧 Dashboard Controls</div>', unsafe_allow_html=True)
    
    if st.session_state.df is None:
        st.session_state.df = load_and_prepare_data()
    
    df = st.session_state.df
    if df is None:
        st.error("❌ Failed to load data from Sample_data.xlsx. Please ensure the file is available.")
        return
    
    page = st.sidebar.selectbox("📍 Navigate to", [
        "📊 Data Overview & Analysis",
        "💡 Business Insights",
        "🔮 Demand Forecasting",
        "⚙️ Model Management"
    ])
    
    st.sidebar.markdown("### 🧠 Model Selection")
    st.session_state.model_choice = st.sidebar.selectbox(
        "Select Forecasting Model",
        ["XGBoost (Daily)", "LSTM (Hourly)"],
        index=["XGBoost (Daily)", "LSTM (Hourly)"].index(st.session_state.model_choice)
    )
    
    st.sidebar.markdown("### 📅 Date Range Filter")
    min_date_data = df['DOB'].min().date()
    max_date_data = df['DOB'].max().date()
    date_range_selected = st.sidebar.date_input(
        "Select Date Range",
        value=[min_date_data, max_date_data],
        min_value=min_date_data,
        max_value=max_date_data
    )
    
    df_filtered = df
    if len(date_range_selected) == 2:
        start_date, end_date = date_range_selected
        df_filtered = df[(df['DOB'].dt.date >= start_date) & (df['DOB'].dt.date <= end_date)]
        if df_filtered.empty:
            st.warning("No data found for the selected date range. Displaying full dataset.")
            df_filtered = df
    
    if page == "📊 Data Overview & Analysis":
        show_data_overview(df_filtered)
    elif page == "💡 Business Insights":
        show_business_insights(df_filtered)
    elif page == "🔮 Demand Forecasting":
        show_forecasting(df)
    else:
        show_model_management(df)

if __name__ == "__main__":
    main()