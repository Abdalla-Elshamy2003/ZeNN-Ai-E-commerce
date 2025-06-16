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

# Custom CSS for professional styling
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

# Utility Functions
@st.cache_data
def load_and_prepare_data(uploaded_file=None):
    """Load and prepare data with comprehensive preprocessing"""
    try:
        if uploaded_file is not None:
            df = pd.read_excel(uploaded_file)
            st.success("✅ Data loaded successfully from uploaded file!")
        else:
            np.random.seed(42)
            dates = pd.date_range(start='2019-12-01', end='2021-12-31', freq='D')
            sample_data = []
            for date in dates:
                for hour in range(9, 22):  # 9 AM to 9 PM
                    demand = np.random.normal(6, 2.5)  # Random demand around 6 kg
                    sample_data.append({
                        'DOB': date,
                        'DAY_NAME': date.strftime('%A'),
                        'Hour': hour,
                        'Demand in Kgs': max(0, demand)
                    })
            df = pd.DataFrame(sample_data)
            st.info("📊 Using generated sample data for demonstration")
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
    
    # Feature engineering
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
    
    X = np.array(X).reshape((X.shape[0], X.shape[1], 1))
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

def generate_lstm_forecast(model, scaler, last_sequence, window_size, hours=24):
    """Generate LSTM forecasts, aligned with model.py"""
    if last_sequence.size == 0:
        st.error("LSTM sequence is empty. Cannot generate forecast.")
        return [], []
    
    forecasts = []
    forecast_timestamps = []
    current_sequence = last_sequence.copy()
    last_timestamp = last_sequence.name[-1]
    
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
    
    # Metrics Cards
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

    # Demand Trends
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

    # Advanced Analytics
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

    # Deeper Dive into Demand Patterns
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
    """Advanced forecasting interface with model selection"""
    st.header("🔮 Demand Forecasting")
    
    if not st.session_state.models_loaded:
        st.warning("⚠️ Please load models first in the **Model Management** section to enable forecasting.")
        return
    
    st.info("Models loaded successfully. Select a model to generate forecasts.")

    if st.session_state.model_info:
        st.subheader("Current Model Information")
        col_m1, col_m2 = st.columns(2)
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
    
    st.markdown("---")

    st.subheader("Generate Forecast")
    model_choice = st.selectbox("Select Forecasting Model", ["XGBoost (Daily)", "LSTM (Hourly)"], key="model_choice")
    
    if model_choice == "XGBoost (Daily)":
        forecast_period = st.slider("Number of Days to Forecast", min_value=1, max_value=30, value=7, key="forecast_period")
        period_label = "Days"
    else:  # LSTM (Hourly)
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
                else:  # LSTM (Hourly)
                    if st.session_state.lstm_model is None or st.session_state.scaler is None:
                        st.error("LSTM model or scaler not loaded.")
                    else:
                        df_hourly = df.set_index('timestamp').sort_index()['Demand in Kgs']
                        window_size = st.session_state.model_info['lstm']['window_size']
                        if len(df_hourly) < window_size:
                            st.error(f"Not enough data to create LSTM sequences with window size {window_size}.")
                        else:
                            last_sequence = df_hourly.values[-window_size:].reshape(-1, 1)
                            last_sequence.name = df_hourly.index[-window_size:]
                            last_sequence_scaled = st.session_state.scaler.transform(last_sequence)
                            forecast_timestamps, forecasts = generate_lstm_forecast(
                                st.session_state.lstm_model,
                                st.session_state.scaler,
                                last_sequence_scaled,
                                window_size,
                                hours=forecast_period
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
            st.error("❌ 'model.py' or 'DemandForecastingModels' not found. Please ensure the following code is in 'model.py':")
            st.code("""
import pandas as pd
import numpy as np
import pickle
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
import warnings
import os
warnings.filterwarnings('ignore')

class DemandForecastingModels:
    def __init__(self):
        self.xgb_model = None
        self.lstm_model = None
        self.scaler = None
        self.model_info = {}
        
    def load_and_prepare_data(self, file_path=None):
        try:
            if file_path and os.path.exists(file_path):
                df = pd.read_excel(file_path)
                print(f"✅ Data loaded successfully from {file_path}")
            else:
                print("⚠️ Creating sample data...")
                dates = pd.date_range(start='2019-12-01', end='2021-12-31', freq='D')
                sample_data = []
                for date in dates:
                    for hour in range(9, 22):
                        demand = np.random.normal(6, 2.5)
                        sample_data.append({
                            'DOB': date,
                            'DAY_NAME': date.strftime('%A'),
                            'Hour': hour,
                            'Demand in Kgs': max(0, demand)
                        })
                df = pd.DataFrame(sample_data)
        except Exception as e:
            print(f"❌ Error loading data: {e}")
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
        
        print(f"📊 Data shape: {df.shape}")
        print(f"📅 Date range: {df['DOB'].min()} to {df['DOB'].max()}")
        
        return df
    
    def prepare_xgb_data(self, df):
        print("🔄 Preparing XGBoost data...")
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
        print(f"✅ XGBoost data prepared: {df_xgb.shape}")
        
        return df_xgb
    
    def train_xgb_model(self, df_xgb):
        print("🚀 Training XGBoost model...")
        
        feature_columns = ['dayofweek', 'is_weekend', 'month', 'quarter', 
                          'lag_1', 'lag_2', 'lag_7', 'rolling_mean_3', 
                          'rolling_mean_7', 'rolling_std_7']
        
        X = df_xgb[feature_columns]
        y = df_xgb['Demand in Kgs']
        
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        self.xgb_model = xgb.XGBRegressor(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=6,
            min_child_weight=1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        
        self.xgb_model.fit(X_train, y_train)
        
        y_pred_train = self.xgb_model.predict(X_train)
        y_pred_test = self.xgb_model.predict(X_test)
        
        train_metrics = self.calculate_metrics(y_train, y_pred_train)
        test_metrics = self.calculate_metrics(y_test, y_pred_test)
        
        self.model_info['xgb'] = {
            'feature_columns': feature_columns,
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'feature_importance': dict(zip(feature_columns, self.xgb_model.feature_importances_))
        }
        
        print("✅ XGBoost model trained successfully!")
        print(f"📊 Test Metrics - MAE: {test_metrics['MAE']:.2f}, RMSE: {test_metrics['RMSE']:.2f}, MAPE: {test_metrics['MAPE']:.2f}%")
        
        return y_test, y_pred_test
    
    def prepare_lstm_data(self, df, window_size=24):
        print("🔄 Preparing LSTM data...")
        
        df_hourly = df.set_index('timestamp').sort_index()['Demand in Kgs']
        
        self.scaler = MinMaxScaler()
        scaled_data = self.scaler.fit_transform(df_hourly.values.reshape(-1, 1))
        
        X, y = [], []
        for i in range(len(scaled_data) - window_size):
            X.append(scaled_data[i:i+window_size])
            y.append(scaled_data[i+window_size])
        
        X, y = np.array(X), np.array(y)
        X = X.reshape((X.shape[0], X.shape[1], 1))
        
        print(f"✅ LSTM data prepared: X shape {X.shape}, y shape {y.shape}")
        
        return X, y, window_size
    
    def train_lstm_model(self, X, y, window_size):
        print("🚀 Training LSTM model...")
        
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        self.lstm_model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(window_size, 1)),
            LSTM(50, return_sequences=False),
            Dense(25),
            Dense(1)
        ])
        
        self.lstm_model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        
        history = self.lstm_model.fit(
            X_train, y_train,
            epochs=100,
            batch_size=32,
            validation_data=(X_test, y_test),
            callbacks=[early_stopping],
            verbose=1
        )
        
        y_pred_train = self.lstm_model.predict(X_train, verbose=0)
        y_pred_test = self.lstm_model.predict(X_test, verbose=0)
        
        y_train_inv = self.scaler.inverse_transform(y_train)
        y_test_inv = self.scaler.inverse_transform(y_test)
        y_pred_train_inv = self.scaler.inverse_transform(y_pred_train)
        y_pred_test_inv = self.scaler.inverse_transform(y_pred_test)
        
        train_metrics = self.calculate_metrics(y_train_inv, y_pred_train_inv)
        test_metrics = self.calculate_metrics(y_test_inv, y_pred_test_inv)
        
        self.model_info['lstm'] = {
            'window_size': window_size,
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'history': history.history
        }
        
        print("✅ LSTM model trained successfully!")
        print(f"📊 Test Metrics - MAE: {test_metrics['MAE']:.2f}, RMSE: {test_metrics['RMSE']:.2f}, MAPE: {test_metrics['MAPE']:.2f}%")
        
        return y_test_inv, y_pred_test_inv
    
    def calculate_metrics(self, y_true, y_pred):
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        return {'MAE': mae, 'RMSE': rmse, 'MAPE': mape}
    
    def save_models(self, models_dir='models'):
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        
        print(f"💾 Saving models to {models_dir}/...")
        
        if self.xgb_model is not None:
            joblib.dump(self.xgb_model, f'{models_dir}/xgb_model.pkl')
            print("✅ XGBoost model saved")
        
        if self.lstm_model is not None:
            self.lstm_model.save(f'{models_dir}/lstm_model.h5')
            print("✅ LSTM model saved")
        
        if self.scaler is not None:
            joblib.dump(self.scaler, f'{models_dir}/scaler.pkl')
            print("✅ Scaler saved")
        
        with open(f'{models_dir}/model_info.pkl', 'wb') as f:
            pickle.dump(self.model_info, f)
        print("✅ Model info saved")
        
        print("🎉 All models and components saved successfully!")
    
    def generate_sample_forecasts(self, df):
        print("🔮 Generating sample forecasts...")
        
        if self.xgb_model is not None:
            df_xgb = self.prepare_xgb_data(df)
            last_row = df_xgb.iloc[-1].copy()
            
            xgb_forecasts = []
            for i in range(7):
                features = last_row[self.model_info['xgb']['feature_columns']].values.reshape(1, -1)
                forecast = self.xgb_model.predict(features)[0]
                xgb_forecasts.append(forecast)
                
                last_row['lag_2'] = last_row['lag_1']
                last_row['lag_1'] = forecast
            
            print(f"📈 XGBoost 7-day forecast: {[f'{x:.2f}' for x in xgb_forecasts]}")
        
        if self.lstm_model is not None and self.scaler is not None:
            df_hourly = df.set_index('timestamp').sort_index()['Demand in Kgs']
            window_size = self.model_info['lstm']['window_size']
            
            last_sequence = df_hourly.values[-window_size:].reshape(-1, 1)
            last_sequence_scaled = self.scaler.transform(last_sequence)
            
            lstm_forecasts = []
            current_sequence = last_sequence_scaled.copy()
            
            for i in range(24):
                X_pred = current_sequence.reshape(1, window_size, 1)
                forecast_scaled = self.lstm_model.predict(X_pred, verbose=0)[0]
                forecast = self.scaler.inverse_transform(forecast_scaled.reshape(-1, 1))[0, 0]
                lstm_forecasts.append(forecast)
                
                current_sequence = np.roll(current_sequence, -1)
                current_sequence[-1] = forecast_scaled
            
            print(f"📈 LSTM 24-hour forecast: {[f'{x:.2f}' for x in lstm_forecasts[:6]]}... (showing first 6 hours)")
            """)
        except Exception as e:
            st.error(f"❌ Error during model retraining: {e}")

# Main Application
def main():
    st.markdown('<div class="main-header">📈 Advanced Demand Forecasting Dashboard</div>', unsafe_allow_html=True)
    st.sidebar.markdown('<div class="sidebar-header">🔧 Dashboard Controls</div>', unsafe_allow_html=True)
    
    uploaded_file = st.sidebar.file_uploader("📁 Upload Excel File", type=['xlsx', 'xls'])
    if uploaded_file is not None:
        st.session_state.df = load_and_prepare_data(uploaded_file)
        st.session_state.models_loaded = False
        st.session_state.insights = []
        st.session_state.xgb_model = None
        st.session_state.lstm_model = None
        st.session_state.scaler = None
        st.session_state.model_info = {}
    elif st.session_state.df is None:
        st.session_state.df = load_and_prepare_data(None)
    
    df = st.session_state.df
    if df is None:
        st.error("❌ Failed to load data. Please upload a valid Excel file.")
        return
    
    page = st.sidebar.selectbox("📍 Navigate to", [
        "📊 Data Overview & Analysis",
        "💡 Business Insights",
        "🔮 Demand Forecasting",
        "⚙️ Model Management"
    ])
    
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