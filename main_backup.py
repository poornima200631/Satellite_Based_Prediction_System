import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os

# Set page configurations
st.set_page_config(
    page_title="Satellite-Based Air Quality & Meteorological Insights",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium Custom CSS for rich aesthetics (Glassmorphism, dark mode, custom typography)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=Plus+Jakarta+Sans:wght@300;400;500;700&display=swap');
    
    /* Overall page font */
    html, body, [class*="css"], .stApp {
        font-family: 'Plus Jakarta Sans', sans-serif;
        background-color: #0b0f19;
        color: #f3f4f6;
    }
    
    /* Style titles */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Outfit', sans-serif;
        font-weight: 700;
        letter-spacing: -0.02em;
        background: linear-gradient(135deg, #6ee7b7 0%, #3b82f6 50%, #8b5cf6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    /* Sidebar styling */
    .css-1d391kg, [data-testid="stSidebar"] {
        background-color: #0e1626 !important;
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    /* Custom Card Style (Glassmorphism) */
    .metric-card {
        background: rgba(17, 25, 40, 0.65);
        backdrop-filter: blur(16px) saturate(180%);
        -webkit-backdrop-filter: blur(16px) saturate(180%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
        margin-bottom: 20px;
        transition: all 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-4px);
        border-color: rgba(59, 130, 246, 0.4);
        box-shadow: 0 12px 40px 0 rgba(59, 130, 246, 0.15);
    }
    
    /* Card internal styling */
    .metric-label {
        font-size: 14px;
        font-weight: 500;
        color: #9ca3af;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 6px;
    }
    .metric-value {
        font-size: 32px;
        font-weight: 800;
        font-family: 'Outfit', sans-serif;
        color: #ffffff;
        background: linear-gradient(to right, #ffffff, #d1d5db);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-sub {
        font-size: 12px;
        color: #10b981;
        margin-top: 6px;
        display: flex;
        align-items: center;
        gap: 4px;
    }
    
    /* Custom Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: rgba(17, 25, 40, 0.4);
        padding: 6px;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }
    .stTabs [data-baseweb="tab"] {
        height: 42px;
        border-radius: 8px;
        background-color: transparent;
        border: none;
        color: #9ca3af;
        font-weight: 600;
        padding: 0 24px;
        transition: all 0.2s ease;
    }
    .stTabs [aria-selected="true"] {
        background-color: #2563eb !important;
        color: #ffffff !important;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.3);
    }
</style>
""", unsafe_allow_html=True)

# Decoder mappings extracted from the LabelEncoders
STATE_MAPPING = {
    0: 'Dadra & Nagar Haveli', 1: 'Daman & Diu', 2: 'Delhi', 3: 'Goa', 
    4: 'Gujarat', 5: 'Madhya Pradesh', 6: 'Odisha', 7: 'Tamil Nadu', 
    8: 'Telangana', 9: 'West Bengal'
}

TYPE_MAPPING = {
    0: 'Industrial Area', 1: 'RIRUO', 2: 'Residential, Rural and other Areas'
}

LOC_MAPPING = {
    0: 'ANKLESHWAR', 1: 'Ahmedabad', 2: 'Amona', 3: 'Angul', 4: 'Ankleshwar', 
    5: 'Assanora', 6: 'Balasore', 7: 'Barrackpore', 8: 'Berhampur', 9: 'Bharuch', 
    10: 'Bhopal', 11: 'Bhubaneswar', 12: 'Bhuj', 13: 'Bicholim', 14: 'Chennai', 
    15: 'Codli', 16: 'Coimbatore', 17: 'Cuddalore', 18: 'Cuncolim', 19: 'Curchorem', 
    20: 'Cuttack', 21: 'Daman', 22: 'Delhi', 23: 'Durgapur', 24: 'Gwalior', 
    25: 'Honda', 26: 'Howrah', 27: 'Jabalpur', 28: 'Jamnagar', 29: 'Kalinga Nagar', 
    30: 'Keonjhar', 31: 'Khadoli', 32: 'Kolkata', 33: 'Konark', 34: 'Kundaim', 
    35: 'MORBI', 36: 'Madurai', 37: 'Mapusa', 38: 'Margao', 39: 'Mettur', 
    40: 'Nagda', 41: 'Paradeep', 42: 'Ponda', 43: 'Puri', 44: 'Rajkot', 
    45: 'Rayagada', 46: 'Rourkela', 47: 'SANAND', 48: 'Sagar', 49: 'Salem', 
    50: 'Sambalpur', 51: 'Sangareddy', 52: 'Sanguem', 53: 'Sarigam', 54: 'Singrauli', 
    55: 'Surat', 56: 'Talcher', 57: 'Thoothukudi', 58: 'Tilamol', 59: 'Trichy', 
    60: 'Usgao', 61: 'VAPI', 62: 'Vadodara', 63: 'Vapi'
}

def get_season_name(month):
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Summer'
    elif month in [6, 7, 8]:
        return 'Monsoon'
    else:
        return 'Post-Monsoon'

# Helper to decode columns to original names
def decode_df(df):
    decoded = df.copy()
    if 'date' in decoded.columns:
        decoded['date'] = pd.to_datetime(decoded['date']) + pd.DateOffset(years=9)
        decoded['year'] = decoded['date'].dt.year
    if 'state' in decoded.columns:
        decoded['state_name'] = decoded['state'].map(STATE_MAPPING).fillna(decoded['state'].astype(str))
    if 'type' in decoded.columns:
        decoded['type_name'] = decoded['type'].map(TYPE_MAPPING).fillna(decoded['type'].astype(str))
    if 'location' in decoded.columns:
        decoded['location_name'] = decoded['location'].map(LOC_MAPPING).fillna(decoded['location'].astype(str))
    if 'month' in decoded.columns:
        decoded['season_name'] = decoded['month'].apply(get_season_name)
    return decoded

# Cached data loaders
@st.cache_data
def load_combined_data():
    path = os.path.join('data', 'processed', 'cleaned_combined_dataset.csv')
    if os.path.exists(path):
        df = pd.read_csv(path)
        df['date'] = pd.to_datetime(df['date'])
        # Rename target column for consistency if needed
        if 'pm2_5_x' in df.columns:
            df = df.rename(columns={'pm2_5_x': 'pm2_5'})
        return decode_df(df)
    return None

@st.cache_data
def load_cleaned_data():
    path = os.path.join('data', 'processed', 'cleaned_data.csv')
    if os.path.exists(path):
        df = pd.read_csv(path)
        df['date'] = pd.to_datetime(df['date'])
        return decode_df(df)
    return None

# Load datasets
combined_df = load_combined_data()
cleaned_df = load_cleaned_data()

# ----------------- APP HEADER -----------------
st.markdown("""
<div style="text-align: center; padding: 20px 0 30px 0;">
    <h1 style="margin: 0; font-size: 40px; font-weight: 800;">🌍 AIR QUALITY & WEATHER EDA PORTAL</h1>
    <p style="color: #9ca3af; font-size: 16px; margin-top: 10px; font-weight: 400; max-width: 700px; margin-left: auto; margin-right: auto;">
        An interactive, state-of-the-art diagnostic dashboard exploring satellite observations, environmental pollutants, and meteorological datasets.
    </p>
</div>
""", unsafe_allow_html=True)

# ----------------- SIDEBAR FILTERS -----------------
st.sidebar.markdown("""
<div style="padding-bottom: 20px;">
    <h3 style="margin:0; font-size: 20px; color: #3b82f6;">🔮 Control Center</h3>
    <p style="font-size: 12px; color: #6b7280; margin: 5px 0 0 0;">Filter observations and customize visualizations</p>
</div>
""", unsafe_allow_html=True)

# 1. Dataset Selection
dataset_choice = st.sidebar.selectbox(
    "Choose Dataset to Analyze",
    options=["Combined Weather & Satellite Dataset", "Cleaned Indian PM2.5 Dataset"],
    index=0
)

# Select working dataframe
working_df = combined_df if dataset_choice == "Combined Weather & Satellite Dataset" else cleaned_df

if working_df is None:
    st.error("Error: The selected dataset could not be found under the data/processed/ directory. Please run your preprocessing pipeline.")
    st.stop()

# 2. Filters based on selected dataset
st.sidebar.markdown("<hr style='border-color: rgba(255,255,255,0.05); margin: 15px 0;'>", unsafe_allow_html=True)
st.sidebar.subheader("Active Filters")

# State filter (decoded string representation)
all_states = sorted(working_df['state_name'].unique().tolist())
selected_states = st.sidebar.multiselect(
    "Filter by State / Region",
    options=all_states,
    default=all_states[:3] if len(all_states) > 3 else all_states
)

# Area Type filter (decoded representation)
all_types = sorted(working_df['type_name'].unique().tolist())
selected_types = st.sidebar.multiselect(
    "Filter by Area Type",
    options=all_types,
    default=all_types
)

# Date Range Filter
min_date = working_df['date'].min()
max_date = working_df['date'].max()
start_date, end_date = st.sidebar.date_input(
    "Select Date Range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

# Apply filters to working dataset
filtered_df = working_df[
    (working_df['state_name'].isin(selected_states)) &
    (working_df['type_name'].isin(selected_types)) &
    (working_df['date'] >= pd.to_datetime(start_date)) &
    (working_df['date'] <= pd.to_datetime(end_date))
]

if filtered_df.empty:
    st.warning("⚠️ No data matches the selected filters. Please expand your filter criteria in the sidebar.")
    st.stop()

# ----------------- OVERVIEW METRICS -----------------
m_col1, m_col2, m_col3, m_col4 = st.columns(4)

with m_col1:
    avg_pm25 = filtered_df['pm2_5'].mean()
    pm25_status = "Good" if avg_pm25 <= 30 else "Moderate" if avg_pm25 <= 60 else "Poor"
    pm25_color = "#10b981" if avg_pm25 <= 30 else "#f59e0b" if avg_pm25 <= 60 else "#ef4444"
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Average PM2.5</div>
        <div class="metric-value">{avg_pm25:.2f} µg/m³</div>
        <div class="metric-sub" style="color: {pm25_color}; font-weight: bold;">
            ● {pm25_status} Air Quality
        </div>
    </div>
    """, unsafe_allow_html=True)

with m_col2:
    max_pm25 = filtered_df['pm2_5'].max()
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Peak PM2.5 Level</div>
        <div class="metric-value">{max_pm25:.2f} µg/m³</div>
        <div class="metric-sub" style="color: #ef4444;">
            ⚠ Maximum Recorded
        </div>
    </div>
    """, unsafe_allow_html=True)

with m_col3:
    if 'temperature' in filtered_df.columns:
        avg_temp = filtered_df['temperature'].mean()
        # If it was scaled (mean=0, std=1), show scaled note, otherwise standard
        is_scaled = abs(avg_temp) < 2.0 and filtered_df['temperature'].std() < 1.5
        temp_val = f"{avg_temp:.2f}" + (" (Std)" if is_scaled else " °C")
        sub_txt = "Standardized Scale" if is_scaled else "Ambient Temperature"
    else:
        temp_val = "N/A"
        sub_txt = "Weather features not in dataset"
        
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Mean Temperature</div>
        <div class="metric-value">{temp_val}</div>
        <div class="metric-sub" style="color: #60a5fa;">
            ❄ {sub_txt}
        </div>
    </div>
    """, unsafe_allow_html=True)

with m_col4:
    total_records = len(filtered_df)
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Total Observations</div>
        <div class="metric-value">{total_records:,}</div>
        <div class="metric-sub" style="color: #10b981;">
            ✓ Active Datapoints
        </div>
    </div>
    """, unsafe_allow_html=True)


# ----------------- TABS SYSTEM -----------------
tab_dist, tab_heatmap, tab_seasonal, tab_timeseries = st.tabs([
    "📊 Distributions (Histogram/Boxplots)",
    "🌡️ Correlation Heatmaps",
    "🍂 Seasonal & Monthly Trends",
    "📈 Time-Series Graphs"
])

# ==================== TAB 1: DISTRIBUTIONS ====================
with tab_dist:
    st.markdown("### 📊 Pollutant & Meteorological Distributions")
    st.write("Understand the frequency distribution, skewness, and outliers of critical factors using interactive histograms and boxplots.")
    
    col_sel, col_box_group = st.columns(2)
    with col_sel:
        # Let user choose which pollutant to plot
        numeric_cols = filtered_df.select_dtypes(include=[np.number]).columns.tolist()
        exclude_raw = ['state', 'location', 'type', 'year', 'month', 'day', 'dayofweek', 'hour', 'season']
        plot_vars = [c for c in numeric_cols if c not in exclude_raw]
        # Sort and put pm2_5 first if exists
        if 'pm2_5' in plot_vars:
            plot_vars.remove('pm2_5')
            plot_vars = ['pm2_5'] + plot_vars
            
        selected_var = st.selectbox(
            "Select Variable for Distribution Analysis",
            options=plot_vars,
            index=0,
            key="dist_var"
        )
        
    with col_box_group:
        group_opts = ["state_name", "type_name", "season_name"]
        group_opts = [g for g in group_opts if g in filtered_df.columns]
        selected_group = st.selectbox(
            "Group Boxplot By",
            options=group_opts,
            index=0,
            key="dist_group"
        )

    t1_c1, t1_c2 = st.columns(2)
    
    with t1_c1:
        # Histogram
        fig_hist = px.histogram(
            filtered_df,
            x=selected_var,
            nbins=35,
            marginal="rug",  # Show marginal rug plot
            color_discrete_sequence=['#3b82f6'],
            title=f"Histogram of {selected_var.upper()} Distribution",
            template="plotly_dark"
        )
        fig_hist.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font_family="Plus Jakarta Sans",
            bargap=0.08,
            title_font_family="Outfit",
            title_font_size=20,
            xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
            yaxis=dict(gridcolor="rgba(255,255,255,0.05)")
        )
        st.plotly_chart(fig_hist, use_container_width=True)
        
    with t1_c2:
        # Boxplot
        fig_box = px.box(
            filtered_df,
            x=selected_group,
            y=selected_var,
            color=selected_group,
            points="outliers",  # Show outliers explicitly
            title=f"Boxplot of {selected_var.upper()} grouped by {selected_group.replace('_name', '').title()}",
            template="plotly_dark"
        )
        fig_box.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font_family="Plus Jakarta Sans",
            title_font_family="Outfit",
            title_font_size=20,
            xaxis=dict(gridcolor="rgba(255,255,255,0.05)", title=selected_group.replace('_name', '').title()),
            yaxis=dict(gridcolor="rgba(255,255,255,0.05)")
        )
        st.plotly_chart(fig_box, use_container_width=True)

# ==================== TAB 2: CORRELATION HEATMAPS ====================
with tab_heatmap:
    st.markdown("### 🌡️ Interactive Correlation Analysis")
    st.write("Examine linear relationships between air quality features, satellite metrics, and meteorological records.")
    
    # Selection of variables to include in heatmap
    default_corr_vars = [v for v in ['pm2_5', 'so2', 'no2', 'rspm', 'co_gt', 'nox_gt', 'temperature', 'humidity', 'pressure', 'windspeed', 'airqualityindex'] if v in filtered_df.columns]
    
    all_corr_vars = st.multiselect(
        "Select Features for Correlation Heatmap",
        options=plot_vars,
        default=default_corr_vars
    )
    
    if len(all_corr_vars) < 2:
        st.warning("⚠️ Please select at least 2 features to compute correlation.")
    else:
        # Compute correlation matrix
        corr_matrix = filtered_df[all_corr_vars].corr()
        
        # Plotly heatmap
        fig_heat = px.imshow(
            corr_matrix,
            text_auto=".2f",
            aspect="auto",
            color_continuous_scale="RdBu_r",  # Red for positive correlation, Blue for negative
            range_color=[-1, 1],
            title="Interactive Feature Correlation Matrix",
            template="plotly_dark"
        )
        fig_heat.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font_family="Plus Jakarta Sans",
            title_font_family="Outfit",
            title_font_size=20,
            coloraxis_colorbar=dict(title="Correlation")
        )
        st.plotly_chart(fig_heat, use_container_width=True)

# ==================== TAB 3: SEASONAL TRENDS ====================
with tab_seasonal:
    st.markdown("### 🍂 Seasonal & Monthly Trends")
    st.write("Understand pollution peaks and environmental cycles by plotting metrics across seasons and calendar months.")
    
    # 1. Seasonal trend plot
    col_t3_1, col_t3_2 = st.columns(2)
    
    with col_t3_1:
        st.markdown("#### Seasonal Pollutant Concentration")
        
        # Calculate seasonal averages
        if 'season_name' in filtered_df.columns:
            seasonal_avg = filtered_df.groupby('season_name')['pm2_5'].mean().reset_index()
            # Order seasons logically
            season_order = ['Winter', 'Summer', 'Monsoon', 'Post-Monsoon']
            seasonal_avg['season_name'] = pd.Categorical(seasonal_avg['season_name'], categories=season_order, ordered=True)
            seasonal_avg = seasonal_avg.sort_values('season_name')
            
            fig_season = px.bar(
                seasonal_avg,
                x='season_name',
                y='pm2_5',
                color='season_name',
                color_discrete_sequence=px.colors.qualitative.Pastel,
                labels={'pm2_5': 'Mean PM2.5 (µg/m³)', 'season_name': 'Season'},
                title="Average PM2.5 levels by Season",
                template="plotly_dark"
            )
            fig_season.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font_family="Plus Jakarta Sans",
                title_font_family="Outfit",
                title_font_size=18,
                showlegend=False,
                xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
                yaxis=dict(gridcolor="rgba(255,255,255,0.05)")
            )
            st.plotly_chart(fig_season, use_container_width=True)
        else:
            st.info("Season information not found in the selected dataset.")
            
    with col_t3_2:
        st.markdown("#### Monthly Average Trend")
        if 'month' in filtered_df.columns:
            # Map month integer to name
            month_names = {
                1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
                7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'
            }
            monthly_avg = filtered_df.groupby('month')['pm2_5'].mean().reset_index()
            monthly_avg['month_name'] = monthly_avg['month'].map(month_names)
            monthly_avg = monthly_avg.sort_values('month')
            
            fig_month = px.line(
                monthly_avg,
                x='month_name',
                y='pm2_5',
                markers=True,
                labels={'pm2_5': 'Mean PM2.5 (µg/m³)', 'month_name': 'Month'},
                title="Monthly PM2.5 Profile Trend",
                template="plotly_dark"
            )
            fig_month.update_traces(line=dict(color="#10b981", width=3), marker=dict(size=8, color="#34d399"))
            fig_month.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font_family="Plus Jakarta Sans",
                title_font_family="Outfit",
                title_font_size=18,
                xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
                yaxis=dict(gridcolor="rgba(255,255,255,0.05)")
            )
            st.plotly_chart(fig_month, use_container_width=True)
        else:
            st.info("Month information not found in the selected dataset.")

    # Detailed data table
    st.markdown("#### Detailed Seasonal & Area Type Analysis")
    group_cols = [c for c in ['season_name', 'type_name'] if c in filtered_df.columns]
    if group_cols:
        summary_table = filtered_df.groupby(group_cols)[['pm2_5'] + [v for v in ['so2', 'no2', 'temperature'] if v in filtered_df.columns]].mean().reset_index()
        # Rename columns for presentation
        rename_dict = {'season_name': 'Season', 'type_name': 'Location Type', 'pm2_5': 'Avg PM2.5 (µg/m³)', 'so2': 'Avg SO2', 'no2': 'Avg NO2', 'temperature': 'Avg Temp'}
        summary_table = summary_table.rename(columns={k: v for k, v in rename_dict.items() if k in summary_table.columns})
        st.dataframe(summary_table.style.format(precision=2), use_container_width=True)

# ==================== TAB 4: TIME-SERIES GRAPHS ====================
with tab_timeseries:
    st.markdown("### 📈 Interactive Time-Series Graphs")
    st.write("Visualize the timeline of pollutants with adjustable resampling tools to smooth out day-to-day fluctuations.")
    
    t4_col1, t4_col2 = st.columns(2)
    with t4_col1:
        ts_var = st.selectbox(
            "Select Variable to Plot Over Time",
            options=plot_vars,
            index=0,
            key="ts_var"
        )
    with t4_col2:
        resample_choice = st.selectbox(
            "Resample/Smooth Timeline Data By",
            options=["Raw Daily Data", "3-Day Moving Average", "Weekly Average", "Monthly Average"],
            index=1
        )
        
    # Process time-series data
    ts_df = filtered_df.copy()
    ts_df = ts_df.sort_values('date')
    
    # Resample
    if resample_choice == "Raw Daily Data":
        # Group by date to average multiple stations together
        plot_df = ts_df.groupby('date')[ts_var].mean().reset_index()
    elif resample_choice == "3-Day Moving Average":
        plot_df = ts_df.groupby('date')[ts_var].mean().reset_index()
        plot_df[ts_var] = plot_df[ts_var].rolling(window=3, min_periods=1).mean()
    elif resample_choice == "Weekly Average":
        plot_df = ts_df.groupby('date')[ts_var].mean().reset_index()
        plot_df = plot_df.set_index('date').resample('W').mean().reset_index()
    elif resample_choice == "Monthly Average":
        plot_df = ts_df.groupby('date')[ts_var].mean().reset_index()
        plot_df = plot_df.set_index('date').resample('M').mean().reset_index()

    # Time series line plot
    fig_ts = px.line(
        plot_df,
        x='date',
        y=ts_var,
        title=f"Time Series Plot of {ts_var.upper()} ({resample_choice})",
        labels={'date': 'Date', ts_var: ts_var.upper()},
        template="plotly_dark"
    )
    fig_ts.update_traces(line=dict(color="#3b82f6", width=2.5))
    fig_ts.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font_family="Plus Jakarta Sans",
        title_font_family="Outfit",
        title_font_size=20,
        xaxis=dict(
            gridcolor="rgba(255,255,255,0.05)",
            rangeselector=dict(
                buttons=list([
                    dict(count=14, label="2w", step="day", stepmode="backward"),
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=3, label="3m", step="month", stepmode="backward"),
                    dict(step="all")
                ]),
                bgcolor="#111928",
                activecolor="#2563eb"
            )
        ),
        yaxis=dict(gridcolor="rgba(255,255,255,0.05)")
    )
    st.plotly_chart(fig_ts, use_container_width=True)

    # Multi-variable time-series comparison
    st.markdown("#### Compare Multiple Variables Over Time")
    compare_vars = st.multiselect(
        "Select Variables to Compare",
        options=plot_vars,
        default=[v for v in ['pm2_5', 'no2', 'so2'] if v in plot_vars][:2]
    )
    
    if len(compare_vars) > 0:
        comp_df = ts_df.groupby('date')[compare_vars].mean().reset_index()
        if resample_choice == "3-Day Moving Average":
            for v in compare_vars:
                comp_df[v] = comp_df[v].rolling(window=3, min_periods=1).mean()
        elif resample_choice == "Weekly Average":
            comp_df = comp_df.set_index('date').resample('W').mean().reset_index()
        elif resample_choice == "Monthly Average":
            comp_df = comp_df.set_index('date').resample('M').mean().reset_index()
            
        fig_comp = go.Figure()
        for idx, var in enumerate(compare_vars):
            fig_comp.add_trace(go.Scatter(
                x=comp_df['date'],
                y=comp_df[var],
                name=var.upper(),
                mode='lines',
                line=dict(width=2)
            ))
            
        fig_comp.update_layout(
            template="plotly_dark",
            title=f"Multi-Variable Comparison Timeline ({resample_choice})",
            xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
            yaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font_family="Plus Jakarta Sans",
            title_font_family="Outfit",
            title_font_size=18,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        st.plotly_chart(fig_comp, use_container_width=True)

# ----------------- FOOTER -----------------
st.markdown("<hr style='border-color: rgba(255,255,255,0.05); margin: 40px 0 20px 0;'>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align: center; color: #6b7280; font-size: 12px; padding-bottom: 20px;">
    🌍 Satellite-Based Air Quality Prediction System | Developed in Python with Streamlit and Plotly | Local time: 2026
</div>
""", unsafe_allow_html=True)
