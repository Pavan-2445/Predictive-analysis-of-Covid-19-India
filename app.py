import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import joblib
from sklearn.preprocessing import MinMaxScaler
import time
import numpy as np

# Enhanced CSS with vibrant colors and no white backgrounds
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700;800&display=swap');
    
    .main {
        font-family: 'Poppins', sans-serif;
        color: white;
    }
    
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 25%, #f093fb 50%, #f5576c 75%, #4facfe 100%);
        background-size: 400% 400%;
        animation: gradientFlow 15s ease infinite;
        min-height: 100vh;
    }
    
    @keyframes gradientFlow {
        0% { background-position: 0% 50%; }
        25% { background-position: 100% 50%; }
        50% { background-position: 100% 100%; }
        75% { background-position: 0% 100%; }
        100% { background-position: 0% 50%; }
    }
    
    .main-header {
        background: linear-gradient(135deg, #ff006e 0%, #8338ec 25%, #3a86ff 50%, #06ffa5 75%, #ffbe0b 100%);
        background-size: 300% 300%;
        animation: rainbowShift 8s ease infinite;
        padding: 3rem 2rem;
        border-radius: 25px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
        box-shadow: 0 25px 50px rgba(0,0,0,0.3);
        border: 2px solid rgba(255,255,255,0.2);
        position: relative;
        overflow: hidden;
    }
    
    .main-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
        animation: shimmer 3s infinite;
    }
    
    @keyframes shimmer {
        0% { left: -100%; }
        100% { left: 100%; }
    }
    
    @keyframes rainbowShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    .metric-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 15px 35px rgba(0,0,0,0.2);
        border: 1px solid rgba(255,255,255,0.1);
        transition: all 0.4s ease;
        animation: slideUp 0.8s ease-out;
        position: relative;
        overflow: hidden;
    }
    
    .metric-container::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, #ff006e, #8338ec, #3a86ff, #06ffa5, #ffbe0b);
        background-size: 300% 100%;
        animation: colorFlow 4s linear infinite;
    }
    
    @keyframes colorFlow {
        0% { background-position: 0% 0; }
        100% { background-position: 300% 0; }
    }
    
    .metric-container:hover {
        transform: translateY(-10px) scale(1.03);
        box-shadow: 0 25px 50px rgba(0,0,0,0.3);
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    }
    
    .metric-box {
        background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 50%, #fecfef 100%);
        color: #2d3436;
        border-radius: 15px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 10px 25px rgba(0,0,0,0.15);
        transition: all 0.3s ease;
        border: 2px solid transparent;
        height: 120px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    
    .metric-box:hover {
        transform: scale(1.05) rotate(2deg);
        border: 2px solid #ffffff;
        box-shadow: 0 15px 35px rgba(0,0,0,0.25);
    }
    
    .metric-box.cases { background: linear-gradient(135deg, #ff6b6b 0%, #ff8e8e 100%); color: white; }
    .metric-box.recovered { background: linear-gradient(135deg, #4ecdc4 0%, #6ee2d8 100%); color: white; }
    .metric-box.total { background: linear-gradient(135deg, #45b7d1 0%, #96c7ed 100%); color: white; }
    .metric-box.rate { background: linear-gradient(135deg, #2ed573 0%, #7bed9f 100%); color: white; }
    
    @keyframes slideUp {
        from {
            opacity: 0;
            transform: translateY(50px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .risk-indicator {
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        text-align: center;
        font-weight: bold;
        font-size: 1.2rem;
        animation: glow 3s ease-in-out infinite;
        border: 2px solid rgba(255,255,255,0.3);
    }
    
    @keyframes glow {
        0%, 100% { 
            box-shadow: 0 0 20px rgba(255,255,255,0.3);
        }
        50% { 
            box-shadow: 0 0 40px rgba(255,255,255,0.6), 0 0 60px rgba(255,255,255,0.4);
        }
    }
    
    .high-risk {
        background: linear-gradient(135deg, #ff4757 0%, #ff3838 50%, #c0392b 100%);
        color: white;
    }
    
    .moderate-risk {
        background: linear-gradient(135deg, #ffa502 0%, #ff6348 50%, #e55039 100%);
        color: white;
    }
    
    .low-risk {
        background: linear-gradient(135deg, #2ed573 0%, #1e90ff 50%, #3742fa 100%);
        color: white;
    }
    
    .stSelectbox > div > div > div {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border-radius: 12px !important;
        border: 2px solid rgba(255,255,255,0.2) !important;
        transition: all 0.3s ease !important;
    }
    
    .stSelectbox > div > div > div:hover {
        border: 2px solid #06ffa5 !important;
        transform: scale(1.02) !important;
        box-shadow: 0 8px 25px rgba(6,255,165,0.3) !important;
    }
    
    .chart-section {
        background: linear-gradient(135deg, rgba(102,126,234,0.9) 0%, rgba(118,75,162,0.9) 100%);
        backdrop-filter: blur(20px);
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 15px 35px rgba(0,0,0,0.2);
        border: 1px solid rgba(255,255,255,0.1);
        animation: fadeInScale 0.8s ease-out;
        min-height: 500px;
    }
    
    @keyframes fadeInScale {
        from { 
            opacity: 0; 
            transform: scale(0.9);
        }
        to { 
            opacity: 1; 
            transform: scale(1);
        }
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background: linear-gradient(135deg, rgba(0,0,0,0.3) 0%, rgba(0,0,0,0.1) 100%);
        padding: 10px;
        border-radius: 15px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 12px;
        padding: 12px 24px;
        transition: all 0.3s ease;
        border: 2px solid transparent;
        font-weight: 600;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.2);
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #06ffa5 0%, #3a86ff 100%);
        transform: scale(1.05);
        border: 2px solid rgba(255,255,255,0.3);
        box-shadow: 0 10px 30px rgba(6,255,165,0.3);
    }
    
    .sidebar-section {
        background: linear-gradient(135deg, rgba(255,0,110,0.9) 0%, rgba(131,56,236,0.9) 100%);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid rgba(255,255,255,0.2);
        box-shadow: 0 10px 25px rgba(0,0,0,0.2);
    }
    
    .state-header {
        background: linear-gradient(135deg, #8338ec 0%, #3a86ff 50%, #06ffa5 100%);
        background-size: 200% 200%;
        animation: gradientMove 6s ease infinite;
        padding: 2rem;
        border-radius: 20px;
        margin: 1rem 0;
        text-align: center;
        color: white;
        box-shadow: 0 20px 40px rgba(0,0,0,0.2);
        border: 2px solid rgba(255,255,255,0.1);
    }
    
    @keyframes gradientMove {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    .equal-height-container {
        display: flex;
        height: 400px;
        gap: 1rem;
        margin: 1rem 0;
    }
    
    .equal-width-section {
        flex: 1;
        background: linear-gradient(135deg, rgba(102,126,234,0.8) 0%, rgba(118,75,162,0.8) 100%);
        border-radius: 15px;
        padding: 1rem;
        border: 1px solid rgba(255,255,255,0.1);
        box-shadow: 0 10px 25px rgba(0,0,0,0.15);
    }
    
    .loading-spinner {
        display: flex;
        justify-content: center;
        align-items: center;
        height: 100px;
    }
    
    .spinner {
        width: 50px;
        height: 50px;
        border: 5px solid rgba(255,255,255,0.3);
        border-top: 5px solid #06ffa5;
        border-radius: 50%;
        animation: spin 1s linear infinite;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 12px;
    }
    
    ::-webkit-scrollbar-track {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #06ffa5 0%, #3a86ff 100%);
        border-radius: 10px;
        border: 2px solid rgba(255,255,255,0.1);
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #ff006e 0%, #8338ec 100%);
    }
</style>
""", unsafe_allow_html=True)

# Load data with enhanced loading animation
@st.cache_data
def load_data():
    df = pd.read_csv("Latest Covid-19 India Status.csv")
    df.columns = df.columns.str.strip()
    return df

# Initialize app
st.set_page_config(
    page_title="🦠 India COVID-19 Dashboard", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced loading spinner
with st.spinner('🔄 Loading dashboard...'):
    time.sleep(1.5)  # Simulate loading time
    df = load_data()

# Data preprocessing
features = ['Total Cases', 'Active', 'Deaths', 'Death Ratio', 'Discharge Ratio']
scaler = MinMaxScaler()
df_scaled = scaler.fit_transform(df[features])
df['Risk Score'] = (df_scaled * [0.2, 0.2, 0.2, 0.2, 0.2]).sum(axis=1)

# Animated main header
st.markdown("""
<div class="main-header">
    <h1 style="font-size: 3.5rem; margin: 0; text-shadow: 3px 3px 6px rgba(0,0,0,0.4); font-weight: 800;">
        🦠 India COVID-19 Predictive Dashboard
    </h1>
    <p style="font-size: 1.4rem; margin: 1rem 0 0 0; opacity: 0.95; font-weight: 300;">
        ⚡ Real-time monitoring and AI-powered risk assessment across Indian states
    </p>
</div>
""", unsafe_allow_html=True)

# Enhanced sidebar
with st.sidebar:
    st.markdown("""
    <div class="sidebar-section">
        <h3 style="margin: 0 0 1rem 0; color: white;">🎛️ Dashboard Controls</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Quick stats in sidebar
    total_cases = df['Total Cases'].sum()
    total_deaths = df['Deaths'].sum()
    total_recovered = df['Discharged'].sum()
    
    st.markdown(f"""
    <div class="sidebar-section">
        <h4 style="color: white; margin: 0 0 1rem 0;">📊 National Overview</h4>
        <div style="display: flex; flex-direction: column; gap: 0.5rem;">
            <div style="background: rgba(255,255,255,0.1); padding: 0.8rem; border-radius: 8px;">
                <strong style="color: #06ffa5;">Total Cases:</strong><br>{total_cases:,}
            </div>
            <div style="background: rgba(255,255,255,0.1); padding: 0.8rem; border-radius: 8px;">
                <strong style="color: #ff6b6b;">Total Deaths:</strong><br>{total_deaths:,}
            </div>
            <div style="background: rgba(255,255,255,0.1); padding: 0.8rem; border-radius: 8px;">
                <strong style="color: #4ecdc4;">Total Recovered:</strong><br>{total_recovered:,}
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Risk level filter
    risk_filter = st.selectbox(
        "🎯 Filter by Risk Level",
        ["All", "High Risk (>0.5)", "Moderate Risk (0.3-0.5)", "Low Risk (<0.3)"],
        help="Filter states based on their calculated risk scores"
    )

# Apply risk filter
filtered_df = df.copy()
if risk_filter == "High Risk (>0.5)":
    filtered_df = df[df['Risk Score'] > 0.5]
elif risk_filter == "Moderate Risk (0.3-0.5)":
    filtered_df = df[(df['Risk Score'] >= 0.3) & (df['Risk Score'] <= 0.5)]
elif risk_filter == "Low Risk (<0.3)":
    filtered_df = df[df['Risk Score'] < 0.3]

# Enhanced overview metrics with equal spacing
st.markdown('<div class="equal-height-container">', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class="equal-width-section">
        <div class="metric-box cases">
            <h3 style="margin: 0; font-size: 1rem;">🔥 Active Cases</h3>
            <h1 style="margin: 0.5rem 0; font-size: 2rem;">{filtered_df['Active'].sum():,}</h1>
            <p style="margin: 0; opacity: 0.8;">Currently active</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="equal-width-section">
        <div class="metric-box recovered">
            <h3 style="margin: 0; font-size: 1rem;">✅ Recovered</h3>
            <h1 style="margin: 0.5rem 0; font-size: 2rem;">{filtered_df['Discharged'].sum():,}</h1>
            <p style="margin: 0; opacity: 0.8;">Total recoveries</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="equal-width-section">
        <div class="metric-box total">
            <h3 style="margin: 0; font-size: 1rem;">📈 Total Cases</h3>
            <h1 style="margin: 0.5rem 0; font-size: 2rem;">{filtered_df['Total Cases'].sum():,}</h1>
            <p style="margin: 0; opacity: 0.8;">Cumulative cases</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    recovery_rate = (filtered_df['Discharged'].sum() / filtered_df['Total Cases'].sum()) * 100
    st.markdown(f"""
    <div class="equal-width-section">
        <div class="metric-box rate">
            <h3 style="margin: 0; font-size: 1rem;">💪 Recovery Rate</h3>
            <h1 style="margin: 0.5rem 0; font-size: 2rem;">{recovery_rate:.1f}%</h1>
            <p style="margin: 0; opacity: 0.8;">Overall recovery</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# Enhanced tabs with vibrant styling
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Cases Analysis", "🥧 Demographics", "⚰️ Mortality", "📋 Data Table", "🗺️ Risk Map"])

with tab1:
    st.markdown('<div class="chart-section">', unsafe_allow_html=True)
    
    # Create vibrant bar chart
    fig1 = px.bar(
        filtered_df.sort_values('Total Cases', ascending=False).head(15),
        x='State/UTs', y='Total Cases',
        title='🔥 Top 15 States by Total Cases',
        color='Risk Score',
        color_continuous_scale='plasma',
        text='Total Cases',
        hover_data=['Active', 'Deaths', 'Discharged']
    )
    fig1.update_traces(texttemplate='%{text:,.0f}', textposition='outside')
    fig1.update_layout(
        title_font_size=24,
        title_font_color='white',
        xaxis_tickangle=-45,
        height=600,
        showlegend=True,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='white',
        xaxis=dict(color='white'),
        yaxis=dict(color='white')
    )
    st.plotly_chart(fig1, use_container_width=True)
    
    # Equal height comparison charts
    st.markdown('<div class="equal-height-container">', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="equal-width-section">', unsafe_allow_html=True)
        fig_scatter = px.scatter(
            filtered_df,
            x='Total Cases',
            y='Active',
            size='Risk Score',
            color='Risk Score',
            hover_name='State/UTs',
            title='🎯 Active Cases vs Total Cases',
            color_continuous_scale='viridis'
        )
        fig_scatter.update_layout(
            height=350,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='white',
            title_font_color='white'
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="equal-width-section">', unsafe_allow_html=True)
        recovery_rates = (filtered_df['Discharged'] / filtered_df['Total Cases'] * 100).fillna(0)
        fig_recovery = px.bar(
            x=recovery_rates.head(10),
            y=filtered_df['State/UTs'].head(10),
            orientation='h',
            title='💚 Recovery Rates by State',
            color=recovery_rates.head(10),
            color_continuous_scale='greens'
        )
        fig_recovery.update_layout(
            height=350,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='white',
            title_font_color='white'
        )
        st.plotly_chart(fig_recovery, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with tab2:
    st.markdown('<div class="chart-section">', unsafe_allow_html=True)
    
    # Enhanced pie chart with vibrant colors
    fig2 = px.pie(
        filtered_df.head(10), 
        names='State/UTs', 
        values='Population',
        title='🏘️ Population Distribution (Top 10 States)',
        hole=0.4,
        color_discrete_sequence=px.colors.qualitative.Vivid
    )
    fig2.update_traces(
        textposition='inside',
        textinfo='percent+label',
        textfont_color='white',
        hovertemplate='<b>%{label}</b><br>Population: %{value:,.0f}<br>Percentage: %{percent}<extra></extra>'
    )
    fig2.update_layout(
        height=600,
        showlegend=True,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='white',
        title_font_color='white',
        title_font_size=24
    )
    st.plotly_chart(fig2, use_container_width=True)
    
    # Population vs Cases correlation
    fig_corr = px.scatter(
        filtered_df,
        x='Population',
        y='Total Cases',
        size='Risk Score',
        color='Death Ratio',
        hover_name='State/UTs',
        title='📊 Population vs Total Cases Correlation',
        trendline='ols',
        color_continuous_scale='turbo'
    )
    fig_corr.update_layout(
        height=500,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='white',
        title_font_color='white'
    )
    st.plotly_chart(fig_corr, use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

with tab3:
    st.markdown('<div class="chart-section">', unsafe_allow_html=True)
    
    st.markdown('<div class="equal-height-container">', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="equal-width-section">', unsafe_allow_html=True)
        fig3 = px.bar(
            filtered_df.sort_values('Deaths', ascending=False).head(10),
            x='State/UTs', y='Deaths',
            title='⚰️ Deaths by State (Top 10)',
            color='Death Ratio',
            color_continuous_scale='reds'
        )
        fig3.update_layout(
            xaxis_tickangle=-45, 
            height=350,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='white',
            title_font_color='white'
        )
        st.plotly_chart(fig3, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="equal-width-section">', unsafe_allow_html=True)
        fig_death_ratio = px.box(
            filtered_df,
            y='Death Ratio',
            title='📈 Death Ratio Distribution',
            points='all',
            color_discrete_sequence=['#ff6b6b']
        )
        fig_death_ratio.update_layout(
            height=350,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='white',
            title_font_color='white'
        )
        st.plotly_chart(fig_death_ratio, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Mortality heatmap
    mortality_data = filtered_df.pivot_table(
        values=['Deaths', 'Death Ratio'], 
        index='State/UTs',
        aggfunc='mean'
    ).head(15)
    
    fig_heatmap = px.imshow(
        mortality_data.T,
        title='🔥 Mortality Heatmap (Top 15 States)',
        color_continuous_scale='plasma',
        aspect='auto'
    )
    fig_heatmap.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='white',
        title_font_color='white'
    )
    st.plotly_chart(fig_heatmap, use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

with tab4:
    st.markdown('<div class="chart-section">', unsafe_allow_html=True)
    
    st.markdown("### 📋 Comprehensive State Data")
    
    # Sort options
    sort_by = st.selectbox(
        "Sort by:",
        ['Risk Score', 'Total Cases', 'Deaths', 'Active', 'Discharge Ratio', 'Death Ratio']
    )
    
    display_df = filtered_df[['State/UTs', 'Total Cases', 'Active', 'Deaths', 'Discharged', 
                             'Death Ratio', 'Discharge Ratio', 'Risk Score']].sort_values(
        by=sort_by, ascending=False
    )
    
    # Enhanced dataframe display
    st.dataframe(
        display_df,
        use_container_width=True,
        height=500
    )
    
    st.markdown('</div>', unsafe_allow_html=True)

with tab5:
    st.markdown('<div class="chart-section">', unsafe_allow_html=True)
    
    # Enhanced treemap with vibrant colors
    fig_risk = px.treemap(
        filtered_df.head(20),
        path=['State/UTs'],
        values='Total Cases',
        color='Risk Score',
        title='🗺️ Risk Assessment Map (Bubble size = Total Cases)',
        color_continuous_scale='turbo'
    )
    fig_risk.update_layout(
        height=600,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='white',
        title_font_color='white',
        title_font_size=24
    )
    st.plotly_chart(fig_risk, use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
# Footer section
st.markdown("""
<div style="text-align: center; margin-top: 2rem; color: white;">
    <p>📅 Today's Date: Saturday, June 21, 2025</p>
    <p>🦠 Stay Safe and Stay Informed!</p>
</div>
""", unsafe_allow_html=True)
