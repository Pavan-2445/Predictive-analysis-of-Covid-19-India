import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import joblib
from sklearn.preprocessing import MinMaxScaler
import time
import numpy as np


st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
    
    .main {
        font-family: 'Poppins', sans-serif;
    }
    
    .stApp {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    background-size: 300% 300%;
    background-attachment: fixed;
    animation: gradientShift 10s ease infinite;
    }
    
    .main-header {
        background: linear-gradient(90deg, #ff9a9e, #fad0c4, #a18cd1);
        background-size: 300% 300%;
        animation: gradientShift 6s ease infinite;
        padding: 2rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        text-align: center;
        color: #1f1f1f;
        box-shadow: 0 20px 40px rgba(0,0,0,0.1);
    }
    
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    .metric-card {
        background: black;
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 0.5rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
        transition: all 0.3s ease;
        animation: slideUp 0.6s ease-out;
    }
    
    .metric-card:hover {
        transform: translateY(-5px) scale(1.02);
    }
    
    @keyframes slideUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .risk-indicator {
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        text-align: center;
        font-weight: bold;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.7; }
        100% { opacity: 1; }
    }
    
    .high-risk {
        background: linear-gradient(45deg, #ff4757, #ff3838);
        color: black;
    }
    
    .moderate-risk {
        background: linear-gradient(45deg, #ffa502, #ff6348);
        color: black;
    }
    
    .low-risk {
        background: linear-gradient(45deg, #2ed573, #1e90ff);
        color: black;
    }
    
    .stSelectbox > div > div > div {
        background: black;
        border-radius: 10px;
        border: 2px solid transparent;
        transition: all 0.3s ease;
    }
    
    .stSelectbox > div > div > div:hover {
        border: 2px solid #4ecdc4;
        transform: scale(1.02);
    }
    
    .chart-container {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 1rem;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        animation: fadeIn 0.8s ease-out;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: linear-gradient(45deg, #667eea, #764ba2);
        color: black;
        border-radius: 10px;
        padding: 0.5rem 1rem;
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(45deg, #4ecdc4, #44a08d);
        transform: scale(1.05);
    }
    
    .loading-spinner {
        display: flex;
        justify-content: center;
        align-items: center;
        height: 100px;
    }
    
    .spinner {
        width: 40px;
        height: 40px;
        border: 4px solid #f3f3f3;
        border-top: 4px solid #4ecdc4;
        border-radius: 50%;
        animation: spin 1s linear infinite;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    df = pd.read_csv("Latest Covid-19 India Status.csv")
    df.columns = df.columns.str.strip()
    return df



st.set_page_config(
    page_title="🛡️India COVID-19 Dashboard - 🇮🇳", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Loading spinner
with st.spinner('🔄 Loading dashboard...'):
    time.sleep(1)  # Simulate loading time
    df = load_data()

# Data preprocessing
features = ['Total Cases', 'Active', 'Deaths', 'Death Ratio', 'Discharge Ratio']
scaler = MinMaxScaler()
df_scaled = scaler.fit_transform(df[features])
df['Risk Score'] = (df_scaled * [0.2, 0.2, 0.2, 0.2, 0.2]).sum(axis=1)

# Main header with animation
st.markdown("""
<div class="main-header">
    <h1 style="font-size: 3rem; margin: 0; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);">
         India COVID-19 Predictive Dashboard
    </h1>
    <p style="font-size: 1.2rem; margin: 0.5rem 0 0 0; opacity: 0.9;">
        Risk assessment across Indian states
    </p>
</div>
""", unsafe_allow_html=True)
st.info("""📌 Note: This analysis is based on COVID-19 data collected up to September 2023.
Please use these insights as a reference to stay aware and take thoughtful precautions for the future.
Let’s stay informed, stay safe, and care for the ones we love. ❤️"""
)

# Sidebar with enhanced styling
with st.sidebar:
    st.markdown("### 🎛️ Dashboard Controls")
    
    # Quick stats in sidebar
    total_cases = df['Total Cases'].sum()
    total_deaths = df['Deaths'].sum()
    total_recovered = df['Discharged'].sum()
    
    
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


col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <h3 style="color: #ff6b6b; margin: 0;">Active Cases</h3>
        <h1 style="margin: 0.5rem 0;">{filtered_df['Active'].sum():,}</h1>
        <p style="color: black; margin: 0;">Currently active</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card">
        <h3 style="color: #4ecdc4; margin: 0;">🛡️Recovered</h3>
        <h1 style="margin: 0.5rem 0;">{filtered_df['Discharged'].sum():,}</h1>
        <p style="color: black; margin: 0;">Total recoveries</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card">
        <h3 style="color: #45b7d1; margin: 0;">Total Cases</h3>
        <h1 style="margin: 0.5rem 0;">{filtered_df['Total Cases'].sum():,}</h1>
        <p style="color: black; margin: 0;">Cumulative cases</p>
    </div>
    """, unsafe_allow_html=True)

with col4:
    recovery_rate = (filtered_df['Discharged'].sum() / filtered_df['Total Cases'].sum()) * 100
    st.markdown(f"""
    <div class="metric-card">
        <h3 style="color: #2ed573; margin: 0;">🛡️Recovery Rate</h3>
        <h1 style="margin: 0.5rem 0;">{recovery_rate:.1f}%</h1>
        <p style="color: black; margin: 0;">Overall recovery</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 Cases Analysis", "⚪ Demographics", "🪦 Mortality", "📋 Data Table", "🗺️ Risk Map"])

with tab1:
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    
    # Create interactive bar chart with better styling
    fig1 = px.bar(
        filtered_df.sort_values('Total Cases', ascending=False).head(10),
        x='State/UTs', y='Total Cases',
        title='📈 Top 10 States by Total Cases',
        color='Risk Score',
        color_continuous_scale='Reds',
        text='Total Cases',
        hover_data=['Active', 'Deaths', 'Discharged']
    )
    fig1.update_traces(texttemplate='%{text:,.0f}', textposition='outside')
    fig1.update_layout(
        title_font_size=20,
        xaxis_tickangle=-45,
        height=600,
        showlegend=True,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig1, use_container_width=True)
    
    fig_comparison = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Active Cases Distribution', 'Recovery Rate by State'),
        specs=[[{"type": "scatter"}, {"type": "bar"}]]
    )
    
    # Scatter plot
    fig_comparison.add_trace(
        go.Scatter(
            x=filtered_df['Total Cases'],
            y=filtered_df['Active'],
            mode='markers',
            marker=dict(
                size=filtered_df['Risk Score']*20,
                color=filtered_df['Risk Score'],
                colorscale='Viridis',
                showscale=True
            ),
            text=filtered_df['State/UTs'],
            hovertemplate='<b>%{text}</b><br>Total: %{x}<br>Active: %{y}<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Recovery rate
    recovery_rates = (filtered_df['Discharged'] / filtered_df['Total Cases'] * 100).fillna(0)
    fig_comparison.add_trace(
        go.Bar(
            x=recovery_rates.head(35),
            y=filtered_df['State/UTs'].head(35),
            orientation='h',
            marker_color=recovery_rates.head(35),
            marker_colorscale='Greens'
        ),
        row=1, col=2
    )
    
    fig_comparison.update_layout(height=500, showlegend=False)
    st.plotly_chart(fig_comparison, use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

with tab2:
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    
    fig_corr = px.scatter(
        filtered_df,
        x='Population',
        y='Total Cases',
        size='Risk Score',
        color='Death Ratio',
        hover_name='State/UTs',
        title='Population vs Total Cases Correlation',
        trendline='ols',
        color_continuous_scale='Reds'
    )
    fig_corr.update_layout(height=500)
    st.plotly_chart(fig_corr, use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

with tab3:
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig3 = px.bar(
            filtered_df.sort_values('Deaths', ascending=False).head(10),
            x='State/UTs', y='Deaths',
            title='Deaths by State (Top 10)',
            color='Death Ratio',
            color_continuous_scale='Reds'
        )
        fig3.update_layout(xaxis_tickangle=-45, height=400)
        st.plotly_chart(fig3, use_container_width=True)
    
    with col2:
        
        fig_death_ratio = px.box(
            filtered_df,
            y='Death Ratio',
            title='Death Ratio Distribution',
            points='all'
        )
        fig_death_ratio.update_layout(height=400)
        st.plotly_chart(fig_death_ratio, use_container_width=True)
    
    # Mortality heatmap
    mortality_data = filtered_df.pivot_table(
        values=['Deaths', 'Death Ratio'], 
        index='State/UTs',
        aggfunc='mean'
    ).head(35)
    
    fig_heatmap = px.imshow(
        mortality_data.T,
        title='Mortality Heatmap',
        color_continuous_scale='Reds',
        aspect='auto'
    )

    
    st.plotly_chart(fig_heatmap, use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

with tab4:
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.subheader("📋 Comprehensive State Data")

    sort_by = st.selectbox(
        "Sort by:",
        ['Risk Score', 'Total Cases', 'Deaths', 'Active', 'Discharge Ratio', 'Death Ratio']
    )
    
    display_df = filtered_df[['State/UTs', 'Total Cases', 'Active', 'Deaths', 'Discharged', 
                             'Death Ratio', 'Discharge Ratio', 'Risk Score']].sort_values(
        by=sort_by, ascending=False
    )
    
    def color_risk_score(val):
        if val > 0.5:
            return 'background-color: #ffebee; color: #c62828'
        elif val > 0.3:
            return 'background-color: #fff3e0; color: #f57c00'
        else:
            return 'background-color: #e8f5e8; color: #2e7d32'
    
    styled_df = display_df.style.applymap(color_risk_score, subset=['Risk Score'])
    st.dataframe(styled_df, use_container_width=True, height=400)
    
    st.markdown('</div>', unsafe_allow_html=True)

with tab5:
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    
    # Risk score visualization
    fig_risk = px.treemap(
        filtered_df.head(35),
        path=['State/UTs'],
        values='Total Cases',
        color='Risk Score',
        title='🗺️ Risk Assessment Map (Bubble size = Total Cases)',
        color_continuous_scale='RdYlGn_r'
    )
    fig_risk.update_layout(height=600)
    st.plotly_chart(fig_risk, use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")

# Enhanced state-specific analysis
st.markdown("📍 Detailed State Analysis")

selected_state = st.selectbox(
    "🎯 Select a State/UT for detailed analysis:",
    options=df['State/UTs'].unique(),
    help="Choose a state to view comprehensive analysis and recommendations"
)

if selected_state:
    state_data = df[df['State/UTs'] == selected_state].iloc[0]
    
    # Animated state header
    st.markdown(f"""
    <div class="main-header" style="margin: 1rem 0;">
        <h2 style="margin: 0;">📌 {selected_state} - Detailed Analysis</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Enhanced metrics display
    col1, col2, col3, col4, col5 = st.columns(5)
    
    metrics = [
        ("Total Cases", int(state_data['Total Cases']), "#ff6b6b"),
        ("Active Cases", int(state_data['Active']), "#ffa502"),
        ("Discharged", int(state_data['Discharged']), "#2ed573"),
        ("Deaths", int(state_data['Deaths']), "#ff4757"),
        ("⚠️ Risk Score", f"{state_data['Risk Score']:.3f}", "#764ba2")
    ]
    
    for i, (title, value, color) in enumerate(metrics):
        with [col1, col2, col3, col4, col5][i]:
            st.markdown(f"""
            <div class="metric-card">
                <h4 style="color: {color}; margin: 0;">{title}</h4>
                <h2 style="margin: 0.5rem 0;">{value}</h2>
            </div>
            """, unsafe_allow_html=True)
    
    # State-specific visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        # Case distribution pie chart
        pie_data = pd.DataFrame({
            'Category': ['Active', 'Discharged', 'Deaths'],
            'Count': [state_data['Active'], state_data['Discharged'], state_data['Deaths']],
            'Color': ['#ffa502', '#2ed573', '#ff4757']
        })
        
        fig4 = px.pie(
            pie_data, 
            values='Count', 
            names='Category',
            title=f'📊 Case Distribution - {selected_state}',
            color='Category',
            color_discrete_map={'Active': '#ffa502', 'Discharged': '#2ed573', 'Deaths': '#ff4757'}
        )
        fig4.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig4, use_container_width=True)
    
    with col2:
        # Risk comparison gauge
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = state_data['Risk Score'],
            title = {'text': f"Risk Score - {selected_state}"},
            delta = {'reference': df['Risk Score'].mean()},
            gauge = {
                'axis': {'range': [None, 1]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 0.3], 'color': "lightgreen"},
                    {'range': [0.3, 0.5], 'color': "yellow"},
                    {'range': [0.5, 1], 'color': "red"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 0.5
                }
            }
        ))
        fig_gauge.update_layout(height=400)
        st.plotly_chart(fig_gauge, use_container_width=True)
    
    # Enhanced precautionary measures with animations
    st.markdown("🛡️ Risk Assessment & Recommendations")
    
    risk_score = state_data['Risk Score']
    
    if risk_score > 0.5:
        st.markdown("""
        <div class="risk-indicator high-risk">
            🚨🫨 HIGH RISK ZONE - IMMEDIATE ACTION REQUIRED
        </div>
        """, unsafe_allow_html=True)
        
        recommendations = [
            "😷 **Mandatory mask wearing** in all public spaces",
            "🏠 **Stay at home** unless absolutely necessary",
            "🧼 **Frequent hand sanitization** - every 15 minutes in public",
            "🚫 **Avoid all gatherings** including family events",
            "📱 **Use contactless payments** and delivery services",
            "🏥 **Monitor symptoms daily** and seek immediate medical help if needed",
            "💉 **Get vaccinated/boosted** if eligible"
        ]
        
    elif risk_score > 0.3:
        st.markdown("""
        <div class="risk-indicator moderate-risk">
            ⚠️😊 MODERATE RISK - ENHANCED PRECAUTIONS NEEDED
        </div>
        """, unsafe_allow_html=True)
        
        recommendations = [
            "😷 **Wear masks** in crowded areas and public transport",
            "🧼 **Regular hand hygiene** with soap/sanitizer",
            "📏 **Maintain social distancing** of 6 feet minimum",
            "🏢 **Limit indoor gatherings** to essential only",
            "🌬️ **Ensure good ventilation** in indoor spaces",
            "🛒 **Shop during off-peak hours** when possible"
        ]
        
    else:
        st.markdown("""
        <div class="risk-indicator low-risk">
            ✅😄 LOW RISK - MAINTAIN STANDARD PRECAUTIONS
        </div>
        """, unsafe_allow_html=True)
        
        recommendations = [
            "😷 **Keep masks handy** for crowded situations",
            "🧼 **Regular hand washing** remains important",
            "📱 **Stay updated** with local health guidelines",
            "💪 **Maintain healthy lifestyle** and immunity",
            "🏥 **Complete vaccination schedule** if pending"
        ]
    
    # Display recommendations with icons
    st.markdown("#### 📋 Personalized Action Plan:")
    for rec in recommendations:
        st.markdown(f"• {rec}")
    
    # Additional insights
    with st.expander("📊 Comparative Analysis"):
        national_avg_risk = df['Risk Score'].mean()
        if state_data['Risk Score'] > national_avg_risk:
            difference = ((state_data['Risk Score'] - national_avg_risk) / national_avg_risk) * 100
            st.error(f"⚠️ Risk score is {difference:.1f}% higher than national average ({national_avg_risk:.3f})")
        else:
            difference = ((national_avg_risk - state_data['Risk Score']) / national_avg_risk) * 100
            st.success(f"✅ Risk score is {difference:.1f}% lower than national average ({national_avg_risk:.3f})")
        
        st.info(f"📈 Discharge Rate: {state_data['Discharge Ratio']:.1f}% (National avg: {df['Discharge Ratio'].mean():.1f}%)")
        st.info(f"📉 Death Rate: {state_data['Death Ratio']:.1f}% (National avg: {df['Death Ratio'].mean():.1f}%)")

# Footer with animations
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 2rem; background: rgba(255,255,255,0.1); border-radius: 15px; margin-top: 2rem;">
    <h4>🔬 COVID-19 Analysis India Dashboard</h4>
    <p>Provides predictive insights for informed decision making and crafted with care, so you can make informed decisions and protect the ones you love.</p>
    <p style="opacity: 0.7;">Built With ❤️ by Pavan Sharma</p>
</div>
""", unsafe_allow_html=True)
