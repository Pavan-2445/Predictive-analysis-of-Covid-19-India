import pandas as pd
import plotly.express as px
import streamlit as st
import joblib
from sklearn.preprocessing import MinMaxScaler

# Load data
df = pd.read_csv("Latest Covid-19 India Status.csv")
df.columns = df.columns.str.strip() 


features = ['Total Cases', 'Active', 'Deaths', 'Death Ratio', 'Discharge Ratio']
scaler = MinMaxScaler()
df_scaled = scaler.fit_transform(df[features])
df['Risk Score'] = (df_scaled * [0.2, 0.2, 0.2, 0.2, 0.2]).sum(axis=1)


st.set_page_config(page_title="India Disease Outbreak Dashboard", layout="wide")
st.title("🦠 India Disease Outbreak Predictive Dashboard")


st.header("📊 Overall State-wise Summary")

tab1, tab2, tab3, tab4 = st.tabs(["Total Cases", "Population Share", "Deaths", "Overview Table"])

with tab1:
    fig1 = px.bar(df, x='State/UTs', y='Total Cases', title='Total Confirmed Cases by State/UT', text='Total Cases')
    st.plotly_chart(fig1, use_container_width=True)

with tab2:
    fig2 = px.pie(df, names='State/UTs', values='Population', title='Population Share by State/UT')
    st.plotly_chart(fig2, use_container_width=True)

with tab3:
    fig3 = px.bar(df, x='State/UTs', y='Deaths', title='Deaths by State/UT', text='Deaths')
    st.plotly_chart(fig3, use_container_width=True)

with tab4:
    st.dataframe(df[['State/UTs', 'Total Cases', 'Deaths', 'Discharge Ratio', 'Death Ratio', 'Risk Score']]
                 .sort_values(by='Risk Score', ascending=False))


st.header("📍 State/UT Specific Analysis")
selected_state = st.selectbox("Select a State/UT", df['State/UTs'].unique())

if selected_state:
    state_data = df[df['State/UTs'] == selected_state].iloc[0]

    st.subheader(f"📌 Details for {selected_state}")
    st.metric("Total Cases", int(state_data['Total Cases']))
    st.metric("Active Cases", int(state_data['Active']))
    st.metric("Discharged", int(state_data['Discharged']))
    st.metric("Deaths", int(state_data['Deaths']))
    st.metric("Risk Score", f"{state_data['Risk Score']:.2f}")

    pie_data = pd.DataFrame({
        'Category': ['Active', 'Discharged', 'Deaths'],
        'Count': [state_data['Active'], state_data['Discharged'], state_data['Deaths']]
    })
    fig4 = px.pie(pie_data, values='Count', names='Category', title='Case Distribution')
    st.plotly_chart(fig4, use_container_width=True)

    
    st.subheader("🛡️ Precautionary Measures")
    if state_data['Risk Score'] > 0.5:
        st.warning("⚠️ High risk zone.")
        st.info("😷 Wear masks")
        st.info("🧼 Sanitize regularly.")
        st.info("🫂🫂 Avoid public gatherings.")
    elif state_data['Risk Score'] > 0.3:
        st.info("🟠 Moderate risk. Follow standard COVID-19 precautions.")
    else:
        st.success("🟢 Low risk. Stay aware and safe!")
