### **Predictive-analysis-of-Covid-19-India**
**PREDICTIVE ANALYSIS FOR DISEASE OUTBREAK**

# Project Title: Predictive analysis for disease outbreak


**Tech Stack:** 

	Language : Python 3
	Data Manipulation : Numpy, Pandas
	Visualization : Plotly , Scipy, Matplotlib, Seaborn
	ML Scaling : Scikit-learn MinMaxScaler
	Web Framework : Streamlit
	Deployment : Streamlit Community Cloud

 
# ABSTRACT
A data driven interactive dashboard to assess, score and visualize COVID-19 risk score across Indian States. This project employs historical state wise COVID-19 data collected up to September 2023 to build an interactive Streamlit dashboard. Using MinMaxScaler and regression, it assigns normalized risk scores to Indian States, enhancing public awareness. This system facilitates insightful visualization and informed decision-making with an intuitive risk-score gauge, heatmaps and trend charts.


# Objective
	
 * To deliver a user-friendly platform that:
 * Highlights high-risk states based on recent COVID-19 activity.
 * Enables government officials, healthcare professionals, and citizens to track hotspots.
 * Supports proactive planning and response strategies through data visualization.


# DATA SOURCE AND PREPROCESSING :
* SOURCE : Data is collected from the Kaggle
* KEY ATTRIBUTES : State, Total Cases, Active Cases, Recovered , Deaths, Population
* Cleaning Steps:
	- Handled missing values and duplicate values of the data set.
	- Extracted population, Confirmed Cases, Deaths and computed derived attributes that is Incidence rate and mortality rate.
	- Normalized active case counts using MinMaxScaler to generate a risk score between 0-1.

# EXPLORATORY DATA ANALYSIS(EDA) :
* Correlation Analysis : Heatmaps reveal relationships between cases, recoveries, and deaths.
* State Comparison : Treemaps and bar charts compare states by active cases and normalized risk.
* Interactive line charts shoeing total cases by state.


# MODELLING :
	No traditional predictive forecasting was used; instead egression-shaped risk is derived from current infection rates using MinMaxScaler. This risk scoring effectively highlights comparative vulnerabilities across states
 
# RISK SCORING MECHANISM :
Normalized active case values yield a continuous score (0–1), displayed on a traffic-light gauge:
•	🟢 Low Risk (0–0.33)
•	🟡 Moderate Risk (0.34–0.66)
•	🔴 High Risk (0.67–1.0)
This scoring enables straightforward interpretation and comparison.


# INTERACTIVE DASHBOARD FEATURES :
•	State Selector: Choose any Indian state to view current risk and trends.
•	Dynamic Plots: Real-time line charts, treemaps, and metrics update upon selection.
•	Download Option : Export customized plos as image files.
•	Live Deployment :  Acessible via Streamlit at the provided URL

![image](https://github.com/user-attachments/assets/40f47e05-eba0-4c84-ab44-8cb2d2f29134)

# KEY INSIGHTS OF THIS PROJECT :
•	Hotspots Identified
•	Trends Tracked
•	Preventive Guidance that is based on risk whether low, moderate and high users receive state-specific safety advisories.

![image](https://github.com/user-attachments/assets/f91e298e-0ec1-4d37-8d94-a020f3d04402)

# FUTURE ENHANCEMENTS:
•	Integrate vaccination and mobility data for enriched risk modelling.
•	 Automate real-time updates of data via dashboards.
•	Deploy behind CI/CD with containerization.
# CONCLUSION :
	This application demonstrates readiness to create data science solutions with measurable impact. It supports public engagement and policy-level decisions through transparent, real-time analytics. The project reflects a strong commitment to socially relevant, real-world problem-solving using data.

# APPENDIX:
	LIVE APP: 🛡️India COVID-19 Dashboard - 🇮🇳 · Streamlit
	GITHUB REPO: https://github.com/Pavan-2445/Predictive-analysis-of-Covid-19-India
	CONTACT : ksvnspavankumar.24@gmail.com
	LinkedIn: www.linkedin.com/in/pavankumarkaravadi24
 Data Set : curl -L -o ~/Downloads/latest-covid19-india-statewise-data.zip\
  https://www.kaggle.com/api/v1/datasets/download/anandhuh/latest-covid19-india-statewise-data

### **OUTPUTS**
 
# HOME PAGE WITH FILTER BY RISK LEVEL

![image](https://github.com/user-attachments/assets/71181280-03fc-4673-b3cc-2ef506eadc66)

 
# OVERALL CASE ANALYSIS

![image](https://github.com/user-attachments/assets/2426ea02-d92e-437c-abab-ced5bcb3f332)
 

# VISUAL REPRESENTATION OF ACTIVE AND RECOVERY RATES

![image](https://github.com/user-attachments/assets/043b0b37-66f5-421a-8074-22910c5a4748)

 
# DETAILED ANALYSIS OF EACH STATE 

 ![image](https://github.com/user-attachments/assets/ed821daa-9cd5-416f-badc-f4656749531f)


# PREVENTIVE GUIDANCE FOR LOW RISK AREAS

![image](https://github.com/user-attachments/assets/51728d54-5136-4542-874c-f2a04779f61d)

 

# PREVENTIVE GUIDANCE FOR HIGH RISK AREAS

 ![image](https://github.com/user-attachments/assets/8b8a2eb5-aa48-4269-9b66-52e0e67a00a0)


# PREVENTIVE GUIDANCE FOR MODERATE RISK AREAS

![image](https://github.com/user-attachments/assets/15a4399c-8415-4ed4-8a8f-df3708fcea26)

 
# COMPARITIVE ANALYSIS W.R.T NATIONAL AVERAGE

![image](https://github.com/user-attachments/assets/c1ead145-605c-48d1-95c1-92b03cfeaece)
 

# POPULATION VS CASES CORREALTION

![image](https://github.com/user-attachments/assets/0fc7ff8f-0799-4689-847f-2dd3246d3f55)

 
# MORTALITY REPRESENTATION

![image](https://github.com/user-attachments/assets/a05816cc-fdfe-4bed-9582-b30c843e311c)

 
# COMPREHENSIVE STATE DATA

![image](https://github.com/user-attachments/assets/d3ac7a1d-eddb-4cf4-8b11-3a6f16e4d212)

 
# SORT BY FEATURE FOR COMPREHENSIVE DATA

![image](https://github.com/user-attachments/assets/118566d0-846d-4d6d-9fb7-e7e0ada83a49)

 
# RISK ASSESSMENT MAP

![image](https://github.com/user-attachments/assets/90443116-e237-41fe-a5f1-6d158f7cd893)

### **ACKNOWLEDGMENTS**

I would like to extend my sincere gratitude to the following platforms and tools that made this project possible:

* Kaggle for providing reliable and structured datasets that served as the backbone of this analysis.
* Google Colab – For offering a powerful cloud-based environment to build models, train data, and validate accuracy seamlessly.
* Streamlit for enabling rapid development and seamless deployment of interactive dashboards with minimal effort.
* Python Ecosystem — including powerful libraries such as Pandas, NumPy, Plotly, and Scikit-learn — for providing the essential tools for data manipulation, visualization, and scaling.

This project reflects my commitment to building real-world data science solutions that are both impactful and accessible.
