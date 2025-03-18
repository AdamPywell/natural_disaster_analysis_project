# Climate Change and Its Effects on Natural Disasters  
**CMPT 353 - Data Analysis Project**  
**Adam Pywell | Morgan Hindy | Jack Clarke**  

---

# Natural Disaster Analysis Project  

This project analyzes the relationship between global temperature changes and the frequency of various natural disasters, including wildfires, tsunamis, floods, storms, epidemics, and insect infestations. The goal is to determine whether rising global temperatures correlate with an increase in the occurrence of these events.  

---

## **Project Structure**  

The project is organized into the following directories and files:  

### **Root Directory**  
- **README.md**: This file, provides an overview of the project.  

### **Data Directory**  
Contains all datasets used in the analysis:  
- **disaster_decloration.csv**: Data on disaster declarations.  
- **disasters.csv**: General disaster data.  
- **global_temp.csv**: Global temperature data.  
- **rainfall.csv**: Rainfall data.  
- **tsunamicsv**: Tsunami data.  

### **Analysis Scripts**  
- **epidemic/epidemics_temp.py**: Script for analyzing epidemics and their correlation with temperature.  
- **floods/**:  
  - **dots.py**: Script for visualizing flood occurrences.  
  - **occurances.py**: Script for analyzing flood frequency.  
- **global_temp/avg_temps.py**: Script for calculating and visualizing global temperature averages.  
- **insect/insects.py**: Script for analyzing insect infestations.  
- **rainfall/**:  
  - **count.py**: Script for analyzing rainfall data.  
  - **heatmap.py**: Script for generating heatmaps of rainfall correlations.  
- **storms/storms.py**: Script for analyzing tropical storms and cyclones.  
- **tsunami/**:  
  - **comparison.py**: Script for comparing tsunami occurrences with global temperatures.  
  - **occurances.py**: Script for analyzing tsunami frequency.  
- **wildfire/Wildfire_Analysis.py**: Script for analyzing wildfire data and its correlation with temperature.  

### **Visualizations**  
- **heatmap.png**: Heatmap visualization of rainfall correlations.  

---

## **Getting Started**  

### **Prerequisites**  
- Python 3.x  
- Required Python libraries: pandas, numpy, matplotlib, seaborn

### **Installation**  
1. Clone the repository:  
   ```bash  
   git clone https://github.com/AdamPywell/natural_disaster_analysis_project.git  
   ```  
2. Navigate to the project directory:  
   ```bash  
   cd natural_disaster_analysis_project  
   ```  
3. Install the required dependencies
   - pandas
   - numpy
   - matplotlib
   - seaborn
     
---

## **Hypothesis**  
The increase in global temperatures has a positive correlation with the rate of occurrence of natural disasters.  

---

## **Introduction**  
Natural disasters are increasingly dominating global news cycles, raising questions about whether their frequency is truly rising or if our perception is skewed by media coverage. This project investigates the relationship between rising global temperatures and the occurrence of natural disasters such as wildfires, tsunamis, floods, storms, epidemics, and insect infestations. Using data analysis, statistical methods, and visualizations, we aim to determine whether climate change is contributing to the increased frequency of these events.  

---

## **Project Structure**  
The project is organized into several sections, each focusing on a specific type of natural disaster:  
1. **Hydrometeorological Events**: Wildfires, tsunamis, floods, and rainfall.  
2. **Meteorological Events**: Tropical storms and cyclones.  
3. **Biological Events**: Epidemics and insect infestations.  

Each section includes data collection, cleaning, analysis, and visualization, followed by statistical testing to determine correlations with global temperature changes.  

---

## **Data Sources**  
The datasets used in this project were primarily sourced from **Kaggle** and other publicly available repositories. Below are the key datasets:  
- **Global Temperature Dataset**: [Climate Change: Earth Surface Temperature Data](https://www.kaggle.com/datasets/berkeleyearth/climate-change-earth-surface-temperature-data)  
- **Wildfire Dataset**: [Our World in Data - Wildfires](https://ourworldindata.org/wildfires)  
- **Tsunami Dataset**: [Tsunami Dataset](https://www.kaggle.com/datasets/andrewmvd/tsunami-dataset)  
- **Flood Dataset**: [US Natural Disaster Declarations](https://www.kaggle.com/datasets/headsortails/us-natural-disaster-declarations)  
- **Rainfall Dataset**: [Rainfall in Pakistan](https://www.kaggle.com/datasets/zusmani/rainfall-in-pakistan)  
- **Cyclone Dataset**: [NOAA Cyclone Tracking Dataset](https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/)  
- **Epidemics and Insect Infestation Dataset**: [EOSDIS Natural Disasters](https://www.kaggle.com/datasets/headsortails/all-natural-disasters-1900-2021-eosdis)  

---

## **Methodology**  
1. **Data Collection**: Datasets were collected from publicly available sources, focusing on global temperature anomalies and disaster occurrences.  
2. **Data Cleaning**: Missing or incomplete entries were removed, and data was filtered to align timelines across datasets.  
3. **Data Analysis**:  
   - **Visualizations**: Bar charts, scatter plots, histograms, and heatmaps were used to identify trends.  
   - **Statistical Analysis**: Linear regression and Pearson correlation coefficients were calculated to quantify relationships between temperature anomalies and disaster occurrences.  
4. **Interpretation**: Results were analyzed to determine the strength and significance of correlations.  

---

## **Key Findings**  
### **1. Wildfires**  
- **Trend**: Wildfire counts showed significant variability, with peaks in 2015 and 2020.  
- **Correlation**: A weak positive correlation (r = 0.223) was found between global temperature anomalies and wildfire occurrences.  
- **Visualization**: Scatter plots and bar charts highlighted regions like North America and Australia as hotspots for wildfires.

### **2. Tsunamis**  
- **Trend**: Tsunami occurrences increased notably from the 1980s onward.  
- **Correlation**: A strong positive correlation (r = 0.91, p < 0.05) was observed between global temperatures and tsunami frequency.  
- **Visualization**: Regression lines and scatter plots confirmed the relationship.  

### **3. Floods**  
- **Trend**: Flood occurrences showed a slight upward trend over time.  
- **Correlation**: Higher temperatures were associated with increased flood counts, particularly in recent decades.  
- **Visualization**: Scatter plots with gradient colors illustrated the relationship.  

### **4. Rainfall**  
- **Trend**: Rainfall in Pakistan showed seasonal peaks in July and August.  
- **Correlation**: Weak correlation (r = 0.28) between rainfall and global temperatures.  
- **Visualization**: Heatmaps and bar charts were used to analyze patterns.  

### **5. Tropical Storms**  
- **Trend**: Storm counts increased from the early 20th century, peaking in the 1980s.  
- **Correlation**: A strong positive correlation (r = 0.76, p < 0.05) was found between global temperatures and storm activity.  
- **Visualization**: Polynomial regression curves highlighted trends.  

### **6. Epidemics and Insect Infestations**  
- **Trend**: Epidemic counts increased significantly from the 1960s onward.  
- **Correlation**: Strong positive correlation (r = 0.916, p < 0.05) between global temperatures and epidemic occurrences.  
- **Visualization**: Regression curves and scatter plots confirmed the relationship.  

---

## **Conclusion**  
The analysis supports the hypothesis that rising global temperatures are positively correlated with the frequency of natural disasters. While some relationships were weak, others, such as tsunamis and epidemics, showed strong statistical significance. These findings highlight the potential impact of climate change on natural disaster patterns and underscore the need for further research and mitigation efforts.  

---

## **Project Limitations**  
- **Data Alignment**: Challenges in aligning datasets with different timelines.  
- **Data Completeness**: Missing or incomplete data in some datasets.  
- **Scope**: Limited to specific types of disasters and regions.  

---

## **Future Work**  
- Expand the analysis to include the severity and economic impact of disasters.  
- Investigate regional variations in disaster occurrences.  
- Incorporate additional environmental factors, such as land use and vegetation types.  

---

## **Team Contributions**  
### **Adam Pywell**  
- Organized communication channels and meetings.
- Gathered data sources and files
- Conducted wildfire data analysis.  
- Edited and structured the final report.  

### **Morgan Hindy**  
- Analyzed epidemics, insect infestations, and tropical storms.  
- Performed statistical tests and visualizations.  
- Contributed to data cleaning and transformation.  

### **Jack Clarke**  
- Gathered initial data and created baseline global temperature graphs.  
- Analyzed tsunamis, floods, and rainfall.  
- Managed GitHub repository and ensured report consistency.  

---

## **Bibliography**  
1. Lempert, K. (2023). *Bad News: Negativity Increases Online News Consumption*. Psychology Today.  
2. Lindsey, R., & Dahlman, L. (2024). *Climate Change: Global Temperature*. Climate.gov.  
3. Wikipedia Contributors. (n.d.). *Ocean*. Wikipedia.  
4. Lindsey, R. (n.d.). *Climate Change: Global Sea Level*. Climate.gov.  

---

## **Resources**  
- **Kaggle**: [https://www.kaggle.com/](https://www.kaggle.com/)  
- **NOAA**: [https://www.ncei.noaa.gov/](https://www.ncei.noaa.gov/)  
- **Our World in Data**: [https://ourworldindata.org/](https://ourworldindata.org/)  

--- 

**END OF README**
