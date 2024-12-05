import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

data = pd.read_csv('climate_change_impact_on_agriculture.csv')

data = data.dropna(subset=['Year'])
data['Year'] = pd.to_numeric(data['Year'], errors='coerce')

data_numeric = data.select_dtypes(include=[np.number])

trend_data = data.groupby('Year').mean(numeric_only=True)

crop_type_counts = data['Crop_Type'].value_counts()
adaptation_strategy_counts = data['Adaptation_Strategies'].value_counts()
region_counts = data['Region'].value_counts()

# 1. Heatmap
correlation_matrix = data_numeric.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
plt.title("Correlation Heatmap")
plt.show()

time_series_data = data.groupby('Year', as_index=False).mean(numeric_only=True)

#Economic Impact Over Time
plt.figure(figsize=(12, 6))
plt.plot(time_series_data['Year'], time_series_data['Economic_Impact_Million_USD'], label='Economic Impact (Million USD)', color='blue', marker='o')
plt.title("Economic Impact Over Time", fontsize=16)
plt.xlabel("Year", fontsize=12)
plt.ylabel("Economic Impact (Million USD)", fontsize=12)
plt.grid(alpha=0.5)
plt.legend()
plt.tight_layout()
plt.show()

#Average Temperature Over Time
plt.figure(figsize=(12, 6))
plt.plot(time_series_data['Year'], time_series_data['Average_Temperature_C'], label='Average Temperature (°C)', color='red', marker='s')
plt.title("Average Temperature Over Time", fontsize=16)
plt.xlabel("Year", fontsize=12)
plt.ylabel("Temperature (°C)", fontsize=12)
plt.grid(alpha=0.5)
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(trend_data.index, trend_data['Total_Precipitation_mm'], label='Total Precipitation (mm)', color='blue')
plt.title("Trend of Total Precipitation Over Time")
plt.xlabel("Year")
plt.ylabel("Total Precipitation (mm)")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(trend_data.index, trend_data['Soil_Health_Index'], label='Soil Health Index', color='green')
plt.title("Trend of Soil Health Index Over Time")
plt.xlabel("Year")
plt.ylabel("Soil Health Index")
plt.legend()
plt.grid(True)
plt.show()

#Line Plot: Year vs. Crop Yield
plt.figure(figsize=(10, 6))
sns.lineplot(data=data, x="Year", y="Crop_Yield_MT_per_HA", ci=None)
plt.title("Crop Yield Over Time")
plt.xlabel("Year")
plt.ylabel("Crop Yield (MT/HA)")
plt.grid(alpha=0.5)
plt.show()

data['Crop_Yield_Category'] = pd.cut(data['Crop_Yield_MT_per_HA'], bins=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
# Calculate the average Economic Impact by Crop Yield Category
avg_economic_impact = data.groupby('Crop_Yield_Category')['Economic_Impact_Million_USD'].mean()

#bar chart
plt.figure(figsize=(10, 6))
avg_economic_impact.plot(kind='bar', color='skyblue', edgecolor='black')
plt.title("Average Economic Impact by Crop Yield Category", fontsize=14)
plt.xlabel("Crop Yield Category", fontsize=12)
plt.ylabel("Average Economic Impact (Million USD)", fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.xticks(rotation=0)
plt.show()


#Economic Impact and Irrigation Access
crop_economic_impact = data.groupby('Crop_Type')['Economic_Impact_Million_USD'].mean().sort_values(ascending=False)
country_irrigation_access = data.groupby('Country')['Irrigation_Access_%'].mean().sort_values(ascending=False)

plt.figure(figsize=(12, 6))
crop_economic_impact.plot(kind='bar', color='skyblue', edgecolor='black')
plt.title("Average Economic Impact by Crop Type")
plt.xlabel("Crop Type")
plt.ylabel("Economic Impact (Million USD)")
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

plt.figure(figsize=(12, 6))
country_irrigation_access.plot(kind='bar', color='orange', edgecolor='black')
plt.title("Average Irrigation Access by Country")
plt.xlabel("Country")
plt.ylabel("Irrigation Access (%)")
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

#Soil Health by Region
region_soil_health_data = data.groupby("Region")["Soil_Health_Index"].mean().sort_values()
plt.figure(figsize=(10, 6))
region_soil_health_data.plot(kind="bar", color="chocolate", edgecolor="black")
plt.title("Soil Health Index by Region")
plt.xlabel("Region")
plt.ylabel("Average Soil Health Index")
plt.xticks(rotation=90)
plt.grid(axis="y", alpha=0.5)
plt.tight_layout()
plt.show()

#Extreme Weather by Region
extreme_weather_region_data = data.groupby("Region")["Extreme_Weather_Events"].sum().sort_values()
plt.figure(figsize=(10, 6))
extreme_weather_region_data.plot(kind="bar", color="salmon", edgecolor="black")
plt.title("Total Extreme Weather Events by Region")
plt.xlabel("Region")
plt.ylabel("Number of Events")
plt.xticks(rotation=90)
plt.grid(axis="y", alpha=0.5)
plt.tight_layout()
plt.show()

#Year vs. Extreme Weather Events
yearly_weather_data = data.groupby("Year", as_index=False).mean(numeric_only=True)
plt.figure(figsize=(10, 6))
sns.barplot(data=yearly_weather_data, x="Year", y="Extreme_Weather_Events", color="blue")
plt.xticks(rotation=45)
plt.title("Year vs. Extreme Weather Events")
plt.xlabel("Year")
plt.ylabel("Extreme Weather Events")
plt.grid(axis="y", alpha=0.5)
plt.show()

#Country vs. Crop Yield
plt.figure(figsize=(10, 6))
country_yield_data = data.groupby("Country", as_index=False).mean(numeric_only=True)
sns.barplot(data=country_yield_data, x="Country", y="Crop_Yield_MT_per_HA", ci=None, color="skyblue")
plt.xticks(rotation=90)
plt.title("Country vs. Crop Yield")
plt.xlabel("Country")
plt.ylabel("Crop Yield (MT/HA)")
plt.grid(axis="y", alpha=0.5)
plt.tight_layout()
plt.show()

#Region vs. Economic Impact
plt.figure(figsize=(10, 6))
region_impact_data = data.groupby("Region", as_index=False).mean(numeric_only=True)
sns.barplot(data=region_impact_data, x="Region", y="Economic_Impact_Million_USD", ci=None, color="lightgreen")
plt.xticks(rotation=90)
plt.title("Region vs. Economic Impact")
plt.xlabel("Region")
plt.ylabel("Economic Impact (Million USD)")
plt.grid(axis="y", alpha=0.5)
plt.tight_layout()
plt.show()

#Adaptation Strategies vs. Economic Impact
adaptation_data = data.groupby("Adaptation_Strategies", as_index=False).mean(numeric_only=True)
plt.figure(figsize=(10, 6))
sns.barplot(data=adaptation_data, x="Adaptation_Strategies", y="Economic_Impact_Million_USD", ci=None, color="orange")
plt.xticks(rotation=45)
plt.title("Adaptation Strategies vs. Economic Impact")
plt.xlabel("Adaptation Strategies")
plt.ylabel("Economic Impact (Million USD)")
plt.grid(axis="y", alpha=0.5)
plt.tight_layout()
plt.show()

#Histograms: Precipitation
plt.figure(figsize=(12, 6))
plt.hist(data['Total_Precipitation_mm'], bins=20, color='steelblue', edgecolor='black', alpha=0.8)
plt.title("Distribution of Total Precipitation (mm)", fontsize=16, weight='bold')
plt.xlabel("Total Precipitation (mm)", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.xticks(ticks=range(0, 3100, 500), fontsize=10)
plt.yticks(fontsize=10)
plt.grid(axis='y', linestyle='--', alpha=0.3)
plt.tight_layout()
plt.show()

#distribution of the Soil Health Index
plt.figure(figsize=(10, 6))
plt.hist(data['Soil_Health_Index'], bins=15, color='green', edgecolor='black', alpha=0.7)
plt.title("Distribution of Soil Health Index", fontsize=14)
plt.xlabel("Soil Health Index", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

#Pie Charts
# Proportion of Crop Types
plt.figure(figsize=(12, 12))
explode_crop = [0.1 if i == max(crop_type_counts.values) else 0 for i in crop_type_counts.values]
plt.pie(crop_type_counts.values, labels=crop_type_counts.index, autopct='%1.1f%%', startangle=140, explode=explode_crop)
plt.title("Proportion of Crop Types", fontsize=16, weight='bold')
plt.tight_layout()
plt.show()

# Proportion of Adaptation Strategies
plt.figure(figsize=(12, 12))
explode_adaptation = [0.1 if i == max(adaptation_strategy_counts.values) else 0 for i in adaptation_strategy_counts.values]
plt.pie(adaptation_strategy_counts.values, labels=adaptation_strategy_counts.index, autopct='%1.1f%%', startangle=140, explode=explode_adaptation)
plt.title("Proportion of Adaptation Strategies", fontsize=16, weight='bold')
plt.tight_layout()
plt.show()

# Proportion of Regions
plt.figure(figsize=(12, 12))
explode_region = [0.1 if i == max(region_counts.values) else 0 for i in region_counts.values]
plt.pie(region_counts.values, labels=region_counts.index, autopct='%1.1f%%', startangle=140, explode=explode_region)
plt.title("Proportion of Regions", fontsize=16, weight='bold')
plt.tight_layout()
plt.show()

#boxplots
sns.boxplot(x='Crop_Type',y='Average_Temperature_C',data=data,palette='muted')
plt.xticks(rotation=45)
plt.title('Average Temperature distribution for each crop type')
plt.show()

sns.boxplot(x='Crop_Type',y='Soil_Health_Index',data=data, palette='muted')
plt.title('Soil Health Index for each crop type')
plt.xticks(rotation=45)
plt.show()

sns.boxplot(x='Country',y='CO2_Emissions_MT',data=data,palette='muted')
plt.xticks(rotation=45)
plt.title("CO2 Emissions for each Country")
plt.show()

# 7. Violin Plot
plt.figure(figsize=(12, 6))
sns.violinplot(x='Crop_Type', y='Irrigation_Access_%', data=data, palette='muted')
plt.title("Irrigation Access by Crop Type")
plt.xlabel("Crop Type")
plt.ylabel("Irrigation Access (%)")
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# 8. Scatter Plots
plt.figure(figsize=(10, 6))
plt.scatter(data['CO2_Emissions_MT'], data['Crop_Yield_MT_per_HA'], alpha=0.7, edgecolor='none')
plt.title("CO2 Emissions vs. Crop Yield")
plt.xlabel("CO2 Emissions (MT)")
plt.ylabel("Crop Yield (MT/HA)")
plt.grid(alpha=0.5)
plt.show()

plt.figure(figsize=(10, 6))
plt.scatter(data['Soil_Health_Index'], data['Economic_Impact_Million_USD'], alpha=0.7, color='green', edgecolor='none')
plt.title("Soil Health vs. Economic Impact")
plt.xlabel("Soil Health Index")
plt.ylabel("Economic Impact (Million USD)")
plt.grid(alpha=0.5)
plt.show()
#Temperature vs. Crop Yield
plt.figure(figsize=(10, 6))
sns.scatterplot(data=data, x="Average_Temperature_C", y="Crop_Yield_MT_per_HA")
plt.title("Temperature vs. Crop Yield")
plt.xlabel("Average Temperature (°C)")
plt.ylabel("Crop Yield (MT/HA)")
plt.grid(alpha=0.5)
plt.tight_layout()
plt.show()

#Precipitation vs. Crop Yield
plt.figure(figsize=(10, 6))
sns.scatterplot(data=data, x="Total_Precipitation_mm", y="Crop_Yield_MT_per_HA")
plt.title("Precipitation vs. Crop Yield")
plt.xlabel("Total Precipitation (mm)")
plt.ylabel("Crop Yield (MT/HA)")
plt.grid(alpha=0.5)
plt.tight_layout()
plt.show()

#CO2 Emissions vs. Crop Yield
plt.figure(figsize=(10, 6))
sns.scatterplot(data=data, x="CO2_Emissions_MT", y="Crop_Yield_MT_per_HA")
plt.title("CO2 Emissions vs. Crop Yield")
plt.xlabel("CO2 Emissions (MT)")
plt.ylabel("Crop Yield (MT/HA)")
plt.grid(alpha=0.5)
plt.tight_layout()
plt.show()

#Soil Health Index vs. Crop Yield
plt.figure(figsize=(10, 6))
sns.scatterplot(data=data, x="Soil_Health_Index", y="Crop_Yield_MT_per_HA")
plt.title("Soil Health Index vs. Crop Yield")
plt.xlabel("Soil Health Index")
plt.ylabel("Crop Yield (MT/HA)")
plt.grid(alpha=0.5)
plt.tight_layout()
plt.show()

#Irrigation Access vs. Economic Impact
plt.figure(figsize=(10, 6))
sns.scatterplot(data=data, x="Irrigation_Access_%", y="Economic_Impact_Million_USD")
plt.title("Irrigation Access vs. Economic Impact")
plt.xlabel("Irrigation Access (%)")
plt.ylabel("Economic Impact (Million USD)")
plt.grid(alpha=0.5)
plt.tight_layout()
plt.show()

#Economic Impact vs. Fertilizer Use by Crop Type
plt.figure(figsize=(12, 6))
sns.scatterplot(
    data=data, 
    x="Fertilizer_Use_KG_per_HA", 
    y="Economic_Impact_Million_USD", 
    hue="Crop_Type", 
    palette="viridis"
)
plt.title("Economic Impact vs. Fertilizer Use by Crop Type")
plt.xlabel("Fertilizer Use (KG/HA)")
plt.ylabel("Economic Impact (Million USD)")
plt.legend(title="Crop Type", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.grid(alpha=0.5)
plt.tight_layout()
plt.show()
#Pesticide Use vs. Economic Impact
plt.figure(figsize=(10, 6))
sns.scatterplot(data=data, x="Pesticide_Use_KG_per_HA", y="Economic_Impact_Million_USD")
plt.title("Pesticide Use vs. Economic Impact")
plt.xlabel("Pesticide Use (KG/HA)")
plt.ylabel("Economic Impact (Million USD)")
plt.grid(alpha=0.5)
plt.tight_layout()
plt.show()

#Extreme Weather Events vs. Crop Yield
plt.figure(figsize=(10, 6))
sns.scatterplot(data=data, x="Extreme_Weather_Events", y="Crop_Yield_MT_per_HA")
plt.title("Extreme Weather Events vs. Crop Yield")
plt.xlabel("Extreme Weather Events")
plt.ylabel("Crop Yield (MT/HA)")
plt.grid(alpha=0.5)
plt.tight_layout()
plt.show()

#Stacked Bar Chart: Region-Wise Crop Types with CO2 Emissions
region_crop_data = data.groupby(['Region', 'Crop_Type'])['CO2_Emissions_MT'].sum().unstack()
region_crop_data.plot(kind='bar', stacked=True, figsize=(12, 6), cmap='coolwarm')
plt.title("Region-Wise Crop Types with CO2 Emissions")
plt.xlabel("Region")
plt.ylabel("CO2 Emissions (MT)")
plt.legend(title="Crop Type", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=90)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

#CO2 Emissions and Economic Impact by Crop Type
plt.figure(figsize=(12, 6))
sns.barplot(
    data=data, 
    x="Crop_Type", 
    y="CO2_Emissions_MT", 
    hue="Adaptation_Strategies", 
    palette="cubehelix"
)
plt.title("CO2 Emissions and Economic Impact by Crop Type")
plt.xlabel("Crop Type")
plt.ylabel("CO2 Emissions (MT)")
plt.legend(title="Adaptation Strategies", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

#Error Bars: CO2 Emissions vs. Crop Yield
co2_mean = data['CO2_Emissions_MT'].mean()
co2_std = data['CO2_Emissions_MT'].std()
crop_yield_mean = data['Crop_Yield_MT_per_HA'].mean()
crop_yield_std = data['Crop_Yield_MT_per_HA'].std()

plt.figure(figsize=(10, 6))
plt.errorbar(co2_mean, crop_yield_mean, xerr=co2_std, yerr=crop_yield_std, fmt='o', color='red', ecolor='red', capsize=5, label='CO2 vs. Crop Yield')
plt.title("Error Bars: CO2 Emissions vs. Crop Yield")
plt.xlabel("CO2 Emissions (MT)")
plt.ylabel("Crop Yield (MT/HA)")
plt.legend()
plt.grid(alpha=0.5)
plt.show()

#Extreme Weather Events by Country
plt.figure(figsize=(12, 6))
sns.countplot(x='Country', data=data, order=data['Country'].value_counts().index, palette='viridis')
plt.title("Number of Extreme Weather Events by Country")
plt.xlabel("Country")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

#Adaptation Strategies Distribution
plt.figure(figsize=(10, 6))
sns.countplot(data=data, x="Adaptation_Strategies", palette="Set2")
plt.xticks(rotation=45)
plt.title("Adaptation Strategies Distribution")
plt.xlabel("Adaptation Strategies")
plt.ylabel("Count")
plt.grid(axis="y", alpha=0.5)
plt.tight_layout()
plt.show()

#Crop Type vs. Adaptation Strategies
plt.figure(figsize=(12, 6))
sns.countplot(data=data, x="Crop_Type", hue="Adaptation_Strategies", palette="pastel")
plt.xticks(rotation=90)
plt.legend(
    title="Adaptation Strategies",
    loc="upper left",
    bbox_to_anchor=(1.05, 1),  # Move legend completely outside the plot
    borderaxespad=0
)
plt.title("Crop Type vs. Adaptation Strategies")
plt.xlabel("Crop Type")
plt.ylabel("Count")
plt.grid(axis="y", alpha=0.5)
plt.tight_layout()
plt.show()

#Regional Heatmap
regional_data = data.groupby('Region')[['Crop_Yield_MT_per_HA', 'Economic_Impact_Million_USD']].mean()
plt.figure(figsize=(10, 8))
sns.heatmap(regional_data, annot=True, fmt=".2f", cmap="YlGnBu", cbar=True)
plt.title("Regional Heatmap: Average Crop Yield and Economic Impact")
plt.xlabel("Metrics")
plt.ylabel("Regions")
plt.show()

#Extreme Weather Events vs. Economic Impact
plt.figure(figsize=(12, 6))
sns.scatterplot(
    data=data,
    x="Extreme_Weather_Events",
    y="Economic_Impact_Million_USD",
    size="Economic_Impact_Million_USD",
    sizes=(20, 200),
    hue="Economic_Impact_Million_USD",
    palette="coolwarm"
)
plt.legend(
    title="Economic Impact (Million USD)",
    loc="upper left",
    bbox_to_anchor=(1.05, 1),  # Move legend outside the plot
    borderaxespad=0
)
plt.title("Extreme Weather Events vs. Economic Impact")
plt.xlabel("Extreme Weather Events")
plt.ylabel("Economic Impact (Million USD)")
plt.grid(alpha=0.5)
plt.tight_layout()
plt.show()





