import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['figure.figsize'] = (10,5)
df = pd.read_csv("delhi_weather_aqi_2025.csv")
print(df.head())
print(df.info())

df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'])
df = df.dropna(subset=['temp_c', 'aqi_index', 'pm2_5', 'pm10'])
df['date_only'] = df['datetime'].dt.date
df['month'] = df['datetime'].dt.month
df['year'] = df['datetime'].dt.year

daily = df.groupby('date_only').agg({
    'temp_c':'mean',
    'aqi_index':'mean',
    'pm2_5':'mean',
    'pm10':'mean'
}).reset_index()

daily['month'] = pd.to_datetime(daily['date_only']).dt.month

monthly = daily.groupby('month').agg({
    'temp_c':'mean',
    'aqi_index':'mean',
    'pm2_5':'mean',
    'pm10':'mean'
}).reset_index()

print(monthly)

print("\nDescriptive Statistics:\n")
print(daily.describe())

plt.plot(monthly['month'], monthly['temp_c'], marker='o')
plt.title("Monthly Average Temperature – Delhi (2025)")
plt.xlabel("Month")
plt.ylabel("Temperature (°C)")
plt.grid(True)
plt.show()

plt.plot(monthly['month'], monthly['aqi_index'], label='AQI')
plt.plot(monthly['month'], monthly['pm2_5'], label='PM2.5')
plt.plot(monthly['month'], monthly['pm10'], label='PM10')
plt.title("Monthly AQI and Pollution Levels – 2025")
plt.xlabel("Month")
plt.ylabel("Concentration / Index")
plt.legend()
plt.grid(True)
plt.show()

plt.scatter(daily['temp_c'], daily['aqi_index'], alpha=0.5)
plt.title("Temperature vs AQI – Delhi 2025")
plt.xlabel("Temperature (°C)")
plt.ylabel("AQI")
plt.grid(True)
plt.show()

corr = daily[['temp_c','aqi_index','pm2_5','pm10']].corr()
print("\nCorrelation Matrix:\n")
print(corr)

sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

def season(month):
    if month in [12,1,2]:
        return 'Winter'
    elif month in [3,4,5]:
        return 'Pre-Summer'
    elif month in [6,7,8]:
        return 'Summer'
    else:
        return 'Post-Monsoon'

daily['season'] = daily['month'].apply(season)

seasonal = daily.groupby('season').mean().reset_index()
print("\nSeasonal Analysis:\n")
print(seasonal)

plt.bar(seasonal['season'], seasonal['temp_c'])
plt.title("Average Temperature by Season – 2025")
plt.ylabel("Temperature (°C)")
plt.show()

plt.bar(seasonal['season'], seasonal['aqi_index'])
plt.title("Average AQI by Season – 2025")
plt.ylabel("AQI")
plt.show()

print("\nKEY INSIGHTS:")
print("Highest average temperature month:", monthly.loc[monthly['temp_c'].idxmax(), 'month'])
print("Lowest average temperature month:", monthly.loc[monthly['temp_c'].idxmin(), 'month'])
print("Worst AQI month:", monthly.loc[monthly['aqi_index'].idxmax(), 'month'])
