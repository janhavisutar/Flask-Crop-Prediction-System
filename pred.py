import tkinter as tk
from tkinter import ttk, messagebox
import requests
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from bs4 import BeautifulSoup
import logging

# Load datasets
crop_df = pd.read_csv('new_dataset_maharashtra.csv')
soil_df = pd.read_csv('sangli_soil_data.csv')

crop_entries = []
soil_entries = []
season_entries = []
combined_soil_entries = []
combined_crop_entries = []
combined_season_entries = []
city_entry = []
temperature_entry = []
humidity_entry = []
rainfall_entry = []

# If 'Season' column is not present in crop dataset, you need to add it.
def determine_season(row):
    temp, rain = row['temperature'], row['rainfall']
    if temp > 30 and rain < 100:
        return 'Summer'
    elif temp < 15 and rain < 50:
        return 'Winter'
    else:
        return 'Rainy'

crop_df['Season'] = crop_df.apply(determine_season, axis=1)

# Instantiate LabelEncoder
label_encoder = LabelEncoder()

# Encode the categorical variable 'Season' in the crop dataset
crop_df['Season'] = label_encoder.fit_transform(crop_df['Season'])



# Prepare the crop data
X_crop = crop_df.drop('label', axis=1)  # Crop features
y_crop = crop_df['label']  # Crop target variable

# Prepare the soil data
X_soil = soil_df.drop('Soil Type', axis=1)  # Soil features
y_soil = soil_df['Soil Type']  # Soil target variable

# Train the crop model
crop_clf = RandomForestClassifier(n_estimators=100, random_state=42)
crop_clf.fit(X_crop, y_crop)

# Train the soil model
soil_clf = RandomForestClassifier(n_estimators=100, random_state=42)
soil_clf.fit(X_soil, y_soil)

# Create a synthetic dataset
def create_synthetic_season_data():
    import numpy as nps
    nps.random.seed(42)
    num_samples = 200
    temperature = nps.random.uniform(0, 40, num_samples)  # Temperature in degrees Celsius
    rainfall = nps.random.uniform(0, 300, num_samples)  # Rainfall in mm
    humidity = nps.random.randint(0, 101, num_samples)  # Humidity as a percentage
    seasons = []

    for temp, rain, hum in zip(temperature, rainfall, humidity):
        if temp > 30 and rain < 100:
            seasons.append('Summer')
        elif temp < 15 and rain < 50:
            seasons.append('Winter')
        else:
            seasons.append('Rainy')

    data = {
        'Temperature': temperature,
        'Rainfall': rainfall,
        'Humidity': humidity,
        'Season': seasons
    }

    return pd.DataFrame(data)

# Generate synthetic season data
season_df = create_synthetic_season_data()

# Prepare the season data
X_season = season_df.drop('Season', axis=1)
y_season = season_df['Season']

# Encode labels if they are categorical
label_encoder = LabelEncoder()
y_season = label_encoder.fit_transform(y_season)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_season, y_season, test_size=0.2, random_state=42)

# Train the season model
season_clf = RandomForestClassifier(n_estimators=100, random_state=42)
season_clf.fit(X_train, y_train)

# Function to predict crop
def predict_crop():
    global X_crop
    from tkinter import ttk, messagebox
    try:
        global crop_entries
        global X_crop
        global crop_clf
        crop_input = [float(entry.get()) for entry in crop_entries]
        crop_input_df = pd.DataFrame([crop_input], columns=X_crop.columns)
        prediction = crop_clf.predict(crop_input_df)
        messagebox.showinfo("Crop Prediction", f"Predicted Crop: {prediction[0]}")
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")

# Function to predict soil type
def predict_soil():
    from tkinter import ttk, messagebox
    try:
        global soil_entries
        global X_soil
        global soil_clf
        soil_input = [float(entry.get()) for entry in soil_entries]
        soil_input_df = pd.DataFrame([soil_input], columns=X_soil.columns)
        prediction = soil_clf.predict(soil_input_df)
        messagebox.showinfo("Soil Prediction", f"Predicted Soil Type: {prediction[0]}")
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")

# Function to predict season with suitable crops
def predict_season_with_crops():
    global season_clf
    global label_encoder
    from tkinter import ttk, messagebox
    try:
        global season_entries
        global season_clf
        global crop_df
        # Get user input for season prediction
        season_input = [float(entry.get()) for entry in season_entries]

        # Make prediction using the trained model
        prediction = season_clf.predict([season_input])
        predicted_season = label_encoder.inverse_transform(prediction)[0]

        # Filter crop dataset to find suitable crops for the predicted season
        suitable_crops = crop_df[crop_df['Season'] == predicted_season]['label'].unique()

        # Show prediction and suitable crops
        messagebox.showinfo("Season Prediction with Suitable Crops", f"Predicted Season: {predicted_season}")
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")

# Function to predict all combined
def predict_all():
    
    from tkinter import ttk, messagebox
    try:
        global season_entries
        global crop_entries
        global soil_entries
        global X_crop
        global X_season
        global X_soil
        global label_encoder

        soil_input = [float(entry.get()) for entry in soil_entries]
        crop_input = [float(entry.get()) for entry in crop_entries]
        season_input = [float(entry.get()) for entry in season_entries]
        
        soil_input_df = pd.DataFrame([soil_input], columns=X_soil.columns)
        crop_input_df = pd.DataFrame([crop_input], columns=X_crop.columns)
        season_input_df = pd.DataFrame([season_input], columns=X_season.columns)
        
        prediction1 = soil_clf.predict(soil_input_df)
        prediction2 = crop_clf.predict(crop_input_df)
        prediction3 = season_clf.predict(season_input_df)
        predicted_season = label_encoder.inverse_transform(prediction3)[0]
        messagebox.showinfo("Combined Prediction", f"Predicted Soil Type: {prediction1[0]}, Predicted Crop: {prediction2[0]}, Predicted Season: {predicted_season}")
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")

# Function to manually set suitable crops for each season
def predict_suitable_crops():
    from tkinter import ttk, messagebox
    try:
        # Get all unique seasons from the dataset
        all_seasons = ['Summer', 'Winter', 'Rainy']
        
        # Define suitable crops for each season
        suitable_crops_dict = {
            'Summer': ['wheat', 'rice'],
            'Winter': ['barley', 'wheat'],
            'Rainy': ['rice', 'maize']
        }
        
        # Display suitable crops for each season
        for season in all_seasons:
            suitable_crops = suitable_crops_dict.get(season, [])
            if len(suitable_crops) == 0:
                messagebox.showinfo("Suitable Crops", f"No suitable crops found for {season}.")
            else:
                messagebox.showinfo("Suitable Crops", f"Suitable Crops for {season}: {', '.join(suitable_crops)}")
    except Exception as e:
        # Show an error message if any exception occurs
        messagebox.showerror("Error", f"An error occurred: {e}")

def get_weather(city_name):
    from tkinter import ttk, messagebox
    from bs4 import BeautifulSoup
    import tkinter as tk
    global requests

    url = f"https://www.weather-forecast.com/locations/{city_name}/forecasts/latest"
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")

    try:
        global city_entry
        global get_weather
        global fetch_weather
        


        # Parsing temperature
        temperature_tag = soup.find("span", {"class": "temp"})
        temperature = temperature_tag.text if temperature_tag else "N/A"

        # Parsing humidity (Note: This is an example, and the exact tag might vary)
        humidity_tag = soup.find("span", {"class": "humidity"})
        humidity = humidity_tag.text if humidity_tag else "N/A"

        # Parsing rainfall (Note: This is an example, and the exact tag might vary)
        rainfall_tag = soup.find("span", {"class": "rainfall"})
        rainfall = rainfall_tag.text if rainfall_tag else "N/A"

        # Parsing weather description
        description_tag = soup.find("span", {"class": "phrase"})
        description = description_tag.text if description_tag else "N/A"

        weather = {
            "temperature": temperature,
            "humidity": humidity,
            "rainfall": rainfall,
            "description": description
        }
        return weather
    except Exception as e:
        print(f"Error parsing weather data: {e}")
        return None
       

# Function to fetch weather data
def fetch_weather():
    global get_weather
    from tkinter import ttk, messagebox
    try:
        global temperature_entry
        global humidity_entry
        global rainfall_entry
        global weather_clf
        global label_encoder_weather


        city_name = city_entry.get()
        weather = get_weather(city_name)
        
        if weather:
            #temperature_entry.delete(0, ttk.END)
            temperature_entry.insert(0, str(weather['temperature']))
            #humidity_entry.delete(0, ttk.END)
            humidity_entry.insert(0, str(weather['humidity']))
            #rainfall_entry.delete(0, ttk.END)
            rainfall_entry.insert(0, str(weather['rainfall']))
            messagebox.showinfo("Weather Info", f"Weather in {city_name}:\nTemperature: {weather['temperature']}°C\nHumidity: {weather['humidity']}%\nRainfall: {weather['rainfall']} mm\nDescription: {weather['description']}")
        else:
            messagebox.showerror("Error", "City not found!")
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")
        
# Synthetic weather data and model
weather_data = {
    'temperature': np.random.uniform(0, 40, 100),
    'humidity': np.random.uniform(0, 100, 100),
    'rainfall': np.random.uniform(0, 300, 100),
    'weather': np.random.choice(['Sunny', 'Rainy', 'Cloudy'], 100)
}

weather_df = pd.DataFrame(weather_data)
X_weather = weather_df[['temperature', 'humidity', 'rainfall']]
y_weather = weather_df['weather']
label_encoder_weather = LabelEncoder()
y_weather_encoded = label_encoder_weather.fit_transform(y_weather)
X_train_weather, X_test_weather, y_train_weather, y_test_weather = train_test_split(X_weather, y_weather_encoded, test_size=0.2, random_state=42)
weather_clf = RandomForestClassifier(n_estimators=100, random_state=42)
weather_clf.fit(X_train_weather, y_train_weather)        
        
def predict_weather():
    from tkinter import ttk, messagebox
    try:
        global get_weather

        temperature = float(temperature_entry.get())
        humidity = float(humidity_entry.get())
        rainfall = float(rainfall_entry.get())
        weather_input_df = pd.DataFrame([[temperature, humidity, rainfall]], columns=['temperature', 'humidity', 'rainfall'])
        prediction = weather_clf.predict(weather_input_df)
        predicted_weather = label_encoder_weather.inverse_transform(prediction)[0]
        messagebox.showinfo("Weather Prediction", f"Predicted Weather: {predicted_weather}")
    except ValueError:
        messagebox.showerror("Input Error", "Please enter valid numerical values for temperature, humidity, and rainfall.")
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")


# Create main window
root = tk.Tk()
root.title("Crop, Soil, and Season Prediction")
root.geometry("600x400")

# Create a notebook for tabs
notebook_style = ttk.Style()
if not notebook_style.theme_names():
    notebook_style.theme_create("mytheme", parent="alt", settings={
        "TNotebook": {"configure": {"tabmargins": [2, 5, 2, 0]}},
        "TNotebook.Tab": {
            "configure": {"padding": [20, 5], "font" : ('Helvetica', 12)},
            "map": {"background": [("selected", "#ff9966")],
                    "foreground": [("selected", "black")],
                    "expand": [("selected", [1, 1, 1, 0])]}
        }
    })
    notebook_style.theme_use("mytheme")

notebook = ttk.Notebook(root)

# Create frames for each tab
tab_frames = {}
tab_names = ["Crop Prediction", "Soil Prediction", "Season Prediction", "Combined Prediction", "Weather Prediction"]
for name in tab_names:
    tab_frames[name] = tk.Frame(notebook, bg="#f0f0f0")
    notebook.add(tab_frames[name], text=name)

# Crop prediction tab
crop_label = tk.Label(tab_frames["Crop Prediction"], text="Crop Prediction", font=('Helvetica', 16, 'bold'), bg="#f0f0f0")
crop_label.grid(row=0, column=0, columnspan=2, pady=10)

crop_columns = list(X_crop.columns)
for i, col in enumerate(crop_columns):
    label = tk.Label(tab_frames["Crop Prediction"], text=f"{col} (Range: {int(X_crop[col].min())} - {int(X_crop[col].max())})", font=('Helvetica', 10), bg="#f0f0f0")
    label.grid(row=i+1, column=1, sticky='w', padx=10, pady=5)
    entry = tk.Entry(tab_frames["Crop Prediction"], font=('Helvetica', 10), width=10, bd=2, relief=tk.GROOVE)
    entry.grid(row=i+1, column=2, padx=10, pady=5)
    entry.insert(0, int(X_crop[col].min()))
    crop_entries.append(entry)

predict_crop_button = tk.Button(tab_frames["Crop Prediction"], text="Predict Crop", command=predict_crop, font=('Helvetica', 12, 'bold'), bg='#ff9966', fg='black', bd=2, relief=tk.RAISED)
predict_crop_button.grid(row=i+1+len(crop_columns), column=0, columnspan=2, padx=10, pady=20, ipadx=20, ipady=10)

# Soil prediction tab
soil_label_heading = tk.Label(tab_frames["Soil Prediction"], text="Soil Prediction", font=('Helvetica', 16, 'bold'), bg="#f0f0f0")
soil_label_heading.grid(row=0, column=0, columnspan=2, pady=10)
soil_columns = list(X_soil.columns)
for i, col in enumerate(soil_columns):
    label = tk.Label(tab_frames["Soil Prediction"], text=f"{col} (Range: {int(X_soil[col].min())} - {int(X_soil[col].max())})", font=('Helvetica', 10), bg="#f0f0f0")
    label.grid(row=i+1, column=0, sticky='w', padx=10, pady=5)
    entry = tk.Entry(tab_frames["Soil Prediction"], font=('Helvetica', 10), width=10, bd=2, relief=tk.GROOVE)
    entry.grid(row=i+1, column=1, padx=10, pady=5)
    entry.insert(0, int(X_soil[col].min()))
    soil_entries.append(entry)

predict_soil_button = tk.Button(tab_frames["Soil Prediction"], text="Predict Soil Type", command=predict_soil, font=('Helvetica', 12, 'bold'), bg='#ff9966', fg='black', bd=2, relief=tk.RAISED)
predict_soil_button.grid(row=i+1+len(soil_columns), column=0, columnspan=2, padx=10, pady=20, ipadx=20, ipady=10)

# Season prediction tab
season_label_heading = tk.Label(tab_frames["Season Prediction"], text="Season Prediction", font=('Helvetica', 16, 'bold'), bg="#f0f0f0")
season_label_heading.grid(row=0, column=0, columnspan=2, pady=10)

season_columns = list(X_season.columns)
for i, col in enumerate(season_columns):
    label = tk.Label(tab_frames["Season Prediction"], text=f"{col} (Range: {int(X_season[col].min())} - {int(X_season[col].max())})", font=('Helvetica', 10), bg="#f0f0f0")
    label.grid(row=i+1, column=0, sticky='w', padx=10, pady=5)
    entry = tk.Entry(tab_frames["Season Prediction"], font=('Helvetica', 10), width=10, bd=2, relief=tk.GROOVE)
    entry.grid(row=i+1, column=1, padx=10, pady=5)
    entry.insert(0, int(X_season[col].min()))
    season_entries.append(entry)


# Button for season prediction
predict_season_button = tk.Button(tab_frames["Season Prediction"], text="Predict Season", command=predict_season_with_crops, font=('Helvetica', 12, 'bold'), bg='#ff9966', fg='black', bd=2, relief=tk.RAISED)
predict_season_button.grid(row=i+1+len(season_columns)+1, column=0, columnspan=2, padx=10, pady=20, ipadx=20, ipady=10)

# Button for suitable crops prediction
suitable_crops_button = tk.Button(tab_frames["Season Prediction"], text="Predict Season with Suitable Crops", command=predict_suitable_crops, font=('Helvetica', 14, 'bold'), bg='#ff9966', fg='black', bd=2, relief=tk.RAISED)
suitable_crops_button.grid(row=i+1+len(season_columns) + 2, columnspan=2, pady=20, ipadx=20, ipady=10, sticky="e")

# Combined prediction tab
combined_label = tk.Label(tab_frames["Combined Prediction"], text="Combined Prediction", font=('Helvetica', 16, 'bold'), bg="#f0f0f0")
combined_label.grid(row=0, column=0, columnspan=2, pady=10)

# Soil entries
combined_soil_columns = list(X_soil.columns)

for i, col in enumerate(combined_soil_columns):
    label = tk.Label(tab_frames["Combined Prediction"], text=f"Soil - {col} (Range: {int(X_soil[col].min())} - {int(X_soil[col].max())})", font=('Helvetica', 10), bg="#f0f0f0")
    label.grid(row=i+1, column=0, sticky='w', padx=10, pady=5)
    entry = tk.Entry(tab_frames["Combined Prediction"], font=('Helvetica', 10), width=10, bd=2, relief=tk.GROOVE)
    entry.grid(row=i+1, column=1, padx=10, pady=5)
    entry.insert(0, int(X_soil[col].min()))
    combined_soil_entries.append(entry)

# Crop entries
combined_crop_columns = list(X_crop.columns)

for i, col in enumerate(combined_crop_columns):
    label = tk.Label(tab_frames["Combined Prediction"], text=f"Crop - {col} (Range: {int(X_crop[col].min())} - {int(X_crop[col].max())})", font=('Helvetica', 10), bg="#f0f0f0")
    label.grid(row=i+1+len(combined_soil_columns), column=0, sticky='w', padx=10, pady=5)
    entry = tk.Entry(tab_frames["Combined Prediction"], font=('Helvetica', 10), width=10, bd=2, relief=tk.GROOVE)
    entry.grid(row=i+1+len(combined_soil_columns), column=1, padx=10, pady=5)
    entry.insert(0, int(X_crop[col].min()))
    combined_crop_entries.append(entry)

# Season entries
combined_season_columns = list(X_season.columns)

for i, col in enumerate(combined_season_columns):
    label = tk.Label(tab_frames["Combined Prediction"], text=f"Season - {col} (Range: {int(X_season[col].min())} - {int(X_season[col].max())})", font=('Helvetica', 10), bg="#f0f0f0")
    label.grid(row=i+1+len(combined_soil_columns)+len(combined_crop_columns), column=0, sticky='w', padx=10, pady=5)
    entry = tk.Entry(tab_frames["Combined Prediction"], font=('Helvetica', 10), width=10, bd=2, relief=tk.GROOVE)
    entry.grid(row=i+1+len(combined_soil_columns)+len(combined_crop_columns), column=1, padx=10, pady=5)
    entry.insert(0, int(X_season[col].min()))
    combined_season_entries.append(entry)


# Button for combined prediction
predict_combined_button = tk.Button(tab_frames["Combined Prediction"], text="Predict All", command=predict_all, font=('Helvetica', 12, 'bold'), bg='#ff9966', fg='black', bd=2, relief=tk.RAISED)
predict_combined_button.grid(row=1+len(combined_soil_columns)+len(combined_crop_columns)+len(combined_season_columns), column=0, columnspan=2, padx=10, pady=20, ipadx=20, ipady=10)


weather_label = tk.Label(tab_frames["Weather Prediction"], text="Weather Prediction", font=('Helvetica', 16, 'bold'), bg="#f0f0f0")
weather_label.grid(row=0, column=0, columnspan=2, pady=10)


# Weather Prediction Frame
weather_frame = tk.Frame(root, bg="#ffcccb", bd=2, relief=tk.GROOVE)
weather_frame.pack(fill='both', expand=True)

#City entry
city_label = tk.Label(tab_frames["Weather Prediction"], text="Enter City:", font=('Helvetica', 12), bg="#f0f0f0")
city_label.grid(row=1, column=0, pady=5)
city_entry = tk.Entry(tab_frames["Weather Prediction"], font=('Helvetica', 12), bd=2, relief=tk.GROOVE)
city_entry.grid(row=1, column=1, pady=5)

fetch_weather_button = tk.Button(tab_frames["Weather Prediction"], text="Fetch Weather", command=fetch_weather, font=('Helvetica', 12, 'bold'), bg='#ff9966', fg='black', bd=2, relief=tk.RAISED)
fetch_weather_button.grid(row=3, column=1, pady=20, ipadx=20, ipady=10)

#Temperature entry
temperature_label = tk.Label(tab_frames["Weather Prediction"], text="Temperature (°C):", font=('Helvetica', 12), bg="#f0f0f0")
temperature_label.grid(row=4, column=0, pady=5)
temperature_entry = tk.Entry(tab_frames["Weather Prediction"], font=('Helvetica', 12), bd=2, relief=tk.GROOVE)
temperature_entry.grid(row=4, column=1, pady=5)

#Humidity entry
humidity_label = tk.Label(tab_frames["Weather Prediction"], text="Humidity (%):", font=('Helvetica', 12), bg="#f0f0f0")
humidity_label.grid(row=5, column=0, pady=5)
humidity_entry = tk.Entry(tab_frames["Weather Prediction"], font=('Helvetica', 12), bd=2, relief=tk.GROOVE)
humidity_entry.grid(row=5, column=1, pady=5)

#Rainfall entry
rainfall_label = tk.Label(tab_frames["Weather Prediction"], text="Rainfall (mm):", font=('Helvetica', 12), bg="#f0f0f0")
rainfall_label.grid(row=6, column=0, pady=5)
rainfall_entry = tk.Entry(tab_frames["Weather Prediction"], font=('Helvetica', 12), bd=2, relief=tk.GROOVE)
rainfall_entry.grid(row=6, column=1, pady=5)

#Buttons for fetching and predicting weather

predict_weather_button = tk.Button(tab_frames["Weather Prediction"], text="Predict Weather", command=predict_weather, font=('Helvetica', 12, 'bold'), bg='#ff9966', fg='black', bd=2, relief=tk.RAISED)
predict_weather_button.grid(row=7, column=1, pady=20, ipadx=20, ipady=10)

suitable_crops_button = tk.Button(tab_frames["Weather Prediction"], text="Predict weather with Suitable Crops", command=predict_suitable_crops, font=('Helvetica', 14, 'bold'), bg='#ff9966', fg='black', bd=2, relief=tk.RAISED)
suitable_crops_button.grid(row=8, column=1, pady=20, ipadx=20, ipady=10)


notebook.pack(expand=True, fill='both')

# Footer frame
footer_frame = tk.Frame(root, bg="#ffcccb", bd=2, relief=tk.GROOVE)
footer_frame.pack(fill='x', side='bottom')

footer_label = tk.Label(footer_frame, text="Agriculture Prediction System © 2024", font=('Helvetica', 10), bg="#ffcccb")
footer_label.pack(pady=5, side='bottom')


root.mainloop()

