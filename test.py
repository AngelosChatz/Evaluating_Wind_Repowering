import ipywidgets as widgets
from IPython.display import display, clear_output, FileLink
import matplotlib.pyplot as plt
import numpy as np
import math
import io
import pandas as pd

def calculate_wind_power(rho, radius, wind_speed, cp):
    area = math.pi * radius**2
    return 0.5 * rho * area * (wind_speed ** 3) * cp

def calculate_energy_and_income(power_watt, hours, price_per_kwh):
    energy_kwh = (power_watt * hours) / 1000
    income = energy_kwh * price_per_kwh
    return energy_kwh, income

def simulate_yearly_production(rho, radius, cp, price_per_kwh):
    np.random.seed(0)
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    avg_wind = np.random.normal(8, 2, 12).clip(3, 15)
    results = []
    for i in range(12):
        power = calculate_wind_power(rho, radius, avg_wind[i], cp)
        energy = (power * 24 * 30) / 1000
        income = energy * price_per_kwh
        results.append({'Month': months[i], 'Wind': avg_wind[i],
                        'Energy (kWh)': energy, 'Income (€)': income})
    return pd.DataFrame(results)

def export_to_excel(df):
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Yearly Data')
    buffer.seek(0)
    with open("/tmp/wind_data.xlsx", "wb") as f:
        f.write(buffer.read())
    return "/tmp/wind_data.xlsx"

scenarios = {
    'Custom': {},
    'Island (low wind)': {'rho': 1.2, 'radius': 35, 'cp': 0.38, 'price': 0.09},
    'Mountain (strong wind)': {'rho': 1.25, 'radius': 50, 'cp': 0.42, 'price': 0.11},
    'Wind Park (efficient)': {'rho': 1.22, 'radius': 60, 'cp': 0.48, 'price': 0.13},
}

scenario_dropdown = widgets.Dropdown(options=list(scenarios.keys()), description='Scenario:')
rho_slider = widgets.FloatSlider(value=1.225, min=1.0, max=1.5, step=0.01, description='Air Density')
radius_slider = widgets.FloatSlider(value=40, min=10, max=80, step=1, description='Blade Radius')
wind_slider = widgets.FloatSlider(value=10, min=1, max=25, step=0.5, description='Wind Speed')
cp_slider = widgets.FloatSlider(value=0.4, min=0.1, max=0.59, step=0.01, description='Cp Efficiency')
hours_slider = widgets.IntSlider(value=24, min=1, max=72, step=1, description='Hours')
price_slider = widgets.FloatSlider(value=0.09, min=0.01, max=0.5, step=0.01, description='€/kWh')
download_button = widgets.Button(description='Download Excel')
out = widgets.Output()

def update_all(change=None):
    with out:
        clear_output(wait=True)
        rho = rho_slider.value
        radius = radius_slider.value
        wind_speed = wind_slider.value
        cp = cp_slider.value
        hours = hours_slider.value
        price = price_slider.value

        power = calculate_wind_power(rho, radius, wind_speed, cp)
        energy, income = calculate_energy_and_income(power, hours, price)

        print(f" Power: {power:.2f} W")
        print(f" Energy for {hours}h: {energy:.2f} kWh")
        print(f" Income: {income:.2f} €")

        df = simulate_yearly_production(rho, radius, cp, price)
        display(df)

        plt.figure(figsize=(8, 4))
        plt.bar(df['Month'], df['Energy (kWh)'], color='skyblue')
        plt.title('Monthly Estimated Energy Production')
        plt.ylabel('Energy (kWh)')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        download_button.on_click(lambda x: display(FileLink(export_to_excel(df))))

def apply_scenario(change):
    preset = scenarios[scenario_dropdown.value]
    if preset:
        rho_slider.value = preset['rho']
        radius_slider.value = preset['radius']
        cp_slider.value = preset['cp']
        price_slider.value = preset['price']

scenario_dropdown.observe(apply_scenario, names='value')
for w in [rho_slider, radius_slider, wind_slider, cp_slider, hours_slider, price_slider]:
    w.observe(update_all, names='value')

ui = widgets.VBox([
    scenario_dropdown,
    rho_slider, radius_slider, wind_slider,
    cp_slider, hours_slider, price_slider,
    download_button
])

display(ui, out)
update_all()