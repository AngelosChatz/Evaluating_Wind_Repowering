# Wind Repowering Evaluation Capacity Model

This model evaluates the potential growth in wind power capacity achievable through repowering across Europe. Using a comprehensive database of existing wind parks operating on European soil, it assesses:

1. Repowering Capacity Potential Model
2. Economic Model
3. Land-use Model

##Getting Started


The following Data list should be created in a folder within the repository named data. 

1. Wind farm database                     - Windfarms_World_20230530.xlsx
2. GIS file with classifications          - gwa3_250_windspeed_100m.tif
3. Elevation map                          - Eurodem.tif
4. ERA5 - Atmospheric climate reanalysis  - era5.nc
5. Spatial boundaries (NUTS regions)      - NUTS_RG_01M_2024_4326_LEVL_3.geojson (Additonal file: custom.geo.json for canditate countries not included in the first file)
6. Power curves                           - RenewablesNinja power curves and a few additional found from litterature (Enercon E-115-3.000	Enercon E126-4000	Nordex 100-3300	Siemens SWT 3.6-107	Vestas V164-9500	Siemens SWT 6.6-170	Vestas V136-3.45)



# Running Sequence

Firstly, install the necessary libraries listed in requirements.txt and then run the submodels in the following prioritized order:


| #  | Script Name                                | Description                                                                                                                                            |
|----|--------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------|
| 1  | Rotor_diameter         | Estimates the rotor diameters of the wind turbines missing that data, based on classification, with a log–log approach.                                |
| 2  | Complex_or_flat                            | Assigns terrain complexity based on slope and the Terrain Ruggedness Index.                                                                            |
| 3  | Area_per_Country                           | Calculates the area per wind park.                                                                                                                     |
| 4  | ClassificationWT_EU                        | Reads a GIS classification map and assigns each park a class.                                                                                          |
| 5  | Repowering_Calculation_Stage_1_(Float)     | Calculates the repowered capacity of every wind park in the EU; number of turbines is a real number (float).                                           |
| 6  | Approach_*_ninja_wt                        | Calculates the repowered capacity of every wind park in the EU; number of turbines is an integer, with four approaches of increasing land flexibility. |
| 7  | Repowering_Capacity_Comaprisons            | Computes successful repowerings per country and average power increase (percentage) per approach.                                                      |
| 8  | Plot_Repowering_Capacity                   | Plots total EU capacity for each approach (with a 1-year construction delay), comparison bar charts per country, and growth rates.                     |
| 9  | Prepare_capacity_factors                   | Computes capacity factors for each wind park using ERA5 meteorological data, turbine power curves, and NUTS-3 region boundaries.                       |
| 10 | Energy_Production_old                      | Assigns each old turbine a representative RenewablesNinja power curve and calculates its capacity factor (same method as #9).                          |
| 11 | Repowered_Energy_Production                | Calculates per-country and scenario energy production for repowered turbines, and relative percentage variation vs. the old fleet.                     |
| 12 | AEP_Maps                                   | Generates energy maps: old production, repowered (approach 2 cap. max), NLH (yield-based), differences, repowered-park locations, coverage of wind energy  |
| 13 | Costs_Model                                | Produces economic plots: LCoE, IRR, NPV, and payback period.                                                                                           |
| 14 | lcoe                                       | Compares LCoE per scenario, country, and learning-rate assumption (sensitivity analysis).                                                              |
| 15 | Additional_Result_plots                    | Plots power density per country, comparative power density per approach, and required land-area comparisons.                                           |




In the following table the input, and output files of each sub-model will be presented

| Script Name                                | Input File(s)                                                                                                                                              | Output File(s)                                                                                   |
|--------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------|
| 1. Rotor_diamete       | <ul><li>`Windfarms_World_20230530.xlsx`</li></ul>                                                                                                           | <ul><li>`Windfarms_World_20230530_final_1.xlsx`</li></ul>                                        |
| 2. Complex_or_flat                         | <ul><li>`Windfarms_World_20230530_final_1.xlsx`</li><li>`eurodem.tif`</li></ul>                                                                              | <ul><li>`Windfarms_World_20230530_with_IEC_Elevation_v2.xlsx`</li></ul>                          |
| 3. Area_per_Country                        | <ul><li>`Windfarms_World_20230530_with_IEC_Elevation_v2.xlsx`</li></ul>                                                                                       | <ul><li>`Windfarms_World_20230530_with_IEC_Elevation_v2_area.xlsx`</li></ul>                     |
| 4. ClassificationWT_EU                     | <ul><li>`Windfarms_World_20230530_with_IEC_Elevation_v2_area.xlsx`</li><li>`gwa3_250_windspeed_100m.tif`</li></ul>                                          | <ul><li>`Windfarms_World_20230530_with_IEC_Elevation_v2_area_classifications.xlsx`</li></ul>     |
| 5. Repowering_Calculation_Stage_1_(Float)  | <ul><li>`Windfarms_World_20230530_with_IEC_Elevation_v2_area_classifications.xlsx`</li></ul>                                                                 | <ul><li>`Repowering_Stage_1_float.xlsx`</li></ul>                                                |
| 6. Approach_*_ninja_wt                     | <ul><li>`Windfarms_World_20230530_with_IEC_Elevation_v2_area_classifications.xlsx`</li></ul>                                                                 | <ul><li>`Approach_*.xlsx` (approaches 1–5)</li></ul>                                             |
| 7. Repowering_Capacity_Comaprisons         | <ul><li>`Approach_*.xlsx` (all approaches)</li></ul>                                                                                                         | <ul><li>Plots</li></ul>                                                                          |
| 8. Plot_Repowering_Capacity                | <ul><li>`Approach_*.xlsx` (all approaches)</li></ul>                                                                                                         | <ul><li>Plots</li></ul>                                                                          |
| 9. Prepare_capacity_factors                | <ul><li>`Approach_*.xlsx` (one approach at a time)</li><li>`era5.nc`</li><li>`NUTS_RG_01M_2016_4326.geojson`</li><li>`custom.geo.json`</li><li>`Power Curves.csv`</li><li>`Power Curves 2.csv`</li></ul> | <ul><li>`Approach_*_Cf.xlsx`</li></ul>                                                            |
| 10. Energy_Production_old                  | <ul><li>`Approach_*.xlsx` (any approach)</li><li>`era5.nc`</li><li>`NUTS_RG_01M_2016_4326.geojson`</li><li>`custom.geo.json`</li><li>`Power Curves.csv`</li></ul> | <ul><li>`Cf_old_updated.xlsx`</li><li>Plots</li></ul>                                             |
| 11. Repowered_Energy_Production            | <ul><li>`Cf_old_updated.xlsx`</li></ul>   <ul><li>`Approach_*.xlsx` (all approaches)`</li></ul>                                                                                                                     | <ul><li>(Plots)</li></ul>                                                               |
| 12. AEP_Maps                               | <ul><li>`Approach_2_Cf.xlsx`</li><li>`Cf_old_updated.xlsx`</li><li>`NUTS_RG_01M_2016_4326.geojson`</li><li>`custom.geo.json`</li></ul>                     | <ul><li>Plots</li></ul>                                                                          |                                                 |
| 13. Costs_Model                            | <ul><li>`Approach_2_Cf.xlsx`</li><li>`Cf_old_updated.xlsx`</li></ul>                                                                                          |  <ul><li>`per_park_results.xlsx`</li><li>Plots</li></ul>                                                                           |
| 14. lcoe                                   | <ul><li>`Approach_2_Cf.xlsx`</li><li>`Cf_old_updated.xlsx`</li></ul>                                                                                          | <ul><li>Plots</li></ul>                                                                          |
| 15. Land_use_Results                       | <ul><li>`Approach_*.xlsx` (all approaches)</li></ul>                                                                                                         | <ul><li>Plots</li></ul>                                                                          |



## Outputs

The resulting output will be an excel file with the following Columns:

| #  | Column Name                     | Description                                                        |
|----|---------------------------------|--------------------------------------------------------------------|
| 1  | `ID`                            | Unique identifier for the wind park.                               |
| 2  | `Continent`                     | Continent where the wind park is located.                          |
| 3  | `ISO code`                      | ISO country code.                                                  |
| 4  | `Country`                       | Name of the country.                                               |
| 5  | `State code`                    | State or regional code within the country.                         |
| 6  | `Area`                          | Specific geographical area or region.                              |
| 7  | `City`                          | Nearest city to the wind park.                                     |
| 8  | `Name`                          | Official wind park name.                                           |
| 9  | `2nd name`                      | Alternative or secondary name of the wind park.                    |
| 10 | `Latitude`                      | Geographical latitude coordinate.                                  |
| 11 | `Longitude`                     | Geographical longitude coordinate.                                 |
| 12 | `Altitude/Depth`                | Altitude (onshore) or depth (offshore) of the installation.        |
| 13 | `Location accuracy`             | Accuracy of the geographical coordinates.                          |
| 14 | `Offshore`                      | Indicates if the wind park is offshore (Yes/No).                   |
| 15 | `Manufacturer`                  | Turbine manufacturer.                                              |
| 16 | `Turbine`                       | Model name of wind turbine installed.                              |
| 17 | `Hub height`                    | Height of turbine hub above ground/sea level.                      |
| 18 | `Number of turbines`            | Total turbines installed.                                          |
| 19 | `Total power`                   | Total installed capacity (MW).                                     |
| 20 | `Developer`                     | Company that developed the wind park.                              |
| 21 | `Operator`                      | Company operating the wind park.                                   |
| 22 | `Owner`                         | Owner of the wind park.                                            |
| 23 | `Commissioning date`            | Date when the wind park began operation.                           |
| 24 | `Status`                        | Operational status of the wind park.                               |
| 25 | `Decommissioning date`          | Planned or actual date of decommissioning.                         |
| 26 | `Link`                          | URL for additional information about the wind park.                |
| 27 | `Update`                        | Date of last data update.                                          |
| 28 | `Rotor Diameter`                | Diameter of turbine rotor blades.                                  |
| 29 | `SingleWT_Capacity`             | Capacity of a single wind turbine (MW).                            |
| 30 | `IEC_Class`                     | Wind turbine classification according to IEC standards.            |
| 31 | `IEC_Class_Num`                 | Numerical IEC classification.                                      |
| 32 | `IEC_Class_Group`               | Grouping of IEC classification.                                    |
| 33 | `TRI`                           | Terrain Ruggedness Index value.                                    |
| 34 | `Slope_deg`                     | Average terrain slope in degrees.                                  |
| 35 | `Terrain_Type`                  | Description of terrain type (e.g., complex, flat).                 |
| 36 | `Active in 2022`                | Indicates whether the wind park was operational in 2022.           |
| 37 | `Turbine Area (m²)`             | Area occupied by a single turbine (m²).                            |
| 38 | `Total Park Area (m²)`          | Total land area occupied by the wind park (m²).                    |
| 39 | `Recommended_WT_Model`          | Recommended turbine model for repowering.                          |
| 40 | `Recommended_WT_Capacity`       | Recommended capacity per turbine for repowering (MW).              |
| 41 | `New_Turbine_Count`             | Suggested number of new turbines after repowering.                 |
| 42 | `Total_New_Capacity`            | Total new capacity achievable after repowering (MW).               |
| 43 | `New_Total_Park_Area (m²)`      | New total area occupied after repowering (m²).                     |



After Costs_model.py is run, additional columns are produced, namely:

| #  | Column Name             | Description                                                    |
|----|-------------------------|----------------------------------------------------------------|
| 1  | `CAPEX_rep`         | Total capital expenditures of each park (EUR).                 |
| 2  | `NPV_rep`           | Net present value of each park (EUR).                          |
| 3  | `IRR_rep`               | Internal rate of return per park (percentage).                 |
| 4  | `Payback_rep`           | Payback period of each park (years).                           |
| 5  | `Capacity_kW_old`  | The total capacity of the old/existing wind park(KW).|
| 6  | `Energy_MWh_old`  | The total energy produced by the old/ existing park (MWh).|
| 7  | `CF_old_pct`  | The capacity factor of the old installed turbines|
| 8  | `LCOE_Eur_per_MWh_rep`  | Levelized cost of electricity (EUR/MWh) for the repowered park.|

The same columns with _dec for decommissioned and replacement strategy data, which are calculated with the old wind turbines and the old energy yield and financial inputs

