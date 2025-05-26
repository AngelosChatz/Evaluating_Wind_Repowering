Wind Repowering Evaluation Capacity Model

This model evaluates the potential growth in wind power capacity achievable through repowering across Europe. Using a comprehensive database of existing wind parks operating on European soil, it assesses:

1.Repowering capacity potential
2.Associated costs
3.Land-use efficiency

Getting Started:

Data: 

1. Wind farm database            - Windfarms_World_20230530.xlsx
2. GIS file with classifications - gwa3_250_windspeed_100m.tif
3. Elevation map                 - Eurodem.tif
4. 
#Running Sequience

Firstly, install the necessary libraries listed in requirements.txt and athen run the submodels in the following prioritized order:

1. Rotor_diameter_(Class influenced)      : Estimates the rotor diameters of the wind turnines missing that data, based on classification, with a log-log approach
2. Complex_or_flat                        : Assigns complexity to terrain based on Slope, and Terrain Ruggedness Index value
3. Area_per_Country                       : Calculates the area per wind park
4. ClassificationWT_EU                    : Reads a GIS classification map and assigins each park a class
5. Repowering_Calculation_Stage_1_(Float) : Calcualtes the repowered capacity of every wind park in the EU, number of turbines is a real number (Float)
6. Approach_*_ninja_wt                    : Calculates the repowered capacity of every wind park in the EU, number of turbines is an integer, with 4 different approaches increasing in land flexibility
7. Repowering_Capacity_Comaprisons        : Succesfull repowerings per country and average power increase (percentage) per apporach
   Additional_Result_plots                : Growth rate incorporation in cummulative EU projection to 2050, Power density per country, Comparative power density per approach, Required land area comparison (2,3,baseline)
   Plot_Repowering_Capacity               : Total Eu Capacity for each approach, with 1 year wind farm construction delay, Approaches comparison bar chart per country, and growth rate per approach 

#Outputs

The resulting output will be an excel file with the following Columns:

1. ID: Unique identifier for the wind park.
2. Continent: Continent where the wind park is located.
3. ISO code: ISO country code.
4. Country: Name of the country.
5. State code: State or regional code within the country.
6. Area: Specific geographical area or region.
7. City: Nearest city to the wind park.
8. Name: Official wind park name.
9. 2nd name: Alternative or secondary name of the wind park.
10. Latitude: Geographical latitude coordinate.
11. Longitude: Geographical longitude coordinate.
12. Altitude/Depth: Altitude (onshore) or depth (offshore) of the installation.
13. Location accuracy: Accuracy of the geographical coordinates.
14. Offshore: Indicates if the wind park is offshore (Yes/No).
15. Manufacturer: Turbine manufacturer.
16. Turbine: Model name of wind turbine installed.
17. Hub height: Height of turbine hub above ground/sea level.
18. Number of turbines: Total turbines installed.
19. Total power: Total installed capacity (MW).
20. Developer: Company that developed the wind park.
21. Operator: Company operating the wind park.
22. Owner: Owner of the wind park.
23. Commissioning date: Date when the wind park began operation.
24. Status: Operational status of the wind park.
25. Decommissioning date: Planned or actual date of decommissioning.
26. Link: URL for additional information about the wind park.
27. Update: Date of last data update.
28. Rotor Diameter: Diameter of turbine rotor blades.
29. SingleWT_Capacity: Capacity of a single wind turbine (MW).
30. IEC_Class: Wind turbine classification according to IEC standards.
31. IEC_Class_Num: Numerical IEC classification.
32. IEC_Class_Group: Grouping of IEC classification.
33. TRI: Terrain Ruggedness Index value.
34. Slope_deg: Average terrain slope in degrees.
35. Terrain_Type: Description of terrain type (e.g., complex, flat).
36. Active in 2022: Indicates whether the wind park was operational in 2022.
37. Turbine Area (m²): Area occupied by a single turbine (m²).
38. Total Park Area (m²): Total land area occupied by the wind park (m²).
39. Recommended_WT_Model: Recommended turbine model for repowering.
40. Recommended_WT_Capacity: Recommended capacity per turbine for repowering (MW).
41. New_Turbine_Count: Suggested number of new turbines after repowering.
42. Total_New_Capacity: Total new capacity achievable after repowering (MW).
43. New_Total_Park_Area (m²): New total area occupied after repowering (m²).

