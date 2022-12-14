# Impact-of-GHG-emissions-on-Electric-Grid
Run The NYC_Greenwashing.py-> Change the paths as necessary.
The data is NYC_Property_Energy.csv. This is the only dataset used. 

There have been several studies on the impact of impact of Grid Purchase, GHG Emissions, Largest Property Use and other energy data to predict Energy Score for building sites but we are looking at the problem from another perspective: We are looking into the data to see if the energy attributes can give us some insight whether the Grid and Total GHG Emissions have some impact on Site Use. This is unique in the sense all the studies either focus on predicting Energy Score or even Smart Grids, but we are looking at the root of the problem. How Grid Purchase and GHG Emissions are highly impactive on the Site Energy Use. Here we are also studying the various effects of other attributes like Electricity Intensity and Gas Intensity to see how, in the absence of Grid Purchase and Total GHG Emissions , the Site Use is impacted.
Approximately 40% of global CO2 emissions are emitted from electricity generation through the combustion of fossil fuels to generate heat needed to power steam turbines. Burning these fuels results in the production of carbon dioxide (CO2)—the primary heat-trapping, “greenhouse gas” responsible for global warming. Applying smart electric grid technologies can potentially reduce CO2 emissions. Electric grid comprises three major sectors: generation, transmission and distribution grid, and consumption. Smart generation includes the use of renewable energy sources (wind, solar, or hydropower). Smart transmission and distribution relies on optimizing the existing assets of overhead transmission lines, underground cables, transformers, and substations such that minimum generating capacities are required in the future. Smart consumption will depend on the use of more efficient equipment like energy-saving lighting lamps, enabling smart homes and hybrid plug-in electric vehicles technologies. 
Global  emissions in 2010 approached 30 gigatons (Gt). Approximately 12 Gt (40%) are emitted from electricity generation sector through the combustion of fossil fuels like coal, oil, and natural gas to generate the heat needed to power steam-driven turbines. Burning these fuels results in the production of carbon dioxide ()—the primary heat-trapping, “greenhouse gas” responsible for global warming, in addition to other nitrogen and sulfur oxides responsible for various environmental impacts.
Over the past two centuries, mankind has increased the concentration of  in the atmosphere from 280 to more than 380 parts per million by volume, and it is growing faster every day. As the concentration of  has risen, so has the average temperature of the planet. Over the past century, the average surface temperature of Earth has increased by about 0.74°C. If we continue to emit carbon without control, temperatures are expected to rise by an additional 3.4°C by the end of this century. Climate change of that magnitude would likely have serious consequences for life on Earth. Sea level rise, droughts, floods, intense storms, forest fires, water scarcity, and cardiorespiratory diseases would be some results. Agricultural systems would be stressed—possibly declined in some parts of the world. There is also the risk that continued warming will push the planet past critical thresholds or “tipping points” —like the large-scale melting of polar ice, the collapse of the Amazon rainforest, or the warming and acidification of the oceans—that will make irreversible climate change. Despite mounting evidence of the dangers posed by climate change, efforts to limit carbon emissions remain insufficient, ineffective, and, in most countries, nonexistent. Given current trends and the best available scientific evidence, mankind probably needs to reduce total  emissions by at least 80% by 2050. Yet each day emissions continue to grow.
Electricity sector is the major source of the total global  emissions responsible for approximately 40% worldwide, followed by transportation, industry, and other sectors.![Image_001](https://user-images.githubusercontent.com/18380810/184597627-7629e374-a4e3-4156-ab9d-3103513cca83.png)
As a result, we will focus on how to decrease the quantities of  emitted from electricity sector by first focussing on how and to which extent the problem is present and how much it is impactful towards site energy use.
We are taking the nyc_benchmarking_disclosure_data where the data definition is given in nyc_benchmarking_disclosure_data_definitions_2017.pdf. Our main focus is on Site EUI (kBtu/ft²), Weather Normalized Site Electricity Intensity (kWh/ft²), Weather Normalized Site Natural Gas Intensity (therms/ft²). Unlike previous studies our target variable here is Site EUI (kBtu/ft²). As our focus here is different, we are using only Gradient Boosting Regressor Model as it has given a decent result. 
<img width="287" alt="Site_EUI_Distribution_vs_Number_of_Buildings" src="https://user-images.githubusercontent.com/18380810/184598581-b4095432-ae38-4963-b6c1-5b0fc180e2a2.png">

Before removing Grid Purchase and Total GHG Emissions :

<img width="275" alt="MAE vs Num_Trees_GBMRegressor" src="https://user-images.githubusercontent.com/18380810/184598662-eafcac3f-235f-4a64-a31a-aea0660cbd31.png">
<img width="290" alt="Mean_R2_vs_num_Trees_Performance_vs_num_trees" src="https://user-images.githubusercontent.com/18380810/184598670-18445075-3b35-4504-8380-40f3dd004214.png">

After removing Grid Purchase and Total GHG Emissions :


<img width="275" alt="After_removing_Grid_Purchase_and_GHG_Emissions_MAE_vs_Num_Trees" src="https://user-images.githubusercontent.com/18380810/184598725-2d4389cb-bdd4-45c6-a956-1fedcd91bf32.png">
<img width="290" alt="After_removing_Grid_Purchase_and_GHG_Emissions_Mean_R2_vs_No_of_Trees" src="https://user-images.githubusercontent.com/18380810/184598727-79eb02b4-95a3-42fc-9d6a-bb578fd48459.png">
<img width="293" alt="After_removing_Grid_Purchase_and_GHG_Emissions_Residuals" src="https://user-images.githubusercontent.com/18380810/184598729-e929ea1c-d340-452b-8c28-4975fe881dbc.png">

Total Correlations:

![All_correlation](https://user-images.githubusercontent.com/18380810/184598731-6b844e53-0e8b-4359-a443-8cb2d18ef47f.png)

Categorical Correlation:

![categorical_correlation](https://user-images.githubusercontent.com/18380810/184599611-f838ee40-f9b8-4702-aea8-f68ee6bce9a3.png)

Numeric Correlation:


![numerics_correlation](https://user-images.githubusercontent.com/18380810/184599683-933dd3c3-120c-4e9e-82ac-2976271b9adf.png)



After removing Grid Purchase and Total GHG Emissions :

<img width="303" alt="After_removing_Grid_Purchase_and_GHG_Emissions_Actuals_vs_Predictions" src="https://user-images.githubusercontent.com/18380810/184598736-f19a6bb7-9fd1-4702-94a4-96cb5d1cbb4e.png">

Before removing Grid Purchase and Total GHG Emissions :

<img width="296" alt="Actuals_vs_Predictions" src="https://user-images.githubusercontent.com/18380810/184599044-aa19b5d6-b88f-45b4-8b0e-dfc30c2984aa.png">
