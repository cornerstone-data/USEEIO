
"""
estimate_margins.py

This script performs balancing of margins for the US Environmentally-Extended Input-Output (USEEIO) model.
It loads benchmark and aggregate data, sets up the environment, and prepares for margin balancing
using the RAS method and aggregation functions.

Author: Richard Wood
Date: 05/06/2024
"""
import os
import numpy as np
import pandas as pd
from sut_ras import gras_scale_table_totals
from sut_ras import gras_scale_table_layers
from collections import defaultdict
from check_balances import compare_tables

# Set R_HOME environment variable
os.environ['R_HOME'] = '' #Your R directory like 'C:/Program Files/R/R-4.2.1'

import rpy2.robjects as ro
from rpy2.robjects import pandas2ri


# Activate the pandas conversion
pandas2ri.activate()

# Define root path and file names
root_path = r'' ## Your useeior/data directory like r'C:/Users/yourname/useeior/data'

redef_type='Before'
#redef_type='After'


# %% Data setup and load

file_names_benchmark = {
    'detail_use_pur_table': 'Detail_Use_2017_PUR_'+redef_type+'Redef_17sch.rda',
    'detail_use_pro_table': 'Detail_Use_2017_PRO_'+redef_type+'Redef_17sch.rda',
    'margins_table': 'Detail_Margins_2017_BeforeRedef_17sch.rda',
}


file_names_meta = {
    'MasterCrosswalk': 'MasterCrosswalk.rda'
}


code_classification = {
    '423100': 'Wholesale',  # Motor vehicle and motor vehicle parts and supplies
    '423400': 'Wholesale',  # Professional and commercial equipment and supplies
    '423600': 'Wholesale',  # Household appliances and electrical and electronic goods 
    '423800': 'Wholesale',  # Machinery, equipment, and supplies
    '423A00': 'Wholesale',  # Other durable goods merchant wholesalers
    '424200': 'Wholesale',  # Drugs and druggists’ sundries
    '424400': 'Wholesale',  # Grocery and related product wholesalers 
    '424700': 'Wholesale',  # Petroleum and petroleum products
    '424A00': 'Wholesale',  # Other nondurable goods merchant wholesalers
    '425000': 'Wholesale',  # Wholesale electronic markets and agents and brokers
    '441000': 'Retail',    # Motor vehicle and parts dealers
    '445000': 'Retail',    # Food and beverage stores
    '452000': 'Retail',    # General merchandise stores
    '444000': 'Retail',    # Building material and garden equipment and supplies dealers
    '446000': 'Retail',    # Health and personal care stores
    '447000': 'Retail',    # Gasoline stations
    '448000': 'Retail',    # Clothing and clothing accessories stores
    '454000': 'Retail',    # Nonstore retailers
    '4B0000': 'Retail',    # All other retail
    '481000': 'Transportation',  # Air transportation
    '482000': 'Transportation',  # Rail transportation
    '483000': 'Transportation',  # Water transportation
    '484000': 'Transportation',  # Truck transportation
    '486000': 'Transportation',  # Pipeline transportation
}
category_to_codes = defaultdict(list)
for code in code_classification.keys():
    cat = code_classification.get(code)
    if cat is not None:
        category_to_codes[cat].append(code)


# ['Transportation', 'Retail', 'Wholesale']
margin_ids_categories = list(set(code_classification.values())) 
margin_ids_detailed = code_classification.keys()


# Load the R data files - detailed SUT benchmarks
for key, file_name in file_names_benchmark.items():
    ro.r['load'](os.path.join(root_path, file_name))
    

# Load the R data files - detailed aggregates such as output
for key, file_name in file_names_meta.items():
    ro.r['load'](os.path.join(root_path, file_name))
    

# Extract the data frames from R environment
detail_use_pro_table = ro.conversion.rpy2py(ro.r['Detail_Use_2017_PRO_'+redef_type+'Redef_17sch'])
detail_use_pur_table = ro.conversion.rpy2py(ro.r['Detail_Use_2017_PUR_'+redef_type+'Redef_17sch'])
margins_table = ro.conversion.rpy2py(ro.r['Detail_Margins_2017_BeforeRedef_17sch'])    
MasterCrosswalk = ro.conversion.rpy2py(ro.r['MasterCrosswalk'])

# %% Data cleaning

##Correct error in reading of last two col names in use_pro
detail_use_pro_table.rename(columns={'NA':'T004','NA.1':'T007'},inplace=True)

# Note that there are more rows in Use Table Producer than Use Table Purchaser. This is due to some commodities being purely margin commodities, so their value across the whole row is 0 in purchaser prices. The BEA exclude these 0 rows from the Purchaser price tables. A simple alignment between the indices can reintroduce these 0 rows in order to align matrices.
detail_use_pur_table_reclass = detail_use_pur_table.reindex(detail_use_pro_table.index)
detail_use_pur_table_reclass =detail_use_pur_table_reclass.fillna(0)

# Define and extract the column names and extract the industry and FD parts
use_columns_all = detail_use_pro_table.columns
industry_list = [col for col in use_columns_all if not col.startswith(('T', 'F'))]
final_demand_list = [col for col in use_columns_all if col.startswith('F')]

# Columns in Use Table Producer and Use Table Purchaser are the same, but in the R data, one is codes, and one is named. The Use Table Purchaser is reclassed to codes.

detail_use_pur_table_reclass = detail_use_pur_table_reclass[use_columns_all]
#detail_use_pur_table_reclass.columns=use_columns_all

#Total margins are extracted from the difference between Use Table Producer and Use Table Purchaser
detail_margin_table = detail_use_pur_table_reclass - detail_use_pro_table

# Extract out the margin totals:
margin_layer_col_totals_df = -detail_margin_table.loc[code_classification.keys(), :].copy()
margin_layer_col_totals_df = margin_layer_col_totals_df.reindex(index=margin_ids_detailed)

# Create a dict of pivot tables, one for each value aggregate margin layer
margin_layer_agg_dict = {
    col: margins_table.pivot_table(
        index='CommodityCode',
        columns='NIPACode',
        values=col,
        aggfunc='sum',
        fill_value=0
    )
    for col in margin_ids_categories
}

# reindex each margin table to match standard index and columns
for category in margin_ids_categories:
    margin_layer_agg_dict[category] = margin_layer_agg_dict[category].reindex(
        index=detail_use_pur_table_reclass.index,
        columns=use_columns_all,
        fill_value=0
    )
    # add in row totals T001,T004,T007
    margin_layer_agg_dict[category]["T001"] = margin_layer_agg_dict[category][industry_list].sum(axis=1)
    margin_layer_agg_dict[category]["T004"] = margin_layer_agg_dict[category][final_demand_list].sum(axis=1)
    margin_layer_agg_dict[category]["T007"] = margin_layer_agg_dict[category]["T001"] + margin_layer_agg_dict[category]["T004"]



# %%  Check for data consistency
# there are two sets of data constraints - col totals, and aggregate layers. Except for sparsity, the only data conflict is then due to inconsistency between these two constraints
# Hence check consistency first:
margin_layer_agg_dict_col_tot={}
for category in margin_ids_categories:
    margin_layer_agg_dict_col_tot[category]=margin_layer_agg_dict[category].sum(axis=0)
margin_layer_agg_dict_col_tot_df=pd.DataFrame(margin_layer_agg_dict_col_tot).T
margin_layer_col_totals_agg = margin_layer_col_totals_df.groupby(code_classification).sum()
errors_data_constraint_abs = margin_layer_agg_dict_col_tot_df-margin_layer_col_totals_agg 
errors_data_constraint_rel = (margin_layer_agg_dict_col_tot_df-margin_layer_col_totals_agg )/(margin_layer_col_totals_agg+1e-20)
# Stack the DataFrame into a Series with MultiIndex (row, column)
errors_abs_sorted = errors_data_constraint_abs.stack().sort_values()
errors_rel_sorted = errors_data_constraint_rel.stack().sort_values()
# Combine into a single DataFrame
errors_sorted = pd.DataFrame({
    'agg layer': margin_layer_agg_dict_col_tot_df.stack(),
    'col total': margin_layer_col_totals_agg.stack(),
    'abs_error': errors_abs_sorted,
    'rel_error': errors_rel_sorted
}).sort_values(by='abs_error')

errors_sorted.to_csv('checks/source_data_errors_margins.csv')
    
# %% Begin data scaling to match the detailed margin totals, and the aggregate margin layers


margin_layer_detailed_dict=dict() # estimate of the detailed layers



#1. **Initial estimate**
#An initial estimate of detailed margin layers is currently constructed by scaling aggregate margin layers to the detailed column totals. These initial estimates can be further refined or replaced with superior data in the future.

for ind_margin_layer in margin_ids_detailed:

    # Select the right df based on classification
    category = code_classification.get(ind_margin_layer) 

    # Prepare the col constraint
    col_constraint = margin_layer_col_totals_df.loc[ind_margin_layer].to_frame().values.T

    margin_layer_arr, R_val = gras_scale_table_totals(margin_layer_agg_dict[category].values, col_constraint, axis=0)

    # Convert NumPy array back to DataFrame with original index and columns
    margin_layer_df = pd.DataFrame(margin_layer_arr,
                                   index=margin_layer_agg_dict[category].index,
                                   columns=margin_layer_agg_dict[category].columns)
    
    #save data 
    margin_layer_detailed_dict[ind_margin_layer] = margin_layer_df


# Now begin the iteration over the two sets of data constraints:
n_iter=10
tolerance = 1e-5
converged = False

R_values = {}# scaling values of totals
R_values_layers=dict() # scaling values of layers
for ind_margin_layer in margin_ids_detailed:
    R_values[ind_margin_layer] = np.zeros((425, n_iter))  # Init full array
    
for ind_margin_layer in margin_ids_categories:
    R_values_layers[ind_margin_layer] = np.zeros((425*408, n_iter))  # Init full array

for i in range(0,n_iter):
    max_change = 0.0  # Track maximum change this iteration
    
    
    # first set of data constraints: detailed margin col totals: 
    # Each detailed margin layer is scaled proportionally so that its column totals match those of the corresponding detailed aggregated margin (as extracted from Use tables).
    
    for ind_margin_layer in margin_ids_detailed:
        # Select the right df based on classification
            
        category = code_classification.get(ind_margin_layer) 
    
        # Prepare the row constraint
        row_constraint = margin_layer_col_totals_df.loc[ind_margin_layer].to_frame().values.T
    
        margin_layer_arr, R_val = gras_scale_table_totals(margin_layer_detailed_dict[ind_margin_layer], row_constraint, axis=0)
    
        # Convert NumPy array back to DataFrame with original index and columns
        margin_layer_detailed_dict[ind_margin_layer] = pd.DataFrame(margin_layer_arr,
                                       index=margin_layer_agg_dict[category].index,
                                       columns=margin_layer_agg_dict[category].columns)
        
        
        # save R values
        R_values[ind_margin_layer][:, i] = R_val
        if i > 0:
            delta = np.max(np.abs(R_val - R_values[ind_margin_layer][:, i - 1]))
            max_change = max(max_change, delta)
    
            
    # Second set of data constraints: layer balancing (i.e. sum over all margin commodities in a group to equal the aggregate margin value)
    # For each margin category, all relevant detailed layers are stacked (summed) and scaled to the aggregated margin data.
    for category, codes in category_to_codes.items():

        # Stack all the detailed layers 
        stacked_layers = np.stack([margin_layer_detailed_dict[code].values for code in codes])
    
        scaled_layers,R_val = gras_scale_table_layers(stacked_layers, margin_layer_agg_dict[category].values )
        
        for j, code in enumerate(codes):
            # save data
            margin_layer_detailed_dict[code] = pd.DataFrame(
                scaled_layers[j],
                index=margin_layer_agg_dict[category].index,
                columns=margin_layer_agg_dict[category].columns
            )
        
        # save R values
        R_values_layers[category][:, i] = R_val        
        if i > 0:
            delta = np.max(np.abs(R_val - R_values_layers[category][:, i - 1]))
            max_change = max(max_change, delta)
        
        
            
    print(f"Iteration {i+1}: max change = {max_change:.3e}")
    if max_change < tolerance and i>1:
        print(f"Converged after {i+1} iterations (tolerance: {tolerance})")
        converged = True
        break

if not converged:
    print("Did not converge within max iterations.")

# Currently there is no convergence criteria, but exits after certain # of iterations

# %% compare adherence of final values 
# final values
stacked_layers_final = {}
stacked_layers_final_col_total=pd.DataFrame()
for category, codes in category_to_codes.items():
    # Stack all the detailed layers
    stacked_layers_final[category] = np.sum([margin_layer_detailed_dict[code].values for code in codes], axis=0)
    stacked_layers_final_col_total[category] = stacked_layers_final[category].sum(axis=0)
    _= compare_tables(stacked_layers_final[category],margin_layer_agg_dict[category].values,category + ' layer ini',category + ' layer fin') 

stacked_layers_final_col_total=stacked_layers_final_col_total.T
stacked_layers_final_col_total.columns=detail_use_pro_table.columns
# Compute the sum for each margin layer and store as a row in a DataFrame
margin_layer_totals_df_final = pd.DataFrame(
    [np.sum(margin_layer_detailed_dict[ind_margin_layer], axis=0) for ind_margin_layer in margin_ids_detailed],
    index=margin_ids_detailed)


_= compare_tables(margin_layer_col_totals_df,margin_layer_totals_df_final,'margins col totals ini','margins col totals fin') 

margin_layer_totals_df_both=pd.concat([margin_layer_col_totals_df,margin_layer_totals_df_final],axis=0)

# %%

os.makedirs('margins', exist_ok=True)

# Example: saving each DataFrame in margin_layer_detailed_dict as CSV
for key, df in margin_layer_detailed_dict.items():
    csv_path = os.path.join('margins', f'{key}.csv')
    df.to_csv(csv_path)
    
