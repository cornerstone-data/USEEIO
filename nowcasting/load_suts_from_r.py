# -----------------------------------------------------------------------------
# load_suts_from_r.py
#
# This script loads detailed and summary Supply and Use Tables (SUTs) and related
# economic data from R .rda files using rpy2, and prepares them for balancing and
# aggregation in Python. It handles both detailed and summary tables, sets up
# aggregation matrices using the master crosswalk, and provides utilities for
# checking consistency between detailed and aggregated data. The script supports
# iterative balancing of SUTs using the SUT-RAS algorithm and outputs balanced
# tables to CSV files for further analysis.
#
# Key features:
# - Loads SUTs, margin, and code mapping data from R data files.
# - Aligns and reclasses tables for consistency.
# - Builds aggregation matrices from a master crosswalk for mapping detailed to
#   summary codes.
# - Performs consistency checks between detailed and summary tables.
# - Runs the SUT-RAS balancing algorithm and writes balanced results
# 
# -----------------------------------------------------------------------------

import os
from pathlib import Path
import appdirs

## Use the nowcasting folder as the working directory

# -----------------------------------------------------------------------------
### USER SPECIFICATION
# Set R_HOME environment variable, in R Studio use R.home()
os.environ['R_HOME'] = 'c:/Programs/R442/'

# Define root path and file names
methodPath = Path(__file__).parent # nowcasting folder
root_path = methodPath.parents[1] / 'useeior' / 'data' # path to useeior, assumes USEEIO and useeior in same parent directory
init_est_path = Path(appdirs.user_data_dir()) / 'USEEIO-input'
output_dir = methodPath / 'output'
output_dir.mkdir(exist_ok=True)

# Choose whether to use specified initial estimate (alternative is to use the benchmark tables directly)
# initial_estimate = 'exogenousdata' or initial_estimate = ''
initial_estimate = 'exogenousdata'
# initial_estimate = ''


year_range_to_estimate =range(2017, 2024)
n_ras_iter=1000


# fast run:
# year_range_to_estimate =range(2017, 2024)
# n_ras_iter=100



# -----------------------------------------------------------------------------
### File inputs
file_names_benchmark = {
    'detail_make_table': 'Detail_Make_2017_AfterRedef_17sch.rda',
    'detail_use_table': 'Detail_Use_2017_PUR_AfterRedef_17sch.rda',
    'detail_use_pro_table': 'Detail_Use_2017_PRO_AfterRedef_17sch.rda',
    'detail_import_table': 'Detail_Import_2017_BeforeRedef_17sch.rda',
    'detail_gross_output_io': 'Detail_GrossOutput_IO_17sch.rda',
    'detail_commodity_code_name': 'Detail_CommodityCodeName_2017.rda'
}

file_names_aggregates = {
    'summary_gross_output_io': 'Summary_GrossOutput_IO_17sch.rda',
    'summary_value_added_code_name': 'Summary_ValueAddedCodeName_2017.rda',
}


file_names_meta = {
    'MasterCrosswalk': 'MasterCrosswalk.rda'
}


# -----------------------------------------------------------------------------
### Housekeeping

from sut_ras import sut_ras
from aggregation_funcs import build_aggregation_matrices
import numpy as np
import pandas as pd    
import sys
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from check_balances import check_row_col_balance
from check_balances import compare_disagg_to_agg
from check_balances import compare_disagg_to_agg_vec
from check_balances import check_and_align_agg_indices
from check_balances import check_import_balance
class Tee(object):
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()
    def flush(self):
        for f in self.files:
            f.flush()
logfile = open("log_sut_nowcast.txt", "w")
sys.stdout = Tee(sys.stdout, logfile)

# Activate the pandas conversion
pandas2ri.activate()


# -----------------------------------------------------------------------------
### Data loading

# Load the R data files - detailed SUT benchmarks
for key, file_name in file_names_benchmark.items():
    ro.r['load'](os.path.join(root_path, file_name))
    
# Load the R data files - detailed aggregates such as output
for key, file_name in file_names_aggregates.items():
    ro.r['load'](os.path.join(root_path, file_name))

# Load the R data files - detailed aggregates such as output
for key, file_name in file_names_meta.items():
    ro.r['load'](os.path.join(root_path, file_name))
    
    
MasterCrosswalk = ro.conversion.rpy2py(ro.r['MasterCrosswalk'])



# Extract the data frames from R environment
detail_make_table = ro.conversion.rpy2py(ro.r['Detail_Make_2017_AfterRedef_17sch'])
# Transpose the make table to get the supply table
detail_supply_table = detail_make_table.transpose()
detail_use_pro_table = ro.conversion.rpy2py(ro.r['Detail_Use_2017_PRO_AfterRedef_17sch'])
detail_use_pur_table = ro.conversion.rpy2py(ro.r['Detail_Use_2017_PUR_AfterRedef_17sch'])
detail_import_table = ro.conversion.rpy2py(ro.r['Detail_Import_2017_BeforeRedef_17sch'])

detail_gross_output_io_raw = ro.conversion.rpy2py(ro.r['Detail_GrossOutput_IO_17sch'])
summary_gross_output_io = ro.conversion.rpy2py(ro.r['Summary_GrossOutput_IO_17sch'])
summary_value_added_code_name = ro.conversion.rpy2py(ro.r['Summary_ValueAddedCodeName_2017'])


detail_commodity_code_name = ro.conversion.rpy2py(ro.r['Detail_CommodityCodeName_2017'])
detail_commodity_code_name_products =  detail_commodity_code_name[0:394]

detail_use_pur_table_reclass = detail_use_pur_table.reindex(detail_use_pro_table.index)
detail_use_pur_table_reclass =detail_use_pur_table_reclass.fillna(0)



# Set up the variables we will use for the initial estimate - Detailed tables used here:
Mraw=detail_make_table.copy()
Uraw=detail_use_pro_table.copy()
Umraw=detail_import_table.copy()

headers_row_make=Mraw.index[0:-1]
headers_col_make=Mraw.columns[0:-1]
headers_row_use=Uraw.index[np.r_[0:402]]  
headers_col_use=Uraw.columns[np.r_[0:402]]  
headers_col_useM=Umraw.columns[np.r_[0:402]]  
headers_va_use=Uraw.index[np.r_[403:406]]  
headers_fd_use=Uraw.columns[np.r_[403:423]]  
headers_fd_useM=Umraw.columns[np.r_[403:423]]  


Mi_ini=Mraw.loc[headers_row_make,headers_col_make]
Vi_ini=Mi_ini.transpose().copy()
Ui_ini=Uraw.loc[headers_row_use,headers_col_use]
Umi_ini=Umraw.loc[headers_row_use,headers_col_useM]
Ufd_ini=Uraw.loc[headers_row_use,headers_fd_use]
Umfd_ini=Umraw.loc[headers_row_use,headers_fd_useM]
Uva_ini=Uraw.loc[headers_va_use,headers_col_use]
Uvafd_ini=Uraw.loc[headers_va_use,headers_fd_use]

Ui_ini_BM = Ui_ini.copy()
Ufd_ini_BM = Ufd_ini.copy()
Vi_ini_BM = Vi_ini.copy()

# Step 1: Reindex to headers_col_make (adds NaNs where missing)
detail_gross_output_io = detail_gross_output_io_raw.reindex(headers_col_use)


# Sign preservation is a core component of RAS, but sometimes it prevents convergences, when, for example stocks need to flip sign from year to year.
U_sign_flex = pd.DataFrame(False, index=Ui_ini.index, columns=Ui_ini.columns)
Uva_sign_flex = pd.DataFrame(False, index=Uva_ini.index, columns=Uva_ini.columns)
Ufd_sign_flex = pd.DataFrame(False, index=Ufd_ini.index, columns=Ufd_ini.columns)

# Allow sign changes in specific industries for Uva and Ufd
U_sign_flex.loc[['S00402'], :] = True
U_sign_flex.loc[:,['S00500']] = True
Uva_sign_flex.loc[['V00200', 'V00300'], :] = True
Ufd_sign_flex.loc[:, [ 'F03000']] = True

    

for yr in year_range_to_estimate:
    
    if initial_estimate == 'exogenousdata':
        
        print("Using exogenous supplied data on com_mix and intermediate Use for initial estimate:")
        Vi_ini=pd.read_excel(init_est_path / 'com_mix.xlsx',
                             sheet_name=f"{yr}_Supply_BAS",index_col=0)/1e6
        Vi_ini.index = Vi_ini.index.str.replace(r'/US$', '', regex=True)
        Vi_ini.columns = (Vi_ini.columns
                       .str.replace(r'^X', '', regex=True)       # remove leading X
                       .str.replace(r'\.US$', '', regex=True)   # remove .US at end
                       .str.replace(r'/US$', '', regex=True))   # remove /US at end
        Vi_ini.index.name = None
        # check for discrepancies
        mask = (Vi_ini == 0) & (Vi_ini_BM != 0)
        indices = np.where(mask)
        Vi_ini[mask] = Vi_ini_BM[mask]
        
        Ui_ini=pd.read_excel(init_est_path / 'intermediate_Use.xlsx',
                             sheet_name=f"{yr}_Inter_U_PRO",index_col=0)/1e6
        Ui_ini.index = Ui_ini.index.str.replace(r'/US$', '', regex=True)
        Ui_ini.columns = (Ui_ini.columns
                       .str.replace(r'^X', '', regex=True)       # remove leading X
                       .str.replace(r'\.US$', '', regex=True)   # remove .US at end
                       .str.replace(r'/US$', '', regex=True))   # remove /US at end
        Ui_ini.index.name = None
        mask = (Ui_ini == 0) & (Ui_ini_BM != 0)
        indices = np.where(mask)
        Ui_ini[mask] = Ui_ini_BM[mask]
        
        
        Ufd_ini=pd.read_csv(init_est_path / f'final_demand_{yr}_PRO.csv',index_col=0)
        Ufd_ini = Ufd_ini.reindex_like(Ufd_ini_BM).combine_first(Ufd_ini_BM)
        Ufd_ini.index.name = None
        mask = (Ufd_ini == 0) & (Ufd_ini_BM != 0)
        indices = np.where(mask)
        Ufd_ini[mask] = Ufd_ini_BM[mask]
        mask = (Ufd_ini > 0) & (Ufd_ini_BM < 0)
        indices = np.where(mask)
        Ufd_ini[mask] = Ufd_ini_BM[mask]
        
        mask = (Ufd_ini < 0) & (Ufd_ini_BM > 0)
        indices = np.where(mask)
        Ufd_ini[mask] = Ufd_ini_BM[mask]
    
    
    use_summary_table =1
    if use_summary_table ==1:
        file_names_yearly = {
            'summary_make_table': f'Summary_Make_{yr}_AfterRedef_17sch.rda',
            'summary_use_table': f'Summary_Use_{yr}_PRO_AfterRedef_17sch.rda',
            'summary_import_table': f'Summary_Import_{yr}_BeforeRedef_17sch.rda',
            'summary_supply_table1': f'Summary_Supply_{yr}_17sch.rda',
            'summary_use_table1': f'Summary_Use_SUT_{yr}_17sch.rda',
        }
    
        for key, file_name in file_names_yearly.items():
            full_path = os.path.join(root_path, file_name)
            if os.path.exists(full_path):
                print(f"Loading {full_path}")
                ro.r['load'](full_path)
            else:
                print(f"File not found: {full_path} — skipping")
    
    
        summary_make_table = ro.conversion.rpy2py(ro.r[f'Summary_Make_{yr}_AfterRedef_17sch'])
        summary_use_table = ro.conversion.rpy2py(ro.r[f'Summary_Use_{yr}_PRO_AfterRedef_17sch'])
        summary_import_table = ro.conversion.rpy2py(ro.r[f'Summary_Import_{yr}_BeforeRedef_17sch'])        
        
    
        # select what tables to work with
        Magg=summary_make_table.copy()
        Uagg=summary_use_table.copy()
        Umagg=summary_import_table.copy()
        
        headers_summary_row_make=Magg.index[0:-1]
        headers_summary_col_make=Magg.columns[0:-1]
        headers_summary_row_use=Uagg.index[np.r_[0:73]]  
        headers_summary_col_use=Uagg.columns[np.r_[0:71]] 
        headers_summary_col_useM=Umagg.columns[np.r_[0:71]]  
        headers_summary_va_use=Uagg.index[np.r_[74:77]]  
        headers_summary_fd_use=Uagg.columns[np.r_[72:92]]  
        headers_summary_fd_useM=Umagg.columns[np.r_[72:92]]  
        
        
        Viagg=Magg.iloc[0:-1,0:-1].transpose().copy()
        Uiagg=Uagg.loc[headers_summary_row_use,headers_summary_col_use]
        Umiagg=Umagg.loc[headers_summary_row_use,headers_summary_col_useM]
        Ufdagg=Uagg.loc[headers_summary_row_use,headers_summary_fd_use]
        Umfdagg=Umagg.loc[headers_summary_row_use,headers_summary_fd_useM]
        Uvaagg=Uagg.loc[headers_summary_va_use,headers_summary_col_use]
        Uvafdagg=Uagg.loc[headers_summary_va_use,headers_summary_fd_use]
        
        ind_out_agg=summary_gross_output_io[str(yr)]
        
        
    
    G_iagg_rows, G_iagg_cols, G_vaagg_rows, G_vaagg_cols, G_fdagg_rows, G_fdagg_cols = build_aggregation_matrices(MasterCrosswalk, Vi_ini, Viagg,
                                    detail_col='BEA_2017_Detail_Code',
                                    summary_col='BEA_2017_Summary_Code',
                                    va=Uva_ini,va_agg=Uvaagg,
                                    fd=Ufd_ini,fd_agg=Ufdagg)
    

    C_ind_out=detail_gross_output_io[str(yr)]
    pro_out=Vi_ini.sum(axis=1) # not used for time series
    
    check_and_align_agg_indices(Viagg, G_iagg_rows, G_iagg_cols)
    
    ### Checks on Data Constraints
    # right now this shows a difference between the aggregated detail ind out and that which is in the Summary Supply Table 
    [diff_in_totals,ind_out_agged,df_diff] = compare_disagg_to_agg_vec(C_ind_out, Viagg.sum(axis=0), G_iagg_cols.to_numpy())
    (methodPath / 'checks').mkdir(exist_ok=True)
    df_diff.to_csv(f'checks/ColTotal_v_SummaryTable_Conflict_{yr}.csv')
    
### RAS BALANCING ###


    # Option 1: Balance tables just using Col totals
    # # Update ini based on detailed ind out
    # [Vi_out,Ui_out,Ufd_out,Uva_out, Umi_out,Umfd_out]=sut_ras(V=Vi_ini,Ui=Ui_ini,Ufd=Ufd_ini,Uva=Uva_ini,Umi=Umi_ini,Umfd=Umfd_ini,
    #                                             ColTot=C_ind_out,Uva_sign_flex=Uva_sign_flex,Ufd_sign_flex=Ufd_sign_flex, n_iter=n_ras_iter)
    
    
    
    # Option 2:  Balance tables just using Col totals Vagg
    # [Vi_out,Ui_out,Ufd_out,Uva_out, Um_out,Umfd_out]=sut_ras(V=Vi_ini,Ui=Ui_ini,Ufd=Ufd_ini,Uva=Uva_ini,Umi=Umi_ini,Umfd=Umfd_ini,
    #                                                 ColTot=C_ind_out,Vagg=Viagg, Uagg=None, Vagg_rows=G_iagg_rows.to_numpy(),Vagg_cols=G_iagg_cols.to_numpy(), n_iter=n_ras_iter)

    
    
    # Option 3:  Balance tables just using ColTot, Vagg and Uagg
    # [Vi_out,Ui_out,Ufd_out,Uva_out, Um_out,Umfd_out]=sut_ras(V=Vi_ini,Ui=Ui_ini,Ufd=Ufd_ini,Uva=Uva_ini,Umi=Umi_ini,Umfd=Umfd_ini,
    #                                                 ColTot=C_ind_out,Vagg=Viagg, Uagg=Uiagg, Vagg_rows=G_iagg_rows.to_numpy(),Vagg_cols=G_iagg_cols.to_numpy(),
    #                                                 Ufdagg=Ufdagg,Uvaagg=Uvaagg,G_va=G_vaagg_rows.to_numpy(),G_fd=G_fdagg_cols.to_numpy(), n_iter=n_ras_iter)
    
    
    # Option 4:  Balance tables just using Vagg and Uagg (no ColTot)
    # [Vi_out,Ui_out,Ufd_out,Uva_out, Um_out,Umfd_out]=sut_ras(V=Vi_ini,Ui=Ui_ini,Ufd=Ufd_ini,Uva=Uva_ini,Umi=Umi_ini,Umfd=Umfd_ini,
    #                                                 ColTot=None,Vagg=Viagg, Uagg=Uiagg, Vagg_rows=G_iagg_rows.to_numpy(),Vagg_cols=G_iagg_cols.to_numpy(),
    #                                                 Ufdagg=Ufdagg,Uvaagg=Uvaagg,G_va=G_vaagg_rows.to_numpy(),G_fd=G_fdagg_cols.to_numpy(), n_iter=n_ras_iter)
    
    
    # Option 5: Balance tables just using Vagg and Uagg (no ColTot) [SIGN FLEX] - with sign flexibility for certain sectors
    # [Vi_out,Ui_out,Ufd_out,Uva_out, Um_out,Umfd_out]=sut_ras(V=Vi_ini,Ui=Ui_ini,Ufd=Ufd_ini,Uva=Uva_ini,Umi=Umi_ini,Umfd=Umfd_ini,
    #                                                 ColTot=None,Vagg=Viagg, Uagg=Uiagg, Vagg_rows=G_iagg_rows.to_numpy(),Vagg_cols=G_iagg_cols.to_numpy(),
    #                                                 Ufdagg=Ufdagg,Uvaagg=Uvaagg,G_va=G_vaagg_rows.to_numpy(),G_fd=G_fdagg_cols.to_numpy(),
    #                                                 Uva_sign_flex=Uva_sign_flex,Ufd_sign_flex=Ufd_sign_flex,U_sign_flex=U_sign_flex, n_iter=n_ras_iter)
    
    
    # Option 6: Balance tables just using Vagg and Uagg including Umagg (imports); (no ColTot) [SIGN FLEX] - with sign flexibility for certain sectors
    # [Vi_out,Ui_out,Ufd_out,Uva_out, Umi_out,Umfd_out]=sut_ras(V=Vi_ini,Ui=Ui_ini,Ufd=Ufd_ini,Uva=Uva_ini,Umi=Umi_ini,Umfd=Umfd_ini,
    #                                                 ColTot=None,Vagg=Viagg, Uagg=Uiagg,Umagg=Umiagg, Vagg_rows=G_iagg_rows.to_numpy(),Vagg_cols=G_iagg_cols.to_numpy(),
    #                                                 Ufdagg=Ufdagg,Umfdagg=Umfdagg,Uvaagg=Uvaagg,G_va=G_vaagg_rows.to_numpy(),G_fd=G_fdagg_cols.to_numpy(),
    #                                                 Uva_sign_flex=Uva_sign_flex,Ufd_sign_flex=Ufd_sign_flex,U_sign_flex=U_sign_flex, n_iter=n_ras_iter)
    
    
    # Option 7: Balance tables just using Vagg and Uagg including Umagg (imports) and with ColTot [SIGN FLEX] - with sign flexibility for certain sectors
    [Vi_out,Ui_out,Ufd_out,Uva_out, Umi_out,Umfd_out]=sut_ras(V=Vi_ini,Ui=Ui_ini,Ufd=Ufd_ini,Uva=Uva_ini,Umi=Umi_ini,Umfd=Umfd_ini,
                                                    ColTot=C_ind_out,Vagg=Viagg, Uagg=Uiagg,Umagg=Umiagg, Vagg_rows=G_iagg_rows.to_numpy(),Vagg_cols=G_iagg_cols.to_numpy(),
                                                    Ufdagg=Ufdagg,Umfdagg=Umfdagg,Uvaagg=Uvaagg,G_va=G_vaagg_rows.to_numpy(),G_fd=G_fdagg_cols.to_numpy(),
                                                    Uva_sign_flex=Uva_sign_flex,Ufd_sign_flex=Ufd_sign_flex,U_sign_flex=U_sign_flex, n_iter=n_ras_iter)
    
    
    
    # Final balance to col totals based on pre-balanced tables 
    [Vi_out,Ui_out,Ufd_out,Uva_out, Um_out,Umfd_out]=sut_ras(V=Vi_out,Ui=Ui_out,Ufd=Ufd_out,Uva=Uva_out,Umi=Umi_out,Umfd=Umfd_out,
                                                             ColTot=C_ind_out,Uva_sign_flex=Uva_sign_flex,Ufd_sign_flex=Ufd_sign_flex,U_sign_flex=U_sign_flex,n_iter=n_ras_iter)
    

    
### END RAS BALANCING ###    

    # Reindex outputs to match the original input DataFrames
    Vi_out = pd.DataFrame(Vi_out, index=Vi_ini.index, columns=Vi_ini.columns)
    Ui_out = pd.DataFrame(Ui_out, index=Ui_ini.index, columns=Ui_ini.columns)
    Ufd_out = pd.DataFrame(Ufd_out, index=Ufd_ini.index, columns=Ufd_ini.columns)
    Uva_out = pd.DataFrame(Uva_out, index=Uva_ini.index, columns=Uva_ini.columns)
    
    u1=pd.concat([Ui_out, Ufd_out], axis=1)
    u2=pd.concat([Uva_out, Uvafd_ini], axis=1)
    U_out = pd.concat([u1,u2], axis=0)
    if Umi_out is not None:
        Umi_out = pd.DataFrame(Umi_out, index=Umi_ini.index, columns=Umi_ini.columns)
        Umfd_out_df = pd.DataFrame(Umfd_out, index=headers_row_use, columns=headers_fd_use)
        
    #checks on output
    check_row_col_balance([Vi_out,Ui_out,Ufd_out,Uva_out, Umi_out])
    
    #check import balance holds
    m,idx,s=check_import_balance(Um_out)
    
    # checks on agg table to disagg table    
    [diff, Vagged,diff_list_V,diff_list_Vagg] = compare_disagg_to_agg(Vi_out, Viagg, G_iagg_rows, G_iagg_cols,matrix_name='Supply')
    [diffU, Uagged,diff_list_u,diff_list_uagg] = compare_disagg_to_agg(Ui_out, Uiagg, G_iagg_rows, G_iagg_cols,matrix_name='Use')
    [diffUfd, Ufdagged,diff_list_ufd,diff_list_ufdaagg] = compare_disagg_to_agg(Ufd_out, Ufdagg, G_iagg_rows, G_fdagg_cols,matrix_name='FD')
    [diffUva, Uvaagged,diff_list_uva,diff_list_uvaagg] = compare_disagg_to_agg(Uva_out, Uvaagg, G_vaagg_rows, G_iagg_cols,matrix_name='VA')
    
    
    # Write the balanced tables to CSV files
    Vi_out.to_csv(os.path.join(output_dir, f'V_out_{yr}.csv'))
    Vi_ini.to_csv(os.path.join(output_dir, f'V_ini_{yr}.csv'))
    # Concatenate Un, Unfd, Unva into a single DataFrame and write to file
   
    U_out.to_csv(os.path.join(output_dir, f'U_out_{yr}.csv'))
    Uraw.to_csv(os.path.join(output_dir, f'U_ini_{yr}.csv'))
    Uagg.to_csv(os.path.join(output_dir, f'U_agg_{yr}.csv'))
    
    Um_out =pd.concat([Umi_out, Umfd_out_df], axis=1)
    Um_out.to_csv(os.path.join(output_dir, f'U_imports_out_{yr}.csv'))
    Umi_ini.to_csv(os.path.join(output_dir, f'U_imports_ini_{yr}.csv'))
    


# # 2017 check - additional checks of final table to the benchmarks

    if yr==2017:
    
        
        def biggest_diffs(df1, df2, n=10, name1="df1", name2="df2"):
            """
            Return top n absolute differences between two DataFrames.
            """
            # stack to long format
            tidy = pd.DataFrame({
                name1: df1.stack(),
                name2: df2.stack()
            })
            tidy["abs_diff"] = (tidy[name1] - tidy[name2]).abs()
            
            # sort by abs difference
            return tidy.sort_values("abs_diff", ascending=False).head(n)
        
        # Example usage:
        top_Vi   = biggest_diffs(Vi_ini_BM, Vi_out, n=10, name1="Vi_ini", name2="Vi_out")
        top_Ui   = biggest_diffs(Ui_ini_BM, Ui_out, n=10, name1="Ui_ini", name2="Ui_out")
        top_Ufd  = biggest_diffs(Ufd_ini_BM, Ufd_out, n=10, name1="Ufd_ini", name2="Ufd_out")
        top_Uva  = biggest_diffs(Uva_ini, Uva_out, n=10, name1="Uva_ini", name2="Uva_out")
        
        # Display
        print("Top differences in Vi:")
        print(top_Vi, "\n")
        
        print("Top differences in Ui:")
        print(top_Ui, "\n")
        
        print("Top differences in Ufd:")
        print(top_Ufd, "\n")
        
        print("Top differences in Uva:")
        print(top_Uva, "\n")
        
        
        def zeros_in_out_not_in_ini(df_ini, df_out):
            """
            Return positions where df_out == 0 but df_ini != 0,
            with indices and values.
            """
            mask = (df_out == 0) & (df_ini != 0)
            
            result = pd.DataFrame({
                "ini": df_ini.stack(),
                "out": df_out.stack()
            })
            result = result[mask.stack()]  # filter only where condition holds
            
            return result
        violations_Vi  = zeros_in_out_not_in_ini(Vi_ini, Vi_out)
        violations_Ui  = zeros_in_out_not_in_ini(Ui_ini, Ui_out)
        violations_Ufd = zeros_in_out_not_in_ini(Ufd_ini, Ufd_out)
        violations_Uva = zeros_in_out_not_in_ini(Uva_ini, Uva_out)
        
        print("Where Vi_out is 0 but Vi_ini is nonzero:")
        print(violations_Vi.head())
        
        print("Where Ui_out is 0 but Ui_ini is nonzero:")
        print(violations_Ui.head())
        
        print("Where Ufd_out is 0 but Ufd_ini is nonzero:")
        print(violations_Ufd.head())
        
        print("Where Ui_out is 0 but Ui_ini is nonzero:")
        print(violations_Uva.head())
