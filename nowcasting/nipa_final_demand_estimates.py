"""
Generate initial estimates for final demand vectors using NIPA tables.
"""

import flowsa # nipa branch
from flowsa.flowbysector import FlowBySector
import pandas as pd
from pathlib import Path

methodPath = Path(__file__).parent
fbsMethodPath = methodPath / "flowsa_fbs"
regenerateFBS = False

#%% Generate PCE dataset
for y in range(2017, 2024):
    # Prepare the NIPA dataset from Table U2.4.5, these values are in Purchaser Price
    nipa_PCE = flowsa.getFlowByActivity('BEA_NIPA', y)
    nipa_PCE[['Table', 'Code_Line']] = nipa_PCE['Description'].str.split(': ', expand=True)
    nipa_PCE[['Code', 'Line']] = nipa_PCE['Code_Line'].str.split(' - ', expand=True)
    nipa_PCE = (nipa_PCE
               .assign(Line = lambda x: x['Line'].astype(int))
               .drop(columns=['Code_Line'])
               .sort_values(by=['Table', 'Line'])
               .assign(SectorConsumedBy = 'F010')
               )
    nipa_PCE = nipa_PCE.query('Table == "U20405"')
    
    # Extract the 2017 PCE Bridge as the basis for attribution
    url = 'https://apps.bea.gov/industry/release/xlsx/PCEBridge_Detail.xlsx'
    bridge = pd.read_excel(url, sheet_name='2017', header=4)
    bridge['Activity'] = bridge['PCE Category'] + ' (' + bridge['NIPA Line'].astype('str') + ')'
    df = (nipa_PCE
          .merge(bridge.filter(['Activity', 'NIPA Line', 'Commodity Code', 'Commodity Description', 'Unnamed: 8']),
                 how='left', left_on='Line', right_on='NIPA Line')
          .rename(columns={'Unnamed: 8': 'Value'}) # Purchasers value
          )
    
    # Attribute the NIPA line items to BEA sectors based on 2017 PCE Bridge
    df = (df
          .assign(
              TotalValue=lambda df: df.groupby('ActivityProducedBy')['Value'].transform('sum'),
              Share=lambda df: df['Value'] / df['TotalValue'],
              DistributedFlowAmount=lambda df: df['FlowAmount'] * df['Share']
              )
          )
    df['Commodity Code'] = df['Commodity Code'].astype('str')
    pce = df.pivot_table(values='DistributedFlowAmount', index='Commodity Code',
                          columns='SectorConsumedBy', aggfunc='sum',
                          margins=False) / 1000000

#%% Generate the FBS for other demands (see NIPA_FD_Common.yaml)
    name=f'NIPA_FD_{y}'
    if regenerateFBS:
        FlowBySector.generateFlowBySector(name, append_sector_names=True,
                                          download_sources_ok=True,
                                          external_config_path=fbsMethodPath)
    fbs = flowsa.getFlowBySector(name)
    
    cw = pd.read_csv('https://raw.githubusercontent.com/USEPA/useeior'
                     '/develop/inst/extdata/USEEIO_Commodity_Meta.csv',
                     usecols=[0, 1], names=['BEA', 'Com_Name'], skiprows=1
                     )
    mapping = (flowsa.common.load_crosswalk('NAICS_to_BEA_Crosswalk_2017')
               .rename(columns={'BEA_2017_Detail_Code': 'BEA',
                                'NAICS_2017_Code': 'Sector'})
               .filter(['BEA', 'Sector'])
               .drop_duplicates()
               .merge(cw, how='left', on='BEA'))
    ## ^^ NOTE FOR MANY TO MANY MAPPINGS this is leading to duplicates and are 
    ## resolved below by removing some of the combinations that do not apply
    
    use = fbs.merge(mapping, left_on='SectorProducedBy', right_on='Sector')
    # for each SPB and SCB, keep only data for the listed BEA sector
    mods = [('531', 'F02R', '531ORE'), # brokers commissions
            ('531', 'F02S', '531ORE'), # brokers commissions
            ('238290', 'F02R', '2334A0'), # Dorms and improvements
            ('238990', 'F02S', '233210'), # Hospitals
            ('236220', 'F02S', '2332A0'), # Offices
            ('236210', 'F02S', '233230'), # Manufacturing
            ('238390', 'F02S', '233262'), # Education
    
            ('923110', 'F10C', 'GSLGE'), 
            ('923120', 'F10C', 'GSLGH'), 
            ('923130', 'F10C', 'GSLGO'), 
    
        # This chunk is not working well because all coming from a single
        # Activity (so BEA use table attribution is jumbled)
        # as a result F06S, F07S, and F10S are of lower quality
            ('238290', 'F06S', '2334A0'), # Dorms and improvements
            ('238990', 'F06S', '233210'), # Hospitals
            ('236220', 'F06S', '2332A0'), # Offices
            ('236210', 'F06S', '233230'), # Manufacturing
            ('238390', 'F06S', '233262'), # Education
            ('238990', 'F07S', '233210'), # Hospitals
            ('236220', 'F07S', '2332A0'), # Offices
            ('236210', 'F07S', '233230'), # Manufacturing
            ('238390', 'F07S', '233262'), # Education
            ('238290', 'F10S', '2334A0'), # Dorms and improvements
            ('238990', 'F10S', '233210'), # Hospitals
            ('236220', 'F10S', '2332A0'), # Offices
            ('236210', 'F10S', '233230'), # Manufacturing
            ('238390', 'F10S', '233262'), # Education
            ]
    for spb, scb, bea in mods:
        use = use.query('~(SectorProducedBy == @spb & SectorConsumedBy == @scb & BEA != @bea)')
    
    use_pivot = use.pivot_table(values='FlowAmount', index='BEA',
                                columns='SectorConsumedBy', aggfunc='sum',
                                margins=False) / 1000000
    
    # Consolidate into a single table
    all_indexes = pce.index.union(use_pivot.index)
    all_columns = pce.columns.union(use_pivot.columns)
    
    pce = pce.reindex(index=all_indexes, columns=all_columns, fill_value=0)
    use_pivot = use_pivot.reindex(index=all_indexes, columns=all_columns, fill_value=0)
    combined = pce.add(use_pivot, fill_value=0)
    combined = combined.add_suffix('00')

    U_17 = pd.read_excel('https://pasteur.epa.gov/uploads/10.23719/1532178/USEEIOv2.5-kinglet-22.xlsx',
                           sheet_name='U')
    U_17 = (U_17
           .assign(BEA = lambda x: x['Unnamed: 0'].str[:-3])
           .set_index('BEA')
           .iloc[:-3]
           )

    # These are the specific PCE margins
    transport = bridge['Unnamed: 5'].sum()
    wholesale = bridge['Wholesale'].sum()
    retail = bridge['Retail'].sum()

    margins_df = (bridge
               .filter(['Commodity Code', 'Unnamed: 4', 'Unnamed: 8'])
               .rename(columns={'Unnamed: 4': 'Producer',
                                'Unnamed: 8': 'Purchaser'})
               .groupby('Commodity Code').agg('sum')
               .reset_index()
               .assign(BEA = lambda x: x['Commodity Code'].astype('str'))
               # .assign(phi = lambda x: x['Producer'] / x['Purchaser'])
               .set_index('BEA')
               .reindex(U_17.index, fill_value=0)
               )
    household_fd = U_17['F01000/US'] # drop value added rows
    margin_ratio = (household_fd / 1000000) / margins_df['Purchaser']
    
    combined = combined.reindex(U_17.index, fill_value=0)
    combined_pro = combined.mul(pd.Series(margin_ratio).fillna(1), axis=0)
   

    ## Margins values need to be added back to final demand for wholesale, transport, and retail sectors
    # transport sectors are somewhat accounted for by the margin_ratio but this is
    # not necessarily a good approach for the time series.
    margins_total = (combined_pro['F01000'] - combined['F01000']).sum()*-1
    margins = {}
    margins['t'] = margins_total * (transport / (transport+wholesale+retail))
    margins['w'] = margins_total * (wholesale / (transport+wholesale+retail))
    margins['r'] = margins_total * (retail / (transport+wholesale+retail))

    sectors= {}
    sectors['t'] = list(combined.index[combined.index.str.startswith(('48', '49'))])
    sectors['t'].remove('491000')
    sectors['r'] = list(combined.index[combined.index.str.startswith(('44', '45', '4B'))])
    sectors['w'] = list(combined.index[combined.index.str.startswith('42')])

    for m in ('t', 'r', 'w'):
        fd = combined.loc[combined.index.isin(sectors[m]), 'F01000'].sum() # SUT value
        new_fd = fd + margins[m]
        subset = pd.DataFrame(household_fd.loc[household_fd.index.isin(sectors[m])])
        subset['pct'] = subset['F01000/US'] / subset['F01000/US'].sum()
        subset['new_fd'] = new_fd * subset['pct']
        combined_pro['F01000'].update(subset['new_fd'])

    # Drop some vectors that have lower quality allocations
    combined_pro = combined_pro.drop(['F06S00', 'F07S00', 'F10C00', 'F10S00'], axis=1)

    # # Recalculate row totals
    # combined_pro['Total'] = combined_pro.sum(axis=1)
    # combined_pro.loc['Total'] = combined_pro.sum()
    
    combined_pro.to_csv(flowsa.settings.paths.local_path.parent / 'USEEIO-input' / f'final_demand_{y}_PRO.csv')
