# Nowcasting

Code in this folder can be used to generate nowcasted IO tables for USEEIO

## Steps

1. Initial estimates are generated using [CalculateIntermediateUseandCommodityMix.R](https://github.com/USEPA/useeior/blob/nowcasting/data-raw/CalculateIntermediateUseAndCommodityMix.R) in useeior
and [nipa_final_demand_estimates.py](nipa_final_demand_estimates.py) in this repository. Outputs of these scripts are placed in the 'Local / USEEIO-input' folder.

2. Run [load_suts_from_r.py](load_suts_from_r.py) to call on data from useeior to generate the estimates.

3. The output files for Make, Use, and Use Imports should be placed in 'Local / USEEIO-input'

4. In useeior, these nowcasted tables can be used to generate useeior data objects using [NowcastedTables.R](https://github.com/USEPA/useeior/blob/nowcasting/data-raw/NowcastedTables.R) and model build can proceed as normal.


## Package requirements
- pandas
- numpy
- flowsa
- rpy2


