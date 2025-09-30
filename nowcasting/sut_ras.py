"""
SUT - GRAS Balancing routine with customised table balances

"""


import numpy as np

def gras_internal(pvalues,nvalues,target):
    # creates an array s of scaling factors, same dimension as target
    # zero values are handled such that a target zero will only enforce 0 on values if there are no positives (to allow for balancing constraints in the data)
   
    # Compute GRAS scaling factor
    with np.errstate(divide='ignore', invalid='ignore'):
        root = np.sqrt(np.maximum(0, target**2 + 4 * pvalues * nvalues))
        denom = 2 * pvalues
        
        # positive-root solution
        s_pos = (target + root) / denom
        # negative-root solution
        s_neg = 1/((-target) / nvalues)

        # if denom <= 0, switch to the negative-root solution
        s = np.where(denom <= 0, s_neg, s_pos)
        s = np.where((denom <= 0) & (nvalues == 0), 1, s)
        s = np.where((target == 0) & (pvalues == 0)  , 0, s)
        


        
    return s

def gras_scale_table_totals(M, target, axis=0, sign_flex=None):
    if axis not in (0, 1):
        raise ValueError("axis must be 0 (columns) or 1 (rows)")

    M_orig = M.copy()
    if axis == 0:
        M = M.T
        M_orig = M_orig.T

    if sign_flex is None:
        sign_flex = np.zeros_like(M, dtype=bool)
    elif axis == 0:
        sign_flex = sign_flex.T

    
    M_opposite_eps = 0#1e-9  # small value for flipped sign entries

    # Construct stack of [M, M_opposite]
    M_opposite = -M * sign_flex * M_opposite_eps
    M_stack = np.stack([M, M_opposite], axis=0)  # shape (2, rows, cols)

    # Positive and negative parts across the stack
    pos_stack = np.where(M_stack > 0, M_stack, 0)
    neg_stack = np.where(M_stack < 0, -M_stack, 0)

    prowtot = pos_stack.sum(axis=0).sum(axis=1)  # sum over columns, then stack
    nrowtot = neg_stack.sum(axis=0).sum(axis=1)

    s = gras_internal(prowtot, nrowtot, target)
    s1d = s.flatten()

    with np.errstate(divide='ignore', invalid='ignore'):
        M_scaled_stack = np.where(
            M_stack > 0,
            M_stack * s1d[np.newaxis, :, np.newaxis],
            M_stack / s1d[np.newaxis, :, np.newaxis]
        )
        M_scaled_stack = np.where(
            s1d[np.newaxis, :, np.newaxis] == 0, 0, M_scaled_stack
        )

    # Collapse back to 2D by summing the stack
    M_new = M_scaled_stack.sum(axis=0)

    # Optional: Prevent sign flip unless sign_flex is True
    preserve_sign_mask = ~sign_flex
    sign_changed = np.sign(M_new) != np.sign(M_orig)
    M_new = np.where(preserve_sign_mask & sign_changed, 0, M_new)

    if axis == 0:
        M_new = M_new.T
        s = s  # unchanged

    return M_new, s





def gras_scale_table_agg(M, target, row_agg, col_agg, sign_flex=None):
    # note "target" value must be same dimension as table    
    
    M_opposite_eps = 1e-4
    
    target = target.to_numpy()
    
    # Separate positive and negative parts
    if sign_flex is not None:
        M_opposite = -M * sign_flex * M_opposite_eps
        M_stack = np.stack([M, M_opposite], axis=0)  # shape (2, i, j)
    else:
        M_stack = M[np.newaxis, :, :]

    pos_stack = np.where(M_stack > 0, M_stack, 0)
    neg_stack = np.where(M_stack < 0, -M_stack, 0)

    pagg = row_agg @ pos_stack.sum(axis=0) @ col_agg.T
    nagg = row_agg @ neg_stack.sum(axis=0) @ col_agg.T


    # Handle the case where both pagg and nagg are zero but target is not zero
    zero_mask = (pagg == 0) & (nagg == 0) & (target != 0)
    eps = 1e-2
    
    # this first check is to add in values if target is non-zero but disagg is all zero. Alternative would be to do a single check at the start of the balancing.
    zero_prob_disagg = row_agg.T @ zero_mask @ col_agg
    target_disagg = row_agg.T @ target @ col_agg
    
    M=np.where(zero_prob_disagg & (target_disagg > 0), eps, M)
    
    
        # Check for mismatch in signs between target and all disagg cells
    M = np.where(zero_prob_disagg & (target_disagg > 0), eps, M)


    # 1. Create masks for where M > 0 and M < 0
    pos_mask = M > 0
    neg_mask = M < 0
    
    # 2. Aggregate masks to target shape to get number of positive/negative contributors per agg cell
    pos_contribs = row_agg @ pos_mask @ col_agg.T
    neg_contribs = row_agg @ neg_mask @ col_agg.T
    total_contribs = row_agg @ ((M != 0).astype(int)) @ col_agg.T
    
    # 3. Check conditions:
    # - Target > 0 and all contributors are negative
    # - Target < 0 and all contributors are positive
    flip_mask = (
        ((target > 0) & (neg_contribs == total_contribs)) |
        ((target < 0) & (pos_contribs == total_contribs))
    )
    
    # 4. Broadcast this flip_mask back to disaggregated size
    flip_mask_disagg = row_agg.T @ flip_mask @ col_agg  # shape = M.shape
    
    # 5. Flip signs in M where needed
    M = np.where(flip_mask_disagg, -M, M)

    
    if sign_flex is not None:
        M_opposite = -M * sign_flex * M_opposite_eps
        M_stack = np.stack([M, M_opposite], axis=0)
    else:
        M_stack = M[np.newaxis, :, :]
    # proceed as normal
    
    pos = np.where(M > 0, M, 0)
    neg = np.where(M < 0, -M, 0)

    pagg = row_agg @ pos @ (col_agg.T)
    nagg = row_agg @ neg @ col_agg.T
    
    # # Add small value to pagg if target > 0, to nagg if target < 0
    # pagg = np.where(zero_mask & (target > 0), eps, pagg)
    # nagg = np.where(zero_mask & (target < 0), eps, nagg)
    
    

    # Get scaling vector - note creates a vector s of scaling factors
    s = gras_internal(pagg, nagg, target)
    
    s_disagg = row_agg.T @ s @ col_agg

    # Apply scaling
    M_new_stack = np.where(M_stack > 0,
                           M_stack * s_disagg[np.newaxis, :, :],
                           M_stack / s_disagg[np.newaxis, :, :])
    M_new_stack = np.where(s_disagg[np.newaxis, :, :] == 0, 0, M_new_stack)

    # Collapse back to 2D
    M_new = M_new_stack.sum(axis=0)
    


    return M_new, s.flatten()


def gras_scale_table_layers(M, target):
    # note "target" value must be same dimension as table    
    
    # this first check is to add in values if target is non-zero but disagg is all zero. Alternative would be to do a single check at the start of the balancing.
    # Separate positive and negative parts
    pos = np.where(M > 0, M, 0)
    neg = np.where(M < 0, -M, 0)

    
    pos_sum = np.sum(pos, axis=0)
    neg_sum = np.sum(neg, axis=0)
    
    # Get scaling vector - note creates a vector s of scaling factors
    s = gras_internal(pos_sum, neg_sum, target)
    

    # Apply scaling
    M = np.where(M > 0, M * s, M / s)
    M = np.where(s == 0, 0, M)


    return M, s.flatten()



def validate_inputs(V, Ui, Ufd, Uva):
    if not (V.shape[1] == Ui.shape[1] ):
        raise ValueError("All input matrices must have the same number of columns.")
    if not (V.shape[0] == Ui.shape[0]):
        raise ValueError("V and Ui must have the same number of rows.")
    
def check_for_negatives(matrix, name, iteration):
    if (matrix < 0).any():
        raise ValueError(f"Negative values detected in {name} at iteration {iteration}.")


def sut_ras(V,Ui,Ufd,Uva,Umi=None,Umfd=None,ColTot=None,RowTot=None,Vagg=None,Uagg=None,Ufdagg=None,Umagg=None,Umfdagg=None,Uvaagg=None,Vagg_rows=None,Vagg_cols=None,G_va=None,G_fd=None,Uva_sign_flex=None,Ufd_sign_flex=None,U_sign_flex=None,n_iter=None):

    """
    Perform the Supply-Use Table RAS (SUT-RAS) balancing algorithm.

    Parameters:
        V (pd.DataFrame): Supply matrix.
        Ui (pd.DataFrame): Intermediate use matrix.
        Ufd (pd.DataFrame): Final demand matrix.
        Uva (pd.DataFrame): Value-added matrix.
        Umi (pd.DataFrame, optional): Imports matrix. Defaults to zero if not provided.
        Umfd (pd.DataFrame, optional): Imports FD matrix. Defaults to zero if not provided.
        ColTot (np.array, optional): Column totals for supply balancing.
        RowTot (np.array, optional): Row totals for use balancing.

    Returns:
        tuple: Balanced matrices (Vn, Un, Unfd, Unva, Umn).
    """

    # Note - EXOGENOUS COLUMN TOTALS ARE OPTIONAL!
    # Note - check for Um, in USIO, this is not included in balance it seems
    if n_iter is None:
        n_iter=1000
    tolerance = 1e-5
    
    #     startloop
    validate_inputs(V, Ui, Ufd, Uva)

    # Make a copy of the input data    
    if hasattr(Ui, "to_numpy"):
        Un = Ui.copy().to_numpy()
    else:
        Un = Ui.copy()
    if hasattr(Uva, "to_numpy"):
        Unva = Uva.copy().to_numpy()
    else:
        Unva = Uva.copy()
    if hasattr(Ufd, "to_numpy"):
        Unfd = Ufd.copy().to_numpy()
    else:
        Unfd = Ufd.copy()
    if hasattr(V, "to_numpy"):
        Vn = V.copy().to_numpy()
    else:
        Vn = V.copy()


    #Imports - set to zero if not provided
    if Umi is None:
        Umi=Ui*0  
    if hasattr(Umi, "to_numpy"):
        Umn = Umi.copy().to_numpy()
    else:
        Umn = Umi.copy()
    
    #Imports FD - set to zero if not provided
    if Umfd is None:
        Umfd=Ufd*0  
    if hasattr(Umfd, "to_numpy"):
        Umnfd = Umfd.copy().to_numpy()
    else:
        Umnfd = Umfd.copy()

    R_values = np.ones((n_iter, Vn.shape[0]))  # Preallocate R values
    Rm_values = np.ones((n_iter, Vn.shape[0]))  # Preallocate R values
    S_values = np.ones((n_iter, Vn.shape[1]))  # Preallocate S values
    S_valuesV = np.ones((n_iter, Vn.shape[1]))  # Preallocate S values
    if Vagg is not None:
        Sagg_values = np.ones((n_iter, Vagg.values.flatten().shape[0]))  # Preallocate S values
    
    if Uagg is not None:
        S_Uagg_values = np.ones((n_iter, Uagg.values.flatten().shape[0]))  # Preallocate S values
        S_valuesVA = np.ones((n_iter, Uvaagg.values.flatten().shape[0]))  # Preallocate S values
        S_valuesFD = np.ones((n_iter, Ufdagg.values.flatten().shape[0]))  # Preallocate S values
    if Umagg is not None:
        S_Umagg_values = np.ones((n_iter, Umagg.values.flatten().shape[0]))  # Preallocate S values
        S_valuesFDm = np.ones((n_iter, Umfdagg.values.flatten().shape[0]))  # Preallocate S values
        

    
    for i in range(1,n_iter):
        
        
        # Aggregate table scaling first, to ensure everything is in the correct ball-park
        if i < (n_iter+10): # note does not apply on last group of iterations in order to force row/col balances.
            # Scale Supply tables to Summary Tables
        #     print(f"{i}")
            if Vagg is not None:
                [Vn,Sagg_values[i]] = gras_scale_table_agg(Vn, Vagg, Vagg_rows, Vagg_cols)
            if Uagg is not None:
                [Un,S_Uagg_values[i]] = gras_scale_table_agg(Un, Uagg, Vagg_rows, Vagg_cols,U_sign_flex)
                [Unfd,S_valuesFD[i]] = gras_scale_table_agg(Unfd, Ufdagg, Vagg_rows, G_fd, sign_flex=Ufd_sign_flex)
                [Unva, S_valuesVA[i]] = gras_scale_table_agg(Unva, Uvaagg, G_va, Vagg_cols, sign_flex=Uva_sign_flex)

            if Umagg is not None:
                [Umn,S_Umagg_values[i]] = gras_scale_table_agg(Umn, Umagg, Vagg_rows, Vagg_cols,U_sign_flex)
                [Umnfd,S_valuesFDm[i]] = gras_scale_table_agg(Umnfd, Umfdagg, Vagg_rows, G_fd, sign_flex=Ufd_sign_flex)
                # force imports less than total use
                Umn = np.clip(Umn, np.minimum(Un, 0) , np.maximum(Un, 0))
                Umnfd = np.clip(Umnfd, np.minimum(Unfd, 0) , np.maximum(Unfd, 0))


        if np.isnan(Unfd).any():
            breakpoint()
        if np.isnan(Un).any():
            breakpoint()
        if np.isnan(Unva).any():
            breakpoint()
            
            
        
        
        # Supply column totals are calculated. 
        # Use column totals are scaled to match Supply column totals.
        if ColTot is not None:
            Vc=Vn.sum(axis=0)
            mask = Vc != 0  # Identify nonzero denominators
        
            # Only update S where Vc is nonzero - no negatives here
            S_valuesV[i, mask] = ColTot[mask] / Vc[mask]
            Vn *= S_valuesV[i]  # Scale Vn columns
            
            check_for_negatives(Vn, "Vn", i)

        # Column Total scaling
        Vc=Vn.sum(axis=0)
        Uc=Un.sum(axis=0)+Unva.sum(axis=0)
        
        combined_use_matrix = np.vstack((Un, Unva))  # Combine Unva and Umn for joint scaling
        if Uva_sign_flex is not None:
            sign_flex_combined = np.vstack((np.zeros_like(Un, dtype=bool),Uva_sign_flex.values if hasattr(Uva_sign_flex, "values") else Uva_sign_flex))
            [scaled_combined_use_matrix,S_values[i]] = gras_scale_table_totals(combined_use_matrix, Vc, axis=0,sign_flex=sign_flex_combined)
        else:
            [scaled_combined_use_matrix,S_values[i]] = gras_scale_table_totals(combined_use_matrix, Vc, axis=0,sign_flex=None)
        
        # Split the scaled matrix back into Unva and Umn
        Un = scaled_combined_use_matrix[:Un.shape[0], :]  # Take the first rows corresponding to Un
        Unva = scaled_combined_use_matrix[Un.shape[0]:, :]  # Take the remaining rows corresponding to Unva
        if np.isnan(scaled_combined_use_matrix).any():
            breakpoint()


        if RowTot is None:
            RowTot = Vn.sum(axis=1)
            
        # Row Total scaling   
        
        # Combine Un and Unfd for joint scaling    
        combined_use_matrix = np.hstack((Un, Unfd))
        combined_use_import_matrix = np.hstack((Umn, Umnfd))
        if Ufd_sign_flex is not None:
            sign_flex_combined = np.hstack((np.zeros_like(Un, dtype=bool),Ufd_sign_flex.values if hasattr(Ufd_sign_flex, "values") else Ufd_sign_flex))
            [scaled_combined_use_matrix,R_values[i]] = gras_scale_table_totals(combined_use_matrix, RowTot, axis=1,sign_flex=sign_flex_combined)
        else:
            [scaled_combined_use_matrix,R_values[i]] = gras_scale_table_totals(combined_use_matrix, RowTot, axis=1,sign_flex=None)
        
        if np.isnan(scaled_combined_use_matrix).any():
            breakpoint()
        
        # Import use table + import FD = 0 in US tables
        [scaled_combined_use_import_matrix,Rm_values[i]] = gras_scale_table_totals(combined_use_import_matrix, RowTot*0, axis=1,sign_flex=None)
        
        if np.isnan(scaled_combined_use_import_matrix).any():
            breakpoint()
        
            
        # Split the scaled matrix back into Un and Unfd
        Un = scaled_combined_use_matrix[:, :Un.shape[1]]
        Unfd = scaled_combined_use_matrix[:, Un.shape[1]:]
        Umn = scaled_combined_use_import_matrix[:, :Un.shape[1]]
        Umnfd = scaled_combined_use_import_matrix[:, Un.shape[1]:]
        

        
        # # Use row totals are calculated.
        Ur=Un.sum(axis=1)+Unfd.sum(axis=1)
        Vr=Vn.sum(axis=1)
        

        # Check for negative values in R_values
        check_for_negatives(R_values, "R_values", i)
        check_for_negatives(Vn, "Vn", i)
        
      

        print(f"Iteration {i}: min/max S_values = {S_values[i].min()}, {S_values[i].max()}")
        print(f"Iteration {i}: min/max R_values = {R_values[i].min()}, {R_values[i].max()}")
        
        
        Vc=Vn.sum(axis=0)
        Uc=Un.sum(axis=0)+Unva.sum(axis=0)
        Ur=Un.sum(axis=1)+Unfd.sum(axis=1)
        Vr=Vn.sum(axis=1)
        

            # Check for convergence
        if np.allclose(Vc, Uc, atol=tolerance,rtol=0) and np.allclose(Vr, Ur, atol=tolerance,rtol=0):
            print(f"Converged at iteration {i}")
            # 1/0
            return Vn, Un, Unfd, Unva, Umn, Umnfd

    else:
        print(f"Did not converge within {n_iter} iterations")
        
        return Vn, Un, Unfd, Unva, Umn, Umnfd

# endloop 
