"""
File for additional tools developed by QCI team
"""
import pandas as pd
import itertools as it
import numpy as np

def get_cross_kerr_table(epr, swp_variable, numeric):
    """
    Function to re-organize the cross-Kerr results once the quantum analysis is finished
        Parameters:
        -------------------
            epr    : Object of QuantumAnalysis class
            swp_variable : the variable swept in data according to which things will be sorted
            numeric : Whether numerical diagonalization of the data was performed
            
        Use notes:
        -------------------
            * It is assumed the epr.analyze_all_variations has already been called and analysis is finished.
    """
    if numeric:
        f1 = epr.results.get_frequencies_ND(vs=swp_variable)
        chis = epr.get_chis(numeric=numeric,swp_variable=swp_variable)
    else:
        f1 = epr.results.get_frequencies_O1(vs=swp_variable)
        chis = epr.get_chis(numeric=numeric,swp_variable=swp_variable)

    swp_indices = chis.index.levels[0]
    mode_indices = chis.index.levels[1]

    mode_combinations = list(zip(mode_indices,mode_indices))
    diff_mode_combinations = list(it.combinations_with_replacement(mode_indices,2))
    mode_combinations.extend(diff_mode_combinations)

    organized_data = pd.DataFrame({swp_variable:swp_indices})
    organized_data.set_index(swp_variable,inplace=True)

    for mode_indx in mode_indices:
        organized_data['f_'+str(mode_indx)+'(GHz)']=np.round(f1.loc[mode_indx].values/1000,3)
        
    for combo_indx in mode_combinations:
        temp_chi_list = [chis.loc[swp_indx].loc[combo_indx] for swp_indx in swp_indices]
        organized_data['chi_'+str(combo_indx[0])+str(combo_indx[1])+' (MHz)']=np.round(temp_chi_list,4)

    return organized_data

def analyze_sweep_no_junctions(epr_hfss):

    modes = range(epr_hfss.n_modes)
    variations = epr_hfss.variations

    all_data = []
    for variation in variations:
        print(f'\n Analyzing variation: ',variation)
        freqs_bare_GHz, Qs_bare = epr_hfss.get_freqs_bare_pd(variation, frame=False)
        SOL = [] #pd.DataFrame()
        for mode in modes:
            print('\n'f'  \033[1mMode {mode} at {"%.2f" % freqs_bare_GHz[mode]} GHz   [{mode+1}/{epr_hfss.n_modes}]\033[0m')
            epr_hfss.set_mode(mode,FieldType='EigenStoredEnergy')
            print('    Calculating ℰ_magnetic', end=',')
            epr_hfss.U_H = epr_hfss.calc_energy_magnetic(variation)
            print('ℰ_electric')
            epr_hfss.U_E = epr_hfss.calc_energy_electric(variation)
            sol = pd.Series({'Frequency':freqs_bare_GHz[mode],'U_H': epr_hfss.U_H, 'U_E': epr_hfss.U_E})
            epr_hfss.omega = 2*np.pi*freqs_bare_GHz[mode]
            for seam in epr_hfss.pinfo.dissipative.seams:
                sol=sol.append(epr_hfss.get_Qseam(seam, mode, variation))
            SOL.append(sol)
        SOL = pd.DataFrame(SOL)
        all_data.append(SOL)
        display(SOL)
        
    all_data = pd.concat(all_data,keys=variations)

    return all_data