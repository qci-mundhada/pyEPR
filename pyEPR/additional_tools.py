"""
File for additional tools developed by QCI team
"""
import pandas as pd
import itertools as it
import numpy as np
import h5py
import itertools as it
from scipy import constants as sc
from scipy import integrate as si
from em_simulations.results import network_data as nd
from pyEPR import ansys

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

    #print(f1)
    #print(chis)


    swp_indices = chis.index.levels[0]
    mode_indices = chis.index.levels[1]

    f1 = f1[swp_indices]

    #print(swp_indices)

    #print(mode_indices)

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

def analyze_sweep_cavity_loss(epr_hfss):
    modes = range(epr_hfss.n_modes)
    variations = epr_hfss.variations

    all_data = []
    for variation in variations:
        print(f'\n Analyzing variation: ',variation)
        freqs_bare_GHz, Qs_bare = epr_hfss.get_freqs_bare_pd(variation, frame=False)
        SOL = []
        for mode in modes:
            print('\n'f'Mode {mode} at {"%.2f" % freqs_bare_GHz[mode]} GHz   [{mode+1}/{epr_hfss.n_modes}]')
            epr_hfss.set_mode(mode,FieldType='EigenStoredEnergy')
            print('Calculating ℰ_magnetic', end=',')
            epr_hfss.U_H = epr_hfss.calc_energy_magnetic(variation)
            print('ℰ_electric')
            epr_hfss.U_E = epr_hfss.calc_energy_electric(variation)

            sol = pd.Series({'Frequency':freqs_bare_GHz[mode],'U_H': epr_hfss.U_H, 'U_E': epr_hfss.U_E})
            epr_hfss.omega = 2*np.pi*freqs_bare_GHz[mode]
            for seam in epr_hfss.pinfo.dissipative.seams:
                sol=sol.append(epr_hfss.get_Qseam(seam, mode, variation))
            for MA_surface in epr_hfss.pinfo.dissipative.dielectric_MA_surfaces:
                sol=sol.append(epr_hfss.get_Qdielectric_MA_surface(MA_surface, mode, variation))
            for resistive_surface in epr_hfss.pinfo.dissipative.resistive_surfaces:
                sol=sol.append(epr_hfss.get_Qcond_surface(resistive_surface, mode, variation))
            SOL.append(sol)

        SOL = pd.DataFrame(SOL)
        display(SOL)
        all_data.append(SOL) 
    all_data = pd.concat(all_data,keys=variations)
    return all_data

def set_h5_attrs(g, kwargs):
    """Sets attributes of HDF5 group/file g according to dict kwargs.
    Args:
        g (HDF5 group or file): Group or file you would like to update.
        kwargs (dict): Dict of data with which to update g.
    """
    for name, value in kwargs.items():
        print(name)
        if name=='hfss_variables' or name=='fock_trunc'or name=='cos_trunc':
            continue
        if isinstance(value, dict):
            sub_g = g.require_group(name)
            set_h5_attrs(sub_g, value)
        else:
            if isinstance(value, (list, np.ndarray)) and len(value) > 0:
                # if isinstance(value[0], (str, unicode)): #python 2 vs python 3 issue
                if isinstance(value[0], (bytes,str)):
                    g.attrs[name] = _byteify(value)
                else:
                    # create or overwrite dataset
                    # this only works if value has the same shape as original dataset
                    array = np.array(value)
                    ds = g.require_dataset(name, shape=array.shape, dtype=array.dtype, exact=True)
                    ds[...] = array
                    # we could instead do the following to overwrite with data of a different shape:
                    # ds = g.require_dataset(name, shape=array.shape, dtype=array.dtype, exact=True)
                    # del ds
                    # g.create_dataset(name, data=array)
            else:
                g.attrs[name] = value
                
def group_to_dict(group):
    """Recursively load the contents of an h5py group into a dict.
    Args:
        group (h5py group): Group from which you want to load all data.
    Returns:
        target (dict): Dict with contents of group loaded into it.
    """
    target = {}
    for key, value in group.items():
        target[key] = {}
        if hasattr(value, 'attrs') and len(value.attrs):
            target[key].update(group_to_dict(value.attrs))
        if hasattr(value, 'keys'):
            target[key].update(group_to_dict(value))
        elif isinstance(value, h5py.Dataset):
            target[key] = np.array(value)
        else:
            target[key] = value
    return target