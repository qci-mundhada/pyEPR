"""
File for additional tools developed by QCI team
"""
import pandas as pd
import itertools as it
import numpy as np
import h5py
import itertools as it

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

    print(f1)
    print(chis)


    swp_indices = chis.index.levels[0]
    mode_indices = chis.index.levels[1]

    print(mode_indices)

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


def get_params_for_forest_calc(epr,variation,ss,N, qubit_index,round_to=3):

    f0 = np.round(epr.results.get_frequencies_O1()[variation].values/1e3,round_to)
    chis = epr.get_chis(numeric=False).loc[variation].values/1e3/N**2
    kappas = f0/epr.results[variation]['Qs'].values
    xi = np.sqrt(ss/(2*chis[qubit_index,qubit_index]))

    return f0, chis, kappas, xi



class forest_calc(object):

    def __init__(self, f0, kappas, chis, qubit_index, ignore_modes = []):
        
        self.f0= f0
        self.kappas = kappas
        self.chis = chis
        self.all_indices = list(range(len(self.f0)))
        self.qubit_index = qubit_index
        self.phis = [(self.chis[i,i]/self.chis[self.qubit_index,self.qubit_index])**0.25 for i in range(len(f0))]

        for mode in ignore_modes:
            self.all_indices.remove(mode)

    def update_mode_indices(ignore_modes):
        
        for mode in ignore_modes:
            self.all_indices.remove(mode)

    def get_processes(self,mode_index):
    
        other_modes = self.all_indices.copy()
        other_modes.remove(mode_index)
        f0 = self.f0

        process_indices = []
        process_individual_freqs = []
        
        process_indices.extend([(mode_index, i) for i in other_modes])
        process_indices.extend([(mode_index, i) for i in other_modes])
        process_individual_freqs = [(f0[mode_index], f0[i]) for i in other_modes] # two mode squeezing
        process_individual_freqs.extend([(-f0[mode_index], f0[i]) for i in other_modes]) # conversion
        
        process_indices.extend([(mode_index, mode_index, i) for i in other_modes])
        process_indices.extend([(mode_index, mode_index, i) for i in other_modes])
        process_individual_freqs.extend([(f0[mode_index], f0[mode_index], f0[i]) for i in other_modes]) # two photon gains on mode
        process_individual_freqs.extend([(-f0[mode_index], -f0[mode_index], f0[i]) for i in other_modes]) # two photon loss on mode
        
        process_indices.extend((mode_index, *combo) for combo in it.combinations_with_replacement(other_modes, 2))
        process_indices.extend((mode_index, *combo) for combo in it.combinations_with_replacement(other_modes, 2))    
        process_individual_freqs.extend((f0[mode_index],*f0[list(combo)]) for combo in it.combinations_with_replacement(other_modes,2)) # one photon loss and multiple modes photon gain
        process_individual_freqs.extend((-f0[mode_index],*f0[list(combo)]) for combo in it.combinations_with_replacement(other_modes,2)) # mode and other modes gain
                
            
        return process_indices, process_individual_freqs

    def get_single_pump_process_params(self,process_indices,process_individual_freqs,xi):
        
        process_freqs = []
        process_bandwidths = []
        process_gs = []

        chis = self.chis
        qubit_index = self.qubit_index
        phis = self.phis
        kappas = self.kappas
        
        
        for (inds, fs) in zip(process_indices, process_individual_freqs):
            
            if len(inds) is 2:
                process_freqs.append(np.abs(np.sum(fs))/2)
                g = phis[inds[0]]*phis[inds[1]]*phis[qubit_index]**2*xi**2*chis[qubit_index,qubit_index] # phi_1*phi_2*phi_q^2*E_J/2 where by 2 since 2 pumps are identical
                if fs[0]==fs[1]:
                    g = g/2 # combinatorial factor for identical modes
                
            if len(inds) is 3:
                process_freqs.append(np.abs(np.sum(fs)))
                g = phis[inds[0]]*phis[inds[1]]*phis[inds[2]]*phis[qubit_index]*xi*chis[qubit_index,qubit_index]*2 # phi_1*phi_2*phi_q^2*E_J 
                if fs[0]==fs[1] and fs[1]==fs[2]:
                    g = g/6 # division by 3! for identical modes
                elif fs[0]==fs[1] or fs[1]==fs[2] or fs[2]==fs[1]:
                    g = g/2 # division by 2! since two modes are identical
                else:
                    g=g
            process_gs.append(g)
            process_bandwidths.append(np.max(kappas[list(inds)]))
            
        return process_freqs, process_bandwidths, process_gs
            

    def get_kappa_induced(self, mode_index, frequency, process_freqs, process_bandwidths, process_gs):
        
        kappa_eff = 0
        kappa_eff_resonant = []
        
        for i in range(len(process_freqs)):
            kappa_eff = kappa_eff + process_gs[i]**2*process_bandwidths[i]/( (process_freqs[i]-frequency)**2 + process_bandwidths[i]**2/4 )
            kappa_eff_resonant.append(4*process_gs[i]**2/(process_bandwidths[i]))
            
        return kappa_eff, kappa_eff_resonant
            
    def get_induced_kappa_vs_frequency(self,mode_index,xi,frequencies):
        
        process_indices, process_individual_freqs = self.get_processes(mode_index)
        process_freqs, process_bandwidths, process_gs = self.get_single_pump_process_params(process_indices, process_individual_freqs,xi)
        
        induced_kappas, kappa_eff_resonant = self.get_kappa_induced(mode_index, frequencies, process_freqs, process_bandwidths, process_gs)
        
        T_induced_resonant = list(1/(2*np.pi*np.asarray(kappa_eff_resonant)))
        
        
        dict_of_things_to_return = {'mode_indices':process_indices,
                                    'process_individual_freqs [GHz]':process_individual_freqs,
                                    'process_freqs [GHz]':process_freqs,
                                    'process_bandwidths [GHz]': np.asarray(process_bandwidths),
                                    'process_gs [GHz]': np.asarray(process_gs),
                                    'kappa_eff_resonant [GHz]': np.asarray(kappa_eff_resonant),
                                    'T_induced_resonant [ns]': np.asarray(T_induced_resonant)
                                }

    #     dict_of_things_to_return = {'mode_indices':process_indices,
    #                                 'process_individual_freqs [GHz]':process_individual_freqs,
    #                                 'process_freqs [GHz]':process_freqs,
    #                                 'process_bandwidths [MHz]': np.asarray(process_bandwidths)*1e3,
    #                                 'process_gs [MHz]': np.asarray(process_gs)*1e3,
    #                                 'kappa_eff_resonant [kHz]': np.asarray(kappa_eff_resonant)*1e6,
    #                                 'T_induced_resonant [us]': np.asarray(T_induced_resonant)/1e3
    #                              }
        
        processes = pd.DataFrame.from_dict(dict_of_things_to_return)

        return induced_kappas, processes




    

