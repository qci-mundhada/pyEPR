"""
File for additional tools developed by QCI team
"""
import pandas as pd
import itertools as it
import numpy as np
import h5py
import itertools as it
from scipy import constants as sc
from em_simulations.results import network_data as nd

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

        # Expecting f0, kappa and chis in GHz
        
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
        


        chis = self.chis
        qubit_index = self.qubit_index
        phis = self.phis
        kappas = self.kappas

        process_dict = {'process_indices':[],
                        'process_individual_freqs': [],
                        'process_freqs': [],
                        'process_bandwidths': [],
                        'process_gs': [],
                        'kappa_eff_resonant':[],
                        'T_induced_resonant':[]   
                        }
        
        for (inds, fs) in zip(process_indices, process_individual_freqs):
            
            if len(inds) is 2:
                process_freq = np.abs(np.sum(fs))/2
                g = phis[inds[0]]*phis[inds[1]]*phis[qubit_index]**2*xi**2*chis[qubit_index,qubit_index] # phi_1*phi_2*phi_q^2*E_J/2 where by 2 since 2 pumps are identical
                if fs[0]==fs[1]:
                    g = g/2 # combinatorial factor for identical modes
                
            if len(inds) is 3:
                process_freq = np.abs(np.sum(fs))
                g = phis[inds[0]]*phis[inds[1]]*phis[inds[2]]*phis[qubit_index]*xi*chis[qubit_index,qubit_index]*2 # phi_1*phi_2*phi_q^2*E_J 
                if fs[0]==fs[1] and fs[1]==fs[2]:
                    g = g/6 # division by 3! for identical modes
                elif fs[0]==fs[1] or fs[1]==fs[2] or fs[2]==fs[1]:
                    g = g/2 # division by 2! since two modes are identical
                else:
                    g=g

            process_bandwidth = np.max(kappas[list(inds)])
            kappa_eff_resonant= 4*g**2/(process_bandwidth)
            T_induced_resonant = (1/(2*np.pi*np.asarray(kappa_eff_resonant)))


            process_dict['process_indices'].append(inds)
            process_dict['process_individual_freqs'].append(fs)        
            process_dict['process_freqs'].append(process_freq)
            process_dict['process_gs'].append(g)
            process_dict['process_bandwidths'].append(process_bandwidth)
            process_dict['kappa_eff_resonant'].append(kappa_eff_resonant)
            process_dict['T_induced_resonant'].append(T_induced_resonant)
 
            
        return process_dict

    def get_double_pump_process_params(self,process_indices,process_individual_freqs,xi_1,xi_2):


        sum_process_dict = {'process_indices':[],
                            'process_individual_freqs': [],
                            'process_freqs': [],
                            'process_bandwidths': [],
                            'process_gs': [],
                            'kappa_eff_resonant':[],
                            'T_induced_resonant':[]   
                           }

        diff_process_dict = {'process_indices':[],
                            'process_individual_freqs': [],
                            'process_freqs': [],
                            'process_bandwidths': [],
                            'process_gs': [],
                            'kappa_eff_resonant':[],
                            'T_induced_resonant':[]   
                           }

        chis = self.chis
        qubit_index = self.qubit_index
        phis = self.phis
        kappas = self.kappas
        
        
        for (inds, fs) in zip(process_indices, process_individual_freqs):
            
            if len(inds) == 2:
                process_freq = np.abs(np.sum(fs))
                g = phis[inds[0]]*phis[inds[1]]*phis[qubit_index]**2*xi_1*xi_2*chis[qubit_index,qubit_index]*2 # phi_1*phi_2*phi_q^2*E_J/2 where by 2 since 2 pumps are identical
                if fs[0]==fs[1]:
                    g = g/2 # combinatorial factor for identical modes

                process_bandwidth = np.max(kappas[list(inds)])
                kappa_eff_resonant = 4*g**2/(process_bandwidth)
                T_induced_resonant = (1/(2*np.pi*np.asarray(kappa_eff_resonant)))

                if fs[0]>0: 
                    process_dict = sum_process_dict
                else:
                    process_dict = diff_process_dict

                process_dict['process_indices'].append(inds)
                process_dict['process_individual_freqs'].append(fs)        
                process_dict['process_freqs'].append(process_freq)
                process_dict['process_gs'].append(g)
                process_dict['process_bandwidths'].append(process_bandwidth)
                process_dict['kappa_eff_resonant'].append(kappa_eff_resonant)
                process_dict['T_induced_resonant'].append(T_induced_resonant)


        return sum_process_dict, diff_process_dict

            

    def _get_kappa_induced(self, frequency, process_dict):
        
        kappa_eff = 0
        kappa_eff_resonant = []

        process_freqs = process_dict['process_freqs']
        process_bandwidths = process_dict['process_bandwidths']
        process_gs = process_dict['process_gs']

        for i in range(len(process_freqs)):
            kappa_eff = kappa_eff + process_gs[i]**2*process_bandwidths[i]/( (process_freqs[i]-frequency)**2 + process_bandwidths[i]**2/4 )
            
        return kappa_eff


    def tidy_up_data_frame(self,df,modify_units=False):

        if not modify_units:

            rename_dict = {'process_freqs':'process_freqs [GHz]', 
                        'process_gs':'process_gs [GHz]',
                        'process_individual_freqs':'process_individual_freqs [GHz]',
                        'process_bandwidths':'process_bandwidths [GHz]',
                        'kappa_eff_resonant':'kappa_eff_resonant [GHz]',
                        'T_induced_resonant':'T_induced_resonant [ns]'}
        else:

            rename_dict = {'process_freqs':'process_freqs [GHz]', 
                        'process_gs':'process_gs [MHz]',
                        'process_individual_freqs':'process_individual_freqs [GHz]',
                        'process_bandwidths':'process_bandwidths [MHz]',
                        'kappa_eff_resonant':'kappa_eff_resonant [kHz]',
                        'T_induced_resonant':'T_induced_resonant [us]'}

            df['process_gs'] = df['process_gs'].map(lambda x: x*1e3)
            df['process_bandwidths'] = df['process_bandwidths'].map(lambda x: x*1e3)
            df['kappa_eff_resonant'] = df['kappa_eff_resonant'].map(lambda x: x*1e6)
            df['T_induced_resonant'] = df['T_induced_resonant'].map(lambda x: x/1e3)

        df.rename(columns=rename_dict,inplace=True)

        return df



    def get_induced_kappa_two_pump(self,mode_index, xi_1, xi_2, frequency_1, frequency_2):
        

        process_indices, process_individual_freqs = self.get_processes(mode_index)
        pump_1_process_dict = self.get_single_pump_process_params(process_indices, process_individual_freqs,xi_1)
        pump_2_process_dict = self.get_single_pump_process_params(process_indices, process_individual_freqs,xi_2)
        sum_process_dict, diff_process_dict = self.get_double_pump_process_params(process_indices,process_individual_freqs,xi_1,xi_2)

        kappa_eff = 0

        process_dicts = [pump_1_process_dict, pump_2_process_dict, sum_process_dict, diff_process_dict]
        frequencies = [frequency_1, frequency_2, np.abs(frequency_1+frequency_2), np.abs(frequency_1-frequency_2)]
        
        partial_kappa_effs = []
        for process_dict, frequency in zip(process_dicts,frequencies):
            tmp =  self._get_kappa_induced(frequency,process_dict)
            kappa_eff = kappa_eff + tmp
            partial_kappa_effs.append(tmp)

        process_dfs = {'pump_1_processes':self.tidy_up_data_frame(pd.DataFrame.from_dict(pump_1_process_dict)),
                       'pump_2_processes':self.tidy_up_data_frame(pd.DataFrame.from_dict(pump_2_process_dict)), 
                       'sum_processes':self.tidy_up_data_frame(pd.DataFrame.from_dict(sum_process_dict)),
                       'diff_processes':self.tidy_up_data_frame(pd.DataFrame.from_dict(diff_process_dict))
                       }
                        
        return kappa_eff, partial_kappa_effs, process_dfs
            
    def get_induced_kappa_vs_frequency(self,mode_index,xi,frequency):
        
        process_indices, process_individual_freqs = self.get_processes(mode_index)
        process_dict= self.get_single_pump_process_params(process_indices, process_individual_freqs,xi)
        
        induced_kappas = self._get_kappa_induced(frequency, process_dict)

        processes = self.tidy_up_data_frame(pd.DataFrame.from_dict(process_dict))

        return induced_kappas, processes

def get_params_for_forest_calc_network(epr,epr_variation,ss,N,qubit_index,modes_to_consider):

    # Always uses the nominal variation for xi_calc also assumes that port for drive pin is 1 and that for junction is 2 

    f0 = epr.results.get_frequencies_O1()[epr_variation].values/1e3
    chis = epr.get_chis(numeric=False).loc[epr_variation].values/1e3/N**2
    xi = np.sqrt(ss/(2*chis[qubit_index,qubit_index]))

    w0 = 2e9*np.pi*f0[modes_to_consider]
    phi_0 = sc.hbar/(2*sc.e)
    LJ = epr.results[epr_variation]['Ljs'][epr.results[epr_variation]['Ljs'].keys()[0]]
    EJ_by_hbar = phi_0**2/LJ/sc.hbar
    phi_zpfs = epr.results[epr_variation]['ZPF'][modes_to_consider][:,0]

    

    return w0, EJ_by_hbar, phi_zpfs, xi


class forest_calc_network(object):

    def __init__(self, w0, EJ_by_hbar, N, phi_zpfs, qubit_index, Z_pp, Z_jp, Z_omegas, R0=50):

        # Expecting f0, kappa and chis in GHz
        
        self.w0= w0  # rad/s
        self.EJ_by_hbar = EJ_by_hbar # rad/s 
        self.N = N
        self.phis = phi_zpfs
        self.qubit_index = qubit_index
        self.Z_pp = Z_pp # Z_port,port Ohm
        self.Z_Jp = Z_jp # Z_junction,port Ohm
        self.Z_ws = Z_omegas # frequencies at which the Zs are evaluated. rad/s
        self.R0 = R0 # 50 Ohm
        self.all_indices = list(range(len(self.w0)))
        self.G = Z_jp/(Z_pp+R0)
        
    def S_phi(self,omega):
        return 2*sc.hbar*self.R0/(omega)

    def G_omega(self,omega):
        return np.interp(omega,self.Z_ws,self.G,left=0, right=0)


    def get_kappa_eff_1pump(self,mode_index,omega_pump,xi):


        
        phi_0 = sc.hbar/(2*sc.e)

        omega_dict = {'pump_frequencies':omega_pump}

        # Conversion to environment
        kappa_conv = 0
        g = (-self.EJ_by_hbar/2)*self.phis[mode_index]*self.phis[self.qubit_index]**2*xi**2/self.N**2
        omega = 2*omega_pump+self.w0[mode_index]
        G = self.G_omega(omega)
        S = self.S_phi(omega)
        kappa_conv += np.abs(g*G/phi_0)**2*S*(omega>0).astype(int)

        omega_dict['omega_conv'] = omega

        omega = -2*omega_pump+self.w0[mode_index]
        G = self.G_omega(omega)
        S = self.S_phi(omega)
        kappa_conv += np.abs(g*G/phi_0)**2*S*(omega>0).astype(int)

        omega_dict['omega_conv_neg_pump'] = omega


        # Two mode squeezing with environement
        kappa_tms = 0
        g = (-self.EJ_by_hbar/2)*self.phis[mode_index]*self.phis[self.qubit_index]**2*xi**2/self.N**2
        omega = 2*omega_pump-self.w0[mode_index]
        G = self.G_omega(omega)
        S = self.S_phi(omega)
        kappa_tms += np.abs(g*G/phi_0)**2*S*(omega>0).astype(int)

        omega_dict['omega_tms'] = omega

        # Simultaneous two-photon loss
        kappa_tpl = 0
        g = (-self.EJ_by_hbar/2)*self.phis[mode_index]**2*self.phis[self.qubit_index]*xi/self.N**2
        omega = omega_pump+2*self.w0[mode_index]
        G = self.G_omega(omega)
        S = self.S_phi(omega)
        kappa_tpl += np.abs(g*G/phi_0)**2*S*(omega>0).astype(int)

        omega_dict['omega_tpl'] = omega

        omega = -omega_pump+2*self.w0[mode_index]
        G = self.G_omega(omega)
        S = self.S_phi(omega)
        kappa_tpl += np.abs(g*G/phi_0)**2*S*(omega>0).astype(int)

        omega_dict['omega_tpl_neg_pump'] = omega

        # Simultaneous two-photon gain
        kappa_tpg = 0
        g = (-self.EJ_by_hbar/2)*self.phis[mode_index]**2*self.phis[self.qubit_index]*xi/self.N**2
        omega = omega_pump-2*self.w0[mode_index]
        G = self.G_omega(omega)
        S = self.S_phi(omega)
        kappa_tpg += np.abs(g*G/phi_0)**2*S*(omega>0).astype(int)

        omega_dict['omega_tpg'] = omega


        tmp = self.all_indices.copy()
        tmp.remove(mode_index)

        kappa_multi_mode = 0

        for other_mode in tmp:

            # Conversion to environment and one other mode
            g = (-self.EJ_by_hbar)*self.phis[mode_index]*self.phis[other_mode]*self.phis[self.qubit_index]*xi/self.N**2
            omega = omega_pump+self.w0[mode_index]-self.w0[other_mode]
            G = self.G_omega(omega)
            S = self.S_phi(omega)
            kappa_multi_mode += np.abs(g*G/phi_0)**2*S*(omega>0).astype(int)

            omega_dict['omega_conv_mode_%d'%other_mode] = omega

            omega = -omega_pump+self.w0[mode_index]-self.w0[other_mode]
            G = self.G_omega(omega)
            S = self.S_phi(omega)
            kappa_multi_mode += np.abs(g*G/phi_0)**2*S*(omega>0).astype(int)

            omega_dict['omega_conv_mode_%d_neg_pump'%other_mode] = omega

            # Exciting the mode of interest, another mode and environment
            g = (-self.EJ_by_hbar)*self.phis[mode_index]*self.phis[other_mode]*self.phis[self.qubit_index]*xi/self.N**2
            omega = omega_pump-self.w0[mode_index]-self.w0[other_mode]
            G = self.G_omega(omega)
            S = self.S_phi(omega)
            kappa_multi_mode += np.abs(g*G/phi_0)**2*S*(omega>0).astype(int)

            omega_dict['omega_tms_mode_%d'%other_mode] = omega

        kappa_eff = kappa_conv + kappa_tms + kappa_tpl + kappa_tpg + kappa_multi_mode

        omega_dict.update({'kappa_conv':kappa_conv,'kappa_tms':kappa_tms,'kappa_tpl':kappa_tpl,'kappa_tpg':kappa_tpg,'kappa_multi_mode':kappa_multi_mode})

        return kappa_eff, pd.DataFrame(omega_dict)


    def get_kappa_eff_2pump(self,mode_index,omega_pump_1,omega_pump_2,xi_1,xi_2):


        
        phi_0 = sc.hbar/(2*sc.e)

        kappa_eff_pump_1, processes_pump_1 = self.get_kappa_eff_1pump(mode_index,omega_pump_1,xi_1)
        kappa_eff_pump_2, processes_pump_2 = self.get_kappa_eff_1pump(mode_index,omega_pump_2,xi_2)

        sum_dict = {'pump_frequency_sum':np.abs(omega_pump_1+omega_pump_2)}
        # Processes enabled by sum of the pumps

        # Conversion to environment
        kappa_sum_conv = 0
        g = (-self.EJ_by_hbar)*self.phis[mode_index]*self.phis[self.qubit_index]**2*xi_1*xi_2/self.N**2
        omega = omega_pump_1 + omega_pump_2 + self.w0[mode_index]
        G = self.G_omega(omega)
        S = self.S_phi(omega)
        kappa_sum_conv += np.abs(g*G/phi_0)**2*S*(omega>0).astype(int)

        sum_dict['omega_sum_conv'] = omega

        omega = -omega_pump_1 - omega_pump_2 + self.w0[mode_index]
        G = self.G_omega(omega)
        S = self.S_phi(omega)
        kappa_sum_conv += np.abs(g*G/phi_0)**2*S*(omega>0).astype(int)

        sum_dict['omega_sum_conv_neg_pump'] = omega

        # Two mode squeezing with environement
        kappa_sum_tms = 0
        g = (-self.EJ_by_hbar)*self.phis[mode_index]*self.phis[self.qubit_index]**2*xi_1*xi_2/self.N**2
        omega = omega_pump_1 + omega_pump_2 -self.w0[mode_index]
        G = self.G_omega(omega)
        S = self.S_phi(omega)
        kappa_sum_tms += np.abs(g*G/phi_0)**2*S*(omega>0).astype(int)

        sum_dict['omega_sum_tms'] = omega
        sum_dict.update({'kappa_sum_conv':kappa_sum_conv,'kappa_sum_tms':kappa_sum_tms})

        kappa_eff_sum = kappa_sum_conv + kappa_sum_tms


        # PRocesses enabled by difference of two pumps

        diff_dict = {'pump_frequency_difference':np.abs(omega_pump_1-omega_pump_2)}
        # Conversion to environment
        kappa_diff_conv = 0
        g = (-self.EJ_by_hbar)*self.phis[mode_index]*self.phis[self.qubit_index]**2*xi_1*xi_2/self.N**2
        omega = omega_pump_1 - omega_pump_2 + self.w0[mode_index]
        G = self.G_omega(omega)
        S = self.S_phi(omega)
        kappa_diff_conv += np.abs(g*G/phi_0)**2*S*(omega>0).astype(int)

        diff_dict['omega_diff_conv'] = omega

        omega = omega_pump_2 - omega_pump_1 + self.w0[mode_index]
        G = self.G_omega(omega)
        S = self.S_phi(omega)
        kappa_diff_conv += np.abs(g*G/phi_0)**2*S*(omega>0).astype(int)

        diff_dict['omega_diff_conv_neg_pump'] = omega

        # Two mode squeezing with environement
        kappa_diff_tms = 0
        g = (-self.EJ_by_hbar)*self.phis[mode_index]*self.phis[self.qubit_index]**2*xi_1*xi_2/self.N**2
        omega = omega_pump_1 - omega_pump_2 -self.w0[mode_index]
        G = self.G_omega(omega)
        S = self.S_phi(omega)
        kappa_diff_tms += np.abs(g*G/phi_0)**2*S*(omega>0).astype(int)

        diff_dict['omega_diff_tms'] = omega

        omega = omega_pump_2 - omega_pump_1 -self.w0[mode_index]
        G = self.G_omega(omega)
        S = self.S_phi(omega)
        kappa_diff_tms += np.abs(g*G/phi_0)**2*S*(omega>0).astype(int)

        diff_dict['omega_diff_tms_neg_pump'] = omega
        diff_dict.update({'kappa_diff_conv':kappa_sum_conv,'kappa_diff_tms':kappa_sum_tms})

        kappa_eff_diff = kappa_diff_conv + kappa_diff_tms

        all_processes = {'pump_1':processes_pump_1,
                         'pump_2':processes_pump_2,
                         'pump_sum':pd.DataFrame(sum_dict),
                         'pump_diff':pd.DataFrame(diff_dict)
                        }



        partial_kappa_effs = [kappa_eff_pump_1,kappa_eff_pump_2,kappa_eff_sum,kappa_eff_diff]
        kappa_eff = kappa_eff_pump_1+kappa_eff_pump_2+kappa_eff_sum+kappa_eff_diff

        return kappa_eff, partial_kappa_effs, all_processes





    



    

        

 

