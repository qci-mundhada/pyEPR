"""
Maxwell 2D analysis module to use pyEPR.

Contains code to conenct to Ansys and to analyze Maxwell 2D files using the EPR method.

This module handles the micowave part of the 2D analysis.

local file- Naz
"""
from __future__ import print_function  # Python 2.7 and 3 compatibility

import pickle
import sys
import time
from collections import OrderedDict
from pathlib import Path
import pandas as pd
import itertools as it
import numpy as np
import h5py
import itertools as it
from scipy import constants as sc
from scipy import integrate as si
from em_simulations.results import network_data as nd
from pyEPR import ansys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import Dict, config, logger
from .ansys import CalcObject, ConstantVecCalcObject, set_property, ureg
from .calcs.constants import epsilon_0, mu_0
from scipy.constants import epsilon_0 as eps0
from scipy.constants import mu_0 as mu0
from .project_info import ProjectInfo
from .reports import (plot_convergence_f_vspass, plot_convergence_max_df,
                      plot_convergence_maxdf_vs_sol,
                      plot_convergence_solved_elem)
from .toolbox.pythonic import print_NoNewLine


class Maxwell2DAnalysis(object):
    def __init__(self, *args, **kwargs):
        '''
        Pass in the arguments for ProjectInfo. See help for `?ProjectInfo`.

        Parameters:
        -------------------
            project_info    : ProjectInfo
                Suplpy the project info or the parameters to create pinfo

        Use notes:
        -------------------
            * If you change the setup or number of eignemodes in HFSS, etc.
              call `update_ansys_info()`


        Example use:
        -------------------

        See the tutorials in the repository.

        .. code-block:: python
            :linenos:

            import pyEPR as epr
            pinfo = epr.ProjectInfo(project_path = path_to_project,
                                    project_name = 'pyEPR_tutorial1',
                                    design_name  = '1. single_transmon')
            eprh = epr.DistributedAnalysis(pinfo)

        Key internal paramters:
        -------------------
            n_modes (int) : Number of eignemodes; e.g., 2
            variations (List[str]) : ['0', '1']
            _list_variations : List of identifier strings for the HFSS variation. Example  block
                .. code-block:: python

                             ("Height='0.06mm' Lj='13.5nH'",   "Height='0.06mm' Lj='15.3nH'")

                A list of solved variations.  An array of strings corresponding to solved variations.
        '''

        # Get the project info
        project_info = None
        if (len(args) == 1) and (args[0].__class__.__name__ == 'ProjectInfo'):
            # isinstance(args[0], ProjectInfo): # fails on module repload with changes
            project_info = args[0]
        else:
            assert len(args) == 0, '''Since you did not pass a ProjectInfo object
                as a arguemnt, we now assuem you are trying to create a project
                info object here by apassing its arguments. See ProjectInfo.
                It does not take any arguments, only kwargs. \N{face with medical mask}'''
            project_info = ProjectInfo(*args, **kwargs)

        # Input
        self.pinfo = project_info  # : project_info: a reference to a Project_Info class
        if self.pinfo.check_connected() is False:
            self.pinfo.connect()

        # hfss connect module
        self.fields = None
        self.solutions = None
        if self.setup:
            self.fields = self.setup.get_fields()
            self.solutions = self.setup.get_solutions()

        # Stores resutls from sims
        self.results = Dict()  # of variations. Saved results
        # TODO: turn into base class shared with analysis!

        # Modes and variations - the following get updated in update_variation_information
        self.n_modes = int(1)  # : Number of eigenmodes
        #: List of variation indecies, which are strings of ints, such as ['0', '1']
        self.variations = []
        self.variations_analyzed = []  # : List of analyzed variations. List of indecies

        self._nominal_variation = ''  # String identifier
        self.variation_nominal_index = '0'  #: index label
        self._list_variations = ("",)  # tuple set of variables
        # container for eBBQ list of varibles; basically the same as _list_variations
        self._hfss_variables = Dict()

        self._previously_analyzed = set()  # previously analyzed variations

        self.update_ansys_info()

        print('Design \"%s\" info:' % self.design.name)
        print('\t%-15s %d\n\t%-15s %d' % ('# eigenmodes', self.n_modes,
                                          '# variations', self.n_variations))

        # Setup data saving
        self.data_dir = None
        self.file_name = None
        self.setup_data()

    @property
    def setup(self):
        return self.pinfo.setup

    @property
    def design(self):
        return self.pinfo.design

    @property
    def project(self):
        return self.pinfo.project

    @property
    def desktop(self):
        return self.pinfo.desktop

    @property
    def app(self):
        return self.pinfo.app

    @property
    def junctions(self):
        return self.pinfo.junctions

    @property
    def ports(self):
        return self.pinfo.ports

    @property
    def options(self):
        return self.pinfo.options

    @property
    def n_variations(self):
        """ Number of variaitons"""
        return len(self._list_variations)

    def setup_data(self):
        '''
        Set up folder paths for saving data to.

        Sets the save filename with the current time.

        Saves to Path(config.root_dir) / self.project.name / self.design.name
        '''

        if len(self.design.name) > 50:
            logger.error('WARNING!   DESIGN FILENAME MAY BE TOO LONG! ')

        self.data_dir = Path(config.root_dir) / \
            self.project.name / self.design.name
        self.data_filename = self.data_dir / (time.strftime(config.save_format,
                                                            time.localtime()) + '.npz')

        if not self.data_dir.is_dir():
            self.data_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_all_objects(self):
        all_groups = self.design.modeler.get_objects_in_group()
        all_objects = []
        for group in all_groups:
            try:
                objs = self.design.modeler.get_objects_in_group(group)
                all_objects.extend(objs)
            except Exception:
                continue  # Skip groups that error
        return list(set(all_objects))

    '''
    def calc_E_stored(self, variation=None, shape=None, smooth=True):
        calcobject = CalcObject([], self.setup)

        # Get E and compute ε·E = D
        A = calcobject.getQty("E")
        if smooth:
            A = A.smooth()
        A = A.times_eps()

        # Get E* (conjugate) again
        B = calcobject.getQty("E")
        if smooth:
            B = B.smooth()
        B = B.conj()

        # Compute ε·E · E* = D · E*
        C = A.dot(B)
        C = C.real()

        # Choose integration method based on model dimensionality
        if "2D" in self.design.solution_type:
            C = C.integrate_surf(name=shape)
        else:
            C = C.integrate_vol(name=shape)

        lv = self._get_lv(variation)
        return C.evaluate(lv=lv)
    '''
    '''
    def calc_E_stored(self, variation="0", shape="AllObjects", smooth=True):
        setup = self.setup
        calcobject = CalcObject(self, setup)

        vecE = calcobject.getQty("E")
        if smooth:
            vecE = vecE.smooth()
        C = vecE.times_eps()
        B = vecE.conj()
        C = C.dot(B)
        C = C.real()
        C = C.integrate_vol(name=volume)

        # 2D or 3D integration
        if self.design.solution_type.startswith("Electrostatic") or "2D" in self.design.solution_type:
            if shape == "AllObjects":
                shape_list = self._get_all_objects()
                for s in shape_list:
                    C_copy = C.copy()
                    C_copy.integrate_surf(name=s)
                    result = C_copy.evaluate(lv=self._get_lv(variation))
                    print(f"Energy in {s}: {result}")
            else:
                C.integrate_surf(name=shape)
                return C.evaluate(lv=self._get_lv(variation))
        else:
            C.integrate_vol(name=shape)
            return C.evaluate(lv=self._get_lv(variation))
    '''

    def add_E_stored(self, shape='AllObjects', smooth=False, lv = None):
        """
        Calculates the stored electric energy in the given shape.

        Parameters
        ----------
        variation : str
            Variation index or key for which to evaluate the result.
        shape : str
            The name of the geometry object or group over which to integrate.
        smooth : bool
            Whether to apply smoothing to the field data.

        Returns
        -------
        float
            The stored electric field energy in the region.
        """
        setup = self.setup 
        calcobject = CalcObject([], setup)

        vecE = calcobject.getQty("E")
        if smooth:
            vecE = vecE.smooth()

        vecD = calcobject.getQty("D")
        if smooth:
            vecD = vecD.smooth("D")

        C = vecE.dot(vecD)
    
     
        # Integrate over the appropriate domain
        if self.design.solution_type == 'electrostatic':
            C = C.integrate_surf(name=shape)
        else:
            C = C.integrate_vol(name=shape)

        # Evaluate at the specified variation point
    
        # print('vecE:', vecE)
        # print('vecD:', vecD)
        # print('C:', C)
        if lv == None:
            lv = self._get_lv(variation)

        quantity_name = f'E_{shape}' 
        C.save_as(quantity_name)

        return quantity_name
        
        # return C.evaluate_mx2d(lv=lv)


    def add_p(self, shape='AllObjects', smooth=False, lv = None):
        """
        Calculates the stored electric energy in the given shape.

        Parameters
        ----------
        variation : str
            Variation index or key for which to evaluate the result.
        shape : str
            The name of the geometry object or group over which to integrate.
        smooth : bool
            Whether to apply smoothing to the field data.

        Returns
        -------
        float
            The stored electric field energy in the region.
        """
        # Add integration for all objects
        # all_int_name = self.add_E_stored(shape='AllObjects', smooth=smooth, lv = lv)

        # # Add integration for specific shape
        # shape_int_name = self.add_E_stored(shape=shape, smooth=smooth, lv = lv)

        # setup = self.setup 
        # calcobject = CalcObject([], setup)
        # shape_integral = calcobject.getQty(shape_int_name)
        # all_integral = calcobject.getQty(all_int_name)
        
        # shape_integral.__div__(all_integral)

        # if lv == None:
        #     lv = self._get_lv(variation)
 
        # quantity_name = f'p_{shape}'
        # shape_integral.save_as(quantity_name)

        # return quantity_name

        setup = self.setup 

        def create_new_calc_object_with_E_dot_D():
            calcobject = CalcObject([], setup)

            vecE = calcobject.getQty("E")
            if smooth:
                vecE = vecE.smooth()

            vecD = calcobject.getQty("D")
            if smooth:
                vecD = vecD.smooth("D")

            C = vecE.dot(vecD)

            return C

        C1 = create_new_calc_object_with_E_dot_D()
        C2 = create_new_calc_object_with_E_dot_D()

     
        # Integrate over the appropriate domain
        if self.design.solution_type == 'electrostatic':
            C1 = C1.integrate_surf(name=shape)
        else:
            C1 = C1.integrate_vol(name=shape)

        # C.write_stack()

        if self.design.solution_type == 'electrostatic':
            C1 = C1.__div__(C2.integrate_surf(name='AllObjects'))
        else:
            C1 = C1.__div__(C2.integrate_vol(name='AllObjects'))

        # Evaluate at the specified variation point
    
        # print('vecE:', vecE)
        # print('vecD:', vecD)
        # print('C:', C)
        if lv == None:
            lv = self._get_lv(variation)

        quantity_name = f'p_{shape}' 
        C1.save_as(quantity_name)

        return quantity_name

    
    def add_participations(self, variation = None, shapes = 'AllObjects', smooth=False, lv = None, group_name = '0'):
        """
       adds participation expression to the fields calculator in ansys

        
        """
        groups_to_check = [group_name]
        for group in groups_to_check:
            group_objects = self.design.modeler.get_objects_in_group(group)

        print("shapes", shapes)

        # Total energy over all objects
        u_total = self.calc_E_stored(variation=variation, shape= 'AllObjects', smooth=smooth, lv = lv)

        print("u_total:", u_total)

        if u_total == 0:
            raise ValueError(f"Total energy is zero for variation '{variation}'. Cannot compute participation.")

        participations = {}
        for shape in shapes:
            u_shape = self.calc_E_stored(variation=variation, shape=shape, smooth=smooth, lv = lv)
            participations[shape] = u_shape / u_total

        return participations



    def calc_energy_magnetic(self,
                                variation=None,
                                volume='AllObjects',
                                smooth=True):

            calcobject = CalcObject([], self.setup)

            vecH = calcobject.getQty("H")
            if smooth:
                vecH = vecH.smooth()
            A = vecH.times_mu()
            B = vecH.conj()
            A = A.dot(B)
            A = A.real()
            A = A.integrate_vol(name=volume)

            lv = self._get_lv(variation)
            return A.evaluate(lv=lv)

    def calc_p_electric_volume(self,
                                name_dielectric3D,
                                relative_to='AllObjects',
                                E_total=None
                                ):
            r'''
            Calculate the dielectric energy-participatio ratio
            of a 3D object (one that has volume) relative to the dielectric energy of
            a list of object objects.

            This is as a function relative to another object or all objects.

            When all objects are specified, this does not include any energy
            that might be stored in any lumped elements or lumped capacitors.

            Returns:
            ---------
                ℰ_object/ℰ_total, (ℰ_object, _total)
            '''

            if E_total is None:
                logger.debug('Calculating ℰ_total')
                ℰ_total = self.calc_energy_electric(volume=relative_to)
            else:
                ℰ_total = E_total

            logger.debug('Calculating ℰ_object')
            ℰ_object = self.calc_energy_electric(volume=name_dielectric3D)

            return ℰ_object/ℰ_total, (ℰ_object, ℰ_total)
    
    def update_ansys_info(self):
        ''''
        Updates all information about the Ansys solved variations and variables.

        n_modes, _list_variations, nominal_variation, n_variations

        '''

        # from oDesign
        self._nominal_variation = self.design.get_nominal_variation()

        if self.setup:
            # from oSetup
            self._list_variations = self._get_list_variations()

            self.variations = [str(i) for i in range(self.n_variations)]

            # get the nominal index
            self.variation_nominal_index = self._get_nominal_variation_index()

            # eigenmodes
            if self.design.solution_type == 'Eigenmode':
                self.n_modes = int(self.setup.n_modes)
            else:
                self.n_modes = 0

        self._update_ansys_variables()

    def _get_list_variations(self):
        """
        Use: Get a list of solved variations.
        Return Value: An array of strings corresponding to solved variations.
        Example: list = oModule.ListVariations("Setup1 : LastAdaptive")
        """   
        #return self.design._solutions.ListVariations(str(self.setup.solution_name))
        
        
        #print(self.design._solutions.ListVariations)
        #return self.design._solutions.ListVariations(str(self.setup.solution_name))
    
    #def _get_list_variations(self):
        
       #def _get_list_variations(self):
        try:
        # Use raw COM object to check design type
            design_type = self.design._design.GetDesignType().lower()
            print(f"[pyEPR] Detected design type: {design_type}")

            if "hfss" in design_type:
            # HFSS path (original)
                return self.design._solutions.ListVariations(str(self.setup.solution_name))
            else:
            # Maxwell or other designs
                #oModule = self.design._design.GetModule("Optimetrics")
                #sweepNames = oModule.GetSetupNames()

                variations = []
                #for sweep in sweepNames:
                # accumulate variations for all sweeps
                    #sweep_variations = oModule.GetAllSolutionVariationNames(sweep)
                    #variations.extend(sweep_variations)

            if variations:
                return variations
            else:
                return ["Nominal"]  # fallback default

        except Exception as e:
            print("[pyEPR] Failed to get variations:", e)
            return ["Nominal"]

    def _get_nominal_variation_index(self):
            """
            Get the nominal variation index number.
            Returns number as str e.g., '0'
            """
            try:
                return str(self._list_variations.index(self._nominal_variation))
            except:
                return '0'
            

    def get_convergence_vs_pass(self, variation='0'):
        '''
        Returns a convergence vs pass number of the eignemode freqs.
        Makes a plot in HFSS that return a pandas dataframe:
            ```
                re(Mode(1)) [g]	re(Mode(2)) [g]	re(Mode(3)) [g]
            Pass []
            1	4.643101	4.944204	5.586289
            2	5.114490	5.505828	6.242423
            3	5.278594	5.604426	6.296777
            ```
        '''
        return self.hfss_report_f_convergence(variation)
    
    def get_convergence(self, variation='0'):
        '''
        Input:
            variation='0' ,'1', ...

        Returns dataframe:
        ```
                Solved Elements	Max Delta Freq. % Pass Number
            1   	    128955	        NaN
            2       	167607	        11.745000
            3       	192746	        3.208600
            4       	199244	        1.524000
        ```
        '''
        variation = self._list_variations[ureg(variation)]
        df, _ = self.setup.get_convergence(variation)
        return df
    def get_mesh_statistics(self, variation='0'):
        '''
        Input:
            variation='0' ,'1', ...

        Returns dataframe:
        ```
                Name	    Num Tets	Min edge    length	    Max edge length	RMS edge length	Min tet vol	Max tet vol	Mean tet vol	Std Devn (vol)
            0	Region	    909451	    0.000243	0.860488	0.037048	    6.006260e-13	0.037352	0.000029	6.268190e-04
            1	substrate	1490356	    0.000270	0.893770	0.023639	    1.160090e-12	0.031253	0.000007	2.309920e-04
        ```
        '''
        variation = self._list_variations[ureg(variation)]
        return self.setup.get_mesh_stats(variation)

    def _update_ansys_variables(self, variations=None):
        """
        Updates the list of ansys hfss variables for the set of sweeps.
        """
        variations = variations or self.variations
        for variation in variations:
            self._hfss_variables[variation] = pd.Series(
                self.get_variables(variation=variation))
        return self._hfss_variables

    def update_ansys_info(self):
        ''''
        Updates all information about the Ansys solved variations and variables.

        n_modes, _list_variations, nominal_variation, n_variations

        '''

        # from oDesign
        self._nominal_variation = self.design.get_nominal_variation()

        if self.setup:
            # from oSetup
            self._list_variations = self._get_list_variations()

            self.variations = [str(i) for i in range(self.n_variations)]

            # get the nominal index
            self.variation_nominal_index = self._get_nominal_variation_index()

            # eigenmodes
            if self.design.solution_type == 'Eigenmode':
                self.n_modes = int(self.setup.n_modes)
            else:
                self.n_modes = 0

        self._update_ansys_variables()

    def has_fields(self, variation=None):
        '''
        Determine if fields exist for a particular solution.

        variation : str | None
        If None, gets the nominal variation
        '''
        if self.solutions:
            return self.solutions.has_fields(variation)
        else:
            return False
        
    def get_variables(self, variation=None):
        """
        Get ansys variables
        """
        lv = self._get_lv(variation)
        variables = OrderedDict()
        for ii in range(int(len(lv)/2)):
            variables['_'+lv[2*ii][:-2]] = lv[2*ii+1]
        #self.variables = variables
        return variables

    def get_variable_vs_variations(self, variable: str, convert: bool = True):
        """
        Get ansys variables

        Return HFSS variable from self.get_ansys_variables() as a
        pandas series vs variations.

            convert : Convert to a numeric quantity if possible using the
                        ureg
        """
        # TODO: These should be common function to the analysis and here!
        # BOth should be subclasses of a base class
        s = self.get_ansys_variables().loc[variable, :]  # : pd.Series
        if convert:
            s = s.apply(lambda x: ureg.Quantity(x).magnitude)
        return s
    
    def _get_lv(self, variation=None):
        '''
        List of variation variables in a format that is used when feeding back to ansys.

        Returns list of var names and var values.

        Such as ['Lj1:=','13nH', 'QubitGap:=','100um']

        Parameters
        -----------
            variation :  string number such as '0' or '1' or ...
        '''

        if variation is None:
            lv = self._nominal_variation
            lv = self._parse_listvariations(lv)
        else:
            lv = self._list_variations[ureg(variation)]
            lv = self._parse_listvariations(lv)
        return lv

    def _get_lv_EM(self, variation):
        if variation is None:
            lv = self._nominal_variation
            #lv = self.parse_listvariations_EM(lv)
        else:
            lv = self._list_variations[ureg(variation)]
            #lv = self.parse_listvariations_EM(lv)
        return str(lv)

    def _parse_listvariations_EM(self, lv):
        lv = str(lv)
        lv = lv.replace("=", ":=,")
        lv = lv.replace(' ', ',')
        lv = lv.replace("'", "")
        lv = lv.split(",")
        return lv

    def _parse_listvariations(self, lv):
        lv = str(lv)
        lv = lv.replace("=", ":=,")
        lv = lv.replace(' ', ',')
        lv = lv.replace("'", "")
        lv = lv.split(",")
        return lv


