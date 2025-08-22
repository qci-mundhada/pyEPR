"""
Maxwell 2D analysis module to use pyEPR.

Contains code to conenct to Ansys and to analyze Maxwell 2D files using the EPR method.

This module handles the micowave part of the 2D analysis.

local file- Naz
"""
from __future__ import print_function  # Python 2.7 and 3 compatibility

from pyEPR.core_distributed_analysis import DistributedAnalysis

class Maxwell2DAnalysis(DistributedAnalysis):

    def _get_list_variations(self):
        return ["Nominal"]

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
        if lv == None:
            lv = self._get_lv(variation)

        quantity_name = f'E_{shape}' 
        C.save_as(quantity_name)

        return quantity_name

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
        if lv == None:
            lv = self._get_lv(variation)

        quantity_name = f'p_{shape}' 
        C1.save_as(quantity_name)

        return quantity_name
