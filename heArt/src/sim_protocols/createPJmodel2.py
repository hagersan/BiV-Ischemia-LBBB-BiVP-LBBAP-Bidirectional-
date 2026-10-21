import sys, shutil, math, pdb
import os as os
import numpy as np
from mpi4py import MPI as pyMPI
from dolfin import *
import dolfin as dolfin
from fenicstools import *
from ..utils.oops_objects_MRC2 import State_Variables
from ..ep.EPmodel_cpp2 import EPmodel

import vtk_py3
import vtk

def createPJmodel(IODet, SimDet):
    """
    Initialize the Purkinje fiber network electrophysiology model.

    This function loads the one-dimensional Purkinje network mesh,
    assigns material and branch identifiers, and initializes
    electrophysiology parameters.

    The model supports spatially varying electrical propagation,
    left bundle branch block (LBBB), and ischemia-related
    modifications.

    Initialization
    --------------
    1. Load the Purkinje network mesh from HDF5.
    2. Initialize material and branch-order identifiers.
    3. Load the spatial ischemia mask.
    4. Initialize simulation state variables.
    5. Configure Purkinje electrical propagation parameters.
    6. Define stimulation locations and LBBB settings.
    7. Construct the Purkinje EP model.

    Parameters
    ----------
    IODet : dict
        Input/output configuration specifying the Purkinje
        mesh location, case identifier, and output directory.

    SimDet : dict
        Simulation parameters containing time integration,
        Purkinje diffusion, pacing locations, branch scaling,
        LBBB settings, ischemia, and solver tolerances.

    Returns
    -------
    EPmodel_ : EPmodel
        Initialized Purkinje electrophysiology model.

    state_obj : State_Variables
        Purkinje simulation state variables.

    EPparams_pj : dict
        Parameter dictionary passed to the Purkinje EP model.
    """
    outputfolder = IODet["outputfolder"]
    folderName = IODet["folderName"] + IODet["caseID"] + "/"
    directory_pj = IODet["directory_pj"]
    casename = IODet["casename_pj"]
    delTat = SimDet["dt"]

    # ============================================================
    # PURKINJE NETWORK MESH AND SPATIAL IDENTIFIERS
    # ============================================================
    #
    # Load the one-dimensional Purkinje fiber network from HDF5.
    #
    # Each network segment may contain information describing:
    #
    #   matid:
    #       Material or regional identifier.
    #
    #   branch_order:
    #       Branch identifier used to describe the network
    #       hierarchy and configure branch-dependent properties.
    #
    #   ischemia_mask:
    #       Identifies network segments belonging to the
    #       prescribed ischemic region.
    #
    # These fields are passed to the Purkinje EP model,
    # where they may influence electrical propagation.
    # ============================================================

    mesh_pj = Mesh()
    comm_pj = mesh_pj.mpi_comm()

    meshfilename_pj = directory_pj + casename + ".hdf5"
    f = HDF5File(comm_pj, meshfilename_pj, "r")
    f.read(mesh_pj, casename, False)
    File(outputfolder + folderName + "mesh_pj.pvd") << mesh_pj

    AHAid_pj = MeshFunction("size_t", mesh_pj, 1, mesh_pj.domains())
    AHAid_pj.set_all(0)

    matid_pj = MeshFunction("size_t", mesh_pj, 1, mesh_pj.domains())
    matid_pj.set_all(0)
    f.read(matid_pj, casename + "/" + "matid")

    branch_pj = MeshFunction("size_t", mesh_pj, 1, mesh_pj.domains())
    branch_pj.set_all(0)
    f.read(branch_pj, casename + "/" + "branch_order")    

    ischemia_pj = MeshFunction("size_t", mesh_pj, 1, mesh_pj.domains())
    ischemia_pj.set_all(0)
    f.read(ischemia_pj, casename + "/" + "ischemia_mask")

    # Define state variables
    #  - - - - - - - - - - - -- - - - - - - - - - - - - - - -- - - - - - -
    state_obj = State_Variables(comm_pj, SimDet)
    state_obj.dt.dt = delTat
    pj_intensity = SimDet["PJ_current_intensity"]

    # ============================================================
    # PURKINJE ELECTROPHYSIOLOGY PARAMETERS
    # ============================================================
    #
    # Configure electrical propagation within the Purkinje
    # fiber network.
    #
    # d_pj:
    #     Baseline diffusion parameter for the Purkinje network.
    #
    # branch_scaling:
    #     Parameter controlling branch-dependent modification
    #     of electrical propagation.
    #
    # branch_scaling_mode:
    #     Selects the branch-scaling formulation.
    #
    # ploc:
    #     Prescribed Purkinje stimulation locations.
    #
    # ploc_mode:
    #     Selects how stimulation locations are interpreted
    #     by the Purkinje EP model.
    #
    # The underlying electrophysiology model determines
    # how these parameters enter the governing equations.
    # ============================================================
    
    EPparams_pj = {
        "EPmesh": mesh_pj,
        "deg": 4,
        "state_obj": state_obj,
        "d_iso": SimDet["d_pj"],
        "ploc": SimDet["ploc"],
        "AHAid": AHAid_pj,
        "matid": matid_pj,
        "branchid": branch_pj,
        "branch_scaling": SimDet["branch_scaling"],
        "branch_scaling_mode": SimDet["branch_scaling_mode"],
        "ploc_mode":SimDet["ploc_mode"],
        # ============================================================
        # LBBB AND ISCHEMIA CONFIGURATION
        # ============================================================
        #
        # The Purkinje network supports disease-related
        # modifications of electrical propagation.
        #
        # lbbb:
        #     Enables the prescribed left bundle branch block.
        #
        # lbbb_location:
        #     Identifies the network region affected by LBBB.
        #
        # lbbb_delay:
        #     Parameter controlling the prescribed conduction
        #     modification in the affected region.
        #
        # Ischemia:
        #     Enables ischemia-related modifications.
        #
        # ischemia_mask:
        #     Identifies Purkinje segments affected by ischemia.
        #
        # The precise conductivity changes and spatial
        # application of these parameters are implemented
        # within EPmodel_cpp2.py.
        # ============================================================        
        "lbbb":SimDet["lbbb"],
        "lbbb_delay": SimDet["lbbb_delay"], 
        "lbbb_location": SimDet["lbbb_location"], 
        "abs_tol": SimDet["abs_tol"],
        "rel_tol": SimDet["rel_tol"], 
        "Ischemia": SimDet["Ischemia"],  
        "ischemia_mask": ischemia_pj,
    }

    if("d_iso_pj" in list(SimDet.keys())):
        EPparams_pj.update({"d_iso_pj": SimDet["ploc"]})

    # Define EP model and solver
    EPmodel_ = EPmodel(EPparams_pj)

    return EPmodel_, state_obj, EPparams_pj