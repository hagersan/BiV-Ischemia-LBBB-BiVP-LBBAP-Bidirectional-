import sys, shutil, math, pdb
import os as os
import numpy as np
from mpi4py import MPI as pyMPI
from dolfin import *
import dolfin as dolfin
from fenicstools import *

import vtk_py3
import vtk

from ..utils.oops_objects_MRC2 import State_Variables
from ..utils.oops_objects_MRC2 import load_ischemia_mask
from ..ep.EPmodel_basic_test import EPmodel

def createEPmodel(IODet, SimDet):
    """
    Initialize the myocardial electrophysiology (EP) model.

    This function loads the myocardial mesh, fiber architecture,
    material identifiers, and ischemic regions before creating
    the electrophysiology model.

    The function also configures electrical conductivity,
    stimulation locations, CRT parameters, and simulation
    state variables.

    Model initialization
    --------------------
    1. Load the myocardial EP mesh and boundary markers.
    2. Load myocardial material and AHA segment identifiers.
    3. Load the ischemic-region mask.
    4. Initialize myocardial fiber, sheet, and normal directions.
    5. Define electrical conductivity parameters.
    6. Configure Purkinje-myocardial and CRT stimulation sites.
    7. Construct the myocardial EP model.

    Parameters
    ----------
    IODet : dict
        Input/output configuration containing mesh filenames,
        directories, case identifiers, and output locations.

    SimDet : dict
        Simulation parameters controlling electrical properties,
        ischemia, pacing, CRT, and time integration.

    Returns
    -------
    EPmodel_ : EPmodel
        Initialized myocardial electrophysiology model.

    state_obj : State_Variables
        Simulation state containing time integration parameters.

    EPparams : dict
        Parameters passed to the myocardial EP model, including
        mesh information, fiber fields, diffusion parameters,
        ischemia settings, and stimulation locations.
    """

    directory_ep = IODet["directory_ep"]
    outputfolder = IODet["outputfolder"]
    folderName = IODet["folderName"] + IODet["caseID"] + "/"
    delTat = SimDet["dt"]
    casename_marked = IODet["casename_ep_marked"]   

    if "casename_ep" in IODet:
        casename = IODet["casename_ep"]
    else:
        casename = IODet["casename"]

    if "isFCH" in list(SimDet.keys()):
        isFCH = SimDet["isFCH"]
    else:
        isFCH = False

    if "iswaorta" in list(SimDet.keys()):
        iswaorta = SimDet["iswaorta"]
    else:
        iswaorta = False  # Default

    # ============================================================
    # MYOCARDIAL EP MESH AND FIBER ARCHITECTURE
    # ============================================================
    #
    # Load the myocardial finite-element mesh from HDF5.
    #
    # The mesh contains:
    #   - Element connectivity and nodal coordinates
    #   - Boundary facet identifiers
    #   - Material identifiers
    #   - AHA segment identifiers, when available
    #
    # A separate marked mesh is used to load the ischemic
    # region mask.
    #
    # The local myocardial coordinate system is defined by:
    #
    #   f0 : Local fiber direction
    #   s0 : Local sheet direction
    #   n0 : Local sheet-normal direction
    #
    # Fiber directions are loaded either from DTI-derived
    # fields or prescribed fiber architecture.
    #
    # These directional fields are subsequently used by
    # the electrophysiology model to construct the
    # anisotropic electrical diffusion tensor.
    # ============================================================

    mesh_ep = Mesh()
    comm_common = mesh_ep.mpi_comm()

    meshfilename_ep = directory_ep + casename + ".hdf5"
    f = HDF5File(comm_common, meshfilename_ep, "r")
    f.read(mesh_ep, casename, False)
    if isFCH:
        mesh_ep.scale(6.5e-2)
    elif iswaorta:
        mesh_ep.scale(1.1)
        # pass

    File(outputfolder + folderName + "mesh_ep.pvd") << mesh_ep

    facetboundaries_ep = MeshFunction("size_t", mesh_ep, 2)
    f.read(facetboundaries_ep, casename + "/" + "facetboundaries")

    matid_ep = MeshFunction("size_t", mesh_ep, mesh_ep.topology().dim())
    AHAid_ep = MeshFunction("size_t", mesh_ep, mesh_ep.topology().dim())

    if f.has_dataset(casename + "/" + "matid"):
        if SimDet.get("function_matid"):
            VQuadelem = FiniteElement("DG", mesh_ep.ufl_cell(), degree=0, quad_scheme="default")
            matid_FS = FunctionSpace(mesh_ep, VQuadelem)
            matid_func = dolfin.Function(matid_FS)
            for cell in cells(mesh_ep):
                matid_ep[cell.index()] = round(matid_func(cell.midpoint()))
        else:
            f.read(matid_ep, casename + "/" + "matid")
    else:
        matid_ep.set_all(0)

    if f.has_dataset(casename + "/" + "AHAid"):
        f.read(AHAid_ep, casename + "/" + "AHAid")
    else:
        AHAid_ep.set_all(0)

    mesh_ep_marked = Mesh()
    comm_common_marked = mesh_ep_marked.mpi_comm()
    meshfilename_ep_marked = directory_ep + casename_marked + ".hdf5"
    f_marked = HDF5File(comm_common_marked, meshfilename_ep_marked, "r")
    f_marked.read(mesh_ep_marked, casename_marked, False)

    # Load ischemia mask for EP mesh
    ischemia_mask_ep = load_ischemia_mask(mesh_ep_marked, directory_ep, casename_marked)

    deg_ep = 4
    Quadelem_ep = FiniteElement("Quadrature", mesh_ep.ufl_cell(), degree=deg_ep, quad_scheme="default"    )
    Quadelem_ep._quad_scheme = "default"
    Quad_ep = FunctionSpace(mesh_ep, Quadelem_ep)

    if "fiber_fspace" in list(SimDet.keys()) and "fiber_fspace_deg" in list(
        SimDet.keys()):
        VQuadelem_ep = VectorElement(SimDet["fiber_fspace"],mesh_ep.ufl_cell(), degree=SimDet["fiber_fspace_deg"],quad_scheme="default",)
        VQuadelem_ep._quad_scheme = "default"
    else:
        VQuadelem_ep = VectorElement(
            "Quadrature", mesh_ep.ufl_cell(), degree=deg_ep, quad_scheme="default"
        )
        VQuadelem_ep._quad_scheme = "default"

    fiberFS_ep = FunctionSpace(mesh_ep, VQuadelem_ep)

    f0_ep = Function(fiberFS_ep)
    s0_ep = Function(fiberFS_ep)
    n0_ep = Function(fiberFS_ep)

    if SimDet["DTI_EP"] is True:
        f.read(f0_ep, casename + "/" + "eF_DTI")
        f.read(s0_ep, casename + "/" + "eS_DTI")
        f.read(n0_ep, casename + "/" + "eN_DTI")
    else:
        f.read(f0_ep, casename + "/" + "eF")
        f.read(s0_ep, casename + "/" + "eS")
        f.read(n0_ep, casename + "/" + "eN")

    f.close()

    comm_ep = mesh_ep.mpi_comm()

    # Define state variables
    #  - - - - - - - - - - - -- - - - - - - - - - - - - - - -- - - - - - -
    state_obj = State_Variables(comm_ep, SimDet)
    state_obj.dt.dt = delTat
    #  - - - - - - - - - - - -- - - - - - - - - - - - - - - -- - - - - - -

    # ============================================================
    # MYOCARDIAL ELECTRICAL CONDUCTIVITY PARAMETERS
    # ============================================================
    #
    # Define the parameters controlling electrical propagation
    # through myocardial tissue.
    #
    # d_iso:
    #   Isotropic diffusion parameter.
    #
    # d_ani_factor:
    #   Parameter controlling the anisotropic contribution
    #   to myocardial electrical diffusion.
    #
    # ani_factor:
    #   Additional scaling parameter used by the
    #   electrophysiology model.
    #
    # The complete diffusion tensor is constructed within
    # the EPmodel class using these parameters and the
    # myocardial fiber architecture.
    #
    # The exact units and parameter relationships should
    # be documented alongside the tensor formulation
    # in EPmodel_basic_test.py.
    # ============================================================

    if "d_iso" in list(SimDet.keys()):
        d_iso = SimDet["d_iso"]
    else:
        d_iso = 0.02

    if "d_ani_factor" in list(SimDet.keys()):
        d_ani_factor = SimDet["d_ani_factor"]
    else:
        d_ani_factor = 0.02

    if "ani_factor" in list(SimDet.keys()):
        ani_factor = SimDet["ani_factor"]
    else:
        ani_factor = 1000.0

    if "CRT" in list(SimDet.keys()):
        CRT = SimDet["CRT"]
    else:
        CRT = False

    if "CRT_current_intensity" in list(SimDet.keys()):
        CRT_current_intensity = SimDet["CRT_current_intensity"]
    else:
        CRT_current_intensity = 0.0        

    if "CRT_pos" in list(SimDet.keys()):
        CRT_pos = SimDet["CRT_pos"]
    else:
        CRT_pos = [0.0,0.0,0.0]

    if "CRT_pacing_timing" in list(SimDet.keys()):
        CRT_pacing_timing = SimDet["CRT_pacing_timing"]
    else:
        CRT_pacing_timing = [[0.0, 0.0]]

    EPparams = {
        "EPmesh": mesh_ep,
        "deg": 4,
        "matid": matid_ep,
        "facetboundaries": facetboundaries_ep,
        "f0": f0_ep,
        "s0": s0_ep,
        "n0": n0_ep,
        "state_obj": state_obj,
        "d_iso": d_iso,
        "d_ani": d_ani_factor,
        "ani_factor": ani_factor,
        "AHAid": AHAid_ep,
        "matid": matid_ep,
        # ============================================================
        # ISCHEMIC REGION CONFIGURATION
        # ============================================================
        #
        # The ischemic region is identified using a spatial mask
        # loaded from the marked myocardial mesh.
        #
        # Ischemia:
        #   Enables the ischemic model configuration.
        #
        # Ischemia_mask:
        #   Identifies the spatial distribution of ischemic tissue.
        #
        # Ischemic_ep_factor:
        #   Parameter controlling ischemia-related modifications
        #   to myocardial electrical properties.
        #
        # Ischemic_stiffness_factor:
        #   Mechanical stiffness parameter passed through the
        #   model configuration.
        #
        # The precise application of these factors is defined
        # within the corresponding EP and mechanics models.
        # ============================================================        
        "Ischemia": SimDet["Ischemia"],
        "Ischemia_mask": ischemia_mask_ep,
        "Ischemic_stiffness_factor": SimDet["Ischemic_stiffness_factor"],  
        "Ischemic_ep_factor": SimDet["Ischemic_ep_factor"],     
        "abs_tol": SimDet["abs_tol"],
        "rel_tol": SimDet["rel_tol"],
        "CRT": CRT,  
        "CRT_current_intensity": CRT_current_intensity,
        "CRT_pos": CRT_pos,   
        "CRT_pacing_timing": CRT_pacing_timing,
        "CRT_eps_factor": SimDet["CRT_eps_factor"],   
    }

    if "ploc" in list(SimDet.keys()):
        EPparams.update({"ploc": SimDet["ploc"]})

    if "pacing_timing" in list(SimDet.keys()):
        EPparams.update({"pacing_timing": SimDet["pacing_timing"]})

    if "isPJ" in SimDet and SimDet["isPJ"] and "pj_tnodes" in SimDet:
        pj_tnodes = SimDet["pj_tnodes"]
    else:
        pj_tnodes = []

    crt_pos = SimDet.get("CRT_pos", []) if SimDet.get("CRT", False) else []

    # ============================================================
    # PURKINJE-MYOCARDIAL AND CRT STIMULATION LOCATIONS
    # ============================================================
    #
    # Myocardial stimulation sites may originate from:
    #
    # 1. Purkinje-myocardial junctions (PMJs)
    #    Electrical activation transmitted from the Purkinje
    #    network to the myocardium.
    #
    # 2. CRT pacing sites
    #    Direct electrical stimulation of the myocardium
    #    through prescribed external pacing locations.
    #
    # Both sets of coordinates are combined into one list
    # used by the myocardial EP model to allocate
    # stimulation sources.
    #
    # The ordering is important:
    #
    #   [PMJ_0, ..., PMJ_(nPJ-1),
    #    CRT_0, ..., CRT_(nCRT-1)]
    #
    # Therefore:
    #
    #   PMJ indices : 0 to nPJ-1
    #   CRT indices : nPJ to nPJ+nCRT-1
    #
    # The coupling classes use this indexing convention
    # to control their respective stimulation sources.
    #
    # This prevents CRT stimulation from directly
    # replacing the PMJ stimulation coefficients.
    #
    # Reverse myocardium-to-Purkinje coupling is handled
    # separately within coupleMyoandPJ_bidirectional.py.
    # ============================================================

    ploc_all = list(pj_tnodes) + list(crt_pos)
    nPJ  = len(pj_tnodes)
    nCRT = len(crt_pos)

    # --- Update EPparams ---
    EPparams.update({"ploc": ploc_all,"nPJ":  nPJ,"nCRT": nCRT})    

    if SimDet["isPJ"] and "PJ_current_intensity" in list(SimDet.keys()):
        EPparams.update({"current_intensity": SimDet["current_intensity"]})

    if "Ischemia" in list(SimDet.keys()):
        EPparams.update({"Ischemia": SimDet["Ischemia"]})

    # Attach to params for use by EPModel
    EPparams["Ischemia_mask"] = ischemia_mask_ep    

    # Define EP model and solver
    EPmodel_ = EPmodel(EPparams)

    return EPmodel_, state_obj, EPparams