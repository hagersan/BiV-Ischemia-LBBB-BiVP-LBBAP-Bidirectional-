import sys, shutil, math, pdb
import os as os
import numpy as np
from mpi4py import MPI as pyMPI

from .coupleEPandPJ import coupleEPandPJ
from .coupleMyoandPJ_bidirectional import coupleMyoandPJ
from .coupleEPandCRT import coupleEPandCRT
from .createEPmodel import createEPmodel
from .createPJmodel2 import createPJmodel
from scipy.spatial import cKDTree

import warnings
from ffc.quadrature.deprecation import QuadratureRepresentationDeprecationWarning
warnings.simplefilter("ignore", QuadratureRepresentationDeprecationWarning)


from dolfin import *
import dolfin
from fenicstools import *

import vtk_py3
import vtk

from ..utils.oops_objects_MRC2 import printout
from ..utils.oops_objects_MRC2 import biventricle_mesh as biv_mechanics_mesh
from ..utils.oops_objects_MRC2 import lv_mesh as lv_mechanics_mesh

from ..utils.oops_objects_MRC2 import State_Variables
from ..utils.oops_objects_MRC2 import update_mesh
from ..utils.oops_objects_MRC2 import exportfiles
from ..utils.oops_objects_MRC2 import load_ischemia_mask
from ..utils.oops_objects_MRC2 import build_active_indicator

from ..utils.mesh_scale_create_fiberFiles import create_EDFibers
from ..utils.oops_objects_MRC2 import json_serialize

from ..ep.EPmodel_basic_test import EPmodel
from ..mechanics.MEmodel3_debug import MEmodel
from .circ import CLmodel
from .circBiV import CLmodel as CLmodel_biv
from ..postprocessing.postprocessdatalib2 import *
import json

from ..mechanics.JRp import *


def run_BiV_ClosedLoop(IODet, SimDet):
    """
    Run a coupled cardiac electrophysiology (bidirectional), mechanics, and
    closed-loop circulation simulation.

    The framework supports left-ventricular and biventricular
    configurations, with optional Purkinje network activation
    and cardiac resynchronization therapy (CRT).

    Model components
    ----------------
    1. Myocardial electrophysiology (EP)
       Simulates electrical excitation in the myocardium.

    2. Purkinje electrophysiology (PJ)
       Simulates excitation propagation through a discrete
       Purkinje fiber network when enabled.

    3. Electromechanical coupling (EP-ME)
       Transfers myocardial electrical activation to the
       mechanics model to determine active contraction.

    4. Cardiac mechanics (ME)
       Computes myocardial deformation and cavity volumes
       using a finite-element formulation.

    5. Closed-loop circulation (CL)
       Describes systemic and pulmonary hemodynamics and
       provides cavity volume constraints.

    6. Cardiac resynchronization therapy (CRT)
       Supports externally applied electrical stimulation,
       with optional interactions with the Purkinje network.

    Simulation workflow
    -------------------
    Initialization:
        - Create the EP and optional Purkinje models.
        - Create the cardiac mechanics model.
        - Initialize the electromechanical coupling.
        - Apply pressure loading to the reference geometry.
        - Initialize the closed-loop circulation.

    Time integration:
        - Update circulatory volumes.
        - Solve for cavity pressures consistent with
          the prescribed volumes.
        - Advance simulation time.
        - Apply electrical stimulation.
        - Solve myocardial and Purkinje electrophysiology.
        - Exchange electrical activation information.
        - Update active contraction in the mechanics model.
        - Export simulation results.

    Parameters
    ----------
    IODet : dict
        Input/output configuration specifying mesh locations,
        case identifiers, and output directories.

    SimDet : dict
        Simulation configuration containing time integration,
        material parameters, electrophysiology, pacing,
        mechanical boundary conditions, and circulation
        parameters.

    Returns
    -------
    None
        Results are written to the configured output directory.

    Notes
    -----
    The simulation uses a partitioned coupling strategy.
    The electrical, mechanical, and circulatory models
    are advanced sequentially.

    See the individual model classes for the governing
    equations and numerical discretizations.
    """

    flags = ["-O3", "-ffast-math", "-march=native"]
    parameters["form_compiler"]["representation"] = "uflacs"

    casename_me = IODet["casename_me"]
    casename_me_marked = IODet["casename_me_marked"]    
    casename_ep = IODet["casename_ep"]
    casename_ep_marked = IODet["casename_ep_marked"]    
    directory_me = IODet["directory_me"]
    outputfolder = IODet["outputfolder"]
    folderName = IODet["folderName"] + IODet["caseID"] + "/"
    delTat = SimDet["dt"]

    if "fiber_fspace_deg" in SimDet:
        deg = SimDet["fiber_fspace_deg"]
    else:
        deg = 4 # Default
        
    parameters["form_compiler"]["quadrature_degree"] = deg

    if "HomogenousActivation" in list(SimDet["GiccioneParams"].keys()):
        ishomo = SimDet["GiccioneParams"]["HomogenousActivation"]
    else:
        ishomo = True  # Default

    if "isLV" in list(SimDet.keys()):
        isLV = SimDet["isLV"]
    else:
        isLV = False  # Default

    if "iswaorta" in list(SimDet.keys()):
        iswaorta = SimDet["iswaorta"]
    else:
        iswaorta = False  # Default

    if "isFCH" in list(SimDet.keys()):
        isFCH = SimDet["isFCH"]
    else:
        isFCH = False  # Default

    if "isBiV" in list(SimDet.keys()):
        isBiV = SimDet["isBiV"]
    else:
        isBiV = False  # Default

    if "isPJ" in list(SimDet.keys()):
        isPJ = SimDet["isPJ"]
    else:
        isPJ = False  # Default

    if not ishomo:
        intensity = SimDet["current_intensity"]

    if isPJ:
        pj_intensity = SimDet["PJ_current_intensity"]

    # Define Purkinje terminal nodes
    if isPJ:
        if "pj_tnodes" in list(SimDet.keys()):
            if isinstance(SimDet["pj_tnodes"], str):
                pj_t_nodes = np.genfromtxt(os.path.join(IODet["directory_pj"], SimDet["pj_tnodes"]), delimiter=',')

            elif isinstance(SimDet["pj_tnodes"], list):
                pj_t_nodes = np.array(SimDet["pj_tnodes"])

            SimDet.update({"pj_tnodes": list(pj_t_nodes)})

        # Create Purkinje PJ model
        EPmodel_pj, state_obj_pj, PJparams = createPJmodel(IODet, SimDet)
        solver_FHN_pj = EPmodel_pj.Solver()

    # Create Tissue EP model
    EPmodel_ep, state_obj, EPparams = createEPmodel(IODet, SimDet)
    solver_FHN_ep = EPmodel_ep.Solver()
    comm_common = EPmodel_ep.mesh.mpi_comm()
    comm_ep = EPmodel_ep.mesh.mpi_comm()

    #  - - - - - - - - - - - -- - - - - - - - - - - - - - - -- - - - - - -
    # Mechanics Mesh
    mesh_me = Mesh()
    mesh_me_params = {
        "directory": directory_me,
        "casename": casename_me,
        "casename_marked": casename_me_marked,        
        "fibre_quad_degree": 4,
        "outputfolder": outputfolder,
        "foldername": folderName,
        "state_obj": state_obj,
        "common_communicator": comm_common,
        "MEmesh": mesh_me,
        "isLV": isLV,
    }

    MEmodel_ = MEmodel(mesh_me_params, SimDet)
    solver_elas = MEmodel_.Solver()
    comm_me = MEmodel_.mesh_me.mpi_comm()

    # Define export for exporting data
    export = exportfiles(comm_me, comm_ep, IODet, SimDet)

    if "AHA_segments" in list(SimDet.keys()):
        AHA_segments = SimDet["AHA_segments"]
    else:
        AHA_segments = [0]

    # Get Unloaded volumes
    V_LV_unload = MEmodel_.GetLVV()
    V_RV_unload = MEmodel_.GetRVV()

    printout("V_LV_unload = " + str(V_LV_unload), comm_me)
    printout("V_RV_unload = " + str(V_RV_unload), comm_me)

    nloadstep = SimDet["nLoadSteps"]

    # ============================================================
    # INITIAL MECHANICAL EQUILIBRIUM AND ED PRESSURE LOADING
    # ============================================================
    #
    # Starting from the prescribed reference configuration,
    # ventricular cavity pressures are incrementally increased
    # toward their target end-diastolic values.
    #
    # The nonlinear mechanics problem is solved at each loading
    # increment to obtain an equilibrated configuration.
    #
    # The resulting displacement state is subsequently used
    # to initialize the mechanical boundary conditions and
    # closed-loop simulation.
    #
    # The spring reference displacement is assigned either:
    #   1. From a user-provided HDF5 field, or
    #   2. From the final loading displacement.
    #
    # Clarify in the mechanics model whether the initial mesh
    # represents a stress-free, unloaded, or otherwise
    # prescribed reference geometry.
    # ============================================================
    MEmodel_.LVCavityvol.vol = MEmodel_.GetLVV()
    MEmodel_.LVCavitypres.pres = 0.0
    MEmodel_.RVCavitypres.pres = 0.0
    MEmodel_.AortaCavitypres.pres = 0.0

    # export.writePV(MEmodel_, 0);
    export.hdf.write(MEmodel_.mesh_me, "ME/mesh")
    export.hdf.write(EPmodel_ep.mesh, "EP/mesh")
    if isPJ:
        export.hdf.write(EPmodel_pj.mesh, "PJ/mesh")

    ## Dump input file
    export.dump_input_file()

    default_params = {
        "EDP": 12.0,
        "maxit": 20,
        "restol": 1e-3,
        "drestol": 1e-4,
        "EDPtol": 1e-1,
        "preinc": 1,
        "LVangle": [60, -60],
    }


    if SimDet.get("EDP"):
        EDP = SimDet["EDP"]
    else:
        EDP = default_params["EDP"]

    LVangle = default_params["LVangle"]
    maxit = default_params["maxit"]
    restol = default_params["restol"]
    drestol = default_params["drestol"]
    EDPtol = default_params["EDPtol"]
    preinc = default_params["preinc"]

    it = 0

    # ============================================================
    # PURKINJE-MYOCARDIUM AND CRT COUPLING
    # ============================================================
    #
    # The myocardium and Purkinje fiber network are represented
    # by separate electrophysiology models.
    #
    # Electrical interactions are handled through coupling
    # operators that exchange activation information between
    # the corresponding discretized domains.
    #
    # Coupling components:
    #
    # 1. coupleEPandPJ:
    #    Handles activation transfer from the Purkinje network
    #    to the myocardium through Purkinje-myocardial junctions.
    #
    # 2. coupleMyoandPJ:
    #    Optional reverse coupling from activated myocardium
    #    to the Purkinje network.
    #    Enabled by SimDet["Myo_to_PJ"].
    #
    # 3. coupleEPandCRT:
    #    Handles CRT stimulation of the myocardial EP model.
    #
    # IMPORTANT:
    # The direction of activation transfer must be distinguished
    # from the direction of electrical propagation within each
    # domain.
    #
    # See the coupling module documentation for activation
    # thresholds, refractory conditions, stimulation currents,
    # and any safeguards against repeated activation.
    # ============================================================

    if isPJ:
        coupleEPandPJ_ = coupleEPandPJ(EPmodel_ep, EPmodel_pj, EPparams, PJparams, state_obj)
        coupleMyoandPJ_ = None
        if SimDet.get("Myo_to_PJ", False):
            coupleMyoandPJ_ = coupleMyoandPJ(EPmodel_ep,EPmodel_pj,EPparams,PJparams,state_obj,SimDet,)

    if SimDet.get("CRT", False):
        coupleEPandCRT_ = coupleEPandCRT(EPmodel_ep, EPparams, state_obj)

    print(SimDet["epiid_Kadj_coeff"])
    print(SimDet["dashpotparam"])    

    while 1:
        printout("Loading", comm_me)
        if not SimDet.get("fch_lumped") and not SimDet.get("lv_lumped"):
            MEmodel_.LVCavitypres.pres += (EDP / 0.0075) / nloadstep 
        if SimDet.get("aorta_pres"):
            MEmodel_.AortaCavitypres.pres += (EDP * 10.0 / 0.0075) / nloadstep

        if isBiV or isFCH:
            if SimDet.get("fch_fe"):
                MEmodel_.RVCavitypres.pres += (EDP / 0.0075) / nloadstep
                MEmodel_.LACavitypres.pres += (EDP / 5.0 / 0.0075) / nloadstep
                MEmodel_.RACavitypres.pres += (EDP / 2.0 / 0.0075) / nloadstep
            elif SimDet.get("fch_lumped"):
                pass
            else:
                if SimDet.get("RVEDPfactor"):
                    MEmodel_.RVCavitypres.pres += (SimDet["RVEDPfactor"] * EDP / 0.0075) / nloadstep
                else:
                    MEmodel_.RVCavitypres.pres += (0.75 * EDP / 0.0075) / nloadstep
        
        if not SimDet.get("fch_lumped") and not SimDet.get("lv_lumped"):
            solver_elas.solvenonlinear()

        export.writePV(MEmodel_, 0)
        export.hdf.write(MEmodel_.GetDisplacement(), "ME/u_loading", it)
        it += 1


        printout(
            "LV Pressure = "
            + str(MEmodel_.GetLVP() * 0.0075)
            + " LV Vol = "
            + str(MEmodel_.GetLVV())  # GetVolumeComputation()),
            + "RV Pressure = "
            + str(MEmodel_.GetRVP() * 0.0075)
            + "RV Vol = "
            + str(MEmodel_.GetRVV()),  # GetVolumeComputation()),
            comm_me,
        )

        if SimDet.get("fch_lumped") or SimDet.get("lv_lumped"):
            break
        if MEmodel_.LVCavitypres.pres * 0.0075 >= EDP:
            break

    # Assign displacement at end of loading (LCL)
    if "springref" in SimDet.keys():
        readh5fieldvar(SimDet["springref"][0], SimDet["springref"][1], MEmodel_.u_me_ED)
    else:
        MEmodel_.u_me_ED.assign(MEmodel_.GetDisplacement())
    MEmodel_.isspringon = 1.0

    printout("volume = " + str(MEmodel_.GetLVV()), comm_me)

    #  - - - - - - - - - - - -- - - - - - - - - - - - - - - -- - - - - - -
    fStrain_uL = MEmodel_.GetFiberstrainUL()
    #  - - - - - - - - - - - -- - - - - - - - - - - - - - - -- - - - - - -

    # Closed-loop phase
    stop_iter = SimDet["closedloopparam"]["stop_iter"]

    # Systemic circulation
    # Pulmonary circulation
    heart_shape = [isLV, iswaorta, isFCH]
    if all(not x for x in heart_shape):
        # if not isLV and not iswaorta and not isFCH:
        Cpa = SimDet["closedloopparam"]["Cpa"]
        Cpv = SimDet["closedloopparam"]["Cpv"]
        Vpa0 = SimDet["closedloopparam"]["Vpa0"]
        Vpv0 = SimDet["closedloopparam"]["Vpv0"]
        Rpv = SimDet["closedloopparam"]["Rpv"]
        Rtv = SimDet["closedloopparam"]["Rtv"]
        Rpa = SimDet["closedloopparam"]["Rpa"]
        Rpvv = SimDet["closedloopparam"]["Rpvv"]
        V_pv = SimDet["closedloopparam"]["V_pv"]
        V_pa = SimDet["closedloopparam"]["V_pa"]
        V_RA = SimDet["closedloopparam"]["V_RA"]

    isrestart = 0
    prev_cycle = 0
    cnt = 0

    Qtv = 0
    Qpa = 0
    Qpv = 0
    Qpvv = 0
    Qlvad = 0
    Qlara = 0

    if "Q_tv" in list(SimDet["closedloopparam"].keys()):
        Qtv = SimDet["closedloopparam"]["Q_tv"]
    if "Q_pa" in list(SimDet["closedloopparam"].keys()):
        Qpa = SimDet["closedloopparam"]["Q_pa"]
    if "Q_pv" in list(SimDet["closedloopparam"].keys()):
        Qpv = SimDet["closedloopparam"]["Q_pv"]
    if "Q_pvv" in list(SimDet["closedloopparam"].keys()):
        Qpvv = SimDet["closedloopparam"]["Q_pvv"]
    if "Q_lvad" in list(SimDet["closedloopparam"].keys()):
        Qlvad = SimDet["closedloopparam"]["Q_lvad"]
    if "Q_lara" in list(SimDet["closedloopparam"].keys()):
        Qlara = SimDet["closedloopparam"]["Q_lara"]

    # Parameters for LVAD #############################################
    LVADrpm = 0
    LVADscale = 0
    if "Q_lvad_rpm" in list(SimDet["closedloopparam"].keys()):
        LVADrpm = SimDet["closedloopparam"]["Q_lvad_rpm"]
    if "Q_lvad_scale" in SimDet["closedloopparam"].keys():
        LVADscale = SimDet["closedloopparam"]["Q_lvad_scale"]
    if "Q_lvad_characteristic" in list(SimDet["closedloopparam"].keys()):
        QLVADFn = SimDet["closedloopparam"]["Q_lvad_characteristic"]

    Qlad = 0
    Qlcx = 0

    # Parameters for Shunt #############################################
    Shuntscale = 0.0
    Rsh = 1e9
    if "Shunt_scale" in list(SimDet["closedloopparam"].keys()):
        Shuntscale = SimDet["closedloopparam"]["Shunt_scale"]
    if "Rsh" in list(SimDet["closedloopparam"].keys()):
        Rsh = SimDet["closedloopparam"]["Rsh"]

    potential_me = Function(FunctionSpace(MEmodel_.mesh_me, "DG", 0))
    writecnt = 0

    P_LV = MEmodel_.GetLVP()  # LVCavitypres.pres
    V_LV = MEmodel_.GetLVV()  # GetVolumeComputation()

    if isBiV or isFCH:
        if SimDet.get("fch_fe"):
            P_LA = MEmodel_.GetLAP()  # set an initial value for P_LA and P_RA
            V_LA = MEmodel_.GetLAV()
            P_RA = MEmodel_.GetRAP()
            V_RA = MEmodel_.GetRAV()

            P_RV = MEmodel_.GetRVP()
            V_RV = MEmodel_.GetRVV()
        elif SimDet.get("fch_lumped"):
            P_LV = EDP / 0.0075
            P_RV = EDP / 0.0075
        else:
            P_RV = MEmodel_.GetRVP()  # LVCavitypres.pres
            V_RV = MEmodel_.GetRVV()  # GetVolumeComputation()

    if isLV or iswaorta:
        if SimDet.get("lv_lumped"):
            CLmodel_ = CLmodel(SimDet)
        else:
            CLmodel_ = CLmodel(SimDet, V_LV)
    elif isBiV or isFCH:
        if SimDet.get("fch_fe"):
            CLmodel_ = CLmodel_biv(SimDet, V_LV, V_RV, V_LA, V_RA)
        elif SimDet.get("fch_lumped"):
            CLmodel_ = CLmodel_biv(SimDet)
        else:
            CLmodel_ = CLmodel_biv(SimDet, V_LV, V_RV)

    it_ = 0

    csv_path = os.path.join(outputfolder + folderName + "output_PV.csv")
    if MPI.rank(comm_me) == 0:
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            headers = [
                "tstep",
                "t",
                "V_LV",
                "P_LV",
                "V_RV",
                "P_RV",
                "V_LA",
                "P_LA",
                "V_RA",
                "P_RA",
                "V_sv",
                "V_sa",
                "V_ad",
                "V_pv",
                "V_pa",
            ]
            writer.writerow(headers)


    while 1:

        if state_obj.cycle > stop_iter:
            break


        if not SimDet.get("fch_lumped") and not SimDet.get("lv_lumped"):
            params = {"P_LV": P_LV,"V_LV": V_LV,"t": state_obj.t,"delTat": state_obj.dt.dt,}
        else:
            params = {"t": state_obj.t,"delTat": state_obj.dt.dt,}

        if isBiV or isFCH:
            if SimDet.get("fch_fe"):
                params.update(
                    {
                        "P_RV": P_RV,
                        "V_RV": V_RV,
                        "P_LA": P_LA,
                        "V_LA": V_LA,
                        "P_RA": P_RA,
                        "V_RA": V_RA,
                    }
                )
            elif SimDet.get("fch_lumped"):
                pass
            else:
                params.update(
                    {
                        "P_RV": P_RV,
                        "V_RV": V_RV,
                    }
                )

        if isLV or iswaorta:
            if SimDet.get("lv_lumped"):
                V_LV, V_LA = CLmodel_.UpdateLVV(params)
            else:
                V_LV = CLmodel_.UpdateLVV(params)

        elif isBiV or isFCH:
            V_LV, V_RV, *extra = CLmodel_.UpdateLVV(params)
            if SimDet.get("fch_fe") or SimDet.get("fch_lumped"):
                V_LA, V_RA = extra

        print_message = (  
            "t = "
            + str(state_obj.t)
            + " VLV = "
            + str(CLmodel_.V_LV)
            + " Psa = "
            + str(CLmodel_.Psa)
            + " PLV = "
            + str(CLmodel_.PLV)
            + " P_LA = "
            + str(CLmodel_.PLA)
            + " Qmv = "
            + str(CLmodel_.Qmv)
            + " Qav = "
            + str(CLmodel_.Qav)
        )
        if isBiV or isFCH:
            if SimDet.get("fch_fe"):
                print_message += " PLA = " + str(P_LA)
            else:
                print_message += " PLA = " + str(CLmodel_.PLA)

            print_message += "\n"
            print_message += " VRV = " + str(V_RV)
            print_message += " Ppa = " + str(CLmodel_.Ppa)
            print_message += " PRV = " + str(CLmodel_.PRV)

            if SimDet.get("fch_fe"):
                print_message += " PRA = " + str(P_RA)
            else:
                print_message += " PRA = " + str(CLmodel_.PRA)

        printout(print_message, comm_me)

        new_row = [
                state_obj.tstep,
                state_obj.t,
                V_LV,
                P_LV*0.0075,
                V_RV,
                P_RV*0.0075,
                CLmodel_.V_LA,
                "-", 
                CLmodel_.V_RA,
                "-", 
                CLmodel_.V_sv,
                CLmodel_.V_sa,
                CLmodel_.V_ad,
                CLmodel_.V_pv,
                CLmodel_.V_pa,
            ]

        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)          
            writer.writerow(new_row)

        with open(outputfolder + folderName + "output_PV.txt", "a") as f_PV:
            if MPI.rank(comm_me) == 0:

                if isLV or iswaorta:
                    if SimDet.get("lv_lumped"):
                        f_PV.write(
                            f"{state_obj.tstep}, {CLmodel_.V_LV}, {CLmodel_.PLV} \n"
                        )
                    else:
                        f_PV.write(f"{state_obj.tstep}, {V_LV}, {P_LV} \n")

                elif isBiV or isFCH:
                    if SimDet.get("fch_lumped"):
                        f_PV.write(
                            f"{state_obj.tstep}, {V_LV}, {CLmodel_.PLV}, {V_RV}, {CLmodel_.PRV}, {V_LA}, {CLmodel_.PLA}, {V_RA}, {CLmodel_.PRA} \n"
                        )
                    elif SimDet.get("fch_fe"):
                        pass
                    else:
                        f_PV.write(
                            f"{state_obj.t}, {V_LV}, {P_LV}, {V_RV}, {P_RV}, {CLmodel_.V_LA}, {CLmodel_.PLA}, {CLmodel_.V_RA}, {CLmodel_.PRA}, {CLmodel_.V_sv}, {CLmodel_.V_sa}, {CLmodel_.V_ad}, {CLmodel_.V_pv}, {CLmodel_.V_pa} \n"
                        )

        MEmodel_.LVCavitypres.pres = P_LV 


        # ============================================================
        # CLOSED-LOOP CIRCULATION - MECHANICS COUPLING
        # ============================================================
        #
        # The circulation model advances the compartment volumes
        # and provides target ventricular cavity volumes.
        #
        # The mechanics model determines cavity deformation and
        # volume in response to applied cavity pressures.
        #
        # To enforce consistency between the two models, cavity
        # pressures are determined by solving a nonlinear
        # pressure-volume matching problem.
        #
        # For the BiV model:
        #
        #   R_LV = V_LV_ME(P_LV, P_RV) - V_LV_CL
        #   R_RV = V_RV_ME(P_LV, P_RV) - V_RV_CL
        #
        # The pressure vector is obtained by finding:
        #
        #   R(P_LV, P_RV) = 0
        #
        # Each residual evaluation requires a nonlinear mechanics
        # solve, making this an iterative coupled problem.
        #
        # The previous pressure state is used as the initial guess.
        #
        # Numerical method:
        #   scipy.optimize.fsolve
        #
        # The full-heart model additionally enforces LA and RA
        # volume consistency.
        # ============================================================

        def Rp(plv, vlv):
            MEmodel_.LVCavitypres.pres = plv
            if SimDet.get("aorta_pres"):
                MEmodel_.AortaCavitypres.pres = CLmodel_.Psa  
            solver_elas.solvenonlinear()
            v_t = MEmodel_.GetLVV()
            return v_t - vlv

        def Rp_biv(plv, prv, lvvc, rvvc):
            MEmodel_.LVCavitypres.pres = plv
            MEmodel_.RVCavitypres.pres = prv
            solver_elas.solvenonlinear()
            vlv = MEmodel_.GetLVV()
            vrv = MEmodel_.GetRVV()
            return [vlv - lvvc, vrv - rvvc]

        def Rp_fch(plv, prv, pla, pra, lvvc, rvvc, lavc, ravc):
            MEmodel_.LVCavitypres.pres = plv
            MEmodel_.RVCavitypres.pres = prv
            MEmodel_.LACavitypres.pres = pla
            MEmodel_.RACavitypres.pres = pra
            solver_elas.solvenonlinear()
            vlv = MEmodel_.GetLVV()
            vrv = MEmodel_.GetRVV()
            vla = MEmodel_.GetLAV()
            vra = MEmodel_.GetRAV()
            return np.array([vlv - lvvc, vrv - rvvc, vla - lavc, vra - ravc])

        # Create the Newton solver
        from scipy.optimize import (newton,fsolve,bisect, root, minimize,)

        def Rp_call(plv):
            return Rp(plv, V_LV) 

        def Rp_biv_call(plrv):
            plv, prv = plrv
            return Rp_biv(plv, prv, V_LV, V_RV)

        def Rp_fch_call(plrva):
            plv, prv, pla, pra = plrva
            return Rp_fch(plv, prv, pla, pra, V_LV, V_RV, V_LA, V_RA)

        if isLV or iswaorta:
            x0 = [P_LV]

        elif isBiV or isFCH:
            if SimDet.get("fch_fe"):
                x0 = [P_LV, P_RV, P_LA, P_RA]
            else:
                x0 = [P_LV, P_RV]

        if isLV or iswaorta:
            if not SimDet.get("lv_lumped"):
                root1, info, ier, msg = fsolve(Rp_call, x0, xtol=1e-4, factor=0.01, full_output=True)

        elif isBiV or isFCH:
            if SimDet.get("fch_fe"):
                root1, info, ier, msg = fsolve(Rp_fch_call, x0, xtol=1e-4, factor=0.01, full_output=True)
            elif SimDet.get("fch_lumped"):
                pass
            else:
                root1, info, ier, msg = fsolve(Rp_biv_call, x0, xtol=1e-4, factor=0.01, full_output=True)

        with open(outputfolder + folderName + "output_nfev.txt", "a") as nfev:
            if (MPI.rank(comm_me) == 0 and not SimDet.get("fch_lumped") and not SimDet.get("lv_lumped")):
                nfev.write(f"t = {state_obj.t}, iter = {info['nfev']}, fun = {info['fvec']}, message = {msg} \n")

        if not SimDet.get("fch_lumped") and not SimDet.get("lv_lumped"):
            P_LV = root1[0]

        if isBiV or isFCH:
            if SimDet.get("fch_fe"):
                P_RV = root1[1]
                P_LA = root1[2]
                P_RA = root1[3]
            elif SimDet.get("fch_lumped"):
                pass
            else:
                P_RV = root1[1]

        # ============================================================
        # TIME AND ACTIVATION STATE UPDATE
        # ============================================================
        # The simulation maintains multiple time representations

        state_obj.tstep = state_obj.tstep + state_obj.dt.dt # Cumulative simulation time
        state_obj.cycle = math.floor(state_obj.tstep / state_obj.BCL) # Number of completed cardiac cycles
        state_obj.t = state_obj.tstep - state_obj.cycle * state_obj.BCL # Time within the current cardiac cycle

        # Compute local time since activation
        t_init = MEmodel_.activeforms.t_init.vector().get_local() # Local electrical activation initiation time
        local_cycle = np.floor((MEmodel_.t_a.vector().get_local() - t_init)/ state_obj.BCL)
        local_cycle[local_cycle < 0] = 0 # Return zero if cycle is less than zero
        MEmodel_.cycle.vector()[:] = local_cycle
        MEmodel_.t_a.vector()[:] = MEmodel_.t_a.vector().get_local() + state_obj.dt.dt # 	Time variable maintained by the mechanics activation model
        MEmodel_.t_since_activation.vector()[:] =  MEmodel_.t_a.vector().get_local()  - t_init - local_cycle * state_obj.BCL
       
        isrestart = 0
        state_obj.dt.dt = delTat

        if isPJ:
           # Reset phi and r in EP at end of diastole ensures iStim > 0 at t=0
           if state_obj.t < state_obj.dt.dt:
               coupleEPandPJ_.reset()
               EPmodel_ep.reset()
               if isPJ:
                   EPmodel_pj.reset()
                   if coupleMyoandPJ_ is not None:
                       coupleMyoandPJ_.reset()

           if SimDet.get("CRT", False):
                coupleEPandCRT_.UpdateCRTandEP()


           if not SimDet.get("lv_lumped") and not ishomo:
                printout("Solving FHN EP", comm_me)
                solver_FHN_ep.solvenonlinear()
        

                if isPJ:
                    # ============================================================
                    # MPI PURKINJE PACING NODE IDENTIFICATION
                    # ============================================================
                    #
                    # The Purkinje mesh may be distributed across MPI processes.
                    # Each process stores only its local mesh vertices.
                    #
                    # For every prescribed pacing location:
                    #
                    # 1. Construct a KD-tree using locally available PJ vertices.
                    # 2. Identify the nearest local vertex.
                    # 3. Gather candidate distances and indices from all ranks.
                    # 4. Determine which rank contains the globally nearest
                    #    candidate vertex.
                    # 5. Apply the stimulus on the selected owner rank.
                    #
                    # The selected index is local to the corresponding MPI rank,
                    # not a global mesh vertex identifier.
                    #
                    # IMPORTANT:
                    # The mapping between mesh vertices, degrees of freedom,
                    # and stimulus-array entries must be consistent.
                    # ============================================================

                    comm = EPmodel_pj.mesh.mpi_comm()
                    rank = comm.Get_rank()
                    size = comm.Get_size()     

                    ploc_pj = SimDet["ploc"]
                    pj_nodes_local = EPmodel_pj.mesh.coordinates()

                    # Build local KDTree
                    pj_tree_local = cKDTree(pj_nodes_local)
                    local_dist, local_idx = pj_tree_local.query(ploc_pj)
                    # Query nearest mesh vertex for each pacing site
                    # distances, pj_node_indices = pj_tree.query(ploc_pj)

                    # Gather all distances and ranks across processes
                    all_dists = comm.allgather(np.atleast_1d(local_dist))
                    all_indices = comm.allgather(np.atleast_1d(local_idx))
                    all_ranks = comm.allgather(np.ones_like(local_idx, dtype=int) * rank)

                    # Find global nearest node for each pacing location
                    all_dists = np.vstack(all_dists)
                    all_indices = np.vstack(all_indices)
                    all_ranks = np.vstack(all_ranks)

                    # Find the global nearest node for each pacing location
                    min_rank_row = np.argmin(all_dists, axis=0)
                    owner_ranks = all_ranks[min_rank_row, np.arange(len(ploc_pj))]
                    owner_local_idx = all_indices[min_rank_row, np.arange(len(ploc_pj))]  

                    if rank == 0:
                        print("owner_ranks:", owner_ranks)
                        print("owner_local_idx:", owner_local_idx)          

                    # Activate PJ fiber network
                    if(state_obj.t > SimDet["pacing_timing"][0][0] and \
                        state_obj.t < SimDet["pacing_timing"][0][0] + SimDet["pacing_timing"][0][1] ):

                        # for idx in pj_node_indices: 
                        for p_idx, (target_rank, local_i) in enumerate(zip(owner_ranks, owner_local_idx)):

                            if rank == target_rank:
                                
                                if "fstim_val_array" in dir(EPmodel_pj) and EPmodel_pj.fstim_val_array is not None:
                                    if 0 <= local_i < len(EPmodel_pj.fstim_array):   
                                        EPmodel_pj.fstim_val_array[local_i] = pj_intensity
                                        printout(f"pacing site {p_idx} via fstim_val_array (rank {rank}, idx {local_i})", comm_me)
                                    else:
                                        print(f"[Rank {rank}] Invalid local_i {local_i} (max {len(EPmodel_pj.fstim_val_array)-1})")
                                
                                elif "fstim_array" in dir(EPmodel_pj) and len(EPmodel_pj.fstim_array) > 0:
                                    # Use pacing-site index instead of mesh node index 
                                    if 0 <= p_idx < len(EPmodel_pj.fstim_array):
                                        EPmodel_pj.fstim_array[p_idx].iStim = pj_intensity
                                        printout(f"pacing site {p_idx} via fstim_array (rank {rank})", comm_me)
                                    else:
                                        print(f"[Rank {rank}] Invalid p_idx {p_idx} (max {len(EPmodel_pj.fstim_array)-1})")
                                
                                else:
                                    print(f"[Rank {rank}] No valid stimulus array found for pacing site {p_idx}")

                            elif state_obj.t >= SimDet["pacing_timing"][0][0] + SimDet["pacing_timing"][0][1]:
                                
                                for p_idx, (target_rank, local_i) in enumerate(zip(owner_ranks, owner_local_idx)):
                                    if rank == target_rank:
                                        
                                        if "fstim_val_array" in dir(EPmodel_pj) and EPmodel_pj.fstim_val_array is not None:
                                            if 0 <= local_i < len(EPmodel_pj.fstim_val_array): 
                                                EPmodel_pj.fstim_val_array[local_i] = 0
                                                printout(f"not pacing site {p_idx} via fstim_val_array (rank {rank}, idx {local_i})", comm_me)
                                            else:
                                                print(f"[Rank {rank}] Invalid local_i {local_i} (max {len(EPmodel_pj.fstim_val_array)-1})")
                                        
                                        elif "fstim_array" in dir(EPmodel_pj) and len(EPmodel_pj.fstim_array) > 0:
                                            if 0 <= p_idx < len(EPmodel_pj.fstim_array):
                                                EPmodel_pj.fstim_array[p_idx].iStim = 0
                                                printout(f"not pacing site {p_idx} via fstim_array (rank {rank})", comm_me)
                                            else:
                                                print(f"[Rank {rank}] Invalid p_idx {p_idx} (max {len(EPmodel_pj.fstim_array)-1})")
                                        
                                        else:
                                            print(f"[Rank {rank}] No valid stimulus array found for pacing site {p_idx}")


                    if("UpdateActivation" in dir(EPmodel_pj)):
                        EPmodel_pj.UpdateActivation()

                    if coupleMyoandPJ_ is not None:
                        # Coupling class for activation of pj network via myocardium threshold 
                        coupleMyoandPJ_.UpdateMyoandPJ()

                    printout("Solving FHN PJ", comm_me)
                    # Solver for pj problem
                    solver_FHN_pj.solvenonlinear()

        else:
            if state_obj.t < state_obj.dt.dt:
                EPmodel_ep.reset()
 
            # Activate EP network
            if(state_obj.t > SimDet["pacing_timing"][0][0] and \
               state_obj.t < SimDet["pacing_timing"][0][0] + SimDet["pacing_timing"][0][1] ):
                for  p in range(0,len(EPmodel_ep.fstim_array)):
                    EPmodel_ep.fstim_array[p].iStim = intensity
                    printout("pacing", comm_me)
            else:
                for  p in range(0,len(EPmodel_ep.fstim_array)):
                    EPmodel_ep.fstim_array[p].iStim = 0.0


            if not SimDet.get("lv_lumped") and not ishomo:
                printout("Solving FHN EP", comm_me)
                # Solver for myocardial EP problem
                solver_FHN_ep.solvenonlinear()

        if isrestart == 0:
            MEmodel_.UpdateVar()  # For damping
            EPmodel_ep.UpdateVar()
           
            if isPJ:
               EPmodel_pj.UpdateVar()

        # Interpolate phi to mechanics mesh
        potential_ref = EPmodel_ep.interpolate_potential_ep2me_phi(V_me=Function(FunctionSpace(MEmodel_.mesh_me, "DG", 0)))
        potential_ref.rename("v_ref", "v_ref")
        potential_me.vector()[:] = potential_ref.vector().get_local()[:]

        #  - - - - - - - - - - - -- - - - - - - - - - - - - - - -- - - - - - -
        if MPI.rank(comm_ep) == 0:
            print("UPdating isActiveField and tInitiationField")

        MEmodel_.activeforms.update_activationTime(potential_n=potential_me, comm=comm_me)

        if isPJ:
            # Coupling class for activation of myocardium via pj terminal nodes                       
            coupleEPandPJ_.UpdatePJandEP()

        if SimDet.get("CRT", False):
            # Coupling class for activation of myocardium via crt activation sites  
            coupleEPandCRT_.UpdateCRTandEP()    


        F_n = MEmodel_.GetFmat()
        fstress_DG = project(
            MEmodel_.Getfstress(),
            FunctionSpace(MEmodel_.mesh_me, "DG", 0),
            form_compiler_parameters={"representation": "uflacs"},
        )
        fstress_DG.rename("fstress", "fstress")

        if "probepts" in list(SimDet.keys()):
            probesfstress = Probes(x.flatten(), FunctionSpace(MEmodel_.mesh_me, "DG", 1))
            probesfstress(fstress_DG)

        Eul_fiber_BiV_DG = project(fStrain_uL,FunctionSpace(MEmodel_.mesh_me, "DG", 0),form_compiler_parameters={"representation": "uflacs"},)

        Eul_fiber_BiV_DG.rename("Eff", "Eff")
        if "probepts" in list(SimDet.keys()):
            probesEul_fiber = Probes(x.flatten(), FunctionSpace(MEmodel_.mesh_me, "DG", 1))
            probesEul_fiber(Eul_fiber_BiV_DG)
            probesE_circ_BiV = Probes(x.flatten(), FunctionSpace(MEmodel_.mesh_me, "DG", 1))
            probesE_long_BiV = Probes(x.flatten(), FunctionSpace(MEmodel_.mesh_me, "DG", 1))
            probesE_radi_BiV = Probes(x.flatten(), FunctionSpace(MEmodel_.mesh_me, "DG", 1))

        # postprocess and write
        #  - - - - - - - - - - - -- - - - - - - - - - - - - - - -- - - - - - -
        export.writePV(MEmodel_, state_obj.tstep, CLmodel = CLmodel_)

        if isLV:
            export.writeQ(MEmodel_, [CLmodel_.Qsa, CLmodel_.Qad, CLmodel_.Qsv, 
                                     CLmodel_.Qmv, CLmodel_.Qav], state_obj.tstep)
            export.writeP(MEmodel_, np.array([CLmodel_.Psa, CLmodel_.Pad, CLmodel_.Psv, 
                                              CLmodel_.PLA, CLmodel_.PLV])*0.0075, state_obj.tstep)
            export.writeV(MEmodel_, np.array([CLmodel_.V_sa, CLmodel_.V_ad, CLmodel_.V_sv, 
                                              CLmodel_.V_LA, CLmodel_.V_LV]), state_obj.tstep)


        if isBiV:
            export.writeQ(MEmodel_, np.array([CLmodel_.Qsa, CLmodel_.Qad, CLmodel_.Qsv, CLmodel_.Qmv, CLmodel_.Qav,
                                              CLmodel_.Qtv, CLmodel_.Qpvv, CLmodel_.Qpa, CLmodel_.Qpv
                                              ]), state_obj.tstep)

            export.writeP(MEmodel_, np.array([CLmodel_.Psa, CLmodel_.Pad, CLmodel_.Psv, CLmodel_.PLA, CLmodel_.PLV,
                                              CLmodel_.Ppa, CLmodel_.Ppv, CLmodel_.PRV,
                                              CLmodel_.PRA])*0.0075, state_obj.tstep)

            export.writeV(MEmodel_, np.array([CLmodel_.V_sa, CLmodel_.V_ad, CLmodel_.V_sv, CLmodel_.V_LA, CLmodel_.V_LV,
                                              CLmodel_.V_pa, CLmodel_.V_pv, CLmodel_.V_RV,
                                              CLmodel_.V_RA]), state_obj.tstep)


        if cnt % SimDet["writeStep"] == 0.0:
            export.writetpt(MEmodel_, state_obj.tstep)
            export.hdf.write(MEmodel_.GetDisplacement(), "ME/u", writecnt)
            export.hdf.write(potential_ref, "ME/potential_ref", writecnt)
            export.hdf.write(MEmodel_.GetSActive(), "ME/Sactive", writecnt)
            export.hdf.write(MEmodel_.Get_t_a(), "ME/t_a", writecnt)
            export.hdf.write(MEmodel_.Get_t_init(), "ME/t_init", writecnt)
            export.hdf.write(MEmodel_.Get_isActive(), "ME/isActive", writecnt)
            export.hdf.write(MEmodel_.Get_local_cycle(), "ME/cycle", writecnt)
            export.hdf.write(MEmodel_.Get_t_since_act(), "ME/t_since_act", writecnt)
            export.hdf.write(Eul_fiber_BiV_DG, "ME/Eff", writecnt)
            export.hdf.write(fstress_DG, "ME/fstress", writecnt)
            export.hdf.write(MEmodel_.GetP(), "ME/imp_constraint", writecnt)

            export.hdf.write(EPmodel_ep.getphivar(), "EP/phi", writecnt)
            export.hdf.write(EPmodel_ep.getrvar(), "EP/r", writecnt)
            export.hdf.write(potential_ref, "EP/potential_ref", writecnt)

            if isPJ:
                export.hdf.write(EPmodel_pj.getphivar(), "PJ/phi", writecnt)
                export.hdf.write(EPmodel_pj.getrvar(), "PJ/r", writecnt)

            writecnt += 1

        if "probepts" in list(SimDet.keys()):
            fIMP = probesIMP.array()
            fIMP2 = probesIMP2.array()
            fIMP3 = probesIMP3.array()
            fStress = probesfstress.array()
            fStrain_vals = probesEul_fiber.array()
            E_circ_BiV = probesE_circ_BiV.array()
            E_long_BiV = probesE_long_BiV.array()
            E_radi_BiV = probesE_radi_BiV.array()

            export.writeIMP(MEmodel_, state_obj.tstep, fIMP)
            export.writeIMP2(MEmodel_, state_obj.tstep, fIMP2)
            export.writeIMP3(MEmodel_, state_obj.tstep, fIMP3)
            export.writefStress(MEmodel_, state_obj.tstep, fStress)
            export.writefStrain(MEmodel_, state_obj.tstep, fStrain_vals)
            export.writeCStrain(MEmodel_, state_obj.tstep, E_circ_BiV)
            export.writeLStrain(MEmodel_, state_obj.tstep, E_long_BiV)
            export.writeRStrain(MEmodel_, state_obj.tstep, E_radi_BiV)

        cnt += 1

        if(state_obj.tstep % state_obj.BCL == 0) :
            export.dump_restart_file(CLmodel_)


#  - - - - - - - - - - - -- - - - - - - - - - - - - - - -- - - - - - -
if __name__ == "__main__":
    print("Testing...")
    run_BiV_TimedGuccione(IODet=IODetails, SimDet=SimDetails)

#  - - - - - - - - - - - -- - - - - - - - - - - - - - - -- - - - - - -
