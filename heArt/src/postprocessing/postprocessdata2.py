import dolfin as df
from .postprocessdatalib2 import *
from ..mechanics.forms_MRC2 import Forms
from ..utils.oops_objects_MRC2 import State_Variables
from ..utils.oops_objects_MRC2 import lv_mesh as lv_mechanics_mesh
from ..utils.oops_objects_MRC2 import biventricle_mesh as biv_mechanics_mesh
from ..mechanics.MEmodel3 import MEmodel
import matplotlib
matplotlib.use("Agg")
from matplotlib.patches import Wedge, Circle
import copy
import numpy as np
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
from tqdm import tqdm
import collections  # <-- This line is required
from matplotlib import pylab as plt
from mpi4py import MPI
import pdb


def postprocessdata(IODet, SimDet, cycle=None):
    directory = IODet["outputfolder"] + "/"
    casename = IODet["caseID"]
    BCL = SimDet["HeartBeatLength"]
    isLV = SimDet["isLV"]

    if cycle is None:
        cycle = SimDet["closedloopparam"]["stop_iter"]

    for ncycle in range(cycle, cycle+1):
        filename = directory + casename + "/" + "BiV_PV.txt"
        homo_tptt, homo_LVP, homo_LVV, homo_RVP, homo_RVV, homo_Qmv = extract_PV(filename, BCL, ncycle, SimDet)

        if SimDet["isLV"]:
            filename = directory + casename + "/" + "BiV_Q.txt"
            (   homo_tptt,
                homo_Qao,
                homo_Qmv,
                homo_Qper,
                homo_Qla,
                homo_Qlad,
                homo_Qlcx,
                Q_lvad,
            ) = extract_Q(filename, BCL, ncycle, SimDet)
        else:
            filename = directory + casename + "/" + "BiV_Q.txt"
            (   homo_tptt,
                homo_Qao,
                homo_Qmv,
                Qsa,
                Qsv,
                homo_Qper,
                homo_Qla,
                homo_Qlad,
                homo_Qlcx,
                Q_lvad,
            ) = extract_Q(filename, BCL, ncycle, SimDet)
     
        filename = directory + casename + "/" + "BiV_P.txt"
        tpt, Psv, PLV, Psa, PLA, Ppv, PRV, Ppa, PRA = extract_P(filename, BCL, ncycle, SimDet)           

        filename = directory + casename + "/" + "BiV_IMP_InC.txt"
        homo_tpt_IMP, homo_IMP = extract_probe(filename, BCL, ncycle)

        filename = directory + casename + "/" + "BiV_fiberStrain.txt"
        homo_tpt_Eff, homo_Eff = extract_probe(filename, BCL, ncycle)

        filename = directory + casename + "/" + "BiV_fiberStress.txt"
        homo_tpt_Sff, homo_Sff = extract_probe(filename, BCL, ncycle)

        ESP, ESV = extractESP(homo_LVP, homo_LVV)
        EDP, EDV = extractEDP(homo_LVP, homo_LVV)

        EDP_RV, EDV_RV = extractEDP(homo_RVP, homo_RVV)        

        if SimDet["isLV"]:
            if max(homo_LVP) < 25:
                EF = 0.0
            else:
                EF = (max(homo_LVV) - min(homo_LVV)) / max(homo_LVV) * 100

        elif SimDet["isBiV"]:

            if max(homo_LVP) < 25:
                EF = 0.0
            else:
                EF = (max(homo_LVV) - min(homo_LVV)) / max(homo_LVV) * 100
                EF_RV = (max(homo_RVV) - min(homo_RVV)) / max(homo_RVV) * 100     

        print(filename)
        print(
            (
                "EF = ",
                EF ,
                "EF_RV = ",
                EF_RV,
                " EDV = ",
                max(homo_LVV),
                " EDV_RV = ",
                max(homo_RVV),                
                " ESV = ",
                min(homo_LVV),
                " ESV_RV = ",
                min(homo_RVV),                
                " EDP = ",
                EDP,
                " EDP_RV = ",
                EDP_RV,                
                # " SBP = ",
                # SBP,
                # " DBP = ",
                # DBP,
            )
        )

        print(("Peak LV pressure = ", max(homo_LVP)))

def compute_activation(IODet, SimDet, cycle=None):
    dt = SimDet["dt"]
    t = dt    
    write_t = SimDet["writeStep"]
    V_thres = 0.9    

    #----------------------------------------------------------------
    # Myocardial EP activation time
    #----------------------------------------------------------------
    mesh = df.Mesh()
    hdf = df.HDF5File(mesh.mpi_comm(), IODet["outputfolder"] + "/" + IODet["caseID"] + "/" + "Data.h5", "r",)
    hdf.read(mesh, "EP/mesh", False)

    phi_arr = extractvtk(IODet["outputfolder"] + "/" + IODet["caseID"], "EP/phi", "CG", 1, IODet["outputfolder"] + "/" + IODet["caseID"] + "/" + "EP_" + "phi", "phi", group="EP", iswrite=False,)

    time_act = df.Function(df.FunctionSpace(mesh, "CG", 1))
    time_act_vec = -1 * np.ones(len(time_act.vector()[:]))

    for phi in phi_arr:
        phi_vec = phi.sub(0).vector().get_local()[::3]
        for idx, (time_act_vec_, phi_vec_) in enumerate(zip(time_act_vec, phi_vec)):
            if phi_vec_ > V_thres and time_act_vec_ == -1:
                time_act_vec[idx] = t

        t += dt * write_t

    time_act.vector()[:] = time_act_vec
    time_act.rename("Activation Time", "Activation Time")

    valid_mask = time_act_vec != -1
    valid_times = time_act_vec[valid_mask]

    t_90_myo = np.percentile(time_act_vec,90)    
    print("90% Activation time myocardium:", t_90_myo, "[ms]")

    act_outdirectory = os.path.join( IODet["outputfolder"], IODet["caseID"], "activation")
    if not os.path.exists(act_outdirectory):
        os.mkdir(act_outdirectory)
    File_act = df.File(os.path.join(act_outdirectory, "act.pvd"))

    if np.all(valid_mask == False):
        time_act.vector()[:]=time_act.vector()[:]
    else: 
        time_act.vector()[:]=time_act.vector()[:]-min(valid_times)

    File_act << time_act

    if not SimDet["isPJ"]:
        with open(act_outdirectory+"/activation_summary_myo.txt", "w") as f:
            f.write("90% Activation Time [ms]\n")
            f.write(f"90% Activation time myocardium:\t{t_90_myo:.3f}[ms]\n")  

    #----------------------------------------------------------------
    # PJ EP activation time
    #----------------------------------------------------------------
    if SimDet["isPJ"]:

        mesh_pj = df.Mesh()
        hdf = df.HDF5File(mesh_pj.mpi_comm(),IODet["outputfolder"] + "/" + IODet["caseID"] + "/" + "Data.h5","r",)
        hdf.read(mesh_pj, "PJ/mesh", False)
        pj_phi_arr = extractvtk(IODet["outputfolder"] + "/" + IODet["caseID"], "PJ/phi", "CG", 1, IODet["outputfolder"] + "/" + IODet["caseID"] + "/" + "PJ_" + "phi", "phi", group="PJ", iswrite=False,)

        pj_time_act = df.Function(df.FunctionSpace(mesh_pj, "CG", 1))
        pj_time_act_vec = -1 * np.ones(len(pj_time_act.vector()[:]))
        t = dt
        for pj_phi in pj_phi_arr:
            pj_phi_vec = pj_phi.sub(0).vector().get_local()[::3]
            for idx, (pj_time_act_vec_, pj_phi_vec_) in enumerate(zip(pj_time_act_vec, pj_phi_vec)):
                if pj_phi_vec_ > V_thres and pj_time_act_vec_ == -1:
                    pj_time_act_vec[idx] = t

            t += dt * write_t

        # ---- shift only valid values ----
        valid = pj_time_act_vec[pj_time_act_vec >= 0]

        if len(valid) > 0:
            min_valid = np.min(valid)

            # create shifted array but keep -1 unchanged
            shifted = np.copy(pj_time_act_vec)
            mask_valid = shifted >= 0
            shifted[mask_valid] -= min_valid

            pj_time_act.vector()[:] = shifted
            t_90_pj = np.percentile(valid, 90)

        else:
            # no PJ activations at all
            pj_time_act.vector()[:] = pj_time_act_vec
            t_90_pj = -1

        pj_time_act.rename("PJ Activation Time", "PJ Activation Time")

        File_act_pj = df.File(os.path.join(act_outdirectory, "pj_act.pvd"))
        File_act_pj << pj_time_act

        with open(act_outdirectory+"/activation_summary.txt", "w") as f:
            f.write("90% Activation Time [ms]\n")
            f.write(f"90% Activation time purkinje:\t{t_90_pj:.3f}[ms]\n")
            f.write(f"90% Activation time myocardium:\t{t_90_myo:.3f}[ms]\n")        


def normalize_directionalbasis(eC0, eL0, eR0, mesh, deg):
    eC0_normalized = eC0 / df.sqrt(df.inner(eC0, eC0))
    eL0_normalized = eL0 / df.sqrt(df.inner(eL0, eL0))
    eR0_normalized = eR0 / df.sqrt(df.inner(eR0, eR0))

    eC0_normalized = (
        df.project(
            eC0_normalized,
            df.VectorFunctionSpace(mesh, "DG", 0),
            form_compiler_parameters={
                "representation": "uflacs",
                "quadrature_degree": deg,
            },
        )
        .vector()
        .get_local()
    )
    isnan_eC0_normalized = np.argwhere(np.isnan(eC0_normalized)).flatten()

    eL0_normalized = (
        df.project(
            eL0_normalized,
            df.VectorFunctionSpace(mesh, "DG", 0),
            form_compiler_parameters={
                "representation": "uflacs",
                "quadrature_degree": deg,
            },
        )
        .vector()
        .get_local()
    )
    isnan_eL0_normalized = np.argwhere(np.isnan(eL0_normalized)).flatten()

    eR0_normalized = (
        df.project(
            eR0_normalized,
            df.VectorFunctionSpace(mesh, "DG", 0),
            form_compiler_parameters={
                "representation": "uflacs",
                "quadrature_degree": deg,
            },
        )
        .vector()
        .get_local()
    )
    isnan_eR0_normalized = np.argwhere(np.isnan(eR0_normalized)).flatten()

    mesh_coordinates = df.FunctionSpace(
        mesh, "DG", 0
    ).tabulate_dof_coordinates()

    np.set_printoptions(threshold=sys.maxsize)
    if isnan_eC0_normalized.size != 0:
        for p in isnan_eC0_normalized:
            list_of_nan_ids = [p // 3 * 3 + i for i in range(0, 3)]
            distances_to_nan_pt = [
                np.linalg.norm(mesh_coordinates[i] - mesh_coordinates[p // 3])
                for i in range(0, len(mesh_coordinates))
            ]
            distances_to_nan_pt[p // 3] = 1000
            closest_pt_id = np.argmin(distances_to_nan_pt, axis=0)
            eC0_normalized[p // 3 * 3 : p // 3 * 3 + 3] = eC0_normalized[
                3 * closest_pt_id : 3 * closest_pt_id + 3
            ]

    if isnan_eL0_normalized.size != 0:
        for p in isnan_eL0_normalized:
            list_of_nan_ids = [p // 3 * 3 + i for i in range(0, 3)]
            distances_to_nan_pt = [
                np.linalg.norm(mesh_coordinates[i] - mesh_coordinates[p // 3])
                for i in range(0, len(mesh_coordinates))
            ]
            distances_to_nan_pt[p // 3] = 1000
            closest_pt_id = np.argmin(distances_to_nan_pt, axis=0)
            eL0_normalized[p // 3 * 3 : p // 3 * 3 + 3] = eL0_normalized[
                3 * closest_pt_id : 3 * closest_pt_id + 3
            ]

    if isnan_eR0_normalized.size != 0:
        for p in isnan_eR0_normalized:
            list_of_nan_ids = [p // 3 * 3 + i for i in range(0, 3)]
            distances_to_nan_pt = [
                np.linalg.norm(mesh_coordinates[i] - mesh_coordinates[p // 3])
                for i in range(0, len(mesh_coordinates))
            ]
            distances_to_nan_pt[p // 3] = 1000
            closest_pt_id = np.argmin(distances_to_nan_pt, axis=0)
            eR0_normalized[p // 3 * 3 : p // 3 * 3 + 3] = eR0_normalized[
                3 * closest_pt_id : 3 * closest_pt_id + 3
            ]

    eC0_normalized_ = df.Function(df.VectorFunctionSpace(mesh, "DG", 0))
    eL0_normalized_ = df.Function(df.VectorFunctionSpace(mesh, "DG", 0))
    eR0_normalized_ = df.Function(df.VectorFunctionSpace(mesh, "DG", 0))

    eC0_normalized_.vector()[:] = eC0_normalized
    eL0_normalized_.vector()[:] = eL0_normalized
    eR0_normalized_.vector()[:] = eR0_normalized

    return eC0_normalized_, eL0_normalized_, eR0_normalized_

def compute_strain(IODet, SimDet, LVid=1, RVid=2, cycle=None):
    isLV = False
    isBiV = False

    mesh = df.Mesh()
    hdf = df.HDF5File(mesh.mpi_comm(),IODet["outputfolder"] + "/" + IODet["caseID"] + "/" + "Data.h5","r",)

    u_arr = extractdisplacement(IODet, SimDet, cycle=None)

    if "isLV" in list(SimDet.keys()):
        isLV = SimDet["isLV"]

        if(isLV):

            mesh_params = {
                "directory": IODet["directory_me"],
                "casename": IODet["casename_me"],
                "outputfolder": IODet["outputfolder"],
                "foldername": IODet["folderName"],
                "isLV": isLV,
            }

            Mesh_obj = lv_mechanics_mesh(mesh_params, SimDet)
            try:
                eC0 = Mesh_obj.eC0
                eL0 = Mesh_obj.eL0
                eR0 = Mesh_obj.eR0

            except AttributeError:

                fiber_angle_param = {
                    "mesh": Mesh_obj.mesh,
                    "facetboundaries": Mesh_obj.facetboundaries,
                    "LV_fiber_angle": [0.01, -0.01],
                    "LV_sheet_angle": [0.1, -0.1],
                    "minztol": Mesh_obj.mesh.hmax()/2.0, # Coarse mesh
                    "isrotatept": False,
                    "isreturn": True,
                    "outfilename": IODet["casename_me"],
                    "outdirectory": IODet["outputfolder"] + IODet["caseID"] + IODet["folderName"],
                    "baseid": SimDet["topid"],
                    "epiid": SimDet["epiid"],
                    "lvid": SimDet["LVendoid"],
                    "degree": SimDet["GiccioneParams"]["deg"]
                }

                eC0, eL0, eR0  = vtk_py.addLVfiber_LDRB(fiber_angle_param)

    if "isBiV" in list(SimDet.keys()):

        isBiV = SimDet["isBiV"]

        if(isBiV):
            mesh_params = {
                "directory": IODet["directory_me"],
                "casename": IODet["casename_me"],
                "casename_marked": IODet["casename_me"],
                "outputfolder": IODet["outputfolder"],
                "foldername": IODet["folderName"],
                "isBiV": isBiV,
            }

            Mesh_obj = biv_mechanics_mesh(mesh_params, SimDet)

    # -------------------------------------------------
    # Fibers
    # -------------------------------------------------
    fiberfile = os.path.join(IODet["directory_me"], "BiV_fibers.h5")

    VQuadelem = df.VectorElement("Quadrature", Mesh_obj.mesh.ufl_cell(), degree=4, quad_scheme="default")
    fiberCLR = df.FunctionSpace(Mesh_obj.mesh, VQuadelem)

    eC0 = df.Function(fiberCLR)
    eL0 = df.Function(fiberCLR)
    eR0 = df.Function(fiberCLR)

    h5 = df.HDF5File(Mesh_obj.mesh.mpi_comm(), fiberfile, "r")
    h5.read(eC0, "BiV_fibers/eC0")
    h5.read(eL0, "BiV_fibers/eL0")
    h5.read(eR0, "BiV_fibers/eR0")
    h5.close()

    deg = SimDet["GiccioneParams"]["deg"]
    eC0_normalized = eC0 / df.sqrt(df.inner(eC0, eC0))
    eL0_normalized = eL0 / df.sqrt(df.inner(eL0, eL0))
    eR0_normalized = eR0 / df.sqrt(df.inner(eR0, eR0))


    if SimDet["Mechanics Discretization"] is "P1P1":
        var_deg = 1
    else:
        var_deg = 2

    udisp = df.Function(df.VectorFunctionSpace(Mesh_obj.mesh, "CG", var_deg))
    udisp.vector()[:] = u_arr[0].vector().get_local()[:]

    GuccioneParams = SimDet["GiccioneParams"]
    params = {
        "mesh": Mesh_obj.mesh,
        "displacement_variable": udisp,
        "material model": GuccioneParams["Passive model"],
        "material params": GuccioneParams["Passive params"],
        "incompressible": GuccioneParams["incompressible"],
        "growth_tensor": None,
    }

    uflforms = Forms(params)
    Fref = df.project(uflforms.Fmat(), df.TensorFunctionSpace(Mesh_obj.mesh, "DG", 0))

    Ecc_outdirectory = os.path.join(IODet["outputfolder"], IODet["caseID"], "Ecc")
    if not os.path.exists(Ecc_outdirectory):
        os.mkdir(Ecc_outdirectory)
    File_Ecc = df.File(os.path.join(Ecc_outdirectory, "Ecc.pvd"))
    Ecc_arr = []
    Ecc_arr_RV = []

    Ell_outdirectory = os.path.join(IODet["outputfolder"], IODet["caseID"], "Ell")
    if not os.path.exists(Ell_outdirectory):
        os.mkdir(Ell_outdirectory)
    File_Ell = df.File(os.path.join(Ell_outdirectory, "Ell.pvd"))
    Ell_arr = []
    Ell_arr_RV = []

    Err_outdirectory = os.path.join(IODet["outputfolder"], IODet["caseID"], "Err")
    if not os.path.exists(Err_outdirectory):
        os.mkdir(Err_outdirectory)
    File_Err = df.File(os.path.join(Err_outdirectory, "Err.pvd"))
    Err_arr = []
    Err_arr_RV = []

    for u_arr_ in u_arr:
        # pdb.set_trace()
        if isinstance(LVid, int):
            wall_vol = df.assemble(
                df.Constant(1.0) * Mesh_obj.dx(LVid),
                form_compiler_parameters={"representation": "uflacs"},
            )
        elif isinstance(LVid, list):
            wall_vol = 0
            for LVid_ in LVid:
                wall_vol += df.assemble(
                    df.Constant(1.0) * Mesh_obj.dx(LVid_),
                    form_compiler_parameters={"representation": "uflacs"},
                )

        wall_vol_RV = df.assemble(
            df.Constant(1.0) * Mesh_obj.dx(RVid),
            form_compiler_parameters={"representation": "uflacs"},
        )

        udisp.vector()[:] = u_arr_.vector().get_local()[:]

        Fmat = uflforms.Fmat()
        F = Fmat * df.inv(Fref)
        Cmat = F.T * F

        Ccc = df.inner(eC0_normalized, Cmat * eC0_normalized)
        Ecc = 0.5 * (1 - 1 / Ccc)

        if isinstance(LVid, int):
            global_Ecc = (
                df.assemble(
                    Ecc * Mesh_obj.dx(LVid),
                    form_compiler_parameters={"representation": "uflacs"},
                )
                / wall_vol
            )
        elif isinstance(LVid, list):
            global_Ecc = 0
            for LVid_ in LVid:
                global_Ecc += (
                    df.assemble(
                        Ecc * Mesh_obj.dx(LVid_),
                        form_compiler_parameters={"representation": "uflacs"},
                    )
                )
            global_Ecc = global_Ecc/wall_vol 

        Ecc_arr.append(global_Ecc)
        if isBiV:
            global_Ecc_RV = (
                df.assemble(
                    Ecc * Mesh_obj.dx(RVid),
                    form_compiler_parameters={"representation": "uflacs"},
                )
                / wall_vol_RV
            )
            Ecc_arr_RV.append(global_Ecc_RV)

        Ecc_field = df.project(
            Ecc,
            df.FunctionSpace(Mesh_obj.mesh, "DG", 0),
            form_compiler_parameters={
                "representation": "uflacs",
                "quadrature_degree": deg,
            },
        )
        Ecc_field.rename("Ecc", "Ecc")
        File_Ecc << Ecc_field
        Ecc_arr.append(global_Ecc)

        Cll = df.inner(eL0_normalized, Cmat * eL0_normalized)
        Ell = 0.5 * (1 - 1 / Cll)
        if isinstance(LVid, int):
            global_Ell = (
                df.assemble(
                    Ell * Mesh_obj.dx(LVid),
                    form_compiler_parameters={"representation": "uflacs"},
                )
                / wall_vol
            )
        elif isinstance(LVid, list):
            global_Ell = 0
            for LVid_ in LVid:
                global_Ell += (
                    df.assemble(
                        Ell * Mesh_obj.dx(LVid_),
                        form_compiler_parameters={"representation": "uflacs"},
                    )
                )
            global_Ell = global_Ell/wall_vol 

        Ell_arr.append(global_Ell)
        if isBiV:
            global_Ell_RV = (
                df.assemble(
                    Ell * Mesh_obj.dx(RVid),
                    form_compiler_parameters={"representation": "uflacs"},
                )
                / wall_vol_RV
            )
            Ell_arr_RV.append(global_Ell_RV)

        Ell_field = df.project(
            Ell,
            df.FunctionSpace(Mesh_obj.mesh, "DG", 0),
            form_compiler_parameters={
                "representation": "uflacs",
                "quadrature_degree": deg,
            },
        )
        Ell_field.rename("Ell", "Ell")
        File_Ell << Ell_field
        Ell_arr.append(global_Ell)

        Crr = df.inner(eR0_normalized, Cmat * eR0_normalized)
        Err = 0.5 * (1 - 1 / Crr)
        if isinstance(LVid, int):
            global_Err = (
                df.assemble(
                    Err * Mesh_obj.dx(LVid),
                    form_compiler_parameters={"representation": "uflacs"},
                )
                / wall_vol
            )
        elif isinstance(LVid, list):
            global_Err = 0
            for LVid_ in LVid:
                global_Err += (
                    df.assemble(
                        Err * Mesh_obj.dx(LVid_),
                        form_compiler_parameters={"representation": "uflacs"},
                    )
                )
            global_Err = global_Err/wall_vol 

        Err_arr.append(global_Err)
        if isBiV:
            global_Err_RV = (
                df.assemble(
                    Err * Mesh_obj.dx(RVid),
                    form_compiler_parameters={"representation": "uflacs"},
                )
                / wall_vol_RV
            )
            Err_arr_RV.append(global_Err_RV)

        Err_field = df.project(
            Err,
            df.FunctionSpace(Mesh_obj.mesh, "DG", 0),
            form_compiler_parameters={
                "representation": "uflacs",
                "quadrature_degree": deg,
            },
        )
        Err_field.rename("Err", "Err")
        File_Err << Err_field
        Err_arr.append(global_Err)

    print("Maximum Ecc :", min(Ecc_arr))
    print("Maximum Ell :", min(Ell_arr))
    print("Maximum Err :", max(Err_arr))

    np.savez(os.path.join(Ecc_outdirectory, "Ecc.npz"), Ecc_arr)
    np.savez(os.path.join(Ell_outdirectory, "Ell.npz"), Ell_arr)
    np.savez(os.path.join(Err_outdirectory, "Err.npz"), Err_arr)

    plt.figure()
    plt.plot(np.arange(0, len(Ecc_arr)), Ecc_arr)
    plt.xlabel("Time point", fontsize=14)
    plt.ylabel("Strain", fontsize=14)
    plt.savefig(os.path.join(Ecc_outdirectory, "Ecc.png"))
    plt.clf()

    plt.figure()
    plt.plot(np.arange(0, len(Ell_arr)), Ell_arr)
    plt.xlabel("Time point", fontsize=14)
    plt.ylabel("Strain", fontsize=14)
    plt.savefig(os.path.join(Ell_outdirectory, "Ell.png"))
    plt.clf()

    plt.figure()
    plt.plot(np.arange(0, len(Err_arr)), Err_arr)
    plt.xlabel("Time point", fontsize=14)
    plt.ylabel("Strain", fontsize=14)
    plt.savefig(os.path.join(Err_outdirectory, "Err.png"))
    plt.clf()

    if isBiV:
        print("Maximum RV Ecc :", min(Ecc_arr_RV))
        print("Maximum RV Ell :", min(Ell_arr_RV))
        print("Maximum RV Err :", max(Err_arr_RV))

        np.savez(os.path.join(Ecc_outdirectory, "Ecc_RV.npz"), Ecc_arr_RV)
        np.savez(os.path.join(Ell_outdirectory, "Ell_RV.npz"), Ell_arr_RV)
        np.savez(os.path.join(Err_outdirectory, "Err_RV.npz"), Err_arr_RV)

        plt.figure()
        plt.plot(np.arange(0, len(Ecc_arr_RV)), Ecc_arr_RV)
        plt.xlabel("Time point", fontsize=14)
        plt.ylabel("Strain", fontsize=14)
        plt.savefig(os.path.join(Ecc_outdirectory, "Ecc_RV.png"))
        plt.clf()

        plt.figure()
        plt.plot(np.arange(0, len(Ell_arr_RV)), Ell_arr_RV)
        plt.xlabel("Time point", fontsize=14)
        plt.ylabel("Strain", fontsize=14)
        plt.savefig(os.path.join(Ell_outdirectory, "Ell_RV.png"))
        plt.clf()

        plt.figure()
        plt.plot(np.arange(0, len(Err_arr_RV)), Err_arr_RV)
        plt.xlabel("Time point", fontsize=14)
        plt.ylabel("Strain", fontsize=14)
        plt.savefig(os.path.join(Err_outdirectory, "Err_RV.png"))
        plt.clf()


def extractdisplacement(IODet, SimDet, cycle=None):
    mesh = df.Mesh()
    hdf = df.HDF5File(mesh.mpi_comm(),IODet["outputfolder"] + "/" + IODet["caseID"] + "/" + "Data.h5","r",)

    # Dump displacement
    if SimDet["Mechanics Discretization"] is "P1P1":
        var_deg = 1
    else:
        var_deg = 2

    try:
        u_arr = extractvtk(IODet["outputfolder"] + "/" + IODet["caseID"],"ME/" + "u","CG",var_deg,IODet["outputfolder"] + "/" + IODet["caseID"] + "/" + "ME_" + "u","u",)
    except RuntimeError:
        print("No attribute for ", var, " found")

    # -------------------------------------------------
    # SAVE NPZ (THIS IS THE ONLY ADDITION)
    # -------------------------------------------------

    print("Saving displacement to npz...", flush=True)

    data = [u.vector().get_local() for u in u_arr]

    out_path = os.path.join(
        IODet["outputfolder"], IODet["caseID"], "ME_u", "u_arr.npz"
    )

    np.savez_compressed(out_path, data=data)

    print(f"Saved: {out_path}", flush=True)         

    return u_arr

def extractdisplacementloading(IODet, SimDet, cycle=None):

    mesh = df.Mesh()
    hdf = df.HDF5File(
        mesh.mpi_comm(),
        IODet["outputfolder"] + "/" + IODet["caseID"] + "/" + "Data.h5",
        "r",
    )

    # Dump displacement
    if SimDet["Mechanics Discretization"] is "P1P1":
        var_deg = 1
    else:
        var_deg = 2

    try:
        u_arr = extractvtk(
            IODet["outputfolder"] + "/" + IODet["caseID"],
            "ME/" + "u_loading",
            "CG",
            var_deg,
            IODet["outputfolder"] + "/" + IODet["caseID"] + "/" + "ME_" + "u_loading",
            "u",
        )
    except RuntimeError:
        print("No attribute for ", var, " found")

    return u_arr

def dumpvtk(IODet, SimDet, cycle=None, ME_var = [], EP_var = [], PJ_var = []):
    hdf = df.HDF5File(df.MPI.comm_world, IODet["outputfolder"] + "/" + IODet["caseID"] + "/" + "Data.h5", "r",)

    list_of_ME_var = ME_var
    list_of_EP_var = EP_var
    list_of_PJ_var = PJ_var

    if hdf.has_dataset("ME"):
        for ME_var in list_of_ME_var:

            var = ME_var[0]
            var_space = ME_var[1]
            var_deg = ME_var[2]

            print("Extracting ME", var, " ", var_space, " ", var_deg)
            gnenericvar = "field"
            try:
                var_arr = extractvtk(
                    IODet["outputfolder"] + "/" + IODet["caseID"],
                    "ME/" + var,
                    var_space,
                    var_deg,
                    IODet["outputfolder"] + "/" + IODet["caseID"] + "/" + "ME_" + var,
                    gnenericvar,
                    group="ME",
                )
            except RuntimeError:
                print("No attribute for ", var, " found")

            if var == "u":
                u_arr = var_arr.copy()

    if hdf.has_dataset("EP"):
        for EP_var in list_of_EP_var:

            var = EP_var[0]
            var_space = EP_var[1]
            var_deg = EP_var[2]
            print("Extracting EP", var)

            try:
                var_arr = extractvtk(IODet["outputfolder"] + "/" + IODet["caseID"],"EP/" + var,var_space,var_deg,IODet["outputfolder"] + "/" + IODet["caseID"] + "/" + "EP_" + var,var,group="EP",)

            except RuntimeError:
                print("No attribute for ", var, " found")

    if hdf.has_dataset("PJ"):
        for PJ_var in list_of_PJ_var:

            var = PJ_var[0]
            var_space = PJ_var[1]
            var_deg = PJ_var[2]
            print("Extracting PJ", var)

            try:
                var_arr = extractvtk(IODet["outputfolder"] + "/" + IODet["caseID"],"PJ/" + var,var_space,var_deg,IODet["outputfolder"] + "/" + IODet["caseID"] + "/" + "PJ_" + var,var,group="PJ",)

            except RuntimeError:
                print("No attribute for ", var, " found")


def plothemodynamics(IODet, SimDet, cycle=None):
    directory = IODet["outputfolder"] + "/"
    casename = IODet["caseID"]
    BCL = SimDet["HeartBeatLength"]

    plt.figure()

    if cycle is None:
        cycle = SimDet["closedloopparam"]["stop_iter"] + 1
        t_systole= np.zeros(cycle)

        for ncycle in range(cycle):
            filename = directory + casename + "/" + "BiV_PV.txt"
            homo_tptt, homo_LVP, homo_LVV, homo_RVP, homo_RVV, homo_Qmv = extract_PV(
                filename, BCL, ncycle, SimDet
            )
            plt.plot(homo_LVV, homo_LVP, label=f"LV")# Cycle = {ncycle}")
            if SimDet.get("isBiV") or SimDet.get("isFCH"):
                plt.plot(homo_RVV, homo_RVP, label=f"RV")# Cycle = {ncycle}")

    else:
        filename = directory + casename + "/" + "BiV_PV.txt"
        homo_tptt, homo_LVP, homo_LVV, homo_RVP, homo_RVV, homo_Qmv = extract_PV(
            filename, BCL, int(cycle), SimDet
        )
        plt.plot(homo_LVV, homo_LVP, label=f"LV")# Cycle = {cycle}")
        if SimDet.get("isBiV") or SimDet.get("isFCH"):
            plt.plot(homo_RVV, homo_RVP, label=f"RV")# Cycle = {cycle}")


    hemodynamics_outdirectory = os.path.join(
        IODet["outputfolder"], IODet["caseID"], "hemodynamics"
    )
    if not os.path.exists(hemodynamics_outdirectory):
        os.mkdir(hemodynamics_outdirectory)

    plt.savefig(os.path.join(hemodynamics_outdirectory, "PV.png"))
    plt.clf()


def plotpressure(IODet, SimDet, cycle=None, compartment="All"):

    directory = IODet["outputfolder"] + "/"
    casename = IODet["caseID"]
    BCL = SimDet["HeartBeatLength"]
    plt.figure()
    filename = directory + casename + "/" + "BiV_P.txt"

    if cycle is None:
        cycle = SimDet["closedloopparam"]["stop_iter"] + 1

        for ncycle in range(cycle):

            tpt, Psv, PLV, Psa, PLA, Ppv, PRV, Ppa, PRA = extract_P(
                filename, BCL, ncycle, SimDet
            )

            if ncycle == cycle - 1:
                meanPsv = time_average(tpt, Psv)
                print(
                    "Mean Psv = ",
                    meanPsv,
                    "mmHg; Peak Psv = ",
                    max(Psv),
                    "mmHg",
                )
                meanPLV = time_average(tpt, PLV)
                print(
                    "Mean PLV = ",
                    meanPLV,
                    "mmHg; Peak PLV = ",
                    max(PLV),
                    "mmHg",
                )
                meanPsa = time_average(tpt, Psa)
                print(
                    "Mean Psa = ",
                    meanPsa,
                    "mmHg; Peak Psa = ",
                    max(Psa),
                    "mmHg",
                )
                meanPLA = time_average(tpt, PLA)
                print(
                    "Mean PLA = ",
                    meanPLA,
                    "mmHg; Peak PLA = ",
                    max(PLA),
                    "mmHg",
                )

            if compartment == "All" or "sv" in compartment:
                plt.plot(tpt, Psv, label=f"Psv Cycle = {ncycle}")
            if compartment == "All" or "lv" in compartment:
                plt.plot(tpt, PLV, label=f"PLV Cycle = {ncycle}")
            if compartment == "All" or "sa" in compartment:
                plt.plot(tpt, Psa, label=f"Psa Cycle = {ncycle}")
            if compartment == "All" or "la" in compartment:
                plt.plot(tpt, PLA, label=f"PLA Cycle = {ncycle}")

            if "isBiV" in list(SimDet.keys()):
                if SimDet["isBiV"]:
                    if compartment == "All" or "pv" in compartment:
                        plt.plot(tpt, Ppv, label=f"Ppv Cycle = {ncycle}")
                    if compartment == "All" or "rv" in compartment:
                        plt.plot(tpt, PRV, label=f"PRV Cycle = {ncycle}")
                    if compartment == "All" or "pa" in compartment:
                        plt.plot(tpt, Ppa, label=f"Ppa Cycle = {ncycle}")
                    if compartment == "All" or "ra" in compartment:
                        plt.plot(tpt, PRA, label=f"PRA Cycle = {ncycle}")

                    if ncycle == cycle - 1:
                        meanPpv = time_average(tpt, Ppv)
                        print(
                            "Mean Ppv = ",
                            meanPpv,
                            "mmHg; Peak Ppv = ",
                            max(Ppv),
                            "mmHg",
                        )
                        meanPRV = time_average(tpt, PRV)
                        print(
                            "Mean PRV = ",
                            meanPRV,
                            "mmHg; Peak PRV = ",
                            max(PRV),
                            "mmHg",
                        )
                        meanPpa = time_average(tpt, Ppa)
                        print(
                            "Mean Ppa = ",
                            meanPpa,
                            "mmHg; Peak Ppa = ",
                            max(Ppa),
                            "mmHg",
                        )
                        meanPRA = time_average(tpt, PRA)
                        print(
                            "Mean PRA = ",
                            meanPRA,
                            "mmHg; Peak PRA = ",
                            max(PRA),
                            "mmHg",
                        )

    else:
        tpt, Psv, PLV, Psa, PLA, Ppv, PRV, Ppa, PRA = extract_P(
            filename, BCL, int(cycle), SimDet
        )

        meanPsv = time_average(tpt, Psv)
        print(
            "Mean Psv = ",
            meanPsv,
            "mmHg; Peak Psv = ",
            max(Psv),
            "mmHg",
        )
        meanPLV = time_average(tpt, PLV)
        print(
            "Mean PLV = ",
            meanPLV,
            "mmHg; Peak PLV = ",
            max(PLV),
            "mmHg",
        )
        meanPsa = time_average(tpt, Psa)
        print(
            "Mean Psa = ",
            meanPsa,
            "mmHg; Peak Psa = ",
            max(Psa),
            "mmHg",
        )
        meanPLA = time_average(tpt, PLA)
        print(
            "Mean PLA = ",
            meanPLA,
            "mmHg; Peak PLA = ",
            max(PLA),
            "mmHg",
        )

        if compartment == "All" or "sv" in compartment:
            plt.plot(tpt, Psv, label=f"Psv")# Cycle = {cycle}")
        if compartment == "All" or "lv" in compartment:
            plt.plot(tpt, PLV, label=f"PLV")# Cycle = {cycle}")
        if compartment == "All" or "sa" in compartment:
            plt.plot(tpt, Psa, label=f"Psa")# Cycle = {cycle}")
        if compartment == "All" or "la" in compartment:
            plt.plot(tpt, PLA, label=f"PLA")# Cycle = {cycle}")

        if "isBiV" in list(SimDet.keys()):
            if SimDet["isBiV"]:
                if compartment == "All" or "pv" in compartment:
                    plt.plot(tpt, Ppv, label=f"Ppv")# Cycle = {cycle}")
                if compartment == "All" or "rv" in compartment:
                    plt.plot(tpt, PRV, label=f"PRV")# Cycle = {cycle}")
                if compartment == "All" or "pa" in compartment:
                    plt.plot(tpt, Ppa, label=f"Ppa")# Cycle = {cycle}")
                if compartment == "All" or "ra" in compartment:
                    plt.plot(tpt, PRA, label=f"PRA")# Cycle = {cycle}")

                meanPpv = time_average(tpt, Ppv)
                print(
                    "Mean Ppv = ",
                    meanPpv,
                    "mmHg; Peak Ppv = ",
                    max(Ppv),
                    "mmHg",
                )
                meanPRV = time_average(tpt, PRV)
                print(
                    "Mean PRV = ",
                    meanPRV,
                    "mmHg; Peak PRV = ",
                    max(PRV),
                    "mmHg",
                )
                meanPpa = time_average(tpt, Ppa)
                print(
                    "Mean Ppa = ",
                    meanPpa,
                    "mmHg; Peak Ppa = ",
                    max(Ppa),
                    "mmHg",
                )
                meanPRA = time_average(tpt, PRA)
                print(
                    "Mean PRA = ",
                    meanPRA,
                    "mmHg; Peak PRA = ",
                    max(PRA),
                    "mmHg",
                )

    hemodynamics_outdirectory = os.path.join(
        IODet["outputfolder"], IODet["caseID"], "hemodynamics"
    )
    if not os.path.exists(hemodynamics_outdirectory):
        os.mkdir(hemodynamics_outdirectory)

    plt.savefig(os.path.join(hemodynamics_outdirectory, "Pressure.png"))
    plt.clf()

def plotflow(IODet, SimDet, cycle=None, compartment="All"):

    directory = IODet["outputfolder"] + "/"
    casename = IODet["caseID"]
    BCL = SimDet["HeartBeatLength"]
    if cycle is None:
        cycle = SimDet["closedloopparam"]["stop_iter"] + 1

    plt.figure()
    for ncycle in range(cycle):
        filename = directory + casename + "/" + "BiV_Q.txt"

        tpt, Qav, Qmv, Qsa, Qsv, Qpvv, Qtv, Qpa, Qpv, Qlvad = extract_Q(
            filename, BCL, ncycle, SimDet
        )

        if ncycle == cycle - 1:
            meanQav = time_average(tpt, Qav)
            print(
                "Mean Qav = ",
                meanQav * 60,
                "L/min; Peak Qav = ",
                max(Qav) * 60,
                "L/min",
            )
            meanQmv = time_average(tpt, Qmv)
            print(
                "Mean Qmv = ",
                meanQmv * 60,
                "L/min; Peak Qmv = ",
                max(Qmv) * 60,
                "L/min",
            )
            meanQsa = time_average(tpt, Qsa)
            print(
                "Mean Qsa = ",
                meanQsa * 60,
                "L/min; Peak Qsa = ",
                max(Qsa) * 60,
                "L/min",
            )
            meanQsv = time_average(tpt, Qsv)
            print(
                "Mean Qsv = ",
                meanQsv * 60,
                "L/min; Peak Qsv = ",
                max(Qsv) * 60,
                "L/min",
            )
            meanQlvad = time_average(tpt, Qlvad)
            print(
                "Mean Qlvad = ",
                meanQlvad * 60,
                "L/min; Peak Qlvad = ",
                max(Qlvad) * 60,
                "L/min",
            )

        if compartment == "All" or "av" in compartment:
            plt.plot(tpt, Qav * 60, label=f"Qav Cycle = {ncycle}")
        if compartment == "All" or "mv" in compartment:
            plt.plot(tpt, Qmv * 60, label=f"Qmv Cycle = {ncycle}")
        if compartment == "All" or "sa" in compartment:
            plt.plot(tpt, Qsa * 60, label=f"Qsa Cycle = {ncycle}")
        if compartment == "All" or "sv" in compartment:
            plt.plot(tpt, Qsv * 60, label=f"Qsv Cycle = {ncycle}")
        if compartment == "All" or "lvad" in compartment:
            plt.plot(tpt, Qlvad * 60, label=f"Qlvad Cycle = {ncycle}")

        if "isBiV" in list(SimDet.keys()):
            if SimDet["isBiV"]:
                if compartment == "All" or "ppv" in compartment:
                    plt.plot(tpt, Qpvv * 60, label=f"Qpvv Cycle = {ncycle}")
                if compartment == "All" or "tv" in compartment:
                    plt.plot(tpt, Qtv * 60, label=f"Qtv Cycle = {ncycle}")
                if compartment == "All" or "pa" in compartment:
                    plt.plot(tpt, Qpa * 60, label=f"Qpa Cycle = {ncycle}")
                if compartment == "All" or "pv" in compartment:
                    plt.plot(tpt, Qpv * 60, label=f"Qpv Cycle = {ncycle}")

                if ncycle == cycle - 1:
                    meanQpvv = time_average(tpt, Qpvv)
                    print(
                        "Mean Qpvv = ",
                        meanQpvv * 60,
                        "L/min; Peak Qpvv = ",
                        max(Qpvv) * 60,
                        "L/min",
                    )
                    meanQtv = time_average(tpt, Qtv)
                    print(
                        "Mean Qtv = ",
                        meanQtv * 60,
                        "L/min; Peak Qtv = ",
                        max(Qtv) * 60,
                        "L/min",
                    )
                    meanQpa = time_average(tpt, Qpa)
                    print(
                        "Mean Qpa = ",
                        meanQpa * 60,
                        "L/min; Peak Qpa = ",
                        max(Qpa) * 60,
                        "L/min",
                    )
                    meanQpv = time_average(tpt, Qpv)
                    print(
                        "Mean Qpv = ",
                        meanQpv * 60,
                        "L/min; Peak Qpv = ",
                        max(Qpv) * 60,
                        "L/min",
                    )

    hemodynamics_outdirectory = os.path.join(
        IODet["outputfolder"], IODet["caseID"], "hemodynamics"
    )
    if not os.path.exists(hemodynamics_outdirectory):
        os.mkdir(hemodynamics_outdirectory)

    # plt.plot(homo_LVV, homo_LVP)
    plt.legend()
    plt.ylabel("Flow rate (L/min)")
    plt.xlabel("Time (s)")
    plt.savefig(os.path.join(hemodynamics_outdirectory, "Flow.png"))
    plt.clf()


def time_average(time_array, data_array):
    """Calculates the time-weighted average of a data array.

    Args:
        time_array (list or numpy array): Array of timestamps.
        data_array (list or numpy array): Array of data values.

    Returns:
        float: Time-weighted average.
    """

    if len(time_array) != len(data_array):
        raise ValueError("Time and data arrays must have the same length.")

    time_array = np.array(time_array)
    data_array = np.array(data_array)

    # Calculate time differences
    time_diffs = np.diff(time_array)

    # Calculate weighted average
    weighted_sum = np.sum(data_array[:-1] * time_diffs)
    total_time = time_array[-1] - time_array[0]

    return weighted_sum / total_time


def plot_PV_comparison(IODet, SimDet, cycle=None):
    """Compare pressure-time curves from two different BiV_PV.txt files."""
    BCL = SimDet["HeartBeatLength"]
    mode = IODet["comparison_plot_mode"]
    
    if cycle is None:
        cycle = SimDet["closedloopparam"]["stop_iter"] + 1

    plt.figure()

    for ncycle in range(cycle):
        filename1 = os.path.join(IODet["outputfolder"]+IODet["directory_baseline"]+"/BiV_PV.txt")
        time1, LVP1, _, _, _, _ = extract_PV(filename1, BCL, ncycle, SimDet)
        plt.plot(time1, LVP1, label=f"LV Baseline")

        if mode == "disease":
            filename2 = os.path.join(IODet["outputfolder"]+IODet["directory_block"]+"/BiV_PV.txt")
            time2, LVP2, _, _, _, _ = extract_PV(filename2, BCL, ncycle, SimDet)
            plt.plot(time2, LVP2,linestyle="dashed", label=f"LV LBBB")

            filename3 = os.path.join(IODet["outputfolder"]+IODet["directory_ischemia_block"]+"/BiV_PV.txt")
            time3, LVP3, _, _, _, _ = extract_PV(filename3, BCL, ncycle, SimDet)
            plt.plot(time3, LVP3,linestyle="dashed", label=f"LV iLBBB")

        elif mode == "pacing":
            filename2 = os.path.join(IODet["outputfolder"]+IODet["directory_bivp"]+"/BiV_PV.txt")
            time2, LVP2, _, _, _, _ = extract_PV(filename2, BCL, ncycle, SimDet)
            plt.plot(time2, LVP2,linestyle="dashed", label=f"LV BiVP")

            filename3 = os.path.join(IODet["outputfolder"]+IODet["directory_lbbap"]+"/BiV_PV.txt")
            time3, LVP3, _, _, _, _ = extract_PV(filename3, BCL, ncycle, SimDet)
            plt.plot(time3, LVP3,linestyle="dashed", label=f"LV LBBAP")    

        elif mode == "ischemic_pacing":           
            filename2 = os.path.join(IODet["outputfolder"]+IODet["directory_ischemia_bivp"]+"/BiV_PV.txt")
            time2, LVP2, _, _, _, _ = extract_PV(filename2, BCL, ncycle, SimDet)
            plt.plot(time2, LVP2,linestyle="dashed", label=f"LV iBiVP")

            filename3 = os.path.join(IODet["outputfolder"]+IODet["directory_ischemia_lbbap"]+"/BiV_PV.txt")
            time3, LVP3, _, _, _, _ = extract_PV(filename3, BCL, ncycle, SimDet)
            plt.plot(time3, LVP3,linestyle="dashed", label=f"LV iLBBAP")  

    # Create output directory if not exists
    hemodynamics_outdirectory = os.path.join(IODet["outputfolder"]+IODet["comparison_plot_mode"])
    if not os.path.exists(hemodynamics_outdirectory):
        os.mkdir(hemodynamics_outdirectory)

    plt.ylabel("Pressure (mmHg)")
    plt.xlabel("Time [ms]")
    plt.savefig(os.path.join(hemodynamics_outdirectory, "Pressure_Comparison_"+IODet["comparison_plot_mode"]+"_simulation.png"))
    plt.show()


def plot_PV_comparison_ind(IODet, SimDet, cycle=None):
    """Compare pressure-time curves from two different BiV_PV.txt files."""
    BCL = SimDet["HeartBeatLength"]
    ncycle = cycle 
    mode = IODet["comparison_plot_mode"]

    plt.figure()
    if mode == "disease":
        filename1 = os.path.join(IODet["outputfolder"]+IODet["directory_baseline"]+"/BiV_PV.txt")
        time1, LVP1, _, RVP1, _, _ = extract_PV(filename1, BCL, ncycle, SimDet)   
        plt.plot(time1, LVP1, label=f"LV Baseline")
        plt.plot(time1, RVP1, label=f"RV Baseline")

        filename2 = os.path.join(IODet["outputfolder"]+IODet["directory_block"]+"/BiV_PV.txt")
        time2, LVP2, _, RVP2, _, _ = extract_PV(filename2, BCL, ncycle, SimDet)
        plt.plot(time2, LVP2, linestyle="dashed", label=f"LV LBBB")
        plt.plot(time2, RVP2, linestyle="dashed", label=f"RV LBBB")

        filename3 = os.path.join(IODet["outputfolder"]+IODet["directory_ischemia_block"]+"/BiV_PV.txt")
        time3, LVP3, _, RVP3, _, _ = extract_PV(filename3, BCL, ncycle, SimDet)
        plt.plot(time3, LVP3, linestyle="dashed", label=f"LV iLBBB")
        plt.plot(time3, RVP3, linestyle="dashed", label=f"RV iLBBB")      
            
    elif mode == "pacing":
        filename1 = os.path.join(IODet["outputfolder"]+IODet["directory_baseline"]+"/BiV_PV.txt")
        time1, LVP1, _, RVP1, _, _ = extract_PV(filename1, BCL, ncycle, SimDet)   
        plt.plot(time1, LVP1, label=f"LV Baseline")
        plt.plot(time1, RVP1, label=f"RV Baseline")

        filename2 = os.path.join(IODet["outputfolder"]+IODet["directory_block"]+"/BiV_PV.txt")
        time2, LVP2, _, RVP2, _, _ = extract_PV(filename2, BCL, ncycle, SimDet)
        plt.plot(time2, LVP2, linestyle="dashed", label=f"LV LBBB")
        plt.plot(time2, RVP2, linestyle="dashed", label=f"RV LBBB")

        filename3 = os.path.join(IODet["outputfolder"]+IODet["directory_bivp"]+"/BiV_PV.txt")
        time3, LVP3, _, RVP3, _, _ = extract_PV(filename3, BCL, ncycle, SimDet)
        plt.plot(time3, LVP3, linestyle="dotted", label=f"LV BiVP")  
        plt.plot(time3, RVP3, linestyle="dotted", label=f"RV BiVP")

        filename4 = os.path.join(IODet["outputfolder"]+IODet["directory_lbbap"]+"/BiV_PV.txt")
        time4, LVP4, _, RVP4, _, _ = extract_PV(filename4, BCL, ncycle, SimDet)
        plt.plot(time4, LVP4, linestyle="dashdot", label=f"LV LBBAP")  
        plt.plot(time4, RVP4, linestyle="dashdot", label=f"RV LBBAP")   

    elif mode == "ischemic_pacing":
        filename1 = os.path.join(IODet["outputfolder"]+IODet["directory_baseline"]+"/BiV_PV.txt")
        time1, LVP1, _, RVP1, _, _ = extract_PV(filename1, BCL, ncycle, SimDet)   
        plt.plot(time1, LVP1, label=f"LV Baseline")
        plt.plot(time1, RVP1, label=f"RV Baseline")

        filename2 = os.path.join(IODet["outputfolder"]+IODet["directory_ischemia_block"]+"/BiV_PV.txt")
        time2, LVP2, _, RVP2, _, _ = extract_PV(filename2, BCL, ncycle, SimDet)
        plt.plot(time2, LVP2, linestyle="dashed", label=f"LV iLBBB")
        plt.plot(time2, RVP2, linestyle="dashed", label=f"RV iLBBB")

        filename3 = os.path.join(IODet["outputfolder"]+IODet["directory_ischemia_bivp"]+"/BiV_PV.txt")
        time3, LVP3, _, RVP3, _, _ = extract_PV(filename3, BCL, ncycle, SimDet)
        plt.plot(time3, LVP3, linestyle="dotted", label=f"LV iBiVP")  
        plt.plot(time3, RVP3, linestyle="dotted", label=f"RV iBiVP")

        filename4 = os.path.join(IODet["outputfolder"]+IODet["directory_ischemia_lbbap"]+"/BiV_PV.txt")
        time4, LVP4, _, RVP4, _, _ = extract_PV(filename4, BCL, ncycle, SimDet)
        plt.plot(time4, LVP4, linestyle="dashdot", label=f"LV iLBBAP")  
        plt.plot(time4, RVP4, linestyle="dashdot", label=f"RV iLBBAP")       


    # Create output directory if not exists
    hemodynamics_outdirectory = os.path.join(IODet["outputfolder"]+IODet["comparison_plot_mode"])
    if not os.path.exists(hemodynamics_outdirectory):
        os.mkdir(hemodynamics_outdirectory)

    plt.legend()
    plt.ylabel("Pressure (mmHg)")
    plt.xlabel("Time (s)")
    plt.savefig(os.path.join(hemodynamics_outdirectory, "Pressure_Comparison_"+IODet["comparison_plot_mode"]+"_cycle_"+str(cycle)+"blocks.png"))
    plt.show()    


def plothemodynamics_comparison_ind(IODet, SimDet, cycle=None):
    """Compare PV loops from two different BiV_PV.txt files."""
    BCL = SimDet["HeartBeatLength"]
    ncycle = cycle 
    mode = IODet["comparison_plot_mode"]    

    plt.figure()
    if mode == "disease":
        filename1 = os.path.join(IODet["outputfolder"]+IODet["directory_baseline"]+"/BiV_PV.txt")
        homo_tptt1, homo_LVP1, homo_LVV1, homo_RVP1, homo_RVV1, homo_Qmv1 = extract_PV(filename1, BCL, ncycle, SimDet)
        plt.plot(homo_LVV1, homo_LVP1, label=f"LV Baseline")
        plt.plot(homo_RVV1, homo_RVP1, label=f"RV Baseline")

        filename2 = os.path.join(IODet["outputfolder"]+IODet["directory_block"]+"/BiV_PV.txt")
        homo_tptt2, homo_LVP2, homo_LVV2, homo_RVP2, homo_RVV2, homo_Qmv2 = extract_PV(filename2, BCL, ncycle, SimDet)
        plt.plot(homo_LVV2, homo_LVP2, linestyle="dashed", label=f"LV LBBB")
        plt.plot(homo_RVV2, homo_RVP2, linestyle="dashed", label=f"RV LBBB")

        filename3 = os.path.join(IODet["outputfolder"]+IODet["directory_ischemia_block"]+"/BiV_PV.txt")
        homo_tptt3, homo_LVP3, homo_LVV3, homo_RVP3, homo_RVV3, homo_Qmv3 = extract_PV(filename3, BCL, ncycle, SimDet)
        plt.plot(homo_LVV3, homo_LVP3, linestyle="dotted",label=f"LV iLBBB")
        plt.plot(homo_RVV3, homo_RVP3, linestyle="dotted",label=f"RV iLBBB")      
            
    elif mode == "pacing":
        filename1 = os.path.join(IODet["outputfolder"]+IODet["directory_baseline"]+"/BiV_PV.txt")
        homo_tptt1, homo_LVP1, homo_LVV1, homo_RVP1, homo_RVV1, homo_Qmv1 = extract_PV(filename1, BCL, ncycle, SimDet)
        plt.plot(homo_LVV1, homo_LVP1, label=f"LV Baseline")
        plt.plot(homo_RVV1, homo_RVP1, label=f"RV Baseline")

        filename2 = os.path.join(IODet["outputfolder"]+IODet["directory_block"]+"/BiV_PV.txt")
        homo_tptt2, homo_LVP2, homo_LVV2, homo_RVP2, homo_RVV2, homo_Qmv2 = extract_PV(filename2, BCL, ncycle, SimDet)
        plt.plot(homo_LVV2, homo_LVP2, linestyle="dashed", label=f"LV LBBB")
        plt.plot(homo_RVV2, homo_RVP2, linestyle="dashed", label=f"RV LBBB")

        filename3 = os.path.join(IODet["outputfolder"]+IODet["directory_bivp"]+"/BiV_PV.txt")
        homo_tptt3, homo_LVP3, homo_LVV3, homo_RVP3, homo_RVV3, homo_Qmv3 = extract_PV(filename3, BCL, ncycle, SimDet)
        plt.plot(homo_LVV3, homo_LVP3, linestyle="dotted",label=f"LV BiVP")
        plt.plot(homo_RVV3, homo_RVP3, linestyle="dotted",label=f"RV BiVP")

        filename4 = os.path.join(IODet["outputfolder"]+IODet["directory_lbbap"]+"/BiV_PV.txt")
        homo_tptt4, homo_LVP4, homo_LVV4, homo_RVP4, homo_RVV4, homo_Qmv4 = extract_PV(filename4, BCL, ncycle, SimDet)
        plt.plot(homo_LVV4, homo_LVP4, linestyle="dashdot",label=f"LV LBBAP")
        plt.plot(homo_RVV4, homo_RVP4, linestyle="dashdot",label=f"RV LBBAP") 

    elif mode == "ischemic_pacing":
        filename1 = os.path.join(IODet["outputfolder"]+IODet["directory_baseline"]+"/BiV_PV.txt")
        homo_tptt1, homo_LVP1, homo_LVV1, homo_RVP1, homo_RVV1, homo_Qmv1 = extract_PV(filename1, BCL, ncycle, SimDet)
        plt.plot(homo_LVV1, homo_LVP1, label=f"LV Baseline")
        plt.plot(homo_RVV1, homo_RVP1, label=f"RV Baseline")

        filename2 = os.path.join(IODet["outputfolder"]+IODet["directory_ischemia_block"]+"/BiV_PV.txt")
        homo_tptt2, homo_LVP2, homo_LVV2, homo_RVP2, homo_RVV2, homo_Qmv2 = extract_PV(filename2, BCL, ncycle, SimDet)
        plt.plot(homo_LVV2, homo_LVP2, linestyle="dashed", label=f"LV iLBBB")
        plt.plot(homo_RVV2, homo_RVP2, linestyle="dashed", label=f"RV iLBBB")

        filename3 = os.path.join(IODet["outputfolder"]+IODet["directory_ischemia_bivp"]+"/BiV_PV.txt")
        homo_tptt3, homo_LVP3, homo_LVV3, homo_RVP3, homo_RVV3, homo_Qmv3 = extract_PV(filename3, BCL, ncycle, SimDet)
        plt.plot(homo_LVV3, homo_LVP3, linestyle="dotted",label=f"LV iBiVP")
        plt.plot(homo_RVV3, homo_RVP3, linestyle="dotted",label=f"RV iBiVP")

        filename4 = os.path.join(IODet["outputfolder"]+IODet["directory_ischemia_lbbap"]+"/BiV_PV.txt")
        homo_tptt4, homo_LVP4, homo_LVV4, homo_RVP4, homo_RVV4, homo_Qmv4 = extract_PV(filename4, BCL, ncycle, SimDet)
        plt.plot(homo_LVV4, homo_LVP4, linestyle="dashdot",label=f"LV iLBBAP")
        plt.plot(homo_RVV4, homo_RVP4, linestyle="dashdot",label=f"RV iLBBAP")       

    # Create output directory if not exists
    hemodynamics_outdirectory = os.path.join(IODet["outputfolder"]+IODet["comparison_plot_mode"])
    if not os.path.exists(hemodynamics_outdirectory):
        os.mkdir(hemodynamics_outdirectory)

    # plt.legend()
    plt.xlabel("Volume (mL)")
    plt.ylabel("Pressure (mmHg)")
    plt.savefig(os.path.join(hemodynamics_outdirectory, "PV_Comparison_"+IODet["comparison_plot_mode"]+"_cycle_"+str(cycle)+"blocks.png"))
    plt.show()


def plothemodynamics_comparison(IODet, SimDet, cycle=None):
    """Compare PV loops from two different BiV_PV.txt files."""
    BCL = SimDet["HeartBeatLength"]
    mode = IODet["comparison_plot_mode"]
    
    if cycle is None:
        cycle = SimDet["closedloopparam"]["stop_iter"] + 1

    plt.figure()

    for ncycle in range(cycle):
        filename1 = os.path.join(IODet["outputfolder"]+IODet["directory_baseline"]+"/BiV_PV.txt")
        homo_tptt1, homo_LVP1, homo_LVV1, homo_RVP1, homo_RVV1, homo_Qmv1 = extract_PV(filename1, BCL, ncycle, SimDet)
        plt.plot(homo_LVV1, homo_LVP1, label=f"LV Baseline")
        plt.plot(homo_RVV1, homo_RVP1, label=f"RV Baseline")

        if mode == "disease":
            filename2 = os.path.join(IODet["outputfolder"]+IODet["directory_block"]+"/BiV_PV.txt")
            homo_tptt2, homo_LVP2, homo_LVV2, homo_RVP2, homo_RVV2, homo_Qmv2 = extract_PV(filename2, BCL, ncycle, SimDet)
            plt.plot(homo_LVV2, homo_LVP2,linestyle="dashed", label=f"LV LBBB")
            plt.plot(homo_RVV2, homo_RVP2,linestyle="dashed", label=f"RV LBBB")

            filename3 = os.path.join(IODet["outputfolder"]+IODet["directory_ischemia_block"]+"/BiV_PV.txt")
            homo_tptt3, homo_LVP3, homo_LVV3, homo_RVP3, homo_RVV3, homo_Qmv3 = extract_PV(filename3, BCL, ncycle, SimDet)
            plt.plot(homo_LVV3, homo_LVP3,linestyle="dashed", label=f"LV iLBBB")
            plt.plot(homo_RVV3, homo_RVP3,linestyle="dashed", label=f"RV iLBBB")

        elif mode == "pacing":
            filename2 = os.path.join(IODet["outputfolder"]+IODet["directory_bivp"]+"/BiV_PV.txt")
            homo_tptt2, homo_LVP2, homo_LVV2, homo_RVP2, homo_RVV2, homo_Qmv2 = extract_PV(filename2, BCL, ncycle, SimDet)
            plt.plot(homo_LVV2, homo_LVP2,linestyle="dashed", label=f"LV BiVP")
            plt.plot(homo_RVV2, homo_RVP2,linestyle="dashed", label=f"RV BiVP")

            filename3 = os.path.join(IODet["outputfolder"]+IODet["directory_lbbap"]+"/BiV_PV.txt")
            homo_tptt3, homo_LVP3, homo_LVV3, homo_RVP3, homo_RVV3, homo_Qmv3 = extract_PV(filename3, BCL, ncycle, SimDet)
            plt.plot(homo_LVV3, homo_LVP3,linestyle="dashed", label=f"LV LBBAP")
            plt.plot(homo_RVV3, homo_RVP3,linestyle="dashed", label=f"RV LBBAP")  

        elif mode == "ischemic_pacing":           
            filename2 = os.path.join(IODet["outputfolder"]+IODet["directory_ischemia_bivp"]+"/BiV_PV.txt")
            homo_tptt2, homo_LVP2, homo_LVV2, homo_RVP2, homo_RVV2, homo_Qmv2 = extract_PV(filename2, BCL, ncycle, SimDet)
            plt.plot(homo_LVV2, homo_LVP2,linestyle="dashed", label=f"LV iBiVP")
            plt.plot(homo_RVV2, homo_RVP2,linestyle="dashed", label=f"RV iBiVP")

            filename3 = os.path.join(IODet["outputfolder"]+IODet["directory_ischemia_lbbap"]+"/BiV_PV.txt")
            homo_tptt3, homo_LVP3, homo_LVV3, homo_RVP3, homo_RVV3, homo_Qmv3 = extract_PV(filename3, BCL, ncycle, SimDet)
            plt.plot(homo_LVV3, homo_LVP3,linestyle="dashed", label=f"LV iLBBAP")
            plt.plot(homo_RVV3, homo_RVP3,linestyle="dashed", label=f"RV iLBBAP") 


    # Create output directory if not exists
    hemodynamics_outdirectory = os.path.join(IODet["outputfolder"]+IODet["comparison_plot_mode"])
    if not os.path.exists(hemodynamics_outdirectory):
        os.mkdir(hemodynamics_outdirectory)

    plt.legend()
    plt.xlabel("Volume (mL)")
    plt.ylabel("Pressure (mmHg)")
    plt.savefig(os.path.join(hemodynamics_outdirectory, "PV_Comparison_"+IODet["comparison_plot_mode"]+"_simulation.png"))
    plt.show()


def compute_strain_AHA_profile(IODet, SimDet, LVid=1, cycle=None, R_apex=1.0):
    isLV = False
    isBiV = False
    BCL = SimDet["HeartBeatLength"] 
    t_vol_min = np.zeros(cycle+1)   

    for ncycle in range(cycle+1):
        filename1 = os.path.join(IODet["outputfolder"]+IODet["caseID"]+"/BiV_PV.txt")
        homo_tptt1, homo_LVP1, homo_LVV1, homo_RVP1, homo_RVV1, homo_Qmv1 = extract_PV(filename1, BCL, ncycle, SimDet)
        t_vol_min[ncycle] = homo_tptt1[np.argmin(homo_LVV1)]

    # ------------------------------------------------------------
    # Read displacement
    # ------------------------------------------------------------
    u_arr = extractdisplacement(IODet, SimDet, cycle=cycle)

    # ------------------------------------------------------------
    # Find EDV frame for selected cycle
    # ------------------------------------------------------------
    if cycle is not None:
        filename_pv = os.path.join(
            IODet["outputfolder"] + IODet["caseID"],
            "BiV_PV.txt"
        )

        homo_tptt, homo_LVP, homo_LVV, homo_RVP, homo_RVV, homo_Qmv = extract_PV(
            filename_pv, BCL, cycle, SimDet
        )

        ref_idx = int(np.argmax(homo_LVV))

        print("EDV reference for regional strain:")
        print(f"  cycle      = {cycle}")
        print(f"  EDV index  = {ref_idx}")
        print(f"  EDV time   = {homo_tptt[ref_idx]}")
        print(f"  EDV volume = {homo_LVV[ref_idx]}")

        if ref_idx >= len(u_arr):
            raise RuntimeError(
                "EDV index from PV file is outside displacement array. "
                "Check that extract_PV(..., cycle) and extractdisplacement(..., cycle) "
                "return the same number of time points."
            )

    else:
        ref_idx = 0
        print("Warning: cycle=None. Using frame 0 as strain reference.")    

    # ------------------------------------------------------------
    # Load mechanics mesh
    # ------------------------------------------------------------
    if "isLV" in list(SimDet.keys()):
        mesh_params = {"directory": IODet["directory_me"],"casename": IODet["casename_me"],"outputfolder": IODet["outputfolder"],"foldername": IODet["folderName"],"isLV": isLV,}
        Mesh_obj = lv_mechanics_mesh(mesh_params, SimDet)  
    elif isBiV:
        mesh_params = {"directory": IODet["directory_me"],"casename": IODet["casename_me"],"casename_marked": IODet["casename_me"],"outputfolder": IODet["outputfolder"],"foldername": IODet["folderName"],"isBiV": isBiV,}
        Mesh_obj = biv_mechanics_mesh(mesh_params, SimDet)

    else:
        raise RuntimeError("Neither SimDet['isLV'] nor SimDet['isBiV'] is True.")              

    mesh = df.Mesh()
    # Fiber/circ/long/radial directions should already be saved in HDF5
    try:
        eC0 = Mesh_obj.eC0
        eL0 = Mesh_obj.eL0
        eR0 = Mesh_obj.eR0
    except AttributeError:
        raise RuntimeError(
            "Could not find eC0/eL0/eR0 in Mesh_obj. "
            "Since these are saved in the mesh HDF5, check that "
            "lv_mechanics_mesh() or biv_mechanics_mesh() reads them correctly."
        )

    deg = SimDet["GiccioneParams"]["deg"]

    # Normalize directions
    eC0_normalized = eC0 / df.sqrt(df.inner(eC0, eC0))
    eL0_normalized = eL0 / df.sqrt(df.inner(eL0, eL0))
    eR0_normalized = eR0 / df.sqrt(df.inner(eR0, eR0))

    # ------------------------------------------------------------
    # AHA segmentation
    # First assign geometric AHA tags, then for BiV keep only LV+septum.
    # ------------------------------------------------------------
    aha_segments_raw = tag_AHA_segments(mesh,apex_axis="z",R_apex=R_apex,reference_point=SimDet["HIS"],rotate=True,plot_debug=False)

    if isBiV:
        aha_segments = df.MeshFunction("size_t", mesh, mesh.topology().dim())
        aha_segments.set_all(0)

        matid = Mesh_obj.matid

        LV_ID = 0
        SEPTUM_ID = 1

        for cell in df.cells(mesh):
            cidx = cell.index()
            m = int(matid[cidx])
            tag = int(aha_segments_raw[cidx])

            if m in [LV_ID, SEPTUM_ID] and 1 <= tag <= 17:
                aha_segments[cidx] = tag
            else:
                aha_segments[cidx] = 0

    else:
        aha_segments = aha_segments_raw

    dx_aha = df.Measure("dx", domain=mesh, subdomain_data=aha_segments)
    n_aha_seg = 17
    aha_volumes = [df.assemble(df.Constant(1.0) * dx_aha(seg))for seg in range(1, n_aha_seg + 1)]

    print("LV/septum AHA segment volumes:")
    for seg, vol in enumerate(aha_volumes, start=1):
        print(f"  AHA segment {seg:2d}: {vol:.6e}")

    # ------------------------------------------------------------
    # RV 8-segment segmentation
    # ------------------------------------------------------------
    if isBiV:
        RV_ID = 2

        rv_segments = tag_RV_8_segments(mesh,Mesh_obj.matid,rv_id=RV_ID,flip_order=False)
        dx_rv = df.Measure("dx", domain=mesh, subdomain_data=rv_segments)

        n_rv_seg = 8

        rv_volumes = [df.assemble(df.Constant(1.0) * dx_rv(seg)) for seg in range(1, n_rv_seg + 1)]

        print("RV 8-segment volumes:")
        for seg, vol in enumerate(rv_volumes, start=1):
            print(f"  RV segment {seg:2d}: {vol:.6e}")

    else:
        rv_segments = None
        dx_rv = None
        n_rv_seg = 0
        rv_volumes = []
    # ------------------------------------------------------------
    # Mechanics displacement space
    # ------------------------------------------------------------

    if SimDet["Mechanics Discretization"] is "P1P1":
        var_deg = 1
    else:
        var_deg = 2
    udisp = df.Function(df.VectorFunctionSpace(Mesh_obj.mesh, "CG", var_deg))

    # ------------------------------------------------------------
    # Set EDV frame as strain reference
    # ------------------------------------------------------------
    udisp.vector()[:] = u_arr[ref_idx].vector().get_local()[:]
    udisp.vector().apply("insert")    

    params = { "mesh": Mesh_obj.mesh,"displacement_variable": udisp,"material model": SimDet["GiccioneParams"]["Passive model"],"material params": SimDet["GiccioneParams"]["Passive params"],"incompressible": SimDet["GiccioneParams"]["incompressible"],"growth_tensor": None,}
    uflforms = Forms(params)
    Fref = df.project(uflforms.Fmat(), df.TensorFunctionSpace(Mesh_obj.mesh, "DG", 0))
    print(f"Using displacement frame {ref_idx} as EDV strain reference.")

    # === Storage ===
    Ecc_AHA_arr = [[] for _ in range(n_aha_seg)]
    Ell_AHA_arr = [[] for _ in range(n_aha_seg)]
    Err_AHA_arr = [[] for _ in range(n_aha_seg)]

    if isBiV:
        Ecc_RV8_arr = [[] for _ in range(n_rv_seg)]
        Ell_RV8_arr = [[] for _ in range(n_rv_seg)]
        Err_RV8_arr = [[] for _ in range(n_rv_seg)]
    else:
        Ecc_RV8_arr = None
        Ell_RV8_arr = None
        Err_RV8_arr = None

    # ------------------------------------------------------------
    # Loop over time
    # ------------------------------------------------------------
    for tidx, u_t in enumerate(u_arr):
        print(f"Processing regional strain time step {tidx + 1}/{len(u_arr)}")

        udisp.vector()[:] = u_t.vector().get_local()[:]
        udisp.vector().apply("insert")

        Fmat = uflforms.Fmat()

        # Relative deformation from EDV reference
        F = Fmat * df.inv(Fref)
        Cmat = F.T * F

        Ccc = df.inner(eC0_normalized, Cmat * eC0_normalized)
        Cll = df.inner(eL0_normalized, Cmat * eL0_normalized)
        Crr = df.inner(eR0_normalized, Cmat * eR0_normalized)

        Ecc = 0.5 * (1.0 - 1.0 / Ccc)
        Ell = 0.5 * (1.0 - 1.0 / Cll)
        Err = 0.5 * (1.0 - 1.0 / Crr)

        # --------------------------------------------------------
        # LV + septum AHA regional average
        # --------------------------------------------------------
        for seg in range(1, n_aha_seg + 1):
            vol = aha_volumes[seg - 1]

            if vol > 0.0:
                Ecc_AHA_arr[seg - 1].append(
                    df.assemble(Ecc * dx_aha(seg)) / vol
                )
                Ell_AHA_arr[seg - 1].append(
                    df.assemble(Ell * dx_aha(seg)) / vol
                )
                Err_AHA_arr[seg - 1].append(
                    df.assemble(Err * dx_aha(seg)) / vol
                )
            else:
                Ecc_AHA_arr[seg - 1].append(np.nan)
                Ell_AHA_arr[seg - 1].append(np.nan)
                Err_AHA_arr[seg - 1].append(np.nan)

        # --------------------------------------------------------
        # RV 8-segment regional average
        # --------------------------------------------------------
        if isBiV:
            for rv_seg in range(1, n_rv_seg + 1):
                vol = rv_volumes[rv_seg - 1]

                if vol > 0.0:
                    Ecc_RV8_arr[rv_seg - 1].append(
                        df.assemble(Ecc * dx_rv(rv_seg)) / vol
                    )
                    Ell_RV8_arr[rv_seg - 1].append(
                        df.assemble(Ell * dx_rv(rv_seg)) / vol
                    )
                    Err_RV8_arr[rv_seg - 1].append(
                        df.assemble(Err * dx_rv(rv_seg)) / vol
                    )
                else:
                    Ecc_RV8_arr[rv_seg - 1].append(np.nan)
                    Ell_RV8_arr[rv_seg - 1].append(np.nan)
                    Err_RV8_arr[rv_seg - 1].append(np.nan)

    # ------------------------------------------------------------
    # Quick EDV-zero check
    # ------------------------------------------------------------
    print("EDV-zero check at reference frame:")
    for seg in range(1, n_aha_seg + 1):
        ecc_ref = Ecc_AHA_arr[seg - 1][ref_idx]
        ell_ref = Ell_AHA_arr[seg - 1][ref_idx]
        err_ref = Err_AHA_arr[seg - 1][ref_idx]
        print(
            f"  AHA {seg:2d}: "
            f"Ecc={ecc_ref:.3e}, Ell={ell_ref:.3e}, Err={err_ref:.3e}"
        )

    if isBiV:
        for seg in range(1, n_rv_seg + 1):
            ecc_ref = Ecc_RV8_arr[seg - 1][ref_idx]
            ell_ref = Ell_RV8_arr[seg - 1][ref_idx]
            err_ref = Err_RV8_arr[seg - 1][ref_idx]
            print(
                f"  RV {seg:2d}: "
                f"Ecc={ecc_ref:.3e}, Ell={ell_ref:.3e}, Err={err_ref:.3e}"
            )    

    # ------------------------------------------------------------
    # Save output
    # ------------------------------------------------------------
    outdir = os.path.join(IODet["outputfolder"], IODet["caseID"])
    os.makedirs(outdir, exist_ok=True)

    np.savez(os.path.join(outdir, "Ecc_AHA.npz"), *Ecc_AHA_arr)
    np.savez(os.path.join(outdir, "Ell_AHA.npz"), *Ell_AHA_arr)
    np.savez(os.path.join(outdir, "Err_AHA.npz"), *Err_AHA_arr)

    if isBiV:
        np.savez(os.path.join(outdir, "Ecc_RV8.npz"), *Ecc_RV8_arr)
        np.savez(os.path.join(outdir, "Ell_RV8.npz"), *Ell_RV8_arr)
        np.savez(os.path.join(outdir, "Err_RV8.npz"), *Err_RV8_arr)

    # ------------------------------------------------------------
    # Save segment fields for ParaView check
    # ------------------------------------------------------------
    V0 = df.FunctionSpace(mesh, "DG", 0)

    aha_fun = df.Function(V0)
    aha_values = np.zeros(V0.dim())

    for cell in df.cells(mesh):
        aha_values[cell.index()] = aha_segments[cell.index()]

    aha_fun.vector().set_local(aha_values)
    aha_fun.vector().apply("insert")

    aha_path = os.path.join(outdir, "AHA_segments_used_for_strain.pvd")
    df.File(aha_path) << aha_fun

    print(f"Saved AHA segment check file: {aha_path}")

    if isBiV:
        rv_fun = df.Function(V0)
        rv_values = np.zeros(V0.dim())

        for cell in df.cells(mesh):
            rv_values[cell.index()] = rv_segments[cell.index()]

        rv_fun.vector().set_local(rv_values)
        rv_fun.vector().apply("insert")

        rv_path = os.path.join(outdir, "RV8_segments_used_for_strain.pvd")
        df.File(rv_path) << rv_fun

        print(f"Saved RV segment check file: {rv_path}")

    print(f"Saved: {os.path.join(outdir, 'Ecc_AHA.npz')}")
    print(f"Saved: {os.path.join(outdir, 'Ell_AHA.npz')}")
    print(f"Saved: {os.path.join(outdir, 'Err_AHA.npz')}")

    if isBiV:
        print(f"Saved: {os.path.join(outdir, 'Ecc_RV8.npz')}")
        print(f"Saved: {os.path.join(outdir, 'Ell_RV8.npz')}")
        print(f"Saved: {os.path.join(outdir, 'Err_RV8.npz')}")


def tag_AHA_segments(mesh, apex_axis="z", R_apex=1.0, reference_point=None, rotate=True, plot_debug=False):
    mesh.init()
    cell_tags = df.MeshFunction("size_t", mesh, mesh.topology().dim())
    cell_tags.set_all(0)

    coords = mesh.coordinates()
    z_vals = coords[:, 2]
    z_min, z_max = z_vals.min(), z_vals.max()
    h = z_max - z_min
    basal_th = z_max - h / 3
    mid_th = z_max - 2 * h / 3

    # Compute theta_shift to align segment 2 with HIS
    theta_shift = 0.0
    if reference_point and rotate:
        his = np.array(reference_point).flatten()
        theta_his = np.degrees(np.arctan2(his[1], his[0])) % 360
        # theta_shift = (theta_his - 330) % 360  # Align end of segment 2 with HIS
        theta_shift = (180 - theta_his) % 360  # Align end of segment 2 with HIS

    for cell in df.cells(mesh):
        mp = cell.midpoint()
        x, y, z = mp.x(), mp.y(), mp.z()

        theta = np.degrees(np.arctan2(y, x)) % 360
        theta_rotated = (theta + theta_shift) % 360

        if z >= basal_th:
            level = "basal"
            tag = int((theta_rotated - 60) % 360 // 60) + 1
        elif z >= mid_th:
            level = "mid"
            tag = int((theta_rotated - 60) % 360 // 60) + 7
        else:
            level = "apical"
            tag = int((theta_rotated - 45) % 360 // 90) + 13

        if np.linalg.norm([x, y, z - z_min]) < R_apex:
            tag = 17

        cell_tags[cell.index()] = tag

    return cell_tags


# def tag_RV_segments(mesh, matid, R_apex=3.0):
#     """
#     Simple RV segmentation for a BiV mesh.

#     Tags:
#         0 = not RV
#         1 = basal RV
#         2 = mid RV
#         3 = apical RV
#     """

#     mesh.init()

#     rv_tags = df.MeshFunction("size_t", mesh, mesh.topology().dim())
#     rv_tags.set_all(0)

#     RV_ID = 2

#     # Collect RV cell midpoint z-values
#     rv_z = []

#     for cell in df.cells(mesh):
#         cidx = cell.index()

#         if int(matid[cidx]) == RV_ID:
#             mp = cell.midpoint()
#             rv_z.append(mp.z())

#     rv_z = np.array(rv_z)

#     if len(rv_z) == 0:
#         print("Warning: no RV cells found.")
#         return rv_tags

#     z_min = rv_z.min()
#     z_max = rv_z.max()
#     h = z_max - z_min

#     basal_th = z_max - h / 3.0
#     mid_th = z_max - 2.0 * h / 3.0

#     rv_counts = np.zeros(3, dtype=int)

#     for cell in df.cells(mesh):
#         cidx = cell.index()

#         if int(matid[cidx]) != RV_ID:
#             rv_tags[cidx] = 0
#             continue

#         mp = cell.midpoint()
#         z = mp.z()

#         if z >= basal_th:
#             tag = 1   # basal RV
#         elif z >= mid_th:
#             tag = 2   # mid RV
#         else:
#             tag = 3   # apical RV

#         rv_tags[cidx] = tag
#         rv_counts[tag - 1] += 1

#     print("RV segment counts:")
#     print(f"  RV basal : {rv_counts[0]}")
#     print(f"  RV mid   : {rv_counts[1]}")
#     print(f"  RV apical: {rv_counts[2]}")

#     return rv_tags   

def tag_RV_8_segments(mesh, matid, rv_id=2, apex_axis="z",
                      center_xy=None, flip_order=False):
    """
    Tag RV according to an 8-segment RV scheme.

    RV tags:
        0 = not RV

        Basal:
        1 = inferior
        2 = lateral
        3 = anterior

        Mid:
        4 = inferior
        5 = lateral
        6 = anterior

        Apical:
        7 = inferior
        8 = anterior

    Parameters
    ----------
    mesh : dolfin.Mesh
    matid : dolfin.MeshFunction
        Cell-wise material ID. RV cells must have value rv_id.
    rv_id : int
        Material ID for RV.
    center_xy : tuple or None
        Angular center. If None, uses mesh bounding-box center.
    flip_order : bool
        If anterior/inferior appear swapped in ParaView, set this to True.
    """

    mesh.init()

    rv_tags = df.MeshFunction("size_t", mesh, mesh.topology().dim())
    rv_tags.set_all(0)

    coords = mesh.coordinates()

    if center_xy is None:
        cx = 0.5 * (coords[:, 0].min() + coords[:, 0].max())
        cy = 0.5 * (coords[:, 1].min() + coords[:, 1].max())
    else:
        cx, cy = center_xy

    # ------------------------------------------------------------
    # Collect RV cell midpoint coordinates
    # ------------------------------------------------------------
    rv_cells = []
    rv_z = []
    rv_theta = []

    for cell in df.cells(mesh):
        cidx = cell.index()

        if int(matid[cidx]) != rv_id:
            continue

        mp = cell.midpoint()
        x, y, z = mp.x(), mp.y(), mp.z()

        theta = np.degrees(np.arctan2(y - cy, x - cx)) % 360.0

        rv_cells.append(cidx)
        rv_z.append(z)
        rv_theta.append(theta)

    rv_z = np.array(rv_z)
    rv_theta = np.array(rv_theta)

    if len(rv_cells) == 0:
        print("Warning: no RV cells found.")
        return rv_tags

    # ------------------------------------------------------------
    # Basal / mid / apical thresholds based only on RV cells
    # ------------------------------------------------------------
    z_min = rv_z.min()
    z_max = rv_z.max()
    h = z_max - z_min

    basal_th = z_max - h / 3.0
    mid_th = z_max - 2.0 * h / 3.0

    # ------------------------------------------------------------
    # Unwrap RV angles
    # This avoids problems if the RV angular span crosses 0 degrees.
    # ------------------------------------------------------------
    theta_rad = np.deg2rad(rv_theta)
    mean_angle = np.arctan2(np.mean(np.sin(theta_rad)),
                            np.mean(np.cos(theta_rad)))
    mean_angle_deg = np.degrees(mean_angle) % 360.0

    def unwrap_relative(theta_deg):
        """
        Return angle relative to RV angular center in range [-180, 180).
        """
        return ((theta_deg - mean_angle_deg + 180.0) % 360.0) - 180.0

    rv_theta_rel = unwrap_relative(rv_theta)

    theta_min = rv_theta_rel.min()
    theta_max = rv_theta_rel.max()
    theta_span = theta_max - theta_min

    basal_mid_1 = theta_min + theta_span / 3.0
    basal_mid_2 = theta_min + 2.0 * theta_span / 3.0

    apical_split = theta_min + theta_span / 2.0

    counts = np.zeros(8, dtype=int)

    # ------------------------------------------------------------
    # Assign RV segment tags
    # ------------------------------------------------------------
    for cell in df.cells(mesh):
        cidx = cell.index()

        if int(matid[cidx]) != rv_id:
            rv_tags[cidx] = 0
            continue

        mp = cell.midpoint()
        x, y, z = mp.x(), mp.y(), mp.z()

        theta = np.degrees(np.arctan2(y - cy, x - cx)) % 360.0
        theta_rel = unwrap_relative(theta)

        # Optional orientation flip
        if flip_order:
            theta_rel = theta_max - (theta_rel - theta_min)

        if z >= basal_th:
            # Basal RV: 1, 2, 3
            if theta_rel < basal_mid_1:
                tag = 1   # basal inferior
            elif theta_rel < basal_mid_2:
                tag = 2   # basal lateral
            else:
                tag = 3   # basal anterior

        elif z >= mid_th:
            # Mid RV: 4, 5, 6
            if theta_rel < basal_mid_1:
                tag = 4   # mid inferior
            elif theta_rel < basal_mid_2:
                tag = 5   # mid lateral
            else:
                tag = 6   # mid anterior

        else:
            # Apical RV: 7, 8
            if theta_rel < apical_split:
                tag = 7   # apical inferior
            else:
                tag = 8   # apical anterior

        rv_tags[cidx] = tag
        counts[tag - 1] += 1

    print("RV 8-segment counts:")
    print(f"  1 basal inferior : {counts[0]}")
    print(f"  2 basal lateral  : {counts[1]}")
    print(f"  3 basal anterior : {counts[2]}")
    print(f"  4 mid inferior   : {counts[3]}")
    print(f"  5 mid lateral    : {counts[4]}")
    print(f"  6 mid anterior   : {counts[5]}")
    print(f"  7 apical inferior: {counts[6]}")
    print(f"  8 apical anterior: {counts[7]}")

    print("RV angular debug:")
    print(f"  mean RV angle = {mean_angle_deg:.2f} deg")
    print(f"  theta_min_rel = {theta_min:.2f} deg")
    print(f"  theta_max_rel = {theta_max:.2f} deg")

    return rv_tags


def bullseye_plot(ax, data, seg_bold=None, cmap="Greys", norm=None):
    data = np.ravel(data)
    data = np.ma.masked_array(data, mask=[False]*16 + [True])

    cmap.set_bad(color='black')

    if seg_bold is None:
        seg_bold = []
    if norm is None:
        norm = Normalize(vmin=data.min(), vmax=data.max())

    r = np.linspace(0.2, 1, 4)
    ax.set(ylim=[0, 1], xticklabels=[], yticklabels=[])
    ax.grid(False)

    for start, stop, r_in, r_out in [
        (0, 6, r[2], r[3]),
        (6, 12, r[1], r[2]),
        (12, 16, r[0], r[1]),
        (16, 17, 0, r[0]),
    ]:
        n = stop - start
        dtheta = 2*np.pi / n
        ax.bar(np.arange(n) * dtheta + np.pi/2, r_out - r_in, dtheta, r_in,
               color=cmap(norm(data[start:stop])))

    for start, stop, r_in, r_out in [
        (0, 6, r[2], r[3]),
        (6, 12, r[1], r[2]),
        (12, 16, r[0], r[1]),
    ]:
        n = stop - start
        dtheta = 2*np.pi / n
        ax.bar(np.arange(n) * dtheta + np.pi/2, r_out - r_in, dtheta, r_in,
               clip_on=False, color="none", edgecolor="black", linewidth=[
                   4 if i + 1 in seg_bold else 1 for i in range(start, stop)])
    ax.plot(np.linspace(0, 2*np.pi), np.linspace(r[0], r[0]), "black",
            linewidth=(4 if 17 in seg_bold else 1))


def check_AHA_plot(IODet, SimDet, target_segment=1, R_apex=3.0,include_septum=True):
    # """
    # Highlight segments 1 and 2 in bullseye plot and in a VTK file.
    # """
    # # === Load mesh ===
    if  SimDet["isLV"]:
        mesh_params = {"directory": IODet["directory_me"],"casename": IODet["casename_me"],"outputfolder": IODet["outputfolder"],"foldername": IODet["folderName"], "isLV": SimDet.get("isLV", True),}
        Mesh_obj = lv_mechanics_mesh(mesh_params, SimDet)
        mesh = Mesh_obj.mesh
        # === Save bullseye plot with segments 1 and 2 highlighted ===
        outdir = os.path.join(IODet["outputfolder"], IODet["caseID"])
        os.makedirs(outdir, exist_ok=True)

        # === Tag AHA segments ===
        aha_segments = tag_AHA_segments(mesh, R_apex=R_apex, reference_point=SimDet["HIS"], rotate=True, plot_debug=True)

        # === Segment counts ===
        num_cells = mesh.num_cells()
        segment_counts = np.zeros(17, dtype=int)
        for cell in df.cells(mesh):
            tag = aha_segments[cell.index()]
            if 1 <= tag <= 17:
                segment_counts[tag - 1] += 1

        # binary_data = np.zeros(17)
        # binary_data[0] = 1.0  # segment 1
        # binary_data[1] = 1.0  # segment 2
        # fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
        # bullseye_plot(ax, data=binary_data, seg_bold=[1, 2],
        #               cmap=cm.get_cmap("Greys"), norm=mpl.colors.Normalize(vmin=0, vmax=1))
        # plt.savefig(os.path.join(outdir, "AHA_bullseye_seg1and2.png"))
        # plt.close()

        # === Create VTK field for segments 1 and 2 ===
        V0 = df.FunctionSpace(mesh, "DG", 0)
        mask = df.Function(V0)
        # mask_values = np.zeros(V0.dim())
        # for cell in df.cells(mesh):
        #     tag = aha_segments[cell.index()]
        #     if tag == 1:
        #         mask_values[cell.index()] = 1.0
        #     elif tag == 2:
        #         mask_values[cell.index()] = 2.0
        # mask.vector().set_local(mask_values)
        # mask.vector().apply("insert")

        # vtkpath = os.path.join(outdir, "AHA_segments_1_and_2_mask.pvd")
        # df.File(vtkpath) << mask
        mask_values = mask.vector().get_local()

        for cell in df.cells(mesh):
            tag = aha_segments[cell.index()]
            mask_values[cell.index()] = tag  # Integer tag directly

        mask.vector().set_local(mask_values)
        mask.vector().apply("insert")

        # Save all segment tags
        vtkpath = os.path.join(outdir, "AHA_segments_all_tags.pvd")
        df.File(vtkpath) << mask

        print(f"✅ All AHA segment tags written to: {vtkpath}")    

    else:
        print("bla")
        # === Load mesh ===
        mesh_params = {
            "directory": IODet["directory_me"],
            "casename": IODet["casename_me"],
            "outputfolder": IODet["outputfolder"],
            "foldername": IODet["folderName"],
            "isLV": SimDet.get("isLV", False),  # BiV case
        }

        Mesh_obj = lv_mechanics_mesh(mesh_params, SimDet)
        mesh = Mesh_obj.mesh

        outdir = os.path.join(IODet["outputfolder"], IODet["caseID"])
        os.makedirs(outdir, exist_ok=True)

        # ------------------------------------------------------------
        # Get material IDs
        # ------------------------------------------------------------
        # You need to adjust this name depending on your mesh object.
        # Common possibilities:
        #   Mesh_obj.cfun
        #   Mesh_obj.matid
        #   Mesh_obj.meshfunction
        #   Mesh_obj.cell_markers
        #
        # For example:
        matid = Mesh_obj.matid

        LV_ID = 0
        SEPTUM_ID = 1
        RV_ID = 2
        ANNULUS_ID = 3

        allowed_ids = [LV_ID]
        if include_septum:
            allowed_ids.append(SEPTUM_ID)

        # ------------------------------------------------------------
        # Compute AHA segments geometrically
        # ------------------------------------------------------------
        # Important: this function probably still computes tags for all cells.
        # We will filter afterward using matid.
        aha_segments = tag_AHA_segments(mesh,R_apex=R_apex,reference_point=SimDet["HIS"],rotate=True,plot_debug=True)

        # ------------------------------------------------------------
        # Count only LV/septum cells
        # ------------------------------------------------------------
        segment_counts = np.zeros(17, dtype=int)

        for cell in df.cells(mesh):
            cidx = cell.index()
            m = matid[cidx]
            tag = aha_segments[cidx]

            if m in allowed_ids and 1 <= tag <= 17:
                segment_counts[tag - 1] += 1

        print("AHA segment counts for LV/septum only:")
        for i, count in enumerate(segment_counts, start=1):
            print(f"  Segment {i:2d}: {count}")

        # ------------------------------------------------------------
        # Create DG0 field
        # ------------------------------------------------------------
        V0 = df.FunctionSpace(mesh, "DG", 0)
        mask = df.Function(V0)
        mask_values = np.zeros(V0.dim())

        for cell in df.cells(mesh):
            cidx = cell.index()
            m = matid[cidx]
            tag = aha_segments[cidx]

            if m in allowed_ids and 1 <= tag <= 17:
                mask_values[cidx] = tag
            else:
                mask_values[cidx] = 0.0

        mask.vector().set_local(mask_values)
        mask.vector().apply("insert")

        vtkpath = os.path.join(outdir, "AHA_segments_all_tags_BiV.pvd")
        df.File(vtkpath) << mask

        print(f"✅ BiV AHA segment tags written to: {vtkpath}")    

        # rv_segments = tag_RV_segments(mesh, matid)
        rv_segments = tag_RV_8_segments(mesh,matid,rv_id=2,flip_order=False)    


        V0 = df.FunctionSpace(mesh, "DG", 0)
        rv_fun = df.Function(V0)
        rv_values = np.zeros(V0.dim())

        for cell in df.cells(mesh):
            rv_values[cell.index()] = rv_segments[cell.index()]

        rv_fun.vector().set_local(rv_values)
        rv_fun.vector().apply("insert")

        rv_vtkpath = os.path.join(outdir, "RV_segments_basal_mid_apical.pvd")
        df.File(rv_vtkpath) << rv_fun

        print(f"✅ RV segment tags written to: {rv_vtkpath}")




def plot_min_bullseye_per_cycle(IODet, SimDet):
    """
    Plots the minimum strain per AHA segment for each cycle using bullseye plots.

    Parameters
    ----------
    IODet : dict
        Dictionary containing I/O details like 'caseID', 'outputfolder'.
    SimDet : dict
        Dictionary containing simulation details.
    """
    outdir = os.path.join(IODet["outputfolder"], IODet["caseID"])
    ecc_path = os.path.join(outdir, "Ecc_AHA.npz")
    ell_path = os.path.join(outdir, "Ell_AHA.npz")    
    ncycle = SimDet["closedloopparam"]["stop_iter"] + 1
    BCL = SimDet["HeartBeatLength"]    
    pv_file = os.path.join(IODet["outputfolder"]+IODet["caseID"]+"/BiV_PV.txt")     
    t_systole= np.zeros(ncycle)   
   
    for cycle in range(ncycle):
        homo_tptt1, homo_LVP1, homo_LVV1, homo_RVP1, homo_RVV1, homo_Qmv1 = extract_PV(pv_file, BCL, cycle, SimDet)
        t_systole[cycle] = homo_tptt1[np.argmin(homo_LVV1)]

    if not os.path.exists(ecc_path):
        raise FileNotFoundError(f"Ecc file not found: {ecc_path}")

    if not os.path.exists(ell_path):
        raise FileNotFoundError(f"Ell file not found: {ell_path}")    

    # Load data
    data_ecc = np.load(ecc_path)
    n_segments_ecc = len(data_ecc.files)
    segment_strains_ecc = [data_ecc[f'arr_{i}'] for i in range(n_segments_ecc)]
    strain_matrix_ecc = np.vstack(segment_strains_ecc)  # shape: (17, N)
    n_timesteps_ecc = strain_matrix_ecc.shape[1]

    data_ell = np.load(ell_path)
    n_segments_ell = len(data_ell.files)
    segment_strains_ell = [data_ell[f'arr_{i}'] for i in range(n_segments_ell)]
    strain_matrix_ell = np.vstack(segment_strains_ell)  # shape: (17, N)
    n_timesteps_ell = strain_matrix_ell.shape[1]

    if n_timesteps_ecc % ncycle != 0:
        raise ValueError("Strain data does not contain correct number of time steps.")

    frames_per_cycle = n_timesteps_ecc // ncycle

    sdi_ecc_all = []
    sdi_ell_all = []   

    for cycle in range(ncycle):
        start = cycle * frames_per_cycle
        end = (cycle + 1) * frames_per_cycle

        min_vals_ecc = np.min(strain_matrix_ecc[:, start:end], axis=1)
        min_vals_ell = np.min(strain_matrix_ell[:, start:end], axis=1)

         # --- Save strain minima and time indices ---
        min_ecc_times = np.argmin(strain_matrix_ecc[:, start:end], axis=1) + start
        min_ell_times = np.argmin(strain_matrix_ell[:, start:end], axis=1) + start

        # convert_frame_index_to_time(min_ecc_times)
        min_ecc_time_ms = convert_frame_index_to_time(min_ecc_times, SimDet["dt"], SimDet["writeStep"], SimDet["dt"])
        min_ell_time_ms = convert_frame_index_to_time(min_ell_times, SimDet["dt"], SimDet["writeStep"], SimDet["dt"])

        valid_min_ecc_times_ms = min_ecc_time_ms[:16]  # segments 1–16 only
        valid_min_ell_times_ms = min_ell_time_ms[:16]  # segments 1–16 only

        print(min_ecc_time_ms)
        print(min_ell_time_ms)
        print(t_systole[cycle])
        print(valid_min_ecc_times_ms - (t_systole[cycle]))
        print(valid_min_ell_times_ms - (t_systole[cycle]))        

        # if cycle == 0:
        SDI_Ecc = np.std(valid_min_ecc_times_ms - (t_systole[cycle])) / BCL * 100
        SDI_Ell = np.std(valid_min_ell_times_ms - (t_systole[cycle])) / BCL * 100
        #else:
            #pdb.set_trace()
            #SDI_Ecc = np.std(valid_min_ecc_times_ms - (t_systole[cycle]-(cycle-1)*BCL)) / BCL * 100
            #SDI_Ell = np.std(valid_min_ell_times_ms - (t_systole[cycle]-(cycle-1)*BCL)) / BCL * 100    

        sdi_ecc_all.append(SDI_Ecc)
        sdi_ell_all.append(SDI_Ell)

        print(f"Ecc SDI: {SDI_Ecc:.4f}")
        print(f"Ell SDI: {SDI_Ell:.4f}")

        # --- Bullseye of time delta between min Ecc and t_systole ---
        delta_time_ecc = np.full(17, np.nan)
        delta_time_ecc[:16] = valid_min_ecc_times_ms - (t_systole[cycle] if cycle == 0 else t_systole[cycle] - (cycle - 1) * BCL)
        # delta_time_ecc[16] = np.nan  # Segment 17 (center) white

        # Normalize color scale to this delta
        vmin_dt, vmax_dt = np.nanmin(delta_time_ecc[:16]), np.nanmax(delta_time_ecc[:16])
        norm_dt = mpl.colors.Normalize(vmin=vmin_dt, vmax=vmax_dt)
        cmap_dt = cm.coolwarm

        fig_dt, ax_dt = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
        bullseye_plot(ax_dt, data=delta_time_ecc, seg_bold=[], cmap=cmap_dt, norm=norm_dt)

        sm_dt = mpl.cm.ScalarMappable(cmap=cmap_dt, norm=norm_dt)
        sm_dt.set_array([])
        cbar_dt = plt.colorbar(sm_dt, ax=ax_dt, orientation='horizontal', pad=0.05, shrink=0.6)
        cbar_dt.set_label("ΔT: Min Ecc vs Systole [ms]", fontsize=12)
        ax_dt.set_title(f"ΔT (Ecc min - systole) - Cycle {cycle + 1}", fontsize=14)

        delta_fig_path = os.path.join(outdir, f"delta_minEcc_from_systole_cycle{cycle+1}.png")
        plt.savefig(delta_fig_path, dpi=300, bbox_inches='tight')
        plt.close(fig_dt)
        print(f"Saved: {delta_fig_path}")

        # --- Bullseye of time delta between min Ell and t_systole ---
        delta_time_ell = np.full(17, np.nan)
        delta_time_ell[:16] = valid_min_ell_times_ms - (t_systole[cycle] if cycle == 0 else t_systole[cycle] - (cycle - 1) * BCL)
        # delta_time_ell[16] = np.nan  # Segment 17 (center) white
        
        # Normalize color scale to this delta
        vmin_dt, vmax_dt = np.nanmin(delta_time_ell[:16]), np.nanmax(delta_time_ell[:16])
        norm_dt = mpl.colors.Normalize(vmin=vmin_dt, vmax=vmax_dt)
        cmap_dt = cm.coolwarm

        fig_dt, ax_dt = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
        bullseye_plot(ax_dt, data=delta_time_ell, seg_bold=[], cmap=cmap_dt, norm=norm_dt)

        sm_dt = mpl.cm.ScalarMappable(cmap=cmap_dt, norm=norm_dt)
        sm_dt.set_array([])
        cbar_dt = plt.colorbar(sm_dt, ax=ax_dt, orientation='horizontal', pad=0.05, shrink=0.6)
        cbar_dt.set_label("ΔT: Min Ell vs Systole [ms]", fontsize=12)
        ax_dt.set_title(f"ΔT (Ell min - systole) - Cycle {cycle + 1}", fontsize=14)

        delta_fig_path = os.path.join(outdir, f"delta_minEll_from_systole_cycle{cycle+1}.png")
        plt.savefig(delta_fig_path, dpi=300, bbox_inches='tight')
        plt.close(fig_dt)
        print(f"Saved: {delta_fig_path}")

        # pdb.set_trace()
        # --- Full bullseye with strain values ---
        vmin_ecc, vmax_ecc = np.nanmin(min_vals_ecc), np.nanmax(min_vals_ecc)
        norm = mpl.colors.Normalize(vmin=vmin_ecc, vmax=vmax_ecc)
        cmap = cm.coolwarm

        fig1, ax1 = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
        bullseye_plot(ax1, data=min_vals_ecc, seg_bold=[], cmap=cmap, norm=norm)

        sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax1, orientation='horizontal', pad=0.05, shrink=0.6)
        cbar.set_label("Minimum circumferential Strain", fontsize=12)
        ax1.set_title(f"Min Ecc Strain - Cycle {cycle + 1}", fontsize=14)

        fullpath = os.path.join(outdir, f"min_Ecc_bullseye_cycle{cycle+1}.png")
        plt.savefig(fullpath, dpi=300, bbox_inches='tight')
        plt.close(fig1)
        print(f"Saved: {fullpath}")

        # --- Full bullseye with strain values ---
        vmin_ell, vmax_ell = np.nanmin(min_vals_ell), np.nanmax(min_vals_ell)
        norm = mpl.colors.Normalize(vmin=vmin_ell, vmax=vmax_ell)
        cmap = cm.coolwarm

        fig2, ax2 = plt.subplots(figsize=(4, 4), subplot_kw=dict(polar=True))
        bullseye_plot(ax2, data=min_vals_ell, seg_bold=[], cmap=cmap, norm=norm)

        sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax2, orientation='horizontal', pad=0.05, shrink=0.4)
        cbar.set_label("Minimum longitudinal Strain", fontsize=12)
        ax2.set_title(f"Min Ell Strain - Cycle {cycle + 1}", fontsize=14)

        fullpath = os.path.join(outdir, f"min_Ell_bullseye_cycle{cycle+1}.png")
        plt.savefig(fullpath, dpi=300, bbox_inches='tight')
        plt.close(fig2)
        print(f"Saved: {fullpath}")

    # After the loop
    with open(os.path.join(outdir, "SDI_summary.txt"), "w") as f:
        f.write("Cycle\tSDI_Ecc (%)\tSDI_Ell (%)\n")
        for i, (ecc_val, ell_val) in enumerate(zip(sdi_ecc_all, sdi_ell_all), 1):
            f.write(f"{i}\t{ecc_val:.4f}\t{ell_val:.4f}\n")


def convert_frame_index_to_time(min_Ecc_time_array, dt, step_interval, t0):
    """
    Converts an array of frame indices to time in ms.
    """
    return np.array([
         (t0 + dt *step_interval * idx) if idx >= 0 else -1
        for idx in min_Ecc_time_array
    ])     

def strain_plot_AHA(IODet,SimDet,mode='all'):
    #directory
    outdir = os.path.join(IODet["outputfolder"], IODet["caseID"])

    data_ecc = np.load(outdir+"/Ecc_AHA_lastcycle.npz")  
    data_ell = np.load(outdir+"/Ell_AHA_lastcycle.npz")     

    ###### data
    ecc_array1 = data_ecc['arr_0'].flatten()
    ecc_array2 = data_ecc['arr_1'].flatten()
    ecc_array3 = data_ecc['arr_2'].flatten()
    ecc_array4 = data_ecc['arr_3'].flatten()
    ecc_array5 = data_ecc['arr_4'].flatten()
    ecc_array6 = data_ecc['arr_5'].flatten()
    ecc_array7 = data_ecc['arr_6'].flatten()
    ecc_array8 = data_ecc['arr_7'].flatten()
    ecc_array9 = data_ecc['arr_8'].flatten()
    ecc_array10 = data_ecc['arr_9'].flatten()
    ecc_array11 = data_ecc['arr_10'].flatten()
    ecc_array12 = data_ecc['arr_11'].flatten()
    ecc_array13 = data_ecc['arr_12'].flatten()
    ecc_array14 = data_ecc['arr_13'].flatten()
    ecc_array15 = data_ecc['arr_14'].flatten()
    ecc_array16 = data_ecc['arr_15'].flatten()
    ecc_array17 = data_ecc['arr_16'].flatten()

    ell_array1 = data_ell['arr_0'].flatten()
    ell_array2 = data_ell['arr_1'].flatten()
    ell_array3 = data_ell['arr_2'].flatten()
    ell_array4 = data_ell['arr_3'].flatten()
    ell_array5 = data_ell['arr_4'].flatten()
    ell_array6 = data_ell['arr_5'].flatten()
    ell_array7 = data_ell['arr_6'].flatten()
    ell_array8 = data_ell['arr_7'].flatten()
    ell_array9 = data_ell['arr_8'].flatten()
    ell_array10 = data_ell['arr_9'].flatten()
    ell_array11 = data_ell['arr_10'].flatten()
    ell_array12 = data_ell['arr_11'].flatten()
    ell_array13 = data_ell['arr_12'].flatten()
    ell_array14 = data_ell['arr_13'].flatten()
    ell_array15 = data_ell['arr_14'].flatten()
    ell_array16 = data_ell['arr_15'].flatten()
    ell_array17 = data_ell['arr_16'].flatten()


    if mode == 'all':
        plt.figure()
        plt.plot(np.arange(0, len(ecc_array1)), ecc_array1)
        plt.plot(np.arange(0, len(ecc_array2)), ecc_array2)
        plt.plot(np.arange(0, len(ecc_array3)), ecc_array3, linestyle="dotted")
        plt.plot(np.arange(0, len(ecc_array4)), ecc_array4, linestyle="dotted")
        plt.plot(np.arange(0, len(ecc_array5)), ecc_array5, linestyle="dotted")
        plt.plot(np.arange(0, len(ecc_array6)), ecc_array6)
        plt.plot(np.arange(0, len(ecc_array7)), ecc_array7)
        plt.plot(np.arange(0, len(ecc_array8)), ecc_array8)
        plt.plot(np.arange(0, len(ecc_array9)), ecc_array9, linestyle="dotted")
        plt.plot(np.arange(0, len(ecc_array10)), ecc_array10, linestyle="dotted")
        plt.plot(np.arange(0, len(ecc_array11)), ecc_array11, linestyle="dotted")
        plt.plot(np.arange(0, len(ecc_array12)), ecc_array12)
        plt.plot(np.arange(0, len(ecc_array13)), ecc_array13)
        plt.plot(np.arange(0, len(ecc_array14)), ecc_array14)
        plt.plot(np.arange(0, len(ecc_array15)), ecc_array15)
        plt.plot(np.arange(0, len(ecc_array16)), ecc_array16)
        plt.plot(np.arange(0, len(ecc_array17)), ecc_array17, linestyle="dotted")
        plt.xlabel("Time point", fontsize=14)
        plt.ylabel("Strain", fontsize=14)
        plt.savefig(os.path.join(outdir,"Ecc_aha.png"))
        plt.clf() 

        plt.figure()
        plt.plot(np.arange(0, len(ell_array1)), ell_array1)
        plt.plot(np.arange(0, len(ell_array2)), ell_array2)
        plt.plot(np.arange(0, len(ell_array3)), ell_array3, linestyle="dotted")
        plt.plot(np.arange(0, len(ell_array4)), ell_array4, linestyle="dotted")
        plt.plot(np.arange(0, len(ell_array5)), ell_array5, linestyle="dotted")
        plt.plot(np.arange(0, len(ell_array6)), ell_array6)
        plt.plot(np.arange(0, len(ell_array7)), ell_array7)
        plt.plot(np.arange(0, len(ell_array8)), ell_array8)
        plt.plot(np.arange(0, len(ell_array9)), ell_array9, linestyle="dotted")
        plt.plot(np.arange(0, len(ell_array10)), ell_array10, linestyle="dotted")
        plt.plot(np.arange(0, len(ell_array11)), ell_array11, linestyle="dotted")
        plt.plot(np.arange(0, len(ell_array12)), ell_array12)
        plt.plot(np.arange(0, len(ell_array13)), ell_array13)
        plt.plot(np.arange(0, len(ell_array14)), ell_array14)
        plt.plot(np.arange(0, len(ell_array15)), ell_array15)
        plt.plot(np.arange(0, len(ell_array16)), ell_array16)
        plt.plot(np.arange(0, len(ell_array17)), ell_array17, linestyle="dotted")
        plt.xlabel("Time point", fontsize=14)
        plt.ylabel("Strain", fontsize=14)
        plt.savefig(os.path.join(outdir,"Ell_aha.png"))
        plt.clf()         

    elif mode == 'split':  
        plt.figure()
        plt.plot(np.arange(0, len(ecc_array1)), ecc_array1)
        plt.plot(np.arange(0, len(ecc_array2)), ecc_array2)
        plt.plot(np.arange(0, len(ecc_array3)), ecc_array3, linestyle="dotted")
        plt.plot(np.arange(0, len(ecc_array4)), ecc_array4, linestyle="dotted")
        plt.plot(np.arange(0, len(ecc_array5)), ecc_array5, linestyle="dotted")
        plt.plot(np.arange(0, len(ecc_array6)), ecc_array6)
        plt.xlabel("Time point", fontsize=14)
        plt.ylabel("Strain", fontsize=14)
        plt.savefig(os.path.join(outdir,"Ecc_aha_base.png"))
        plt.clf() 

        plt.figure()
        plt.plot(np.arange(0, len(ell_array1)), ell_array1)
        plt.plot(np.arange(0, len(ell_array2)), ell_array2)
        plt.plot(np.arange(0, len(ell_array3)), ell_array3, linestyle="dotted")
        plt.plot(np.arange(0, len(ell_array4)), ell_array4, linestyle="dotted")
        plt.plot(np.arange(0, len(ell_array5)), ell_array5, linestyle="dotted")
        plt.plot(np.arange(0, len(ell_array6)), ell_array6)
        plt.xlabel("Time point", fontsize=14)
        plt.ylabel("Strain", fontsize=14)
        plt.savefig(os.path.join(outdir,"Ell_aha_base.png"))
        plt.clf()  

        plt.figure()
        plt.plot(np.arange(0, len(ecc_array7)), ecc_array7)
        plt.plot(np.arange(0, len(ecc_array8)), ecc_array8)
        plt.plot(np.arange(0, len(ecc_array9)), ecc_array9, linestyle="dotted")
        plt.plot(np.arange(0, len(ecc_array10)), ecc_array10, linestyle="dotted")
        plt.plot(np.arange(0, len(ecc_array11)), ecc_array11, linestyle="dotted")
        plt.plot(np.arange(0, len(ecc_array12)), ecc_array12)
        plt.xlabel("Time point", fontsize=14)
        plt.ylabel("Strain", fontsize=14)
        plt.savefig(os.path.join(outdir,"Ecc_aha_mid.png"))
        plt.clf() 

        plt.figure()
        plt.plot(np.arange(0, len(ell_array7)), ell_array7)
        plt.plot(np.arange(0, len(ell_array8)), ell_array8)
        plt.plot(np.arange(0, len(ell_array9)), ell_array9, linestyle="dotted")
        plt.plot(np.arange(0, len(ell_array10)), ell_array10, linestyle="dotted")
        plt.plot(np.arange(0, len(ell_array11)), ell_array11, linestyle="dotted")
        plt.plot(np.arange(0, len(ell_array12)), ell_array12) 
        plt.xlabel("Time point", fontsize=14)
        plt.ylabel("Strain", fontsize=14)
        plt.savefig(os.path.join(outdir,"Ell_aha_mid.png"))
        plt.clf()  

        plt.figure()
        plt.plot(np.arange(0, len(ecc_array13)), ecc_array13)
        plt.plot(np.arange(0, len(ecc_array14)), ecc_array14)
        plt.plot(np.arange(0, len(ecc_array15)), ecc_array15)
        plt.plot(np.arange(0, len(ecc_array16)), ecc_array16)
        plt.plot(np.arange(0, len(ecc_array17)), ecc_array17, linestyle="dotted")
        plt.xlabel("Time point", fontsize=14)
        plt.ylabel("Strain", fontsize=14)
        plt.savefig(os.path.join(outdir,"Ecc_aha_apex.png"))
        plt.clf() 

        plt.figure()
        plt.plot(np.arange(0, len(ell_array13)), ell_array13)
        plt.plot(np.arange(0, len(ell_array14)), ell_array14)
        plt.plot(np.arange(0, len(ell_array15)), ell_array15)
        plt.plot(np.arange(0, len(ell_array16)), ell_array16)
        plt.plot(np.arange(0, len(ell_array17)), ell_array17, linestyle="dotted")
        plt.xlabel("Time point", fontsize=14)
        plt.ylabel("Strain", fontsize=14)
        plt.savefig(os.path.join(outdir,"Ell_aha_apex.png"))
        plt.clf()               

def plot_min_bullseye_per_cycle_comparison(IODet, SimDet):
    """
    Plot a 3x3 grid of bullseye plots (17 AHA segments each) with a shared colorbar.

    Parameters
    ----------
    data_list : list of ndarray
        List of 9 numpy arrays, each of shape (17,) representing AHA segment data.
    save_path : str or None
        If provided, the figure will be saved to this path.
    vmin, vmax : float or None
        If provided, sets the color scale limits for all subplots.
    cmap : colormap
        Colormap for all plots.
    titles : list of str
        Optional titles for each subplot.
    """
    ncycle = SimDet["closedloopparam"]["stop_iter"] + 1
    BCL = SimDet["HeartBeatLength"]  
    # outputdir = IODet["block"] 

    #################################################################
    ########### baseline
    #################################################################      
    outdir_baseline = os.path.join(IODet["outputfolder"], IODet["caseID"])
    ecc_path_baseline = os.path.join(outdir_baseline, "Ecc_AHA.npz")
    ell_path_baseline = os.path.join(outdir_baseline, "Ell_AHA.npz")    

    pv_file_baseline = os.path.join(IODet["outputfolder"]+IODet["caseID"]+"/BiV_PV.txt")     
    t_systole_baseline= np.zeros(ncycle)   
   
    for cycle in range(ncycle):
        homo_tptt1, homo_LVP1, homo_LVV1, homo_RVP1, homo_RVV1, homo_Qmv1 = extract_PV(pv_file_baseline, BCL, cycle, SimDet)
        t_systole_baseline[cycle] = homo_tptt1[np.argmin(homo_LVV1)]    

    # Load data baseline
    data_ecc_baseline = np.load(ecc_path_baseline)
    n_segments_ecc_baseline = len(data_ecc_baseline.files)
    segment_strains_ecc_baseline = [data_ecc_baseline[f'arr_{i}'] for i in range(n_segments_ecc_baseline)]
    strain_matrix_ecc_baseline = np.vstack(segment_strains_ecc_baseline)  # shape: (17, N)
    n_timesteps_ecc_baseline = strain_matrix_ecc_baseline.shape[1]

    data_ell_baseline = np.load(ell_path_baseline)
    n_segments_ell_baseline = len(data_ell_baseline.files)
    segment_strains_ell_baseline = [data_ell_baseline[f'arr_{i}'] for i in range(n_segments_ell_baseline)]
    strain_matrix_ell_baseline = np.vstack(segment_strains_ell_baseline)  # shape: (17, N)
    n_timesteps_ell_baseline = strain_matrix_ell_baseline.shape[1]
    frames_per_cycle_baseline = n_timesteps_ecc_baseline // ncycle

    #################################################################
    ########### lafb
    #################################################################    
    outdir_lafb = "/mnt/home/hagersan/GITLAB/heArt/demo/LBBB-fine/LVelectromechanics-PJ-lbbb8-new"
    ecc_path_lafb = os.path.join(outdir_lafb, "Ecc_AHA.npz")
    ell_path_lafb = os.path.join(outdir_lafb, "Ell_AHA.npz") 

    pv_file_lafb = os.path.join(outdir_lafb+"/BiV_PV.txt")     
    t_systole_lafb= np.zeros(ncycle)   
   
    for cycle in range(ncycle):
        homo_tptt1, homo_LVP1, homo_LVV1, homo_RVP1, homo_RVV1, homo_Qmv1 = extract_PV(pv_file_lafb, BCL, cycle, SimDet)
        t_systole_lafb[cycle] = homo_tptt1[np.argmin(homo_LVV1)] 

    # Load data lafb
    data_ecc_lafb = np.load(ecc_path_lafb)
    n_segments_ecc_lafb = len(data_ecc_lafb.files)
    segment_strains_ecc_lafb = [data_ecc_lafb[f'arr_{i}'] for i in range(n_segments_ecc_lafb)]
    strain_matrix_ecc_lafb = np.vstack(segment_strains_ecc_lafb)  # shape: (17, N)
    n_timesteps_ecc_lafb = strain_matrix_ecc_lafb.shape[1]

    data_ell_lafb = np.load(ell_path_lafb)
    n_segments_ell_lafb = len(data_ell_lafb.files)
    segment_strains_ell_lafb = [data_ell_lafb[f'arr_{i}'] for i in range(n_segments_ell_lafb)]
    strain_matrix_ell_lafb = np.vstack(segment_strains_ell_lafb)  # shape: (17, N)
    n_timesteps_ell_lafb = strain_matrix_ell_lafb.shape[1]
    frames_per_cycle_lafb = n_timesteps_ecc_lafb // ncycle

    #################################################################
    ########### septal-apical block
    #################################################################    
    outdir_sa = "/mnt/home/hagersan/GITLAB/heArt/demo/LBBB-fine/LVelectromechanics-PJ-lbbb398-new"
    ecc_path_sa = os.path.join(outdir_sa, "Ecc_AHA.npz")
    ell_path_sa = os.path.join(outdir_sa, "Ell_AHA.npz") 

    pv_file_sa = os.path.join(outdir_sa+"/BiV_PV.txt")     
    t_systole_sa= np.zeros(ncycle)   
   
    for cycle in range(ncycle):
        homo_tptt1, homo_LVP1, homo_LVV1, homo_RVP1, homo_RVV1, homo_Qmv1 = extract_PV(pv_file_sa, BCL, cycle, SimDet)
        t_systole_sa[cycle] = homo_tptt1[np.argmin(homo_LVV1)] 

    # Load data sa
    data_ecc_sa = np.load(ecc_path_sa)
    n_segments_ecc_sa = len(data_ecc_sa.files)
    segment_strains_ecc_sa = [data_ecc_sa[f'arr_{i}'] for i in range(n_segments_ecc_sa)]
    strain_matrix_ecc_sa = np.vstack(segment_strains_ecc_sa)  # shape: (17, N)
    n_timesteps_ecc_sa = strain_matrix_ecc_sa.shape[1]

    data_ell_sa = np.load(ell_path_sa)
    n_segments_ell_sa = len(data_ell_sa.files)
    segment_strains_ell_sa = [data_ell_sa[f'arr_{i}'] for i in range(n_segments_ell_sa)]
    strain_matrix_ell_sa = np.vstack(segment_strains_ell_sa)  # shape: (17, N)
    n_timesteps_ell_sa = strain_matrix_ell_sa.shape[1]
    frames_per_cycle_sa = n_timesteps_ecc_sa // ncycle    

    #################################################################
    ########### partial posterior block
    #################################################################    
    outdir_pp = "/mnt/home/hagersan/GITLAB/heArt/demo/LBBB-fine/LVelectromechanics-PJ-lbbb954-new"
    ecc_path_pp = os.path.join(outdir_pp, "Ecc_AHA.npz")
    ell_path_pp = os.path.join(outdir_pp, "Ell_AHA.npz") 

    pv_file_pp = os.path.join(outdir_pp+"/BiV_PV.txt")     
    t_systole_pp= np.zeros(ncycle)   
   
    for cycle in range(ncycle):
        homo_tptt1, homo_LVP1, homo_LVV1, homo_RVP1, homo_RVV1, homo_Qmv1 = extract_PV(pv_file_pp, BCL, cycle, SimDet)
        t_systole_pp[cycle] = homo_tptt1[np.argmin(homo_LVV1)] 

    # Load data pp
    data_ecc_pp = np.load(ecc_path_pp)
    n_segments_ecc_pp = len(data_ecc_pp.files)
    segment_strains_ecc_pp = [data_ecc_pp[f'arr_{i}'] for i in range(n_segments_ecc_pp)]
    strain_matrix_ecc_pp = np.vstack(segment_strains_ecc_pp)  # shape: (17, N)
    n_timesteps_ecc_pp = strain_matrix_ecc_pp.shape[1]

    data_ell_pp = np.load(ell_path_pp)
    n_segments_ell_pp = len(data_ell_pp.files)
    segment_strains_ell_pp = [data_ell_pp[f'arr_{i}'] for i in range(n_segments_ell_pp)]
    strain_matrix_ell_pp = np.vstack(segment_strains_ell_pp)  # shape: (17, N)
    n_timesteps_ell_pp = strain_matrix_ell_pp.shape[1]
    frames_per_cycle_pp = n_timesteps_ecc_pp // ncycle    

    #################################################################
    ########### lbbb
    #################################################################      
    outdir_lbbb = "/mnt/home/hagersan/GITLAB/heArt/demo/LBBB-fine/LVelectromechanics-PJ-block-test"
    ecc_path_lbbb = os.path.join(outdir_lbbb, "Ecc_AHA.npz")
    ell_path_lbbb = os.path.join(outdir_lbbb, "Ell_AHA.npz") 

    pv_file_lbbb = os.path.join(outdir_lbbb+"/BiV_PV.txt")     
    t_systole_lbbb= np.zeros(ncycle)   
   
    for cycle in range(ncycle):
        homo_tptt1, homo_LVP1, homo_LVV1, homo_RVP1, homo_RVV1, homo_Qmv1 = extract_PV(pv_file_lbbb, BCL, cycle, SimDet)
        t_systole_lbbb[cycle] = homo_tptt1[np.argmin(homo_LVV1)]  

    # Load data lbbb
    data_ecc_lbbb = np.load(ecc_path_lbbb)
    n_segments_ecc_lbbb = len(data_ecc_lbbb.files)
    segment_strains_ecc_lbbb = [data_ecc_lbbb[f'arr_{i}'] for i in range(n_segments_ecc_lbbb)]
    strain_matrix_ecc_lbbb = np.vstack(segment_strains_ecc_lbbb)  # shape: (17, N)
    n_timesteps_ecc_lbbb = strain_matrix_ecc_lbbb.shape[1]

    data_ell_lbbb = np.load(ell_path_lbbb)
    n_segments_ell_lbbb = len(data_ell_lbbb.files)
    segment_strains_ell_lbbb = [data_ell_lbbb[f'arr_{i}'] for i in range(n_segments_ell_lbbb)]
    strain_matrix_ell_lbbb = np.vstack(segment_strains_ell_lbbb)  # shape: (17, N)
    n_timesteps_ell_lbbb = strain_matrix_ell_lbbb.shape[1]
    frames_per_cycle_lbbb = n_timesteps_ecc_lbbb // ncycle


    #################################################################
    ######### Calculation

    for cycle in range(ncycle):
        start = cycle * frames_per_cycle_baseline
        end = (cycle + 1) * frames_per_cycle_baseline

        # --- Save strain minima and time indices - baseline ---
        min_ecc_times_baseline = np.argmin(strain_matrix_ecc_baseline[:, start:end], axis=1) + start
        min_ell_times_baseline = np.argmin(strain_matrix_ell_baseline[:, start:end], axis=1) + start

        # convert_frame_index_to_time(min_ecc_times)
        min_ecc_time_ms_baseline = convert_frame_index_to_time(min_ecc_times_baseline, SimDet["dt"], SimDet["writeStep"], SimDet["dt"])
        min_ell_time_ms_baseline = convert_frame_index_to_time(min_ell_times_baseline, SimDet["dt"], SimDet["writeStep"], SimDet["dt"])

        valid_min_ecc_times_ms_baseline = min_ecc_time_ms_baseline[:16]  # segments 1–16 only
        valid_min_ell_times_ms_baseline = min_ell_time_ms_baseline[:16]  # segments 1–16 only

        # --- Bullseye of time delta between min Ecc and t_systole ---
        delta_time_ecc_baseline = np.full(17, np.nan)
        delta_time_ecc_baseline[:16] = valid_min_ecc_times_ms_baseline - (t_systole_baseline[cycle] if cycle == 0 else t_systole_baseline[cycle] - (cycle - 1) * BCL)

        # --- Bullseye of time delta between min Ecc and t_systole ---
        delta_time_ell_baseline = np.full(17, np.nan)
        delta_time_ell_baseline[:16] = valid_min_ell_times_ms_baseline - (t_systole_baseline[cycle] if cycle == 0 else t_systole_baseline[cycle] - (cycle - 1) * BCL)        

        # --- Save strain minima and time indices  - lafb ---
        min_ecc_times_lafb = np.argmin(strain_matrix_ecc_lafb[:, start:end], axis=1) + start
        min_ell_times_lafb = np.argmin(strain_matrix_ell_lafb[:, start:end], axis=1) + start

        # convert_frame_index_to_time(min_ecc_times)
        min_ecc_time_ms_lafb = convert_frame_index_to_time(min_ecc_times_lafb, SimDet["dt"], SimDet["writeStep"], SimDet["dt"])
        min_ell_time_ms_lafb = convert_frame_index_to_time(min_ell_times_lafb, SimDet["dt"], SimDet["writeStep"], SimDet["dt"])

        valid_min_ecc_times_ms_lafb = min_ecc_time_ms_lafb[:16]  # segments 1–16 only
        valid_min_ell_times_ms_lafb = min_ell_time_ms_lafb[:16]  # segments 1–16 only

        # --- Bullseye of time delta between min Ecc and t_systole ---
        delta_time_ecc_lafb = np.full(17, np.nan)
        delta_time_ecc_lafb[:16] = valid_min_ecc_times_ms_lafb - (t_systole_lafb[cycle] if cycle == 0 else t_systole_lafb[cycle] - (cycle - 1) * BCL)

        # --- Bullseye of time delta between min Ecc and t_systole ---
        delta_time_ell_lafb = np.full(17, np.nan)
        delta_time_ell_lafb[:16] = valid_min_ell_times_ms_lafb - (t_systole_lafb[cycle] if cycle == 0 else t_systole_lafb[cycle] - (cycle - 1) * BCL)   

        # --- Save strain minima and time indices - sa ---
        min_ecc_times_sa = np.argmin(strain_matrix_ecc_sa[:, start:end], axis=1) + start
        min_ell_times_sa = np.argmin(strain_matrix_ell_sa[:, start:end], axis=1) + start

        # convert_frame_index_to_time(min_ecc_times)
        min_ecc_time_ms_sa = convert_frame_index_to_time(min_ecc_times_sa, SimDet["dt"], SimDet["writeStep"], SimDet["dt"])
        min_ell_time_ms_sa = convert_frame_index_to_time(min_ell_times_sa, SimDet["dt"], SimDet["writeStep"], SimDet["dt"])

        valid_min_ecc_times_ms_sa = min_ecc_time_ms_sa[:16]  # segments 1–16 only
        valid_min_ell_times_ms_sa = min_ell_time_ms_sa[:16]  # segments 1–16 only

        # --- Bullseye of time delta between min Ecc and t_systole ---
        delta_time_ecc_sa = np.full(17, np.nan)
        delta_time_ecc_sa[:16] = valid_min_ecc_times_ms_sa - (t_systole_sa[cycle] if cycle == 0 else t_systole_sa[cycle] - (cycle - 1) * BCL)

        # --- Bullseye of time delta between min Ecc and t_systole ---
        delta_time_ell_sa = np.full(17, np.nan)
        delta_time_ell_sa[:16] = valid_min_ell_times_ms_sa - (t_systole_sa[cycle] if cycle == 0 else t_systole_sa[cycle] - (cycle - 1) * BCL)        

        # --- Save strain minima and time indices - pp ---
        min_ecc_times_pp = np.argmin(strain_matrix_ecc_pp[:, start:end], axis=1) + start
        min_ell_times_pp = np.argmin(strain_matrix_ell_pp[:, start:end], axis=1) + start

        # convert_frame_index_to_time(min_ecc_times)
        min_ecc_time_ms_pp = convert_frame_index_to_time(min_ecc_times_pp, SimDet["dt"], SimDet["writeStep"], SimDet["dt"])
        min_ell_time_ms_pp = convert_frame_index_to_time(min_ell_times_pp, SimDet["dt"], SimDet["writeStep"], SimDet["dt"])

        valid_min_ecc_times_ms_pp = min_ecc_time_ms_pp[:16]  # segments 1–16 only
        valid_min_ell_times_ms_pp = min_ell_time_ms_pp[:16]  # segments 1–16 only

        # --- Bullseye of time delta between min Ecc and t_systole ---
        delta_time_ecc_pp = np.full(17, np.nan)
        delta_time_ecc_pp[:16] = valid_min_ecc_times_ms_pp - (t_systole_pp[cycle] if cycle == 0 else t_systole_pp[cycle] - (cycle - 1) * BCL)

        # --- Bullseye of time delta between min Ecc and t_systole ---
        delta_time_ell_pp = np.full(17, np.nan)
        delta_time_ell_pp[:16] = valid_min_ell_times_ms_pp - (t_systole_pp[cycle] if cycle == 0 else t_systole_pp[cycle] - (cycle - 1) * BCL)        

        # --- Save strain minima and time indices - lbbb---
        min_ecc_times_lbbb = np.argmin(strain_matrix_ecc_lbbb[:, start:end], axis=1) + start
        min_ell_times_lbbb = np.argmin(strain_matrix_ell_lbbb[:, start:end], axis=1) + start

        # convert_frame_index_to_time(min_ecc_times)
        min_ecc_time_ms_lbbb = convert_frame_index_to_time(min_ecc_times_lbbb, SimDet["dt"], SimDet["writeStep"], SimDet["dt"])
        min_ell_time_ms_lbbb = convert_frame_index_to_time(min_ell_times_lbbb, SimDet["dt"], SimDet["writeStep"], SimDet["dt"])

        valid_min_ecc_times_ms_lbbb = min_ecc_time_ms_lbbb[:16]  # segments 1–16 only
        valid_min_ell_times_ms_lbbb = min_ell_time_ms_lbbb[:16]  # segments 1–16 only

        # --- Bullseye of time delta between min Ecc and t_systole ---
        delta_time_ecc_lbbb = np.full(17, np.nan)
        delta_time_ecc_lbbb[:16] = valid_min_ecc_times_ms_lbbb - (t_systole_lbbb[cycle] if cycle == 0 else t_systole_lbbb[cycle] - (cycle - 1) * BCL)

        # --- Bullseye of time delta between min Ecc and t_systole ---
        delta_time_ell_lbbb = np.full(17, np.nan)
        delta_time_ell_lbbb[:16] = valid_min_ell_times_ms_lbbb - (t_systole_lbbb[cycle] if cycle == 0 else t_systole_lbbb[cycle] - (cycle - 1) * BCL)  

        # --- Bullseye of time delta between min Ecc and t_systole ---
        # Normalize color scale to this delta
        vmin_dt, vmax_dt = np.nanmin(delta_time_ecc_lbbb[:16]), np.nanmax(delta_time_ecc_lbbb[:16])
        norm_dt = mpl.colors.Normalize(vmin=vmin_dt, vmax=vmax_dt)
        cmap_dt = cm.coolwarm

        fig_dt, axes_dt = plt.subplots(2, 5, figsize=(8, 4), subplot_kw=dict(polar=True),gridspec_kw={"wspace": 0.1, "hspace": 0.1})
        axes_dt = axes_dt.reshape(2, 5)  # Ensure shape is consistent

        bullseye_plot(axes_dt[0, 0], data=delta_time_ecc_baseline, seg_bold=[], cmap=cmap_dt, norm=norm_dt)
        bullseye_plot(axes_dt[0, 1], data=delta_time_ecc_lafb, seg_bold=[], cmap=cmap_dt, norm=norm_dt)
        bullseye_plot(axes_dt[0, 2], data=delta_time_ecc_sa, seg_bold=[], cmap=cmap_dt, norm=norm_dt)
        bullseye_plot(axes_dt[0, 3], data=delta_time_ecc_pp, seg_bold=[], cmap=cmap_dt, norm=norm_dt)
        bullseye_plot(axes_dt[0, 4], data=delta_time_ecc_lbbb, seg_bold=[], cmap=cmap_dt, norm=norm_dt)

        bullseye_plot(axes_dt[1, 0], data=delta_time_ell_baseline, seg_bold=[], cmap=cmap_dt, norm=norm_dt)
        bullseye_plot(axes_dt[1, 1], data=delta_time_ell_lafb, seg_bold=[], cmap=cmap_dt, norm=norm_dt)
        bullseye_plot(axes_dt[1, 2], data=delta_time_ell_sa, seg_bold=[], cmap=cmap_dt, norm=norm_dt)
        bullseye_plot(axes_dt[1, 3], data=delta_time_ell_pp, seg_bold=[], cmap=cmap_dt, norm=norm_dt)        
        bullseye_plot(axes_dt[1, 4], data=delta_time_ell_lbbb, seg_bold=[], cmap=cmap_dt, norm=norm_dt)

        # Remove radial labels and ticks for all
        for ax in axes_dt.ravel():
            ax.set_xticks([])
            ax.set_yticks([])        

        # Shared colorbar (optional)
        sm_dt = mpl.cm.ScalarMappable(cmap=cmap_dt, norm=norm_dt)
        sm_dt.set_array([])
        #cbar_dt = fig_dt.colorbar(sm_dt, ax=axes_dt.ravel().tolist(), orientation='horizontal', pad=0.1, shrink=0.85)
        cbar_dt = fig_dt.colorbar(sm_dt, ax=axes_dt.ravel().tolist(),
                          orientation='horizontal',
                          pad=0.08,   # space between subplots and colorbar
                          fraction=0.035,  # relative height of colorbar
                          aspect=30,  # length/height ratio of colorbar
                          shrink=0.65)  # shrink overall size of colorbar
         
        cbar_dt.set_label("ΔT: Min Peak Strain vs Systole [ms]", fontsize=10, labelpad=4)
        # axes_dt.set_title(f"ΔT (Peakstrain min - systole) - Cycle {cycle + 1}", fontsize=14)

        # Save or show
        # plt.tight_layout()
        delta_fig_path = os.path.join(IODet["outputfolder"]+outputdir, f"delta_minPeak_from_systole_cycle{cycle+1}.png")
        plt.savefig(delta_fig_path, dpi=600, bbox_inches='tight')
        plt.close(fig_dt)
        # plt.savefig("all_bullseyes_grid.png", dpi=300)
        # plt.show()
            

def peak_strain_analysis(u_arr, mesh, udisp, uflforms, Fref, eC0_norm, aha_tags):
    V_dg0 = df.FunctionSpace(mesh, "DG", 0)
    Ecc_proj = df.Function(V_dg0)
    
    min_Ecc_AHA = np.full(18, np.inf)
    min_Ecc_time = np.full(18, -1, dtype=int)

    tag_array = aha_tags.array()
    # num_cells = mesh.topology.index_map(mesh.topology.dim()).size_local
    num_cells = mesh.num_cells()

    for t_idx, u in enumerate(tqdm(u_arr, desc="Computing peak Ecc")):
        udisp.vector()[:] = u.vector().get_local()
        F = uflforms.Fmat() * df.inv(Fref)
        C = F.T * F
        Ecc_expr = 0.5 * (1 - 1 / df.inner(eC0_norm, C * eC0_norm))

        # Ecc_proj.interpolate(Ecc_expr)
        Ecc_proj.assign(df.project(Ecc_expr, Ecc_proj.function_space()))
        cell_values = Ecc_proj.vector().get_local()

        for seg_id in range(1, 18):
            segment_cells = np.where(tag_array[:num_cells] == seg_id)[0]
            if segment_cells.size == 0:
                continue
            segment_vals = cell_values[segment_cells]
            min_val = np.min(segment_vals)
            if min_val < min_Ecc_AHA[seg_id]:
                min_Ecc_AHA[seg_id] = min_val
                min_Ecc_time[seg_id] = t_idx

    return {
        seg: min_Ecc_AHA[seg] for seg in range(1, 18)
    }, {
        seg: min_Ecc_time[seg] for seg in range(1, 18)
    }

def compute_strain_AHA(IODet, SimDet, LVid=1, RVid=2, cycle=None, ref_point=None, aha_tags=None): #serial

    outdir = os.path.join(IODet["outputfolder"], IODet["caseID"])

    # --- Mesh and displacement ---
    mesh = df.Mesh()
    hdf = df.HDF5File(mesh.mpi_comm(), os.path.join(IODet["outputfolder"], IODet["caseID"], "Data.h5"), "r")
    hdf.read(mesh, "ME/mesh", False)
    ncycle = cycle
    cycle = cycle + 1

    u_arr = extractdisplacement(IODet, SimDet, cycle)

    n_frames = len(u_arr)
    frames_per_cycle = n_frames // (cycle) # = 320 if n_frames = 640

    start = ncycle * frames_per_cycle
    end = (cycle) * frames_per_cycle    
    
    # pdb.set_trace()
    # Extract second cycle
    u_arr_cycle1 = u_arr[start:end]    

    # --- Mechanics mesh setup ---
    mesh_params = {
        "directory": IODet["directory_me"],
        "casename": IODet["casename_me"],
        "outputfolder": IODet["outputfolder"],
        "foldername": IODet["folderName"],
        "isLV": True,
    }
    Mesh_obj = lv_mechanics_mesh(mesh_params, SimDet)
    mesh = Mesh_obj.mesh
    var_deg = 1 if SimDet["Mechanics Discretization"] == "P1P1" else 2
    udisp = df.Function(df.VectorFunctionSpace(mesh, "CG", var_deg))

    try:
        eC0 = Mesh_obj.eC0
        eL0 = Mesh_obj.eL0
        eR0 = Mesh_obj.eR0

    except AttributeError:
        fiber_angle_param = {   "mesh": Mesh_obj.mesh,
                                "facetboundaries": Mesh_obj.facetboundaries,
                                "LV_fiber_angle": [0.01, -0.01],
                                "LV_sheet_angle": [0.1, -0.1],
                                "minztol": Mesh_obj.mesh.hmax()/2.0, # Coarse mesh
                                "isrotatept": False,
                                "isreturn": True,
                                "outfilename": IODet["casename_me"],
                                "outdirectory": IODet["outputfolder"] + IODet["caseID"] + IODet["folderName"],
                                "baseid": SimDet["topid"],
                                "epiid": SimDet["epiid"],
                                "lvid": SimDet["LVendoid"],
                                "degree": SimDet["GiccioneParams"]["deg"]}

        eC0, eL0, eR0  = vtk_py.addLVfiber_LDRB(fiber_angle_param)

    eC0_norm = eC0 / df.sqrt(df.inner(eC0, eC0))
    eL0_norm = eL0 / df.sqrt(df.inner(eL0, eL0))
    eR0_norm = eR0 / df.sqrt(df.inner(eR0, eR0))

    GuccioneParams = SimDet["GiccioneParams"]
    uflforms = Forms({
        "mesh": mesh,
        "displacement_variable": udisp,
        "material model": GuccioneParams["Passive model"],
        "material params": GuccioneParams["Passive params"],
        "incompressible": GuccioneParams["incompressible"],
        "growth_tensor": None,
    })

    Fref = df.project(uflforms.Fmat(), df.TensorFunctionSpace(mesh, "DG", 0))

    # --- AHA tag assignment ---
    if aha_tags is None:
        if ref_point is None:
            ref_point = SimDet["HIS"]
        aha_tags = assign_aha_tags(mesh, ref_point)

    aha_counts = collections.Counter()

    for cell in df.cells(mesh):
        tag = aha_tags[cell]
        aha_counts[tag] += 1

    # Print the counts
    print("AHA Segment Element Counts:")
    for tag in sorted(aha_counts):
        print(f"  Segment {tag}: {aha_counts[tag]} elements")

    # --- Strain computation ---
    min_Ecc_AHA1, min_Ecc_time1 = peak_strain_analysis(u_arr_cycle1, mesh, udisp, uflforms, Fref, eC0_norm, aha_tags)
    min_Ell_AHA1, min_Ell_time1 = peak_strain_analysis(u_arr_cycle1, mesh, udisp, uflforms, Fref, eL0_norm, aha_tags)
   
    min_Ecc_time1_ms = convert_frame_index_to_time(min_Ecc_time1)
    min_Ell_time1_ms = convert_frame_index_to_time(min_Ell_time1)

    # pdb.set_trace

    # Filter out invalid values (e.g., -1)
    valid_Ecc_times = [t for t in min_Ecc_time1_ms.values() if t != -1]
    valid_Ell_times = [t for t in min_Ell_time1_ms.values() if t != -1]

    # Convert to NumPy array for convenience
    Ecc_times_array = np.array(valid_Ecc_times)    
    Ell_times_array = np.array(valid_Ell_times)  

    # --- Compute SDI ---
    # pdb.set_trace()
    t_systole= np.zeros(cycle)   
    BCL = SimDet["HeartBeatLength"]    

    for ncycle in range(cycle):
        filename1 = os.path.join(IODet["outputfolder"]+IODet["caseID"]+"/BiV_PV.txt")
        homo_tptt1, homo_LVP1, homo_LVV1, homo_RVP1, homo_RVV1, homo_Qmv1 = extract_PV(filename1, BCL, ncycle, SimDet)
        t_systole[ncycle] = homo_tptt1[np.argmin(homo_LVV1)]

    if t_systole is None or len(t_systole) < cycle:
        raise ValueError("SimDet must include 't_min_vol' with at least two entries")

    # # Absolute differences from the reference
    # Ecc_differences = np.abs(Ecc_times_array - (t_systole[1]-800))
    # Ell_differences = np.abs(Ell_times_array - (t_systole[1]-800))

    SDI_Ecc = np.std(Ecc_times_array - (t_systole[cycle-1]-BCL)) / BCL * 100
    SDI_Ell = np.std(Ell_times_array - (t_systole[cycle-1]-BCL)) / BCL * 100    

    print(f"Ecc SDI: {SDI_Ecc:.4f}")
    print(f"Ell SDI: {SDI_Ell:.4f}")

    with open(outdir+"/SDI.txt", "w") as f:
        f.write("SDI\n")
        f.write(f"Ecc SDI:\t{SDI_Ecc:.4f}[%]\n")  
        f.write(f"Ell SDI:\t{SDI_Ell:.4f}[%]\n")  

    # Convert dictionary to ordered list of 17 values
    Ecc_data_dict = min_Ecc_AHA1
    Ecc_data_list = [Ecc_data_dict.get(i, np.nan) for i in range(1, 18)]  # 1-based AHA segments

    Ecc_min_val = np.nanmin(Ecc_data_list)
    Ecc_max_val = np.nanmax(Ecc_data_list)

    # Optional: define normalization and segments to highlight
    Ecc_norm = mpl.colors.Normalize(vmin=min(Ecc_data_list), vmax=max(Ecc_data_list))  # scale colors
    Ecc_cmap = plt.cm.viridis

    # Create figure and polar axes
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    # Call the plotting function
    bullseye_plot(ax, Ecc_data_list, seg_bold=[], cmap=Ecc_cmap, norm=Ecc_norm)

    # Add colorbar (legend)
    # Create dummy mappable for colorbar
    ticks = np.linspace(Ecc_min_val, Ecc_max_val, 5)
    sm = mpl.cm.ScalarMappable(cmap=Ecc_cmap, norm=Ecc_norm)
    sm.set_array([])  # Required for compatibility
    cbar = plt.colorbar(sm, ax=ax, orientation='horizontal', pad=0.05, shrink=0.6)
    cbar.set_label("Peak Ecc Strain", fontsize=12)
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([f"{t:.3f}" for t in ticks])  # formatted ticks


    # Add title and save
    plt.title("Peak concentric Strain - Cycle 2", fontsize=14)
    plt.savefig(outdir+"/peak_Ecc_bullseye.png", dpi=300, bbox_inches='tight')
    plt.close(fig)

    print("Saved:peak_Ecc_bullseye.png")

    # Convert dictionary to ordered list of 17 values
    Ell_data_dict = min_Ell_AHA1
    Ell_data_list = [Ell_data_dict.get(i, np.nan) for i in range(1, 18)]  # 1-based AHA segments

    Ell_min_val = np.nanmin(Ell_data_list)
    Ell_max_val = np.nanmax(Ell_data_list)

    # Optional: define normalization and segments to highlight
    Ell_norm = mpl.colors.Normalize(vmin=min(Ell_data_list), vmax=max(Ell_data_list))  # scale colors
    Ell_cmap = plt.cm.viridis

    # Create figure and polar axes
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    # Call the plotting function
    bullseye_plot(ax, Ell_data_list, seg_bold=[], cmap=Ell_cmap, norm=Ell_norm)

    # Add colorbar (legend)
    # Create dummy mappable for colorbar
    ticks = np.linspace(Ell_min_val, Ell_max_val, 5)
    sm = mpl.cm.ScalarMappable(cmap=Ell_cmap, norm=Ell_norm)
    sm.set_array([])  # Required for compatibility
    cbar = plt.colorbar(sm, ax=ax, orientation='horizontal', pad=0.05, shrink=0.6)
    cbar.set_label("Peak longitudinal Strain", fontsize=12)
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([f"{t:.3f}" for t in ticks])  # formatted ticks

    # Add title and save
    plt.title("Peak longitudinal Strain - Cycle 2", fontsize=14)
    plt.savefig(outdir+"/peak_Ell_bullseye.png", dpi=300, bbox_inches='tight')
    plt.close(fig)

    print("Saved:peak_Ell_bullseye.png")    

    # Prepare time and strain data lists (1–17)
    Ecc_strain_dict = min_Ecc_AHA1
    Ecc_time_dict = min_Ecc_time1_ms

    segments = list(range(1, 18))
    Ecc_strain_list = [Ecc_strain_dict.get(i, np.nan) for i in segments]
    Ecc_time_list = [Ecc_time_dict.get(i, np.nan) for i in segments]

    # ----- Save data to TXT -----
    output_txt = "/peak_Ecc_data.txt"
    with open(outdir+output_txt, "w") as f:
        f.write("Segment\tPeak_Ecc\tTime_to_Peak_Ecc (ms)\n")
        for i, strain, time in zip(segments, Ecc_strain_list, Ecc_time_list):
            f.write(f"{i}\t{strain:.4f}\t{time:.1f}\n")

    print(f"Saved: {output_txt}")

    # ----- Plot: Time to peak Ecc -----
    time_min = np.nanmin(Ecc_time_list)
    time_max = np.nanmax(Ecc_time_list)
    time_norm = mpl.colors.Normalize(vmin=time_min, vmax=time_max)
    cmap_time = plt.cm.viridis

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    bullseye_plot(ax, Ecc_time_list, seg_bold=[], cmap=cmap_time, norm=time_norm)

    # Colorbar with time scale
    ticks_time = np.linspace(time_min, time_max, 5)
    sm_time = mpl.cm.ScalarMappable(cmap=cmap_time, norm=time_norm)
    sm_time.set_array([])
    cbar = plt.colorbar(sm_time, ax=ax, orientation='horizontal', pad=0.05, shrink=0.6)
    cbar.set_label("Time to Peak concentric Strain (ms)", fontsize=12)
    cbar.set_ticks(ticks_time)
    cbar.set_ticklabels([f"{t:.0f}" for t in ticks_time])

    plt.title("Time to Peak concentric Strain - Cycle 2", fontsize=14)
    plt.savefig(outdir+"/time_to_peak_Ecc_bullseye.png", dpi=300, bbox_inches='tight')
    plt.close(fig)

    print("Saved: cycle2_time_to_peak_Ecc_bullseye.png")

    # Prepare time and strain data lists (1–17)
    Ell_strain_dict = min_Ell_AHA1
    Ell_time_dict = min_Ell_time1_ms

    segments = list(range(1, 18))
    Ell_strain_list = [Ell_strain_dict.get(i, np.nan) for i in segments]
    Ell_time_list = [Ell_time_dict.get(i, np.nan) for i in segments]

    # ----- Save data to TXT -----
    output_txt = "/peak_Ell_data.txt"
    with open(outdir+output_txt, "w") as f:
        f.write("Segment\tPeak_Ell\tTime_to_Peak_Ell (ms)\n")
        for i, strain, time in zip(segments, Ell_strain_list, Ell_time_list):
            f.write(f"{i}\t{strain:.4f}\t{time:.1f}\n")

    print(f"Saved: {output_txt}")

    # ----- Plot: Time to peak Ell -----
    time_min = np.nanmin(Ell_time_list)
    time_max = np.nanmax(Ell_time_list)
    time_norm = mpl.colors.Normalize(vmin=time_min, vmax=time_max)
    cmap_time = plt.cm.viridis

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    bullseye_plot(ax, Ell_time_list, seg_bold=[], cmap=cmap_time, norm=time_norm)

    # Colorbar with time scale
    ticks_time = np.linspace(time_min, time_max, 5)
    sm_time = mpl.cm.ScalarMappable(cmap=cmap_time, norm=time_norm)
    sm_time.set_array([])
    cbar = plt.colorbar(sm_time, ax=ax, orientation='horizontal', pad=0.05, shrink=0.6)
    cbar.set_label("Time to Peak longitudinal Strain (ms)", fontsize=12)
    cbar.set_ticks(ticks_time)
    cbar.set_ticklabels([f"{t:.0f}" for t in ticks_time])

    plt.title("Time to Peak longitudinal Strain - Cycle 2", fontsize=14)
    plt.savefig(outdir+"/time_to_peak_Ell_bullseye.png", dpi=300, bbox_inches='tight')
    plt.close(fig)

    print("Saved: cycle2_time_to_peak_Ell_bullseye.png")    

    return min_Ecc_AHA1, min_Ecc_time1, min_Ecc_time1_ms, SDI_Ecc, min_Ell_AHA1, min_Ell_time1, min_Ell_time1_ms, SDI_Ell


def convert_frame_index_to_time(min_Ecc_time, dt=0.5, step_interval=5, t0=0.5):
    return {seg: (t0 + dt * step_interval * t_idx) if t_idx >= 0 else -1
            for seg, t_idx in min_Ecc_time.items()}


def bullseye_plot(ax, data, seg_bold=None, cmap="viridis", norm=None):
    """
    Bullseye representation for the left ventricle.

    Parameters
    ----------
    ax : Axes
    data : list[float]
        The intensity values for each of the 17 segments.
    seg_bold : list[int], optional
        A list with the segments to highlight.
    cmap : colormap, default: "viridis"
        Colormap for the data.
    norm : Normalize or None, optional
        Normalizer for the data.

    Notes
    -----
    This function creates the 17 segment model for the left ventricle according
    to the American Heart Association (AHA) [1]_

    References
    ----------
    .. [1] M. D. Cerqueira, N. J. Weissman, V. Dilsizian, A. K. Jacobs,
        S. Kaul, W. K. Laskey, D. J. Pennell, J. A. Rumberger, T. Ryan,
        and M. S. Verani, "Standardized myocardial segmentation and
        nomenclature for tomographic imaging of the heart",
        Circulation, vol. 105, no. 4, pp. 539-542, 2002.
    """

    data = np.ravel(data)
    if seg_bold is None:
        seg_bold = []
    if norm is None:
        norm = mpl.colors.Normalize(vmin=data.min(), vmax=data.max())

    r = np.linspace(0.2, 1, 4)

    ax.set(ylim=[0, 1], xticklabels=[], yticklabels=[])
    ax.grid(False)  # Remove grid

    # Fill segments 1-6, 7-12, 13-16.
    for start, stop, r_in, r_out in [
            (0, 6, r[2], r[3]),
            (6, 12, r[1], r[2]),
            (12, 16, r[0], r[1]),
            (16, 17, 0, r[0]),
    ]:
        n = stop - start
        dtheta = 2*np.pi / n
        ax.bar(np.arange(n) * dtheta + np.pi/2, r_out - r_in, dtheta, r_in,
               color=cmap(norm(data[start:stop])))

    # Now, draw the segment borders.  In order for the outer bold borders not
    # to be covered by inner segments, the borders are all drawn separately
    # after the segments have all been filled.  We also disable clipping, which
    # would otherwise affect the outermost segment edges.
    # Draw edges of segments 1-6, 7-12, 13-16.
    for start, stop, r_in, r_out in [
            (0, 6, r[2], r[3]),
            (6, 12, r[1], r[2]),
            (12, 16, r[0], r[1]),
    ]:
        n = stop - start
        dtheta = 2*np.pi / n
        ax.bar(np.arange(n) * dtheta + np.pi/2, r_out - r_in, dtheta, r_in,
               clip_on=False, color="none", edgecolor="k", linewidth=[
                   4 if i + 1 in seg_bold else 2 for i in range(start, stop)])
    # Draw edge of segment 17 -- here; the edge needs to be drawn differently,
    # using plot().
    ax.plot(np.linspace(0, 2*np.pi), np.linspace(r[0], r[0]), "k",
            linewidth=(4 if 17 in seg_bold else 2))


def plot_strain_comparison(IODet, SimDet, cycle=None):
    """Compare strain-time curves from two different Ecc.nz, Ell.nz and Err.nz files."""
    BCL = SimDet["HeartBeatLength"]
    # block = IODet["block"]   
    crt = IODet["caseID"]
    mode = IODet["comparison_plot_mode"]
    
    if cycle is None:
        ncycle = SimDet["closedloopparam"]["stop_iter"] +1 


    if mode == "disease":
        #### Ecc LV profile
        plt.figure()   
        Ecc_baseline_filename = os.path.join(IODet["outputfolder"]+IODet["directory_baseline"]+"/ECC/Ecc_LV_lastcycle.npz")
        Ecc_baseline_load = np.load(Ecc_baseline_filename)
        Ecc_baseline = Ecc_baseline_load[Ecc_baseline_load.files[0]]
    
        Ecc_block_filename = os.path.join(IODet["outputfolder"]+IODet["directory_block"]+"/ECC/Ecc_LV_lastcycle.npz")
        Ecc_block_load = np.load(Ecc_block_filename)
        Ecc_block = Ecc_block_load[Ecc_block_load.files[0]]
    
        Ecc_case_filename = os.path.join(IODet["outputfolder"]+IODet["directory_ischemia_block"]+"/ECC/Ecc_LV_lastcycle.npz")
        Ecc_case_load = np.load(Ecc_case_filename)
        Ecc_case = Ecc_case_load[Ecc_case_load.files[0]]    
    
        x = np.arange(len(Ecc_block))    
    
        plt.plot(x, Ecc_baseline, label=f"Baseline")
        plt.plot(x, Ecc_block, linestyle="dashed", label=f"LBBB")           
        plt.plot(x, Ecc_case, linestyle="dashed", label=f"iLBBB")   
 
    
        hemodynamics_outdirectory = os.path.join(IODet["outputfolder"]+IODet["comparison_plot_mode"])
        if not os.path.exists(hemodynamics_outdirectory):
            os.mkdir(hemodynamics_outdirectory)
    
        plt.xticks([0, 40, 80, 120, 160, 200, 240, 280, 320])
        plt.xlabel("Time point", fontsize=14)
        plt.ylabel("Strain", fontsize=14)
        plt.savefig(os.path.join("./Ecc_comparison.png"))
        plt.clf()    
    
    
        #### Ecc RV profile
        plt.figure()   
        RV_Ecc_baseline_filename = os.path.join(IODet["outputfolder"]+IODet["directory_baseline"]+"/ECC/Ecc_RV_lastcycle.npz")
        RV_Ecc_baseline_load = np.load(RV_Ecc_baseline_filename)
        RV_Ecc_baseline= Ecc_baseline_load[RV_Ecc_baseline_load.files[0]]
    
        RV_Ecc_block_filename = os.path.join(IODet["outputfolder"]+IODet["directory_block"]+"/ECC/Ecc_RV_lastcycle.npz")
        RV_Ecc_block_load = np.load(RV_Ecc_block_filename)
        RV_Ecc_block = Ecc_block_load[RV_Ecc_block_load.files[0]]    
    
        RV_Ecc_case_filename = os.path.join(IODet["outputfolder"]+IODet["directory_ischemia_block"]+"/ECC/Ecc_RV_lastcycle.npz")
        RV_Ecc_case_load = np.load(RV_Ecc_case_filename)
        RV_Ecc_case = Ecc_case_load[RV_Ecc_case_load.files[0]]  
    
        plt.plot(x, RV_Ecc_baseline, label=f"Baseline")
        plt.plot(x, RV_Ecc_block, linestyle="dashed", label=f"LBBB")            
        plt.plot(x, RV_Ecc_case, linestyle="dashed", label=f"iLBBB")   

    
        hemodynamics_outdirectory = os.path.join(IODet["outputfolder"]+IODet["comparison_plot_mode"])
        if not os.path.exists(hemodynamics_outdirectory):
            os.mkdir(hemodynamics_outdirectory)
    
        plt.xticks([0, 40, 80, 120, 160, 200, 240, 280, 320])
        plt.xlabel("Time point", fontsize=14)
        plt.ylabel("Strain", fontsize=14)
        plt.savefig(os.path.join("./RV_Ecc_comparison.png"))
        plt.clf()      
    
        #### Ell LV profile
        plt.figure()   
        Ell_baseline_filename = os.path.join(IODet["outputfolder"]+IODet["directory_baseline"]+"/ELL/Ell_LV_lastcycle.npz")
        Ell_baseline_load = np.load(Ell_baseline_filename)
        Ell_baseline = Ell_baseline_load[Ell_baseline_load.files[0]]
    
        Ell_block_filename = os.path.join(IODet["outputfolder"]+IODet["directory_block"]+"/ELL/Ell_LV_lastcycle.npz")
        Ell_block_load = np.load(Ell_block_filename)
        Ell_block = Ell_block_load[Ell_block_load.files[0]]    
    
        Ell_case_filename = os.path.join(IODet["outputfolder"]+IODet["directory_ischemia_block"]+"/ELL/Ell_LV_lastcycle.npz")
        Ell_case_load = np.load(Ell_case_filename)
        Ell_case = Ell_case_load[Ell_case_load.files[0]]   
    
    
        plt.plot(x, Ell_baseline, label=f"Baseline")
        plt.plot(x, Ell_block, linestyle="dashed", label=f"LBBB")          
        plt.plot(x, Ell_case, linestyle="dashed", label=f"iLBBB")   
  
    
        hemodynamics_outdirectory = os.path.join(IODet["outputfolder"]+IODet["comparison_plot_mode"])
        if not os.path.exists(hemodynamics_outdirectory):
            os.mkdir(hemodynamics_outdirectory)
    
        plt.xticks([0, 40, 80, 120, 160, 200, 240, 280, 320])
        plt.xlabel("Time point", fontsize=14)
        plt.ylabel("Strain", fontsize=14)
        plt.savefig(os.path.join("./Ell_comparison.png"))
        plt.clf()           
    
        #### Ell RV profile
        plt.figure()
        RV_Ell_baseline_filename = os.path.join(IODet["outputfolder"]+IODet["directory_baseline"]+"/ELL/Ell_RV_lastcycle.npz")
        RV_Ell_baseline_load = np.load(RV_Ell_baseline_filename)
        RV_Ell_baseline = Ell_baseline_load[RV_Ell_baseline_load.files[0]]   
    
        RV_Ell_block_filename = os.path.join(IODet["outputfolder"]+IODet["directory_block"]+"/ELL/Ell_RV_lastcycle.npz")
        RV_Ell_block_load = np.load(RV_Ell_block_filename)
        RV_Ell_block = Ell_block_load[RV_Ell_block_load.files[0]]       
    
        RV_Ell_case_filename = os.path.join(IODet["outputfolder"]+IODet["directory_ischemia_block"]+"/ELL/Ell_RV_lastcycle.npz")
        RV_Ell_case_load = np.load(RV_Ell_case_filename)
        RV_Ell_case = Ell_case_load[RV_Ell_case_load.files[0]]     
       
        plt.plot(x, RV_Ell_baseline, label=f"Baseline")
        plt.plot(x, RV_Ell_block, linestyle="dashed", label=f"LBBB")           
        plt.plot(x, RV_Ell_case, linestyle="dashed", label=f"iLBBB")   
    
        hemodynamics_outdirectory = os.path.join(IODet["outputfolder"])
        if not os.path.exists(hemodynamics_outdirectory):
            os.mkdir(hemodynamics_outdirectory)
    
        plt.xticks([0, 40, 80, 120, 160, 200, 240, 280, 320])
        plt.xlabel("Time point", fontsize=14)
        plt.ylabel("Strain", fontsize=14)
        plt.savefig(os.path.join("./RV_Ell_comparison.png"))
        plt.clf()        
        
    elif mode == "pacing":
        filename1 = os.path.join(IODet["outputfolder"]+IODet["directory_baseline"]+"/BiV_PV.txt")
        time1, LVP1, _, RVP1, _, _ = extract_PV(filename1, BCL, ncycle, SimDet)   
        plt.plot(time1, LVP1, label=f"LV Baseline")
        plt.plot(time1, RVP1, label=f"RV Baseline")

        filename2 = os.path.join(IODet["outputfolder"]+IODet["directory_block"]+"/BiV_PV.txt")
        time2, LVP2, _, RVP2, _, _ = extract_PV(filename2, BCL, ncycle, SimDet)
        plt.plot(time2, LVP2, linestyle="dashed", label=f"LV LBBB")
        plt.plot(time2, RVP2, linestyle="dashed", label=f"RV LBBB")

        filename3 = os.path.join(IODet["outputfolder"]+IODet["directory_bivp"]+"/BiV_PV.txt")
        time3, LVP3, _, RVP3, _, _ = extract_PV(filename3, BCL, ncycle, SimDet)
        plt.plot(time3, LVP3, linestyle="dotted", label=f"LV BiVP")  
        plt.plot(time3, RVP3, linestyle="dotted", label=f"RV BiVP")

        filename4 = os.path.join(IODet["outputfolder"]+IODet["directory_lbbap"]+"/BiV_PV.txt")
        time4, LVP4, _, RVP4, _, _ = extract_PV(filename4, BCL, ncycle, SimDet)
        plt.plot(time4, LVP4, linestyle="dashdot", label=f"LV LBBAP")  
        plt.plot(time4, RVP4, linestyle="dashdot", label=f"RV LBBAP")   

    elif mode == "ischemic_pacing":
        filename1 = os.path.join(IODet["outputfolder"]+IODet["directory_baseline"]+"/BiV_PV.txt")
        time1, LVP1, _, RVP1, _, _ = extract_PV(filename1, BCL, ncycle, SimDet)   
        plt.plot(time1, LVP1, label=f"LV Baseline")
        plt.plot(time1, RVP1, label=f"RV Baseline")

        filename2 = os.path.join(IODet["outputfolder"]+IODet["directory_ischemia_block"]+"/BiV_PV.txt")
        time2, LVP2, _, RVP2, _, _ = extract_PV(filename2, BCL, ncycle, SimDet)
        plt.plot(time2, LVP2, linestyle="dashed", label=f"LV iLBBB")
        plt.plot(time2, RVP2, linestyle="dashed", label=f"RV iLBBB")

        filename3 = os.path.join(IODet["outputfolder"]+IODet["directory_ischemia_bivp"]+"/BiV_PV.txt")
        time3, LVP3, _, RVP3, _, _ = extract_PV(filename3, BCL, ncycle, SimDet)
        plt.plot(time3, LVP3, linestyle="dotted", label=f"LV iBiVP")  
        plt.plot(time3, RVP3, linestyle="dotted", label=f"RV iBiVP")

        filename4 = os.path.join(IODet["outputfolder"]+IODet["directory_ischemia_lbbap"]+"/BiV_PV.txt")
        time4, LVP4, _, RVP4, _, _ = extract_PV(filename4, BCL, ncycle, SimDet)
        plt.plot(time4, LVP4, linestyle="dashdot", label=f"LV iLBBAP")  
        plt.plot(time4, RVP4, linestyle="dashdot", label=f"RV iLBBAP")        

    # Create output directory if not exists
    hemodynamics_outdirectory = os.path.join(IODet["outputfolder"]+IODet["comparison_plot_mode"])
    if not os.path.exists(hemodynamics_outdirectory):
        os.mkdir(hemodynamics_outdirectory)

    plt.legend()
    plt.ylabel("Pressure (mmHg)")
    plt.xlabel("Time (s)")
    plt.savefig(os.path.join(hemodynamics_outdirectory, "Pressure_Comparison_"+IODet["comparison_plot_mode"]+"_cycle_"+str(cycle)+"blocks.png"))
    plt.show()


def plot_strain_comparison_ind(IODet, SimDet, cycle=None):
    """Compare strain-time curves from baseline, block, and case for Ecc and Ell."""
    BCL = SimDet["HeartBeatLength"]
    block = IODet["directory_block"]
    crt = IODet["caseID"]
    mode = IODet["comparison_plot_mode"]
    # Time window restriction
    start = 0 #319
    end = 319 #640

    if cycle is None:
        cycle = SimDet["closedloopparam"]["stop_iter"]

    # Helper loader
    def load_npz(path):
        arr = np.load(path)
        return arr[arr.files[0]]

    Ecc_baseline = load_npz(os.path.join(IODet["outputfolder"], IODet["directory_baseline"], "ECC/Ecc_LV_lastcycle.npz"))
    RV_Ecc_baseline = load_npz(os.path.join(IODet["outputfolder"], IODet["directory_baseline"], "ECC/Ecc_RV_lastcycle.npz"))
    Ell_baseline = load_npz(os.path.join(IODet["outputfolder"], IODet["directory_baseline"], "ELL/Ell_LV_lastcycle.npz"))  
    RV_Ell_baseline = load_npz(os.path.join(IODet["outputfolder"], IODet["directory_baseline"], "ELL/Ell_RV_lastcycle.npz"))  

    x = np.arange(start, end)    
    
    if mode == "disease":
        # -----------------------------------------------------------
        # ===============     Ecc LV     ============================
        # -----------------------------------------------------------   
        Ecc_block = load_npz(os.path.join(IODet["outputfolder"], IODet["directory_block"], "ECC/Ecc_LV_lastcycle.npz"))
        Ecc_ischemia_block = load_npz(os.path.join(IODet["outputfolder"], IODet["directory_ischemia_block"], "ECC/Ecc_LV_lastcycle.npz"))  

        plt.figure()
        plt.plot(x, Ecc_baseline[start:end], label="LV Baseline")
        plt.plot(x, Ecc_block[start:end], linestyle="dashed", label=f"LV LBBB")    
        plt.plot(x, Ecc_ischemia_block[start:end], linestyle="dotted", label=f"LV iLBBB")

        plt.xlabel("Time point", fontsize=14)
        plt.ylabel("Strain", fontsize=14)
        plt.legend()
        plt.savefig(os.path.join(IODet["outputfolder"], IODet["comparison_plot_mode"], "Ecc_comparisonind_disease.png"))
        plt.clf()

        # -----------------------------------------------------------
        # ===============     Ecc RV     ============================
        # -----------------------------------------------------------
        RV_Ecc_block = load_npz(os.path.join(IODet["outputfolder"], IODet["directory_block"], "ECC/Ecc_RV_lastcycle.npz"))
        RV_Ecc_ischemia_block = load_npz(os.path.join(IODet["outputfolder"], IODet["directory_ischemia_block"], "ECC/Ecc_RV_lastcycle.npz"))  
       
        plt.figure()
        plt.plot(x, RV_Ecc_baseline[start:end], label="RV Baseline")
        plt.plot(x, RV_Ecc_block[start:end], linestyle="dashed", label=f"RV LBBB")    
        plt.plot(x, RV_Ecc_ischemia_block[start:end], linestyle="dashed", label=f"RV iLBBB") 

        plt.xlabel("Time point", fontsize=14)
        plt.ylabel("Strain", fontsize=14)
        plt.legend()
        plt.savefig(os.path.join(IODet["outputfolder"], IODet["comparison_plot_mode"], "RV_Ecc_comparisonind_disease.png"))
        plt.clf()        

        # -----------------------------------------------------------
        # ===============     Ell LV     ============================
        # -----------------------------------------------------------
        Ell_block = load_npz(os.path.join(IODet["outputfolder"], IODet["directory_block"], "ELL/Ell_LV_lastcycle.npz"))
        Ell_ischemia_block = load_npz(os.path.join(IODet["outputfolder"], IODet["directory_ischemia_block"], "ELL/Ell_LV_lastcycle.npz"))

        plt.figure()
        plt.plot(x, Ell_baseline[start:end], label="LV Baseline")
        plt.plot(x, Ell_block[start:end], linestyle="dashed", label=f"LV LBBB")    
        plt.plot(x, Ell_ischemia_block[start:end], linestyle="dashed", label=f"LV iLBBB") 

        plt.xlabel("Time point", fontsize=14)
        plt.ylabel("Strain", fontsize=14)
        plt.legend()
        plt.savefig(os.path.join(IODet["outputfolder"], IODet["comparison_plot_mode"], "Ell_comparisonind_disease.png"))
        plt.clf()

        # -----------------------------------------------------------
        # ===============     Ell RV     ============================
        # -----------------------------------------------------------
        RV_Ell_block = load_npz(os.path.join(IODet["outputfolder"], IODet["directory_block"], "ELL/Ell_RV_lastcycle.npz"))
        RV_Ell_ischemia_block = load_npz(os.path.join(IODet["outputfolder"], IODet["directory_ischemia_block"], "ELL/Ell_RV_lastcycle.npz"))

        plt.figure()
        plt.plot(x, RV_Ell_baseline[start:end], label="RV Baseline")
        plt.plot(x, RV_Ell_block[start:end], linestyle="dashed", label=f"RV LBBB")    
        plt.plot(x, RV_Ell_ischemia_block[start:end], linestyle="dashed", label=f"RV iLBBB") 

        plt.xlabel("Time point", fontsize=14)
        plt.ylabel("Strain", fontsize=14)
        plt.legend()
        plt.savefig(os.path.join(IODet["outputfolder"], IODet["comparison_plot_mode"], "RV_Ell_comparisonind_disease.png"))
        plt.clf()        

    elif mode == "pacing":
        Ecc_lbbb = load_npz(os.path.join(IODet["outputfolder"], IODet["directory_block"], "ECC/Ecc_LV_lastcycle.npz"))
        Ecc_bivp = load_npz(os.path.join(IODet["outputfolder"], IODet["directory_bivp"], "ECC/Ecc_LV_lastcycle.npz"))
        Ecc_lbbap = load_npz(os.path.join(IODet["outputfolder"], IODet["directory_lbbap"], "ECC/Ecc_LV_lastcycle.npz"))

        plt.figure()
        plt.plot(x, Ecc_baseline[start:end], label="LV Baseline")
        plt.plot(x, Ecc_lbbb[start:end], linestyle="dashed", label=f"LV LBBB")    
        plt.plot(x, Ecc_bivp[start:end], linestyle="dotted", label=f"LV BiVP")
        plt.plot(x, Ecc_lbbap[start:end], linestyle="dotted", label=f"LV LBBAP")

        plt.xlabel("Time point", fontsize=14)
        plt.ylabel("Strain", fontsize=14)
        plt.legend()
        plt.savefig(os.path.join(IODet["outputfolder"], IODet["comparison_plot_mode"], "Ecc_comparisonind_pacing.png"))
        plt.clf()  


        RV_Ecc_lbbb = load_npz(os.path.join(IODet["outputfolder"], IODet["directory_block"], "ECC/Ecc_RV_lastcycle.npz"))
        RV_Ecc_bivp = load_npz(os.path.join(IODet["outputfolder"], IODet["directory_bivp"], "ECC/Ecc_RV_lastcycle.npz"))
        RV_Ecc_lbbap = load_npz(os.path.join(IODet["outputfolder"], IODet["directory_lbbap"], "ECC/Ecc_RV_lastcycle.npz"))
        
        plt.figure()
        plt.plot(x, RV_Ecc_baseline[start:end], label="RV Baseline")
        plt.plot(x, RV_Ecc_lbbb[start:end], linestyle="dashed", label=f"RV LBBB")    
        plt.plot(x, RV_Ecc_bivp[start:end], linestyle="dotted", label=f"RV BiVP")
        plt.plot(x, RV_Ecc_lbbap[start:end], linestyle="dotted", label=f"RV LBBAP")

        plt.xlabel("Time point", fontsize=14)
        plt.ylabel("Strain", fontsize=14)
        plt.legend()
        plt.savefig(os.path.join(IODet["outputfolder"], IODet["comparison_plot_mode"], "RV_Ecc_comparisonind_pacing.png"))
        plt.clf()

        
        Ell_lbbb = load_npz(os.path.join(IODet["outputfolder"], IODet["directory_block"], "ELL/Ell_LV_lastcycle.npz"))
        Ell_bivp = load_npz(os.path.join(IODet["outputfolder"], IODet["directory_bivp"], "ELL/Ell_LV_lastcycle.npz"))
        Ell_lbbap = load_npz(os.path.join(IODet["outputfolder"], IODet["directory_lbbap"], "ELL/Ell_LV_lastcycle.npz"))
        
        plt.figure()        
        plt.plot(x, Ell_baseline[start:end], label="LV Baseline")
        plt.plot(x, Ell_lbbb[start:end], linestyle="dashed", label=f"LV LBBB")    
        plt.plot(x, Ell_bivp[start:end], linestyle="dotted", label=f"LV BiVP")
        plt.plot(x, Ell_lbbap[start:end], linestyle="dotted", label=f"LV LBBAP")

        plt.xlabel("Time point", fontsize=14)
        plt.ylabel("Strain", fontsize=14)
        plt.legend()
        plt.savefig(os.path.join(IODet["outputfolder"], IODet["comparison_plot_mode"], "Ell_comparisonind_pacing.png"))
        plt.clf()        
        

        RV_Ell_lbbb = load_npz(os.path.join(IODet["outputfolder"], IODet["directory_block"], "ELL/Ell_RV_lastcycle.npz"))
        RV_Ell_bivp = load_npz(os.path.join(IODet["outputfolder"], IODet["directory_bivp"], "ELL/Ell_RV_lastcycle.npz"))
        RV_Ell_lbbap = load_npz(os.path.join(IODet["outputfolder"], IODet["directory_lbbap"], "ELL/Ell_RV_lastcycle.npz"))
        
        plt.figure()
        plt.plot(x, RV_Ell_baseline[start:end], label="RV Baseline")
        plt.plot(x, RV_Ell_lbbb[start:end], linestyle="dashed", label=f"RV LBBB")    
        plt.plot(x, RV_Ell_bivp[start:end], linestyle="dotted", label=f"RV BiVP")
        plt.plot(x, RV_Ell_lbbap[start:end], linestyle="dotted", label=f"RV LBBAP")

        plt.xlabel("Time point", fontsize=14)
        plt.ylabel("Strain", fontsize=14)
        plt.legend()
        plt.savefig(os.path.join(IODet["outputfolder"], IODet["comparison_plot_mode"], "RV_Ell_comparisonind_pacing.png"))
        plt.clf()  


    elif mode == "ischemic_pacing":
        Ecc_ilbbb = load_npz(os.path.join(IODet["outputfolder"], IODet["directory_ischemia_block"], "ECC/Ecc_LV_lastcycle.npz"))
        Ecc_ibivp = load_npz(os.path.join(IODet["outputfolder"], IODet["directory_ischemia_bivp"], "ECC/Ecc_LV_lastcycle.npz"))
        Ecc_ilbbap = load_npz(os.path.join(IODet["outputfolder"], IODet["directory_ischemia_lbbap"], "ECC/Ecc_LV_lastcycle.npz"))

        plt.figure()
        plt.plot(x, Ecc_baseline[start:end], label="LV Baseline")
        plt.plot(x, Ecc_ilbbb[start:end], linestyle="dashed", label=f"LV iLBBB")    
        plt.plot(x, Ecc_ibivp[start:end], linestyle="dotted", label=f"LV iBiVP")
        plt.plot(x, Ecc_ilbbap[start:end], linestyle="dotted", label=f"LV iLBBAP")

        plt.xlabel("Time point", fontsize=14)
        plt.ylabel("Strain", fontsize=14)
        plt.legend()
        plt.savefig(os.path.join(IODet["outputfolder"], IODet["comparison_plot_mode"], "Ecc_comparisonind_ipacing.png"))
        plt.clf()  


        RV_Ecc_ilbbb = load_npz(os.path.join(IODet["outputfolder"], IODet["directory_ischemia_block"], "ECC/Ecc_RV_lastcycle.npz"))
        RV_Ecc_ibivp = load_npz(os.path.join(IODet["outputfolder"], IODet["directory_ischemia_bivp"], "ECC/Ecc_RV_lastcycle.npz"))
        RV_Ecc_ilbbap = load_npz(os.path.join(IODet["outputfolder"], IODet["directory_ischemia_lbbap"], "ECC/Ecc_RV_lastcycle.npz"))
        
        plt.figure()
        plt.plot(x, RV_Ecc_baseline[start:end], label="RV Baseline")
        plt.plot(x, RV_Ecc_ilbbb[start:end], linestyle="dashed", label=f"RV iLBBB")    
        plt.plot(x, RV_Ecc_ibivp[start:end], linestyle="dotted", label=f"RV iBiVP")
        plt.plot(x, RV_Ecc_ilbbap[start:end], linestyle="dotted", label=f"RV iLBBAP")

        plt.xlabel("Time point", fontsize=14)
        plt.ylabel("Strain", fontsize=14)
        plt.legend()
        plt.savefig(os.path.join(IODet["outputfolder"], IODet["comparison_plot_mode"], "RV_Ecc_comparisonind_ipacing.png"))
        plt.clf()

        
        Ell_ilbbb = load_npz(os.path.join(IODet["outputfolder"], IODet["directory_ischemia_block"], "ELL/Ell_LV_lastcycle.npz"))
        Ell_ibivp = load_npz(os.path.join(IODet["outputfolder"], IODet["directory_ischemia_bivp"], "ELL/Ell_LV_lastcycle.npz"))
        Ell_ilbbap = load_npz(os.path.join(IODet["outputfolder"], IODet["directory_ischemia_lbbap"], "ELL/Ell_LV_lastcycle.npz"))
        
        plt.figure()        
        plt.plot(x, Ell_baseline[start:end], label="LV Baseline")
        plt.plot(x, Ell_ilbbb[start:end], linestyle="dashed", label=f"LV iLBBB")    
        plt.plot(x, Ell_ibivp[start:end], linestyle="dotted", label=f"LV iBiVP")
        plt.plot(x, Ell_ilbbap[start:end], linestyle="dotted", label=f"LV iLBBAP")

        plt.xlabel("Time point", fontsize=14)
        plt.ylabel("Strain", fontsize=14)
        plt.legend()
        plt.savefig(os.path.join(IODet["outputfolder"], IODet["comparison_plot_mode"], "Ell_comparisonind_ipacing.png"))
        plt.clf()        
        

        RV_Ell_ilbbb = load_npz(os.path.join(IODet["outputfolder"], IODet["directory_ischemia_block"], "ELL/Ell_RV_lastcycle.npz"))
        RV_Ell_ibivp = load_npz(os.path.join(IODet["outputfolder"], IODet["directory_ischemia_bivp"], "ELL/Ell_RV_lastcycle.npz"))
        RV_Ell_ilbbap = load_npz(os.path.join(IODet["outputfolder"], IODet["directory_ischemia_lbbap"], "ELL/Ell_RV_lastcycle.npz"))
        
        plt.figure()
        plt.plot(x, RV_Ell_baseline[start:end], label="RV Baseline")
        plt.plot(x, RV_Ell_ilbbb[start:end], linestyle="dashed", label=f"RV iLBBB")    
        plt.plot(x, RV_Ell_ibivp[start:end], linestyle="dotted", label=f"RV iBiVP")
        plt.plot(x, RV_Ell_ilbbap[start:end], linestyle="dotted", label=f"RV iLBBAP")

        plt.xlabel("Time point", fontsize=14)
        plt.ylabel("Strain", fontsize=14)
        plt.legend()
        plt.savefig(os.path.join(IODet["outputfolder"], IODet["comparison_plot_mode"], "RV_Ell_comparisonind_ipacing.png"))
        plt.clf() 


def extract_terminal_points_from_PJ_mesh(IODet, SimDet, coord_cols=(0, 1, 2)):
    """
    Load PJ terminal node coordinates from CSV.

    By default assumes columns:
        x, y, z

    If your CSV is:
        id, x, y, z
    then call with:
        coord_cols=(1, 2, 3)
    """

    csv_path = os.path.join(
        IODet["directory_pj"],
        SimDet["pj_tnodes"]
    )

    if not os.path.exists(csv_path):
        raise RuntimeError(f"Could not find PJ terminal CSV: {csv_path}")

    data = np.genfromtxt(csv_path, delimiter=",", comments="#")

    # Remove header/empty rows if genfromtxt produced NaNs
    if data.ndim == 1:
        data = data.reshape(1, -1)

    data = data[~np.isnan(data).all(axis=1)]

    if data.shape[1] <= max(coord_cols):
        raise RuntimeError(
            f"CSV has only {data.shape[1]} columns, but coord_cols={coord_cols}. "
            "Check whether the file is x,y,z or id,x,y,z."
        )

    terminal_points = data[:, coord_cols]

    print(f"Loaded PJ terminal CSV: {csv_path}")
    print(f"Number of PJ terminal points: {terminal_points.shape[0]}")
    print("First few terminal points:")
    print(terminal_points[:5, :])

    return terminal_points    


def count_PJ_terminals_by_segments(IODet, SimDet, terminal_points,
                                   R_apex=1.0,
                                   write_csv=True):
    """
    Count PJ terminal nodes in:
        LV + septum AHA segments 1-17
        RV segments 1-8

    Assumes BiV material IDs:
        0 = LV
        1 = septum
        2 = RV
        3 = annulus/base

    If a PJ terminal maps to matid 3, it is reassigned to the
    closest valid LV/septum AHA or RV segment.
    """

    isLV = SimDet.get("isLV", False)
    isBiV = SimDet.get("isBiV", False)

    if isLV:
        mesh_params = {
            "directory": IODet["directory_me"],
            "casename": IODet["casename_me"],
            "outputfolder": IODet["outputfolder"],
            "foldername": IODet["folderName"],
            "isLV": isLV,
        }
        Mesh_obj = lv_mechanics_mesh(mesh_params, SimDet)

    elif isBiV:
        mesh_params = {
            "directory": IODet["directory_me"],
            "casename": IODet["casename_me"],
            "casename_marked": IODet["casename_me"],
            "outputfolder": IODet["outputfolder"],
            "foldername": IODet["folderName"],
            "isBiV": isBiV,
        }
        Mesh_obj = biv_mechanics_mesh(mesh_params, SimDet)

    else:
        raise RuntimeError("Neither SimDet['isLV'] nor SimDet['isBiV'] is True.")

    mesh = Mesh_obj.mesh
    mesh.init()

    terminal_points = np.asarray(terminal_points)

    LV_ID = 0
    SEPTUM_ID = 1
    RV_ID = 2
    ANNULUS_ID = 3

    if isBiV:
        matid = Mesh_obj.matid
    else:
        matid = None

    # ------------------------------------------------------------
    # AHA tags for LV + septum
    # ------------------------------------------------------------
    aha_raw = tag_AHA_segments(
        mesh,
        apex_axis="z",
        R_apex=R_apex,
        reference_point=SimDet["HIS"],
        rotate=True,
        plot_debug=False
    )

    if isBiV:
        aha_segments = df.MeshFunction("size_t", mesh, mesh.topology().dim())
        aha_segments.set_all(0)

        for cell in df.cells(mesh):
            cidx = cell.index()
            m = int(matid[cidx])
            tag = int(aha_raw[cidx])

            if m in [LV_ID, SEPTUM_ID] and 1 <= tag <= 17:
                aha_segments[cidx] = tag
            else:
                aha_segments[cidx] = 0
    else:
        aha_segments = aha_raw

    # ------------------------------------------------------------
    # RV 8 tags
    # ------------------------------------------------------------
    if isBiV:
        rv_segments = tag_RV_8_segments(
            mesh,
            matid,
            rv_id=RV_ID,
            flip_order=False
        )
    else:
        rv_segments = None

    # ------------------------------------------------------------
    # Bounding box tree
    # ------------------------------------------------------------
    tree = mesh.bounding_box_tree()
    n_cells = mesh.num_cells()

    # ------------------------------------------------------------
    # Precompute valid AHA and RV cell midpoints.
    # Used when a terminal maps to matid 3 annulus/base.
    # ------------------------------------------------------------
    aha_cell_points = []
    aha_cell_tags = []

    rv_cell_points = []
    rv_cell_tags = []

    for cell in df.cells(mesh):
        cidx = cell.index()
        mp = cell.midpoint()

        if isBiV:
            m_cell = int(matid[cidx])

            aha_tag_cell = int(aha_segments[cidx])
            if m_cell in [LV_ID, SEPTUM_ID] and 1 <= aha_tag_cell <= 17:
                aha_cell_points.append([mp.x(), mp.y(), mp.z()])
                aha_cell_tags.append(aha_tag_cell)

            rv_tag_cell = int(rv_segments[cidx])
            if m_cell == RV_ID and 1 <= rv_tag_cell <= 8:
                rv_cell_points.append([mp.x(), mp.y(), mp.z()])
                rv_cell_tags.append(rv_tag_cell)

        else:
            aha_tag_cell = int(aha_segments[cidx])
            if 1 <= aha_tag_cell <= 17:
                aha_cell_points.append([mp.x(), mp.y(), mp.z()])
                aha_cell_tags.append(aha_tag_cell)

    aha_cell_points = np.asarray(aha_cell_points)
    aha_cell_tags = np.asarray(aha_cell_tags, dtype=int)

    rv_cell_points = np.asarray(rv_cell_points)
    rv_cell_tags = np.asarray(rv_cell_tags, dtype=int)

    if len(aha_cell_points) == 0:
        raise RuntimeError("No valid LV/septum AHA cells found for annulus remapping.")

    if isBiV and len(rv_cell_points) == 0:
        raise RuntimeError("No valid RV segment cells found for annulus remapping.")

    def closest_AHA_or_RV_segment_to_point(p):
        """
        For annulus/base terminals, decide whether the terminal is closer
        to LV/septum AHA myocardium or RV myocardium.

        Returns:
            region_type, tag
            region_type = "AHA" or "RV"
        """

        p = np.asarray(p, dtype=float)

        d2_aha = np.sum((aha_cell_points - p)**2, axis=1)
        idx_aha = int(np.argmin(d2_aha))
        min_d2_aha = d2_aha[idx_aha]
        aha_tag = int(aha_cell_tags[idx_aha])

        if not isBiV:
            return "AHA", aha_tag

        d2_rv = np.sum((rv_cell_points - p)**2, axis=1)
        idx_rv = int(np.argmin(d2_rv))
        min_d2_rv = d2_rv[idx_rv]
        rv_tag = int(rv_cell_tags[idx_rv])

        if min_d2_aha <= min_d2_rv:
            return "AHA", aha_tag
        else:
            return "RV", rv_tag

    # ------------------------------------------------------------
    # Count terminal points
    # ------------------------------------------------------------
    aha_counts = np.zeros(17, dtype=int)
    rv_counts = np.zeros(8, dtype=int)

    outside_count = 0
    unassigned_count = 0

    terminal_assignments = []
    unassigned_by_matid = {}
    assigned_by_matid = {}

    annulus_to_aha_count = 0
    annulus_to_rv_count = 0

    annulus_to_aha_segments = np.zeros(17, dtype=int)
    annulus_to_rv_segments = np.zeros(8, dtype=int)

    for i, p in enumerate(terminal_points):
        point = df.Point(float(p[0]), float(p[1]), float(p[2]))

        cell_id = tree.compute_first_entity_collision(point)

        # PJ terminals may lie exactly on/just outside endocardium.
        # If not found inside a cell, assign to nearest cell.
        if cell_id >= n_cells:
            try:
                cell_id = tree.compute_closest_entity(point)[0]
            except Exception:
                outside_count += 1
                terminal_assignments.append([i, p[0], p[1], p[2], -1, -1, 0, 0])
                continue

        if cell_id < 0 or cell_id >= n_cells:
            outside_count += 1
            terminal_assignments.append([i, p[0], p[1], p[2], -1, -1, 0, 0])
            continue

        if isBiV:
            m = int(matid[cell_id])
            assigned_by_matid[m] = assigned_by_matid.get(m, 0) + 1

            if m in [LV_ID, SEPTUM_ID]:
                aha_tag = int(aha_segments[cell_id])
                rv_tag = 0

                if 1 <= aha_tag <= 17:
                    aha_counts[aha_tag - 1] += 1
                else:
                    unassigned_count += 1
                    unassigned_by_matid[m] = unassigned_by_matid.get(m, 0) + 1

            elif m == RV_ID:
                aha_tag = 0
                rv_tag = int(rv_segments[cell_id])

                if 1 <= rv_tag <= 8:
                    rv_counts[rv_tag - 1] += 1
                else:
                    unassigned_count += 1
                    unassigned_by_matid[m] = unassigned_by_matid.get(m, 0) + 1

            elif m == ANNULUS_ID:
                region_type, seg_tag = closest_AHA_or_RV_segment_to_point(p)

                if region_type == "AHA":
                    aha_tag = seg_tag
                    rv_tag = 0

                    aha_counts[aha_tag - 1] += 1
                    annulus_to_aha_segments[aha_tag - 1] += 1
                    annulus_to_aha_count += 1

                elif region_type == "RV":
                    aha_tag = 0
                    rv_tag = seg_tag

                    rv_counts[rv_tag - 1] += 1
                    annulus_to_rv_segments[rv_tag - 1] += 1
                    annulus_to_rv_count += 1

                else:
                    aha_tag = 0
                    rv_tag = 0
                    unassigned_count += 1
                    unassigned_by_matid[m] = unassigned_by_matid.get(m, 0) + 1

            else:
                aha_tag = 0
                rv_tag = 0
                unassigned_count += 1
                unassigned_by_matid[m] = unassigned_by_matid.get(m, 0) + 1

        else:
            m = LV_ID
            aha_tag = int(aha_segments[cell_id])
            rv_tag = 0

            if 1 <= aha_tag <= 17:
                aha_counts[aha_tag - 1] += 1
            else:
                unassigned_count += 1
                unassigned_by_matid[m] = unassigned_by_matid.get(m, 0) + 1

        terminal_assignments.append([
            i, p[0], p[1], p[2], cell_id, m, aha_tag, rv_tag
        ])

    # ------------------------------------------------------------
    # Print diagnostics
    # ------------------------------------------------------------
    print("PJ terminal counts by AHA segment:")
    for seg in range(1, 18):
        print(f"  AHA {seg:2d}: {aha_counts[seg - 1]}")

    if isBiV:
        print("PJ terminal counts by RV segment:")
        for seg in range(1, 9):
            print(f"  RV {seg:2d}: {rv_counts[seg - 1]}")

    print(f"Outside mesh count    : {outside_count}")
    print(f"Unassigned count      : {unassigned_count}")
    print(f"Total terminal points : {len(terminal_points)}")

    print("PJ terminal closest-cell material counts:")
    for m, count in sorted(assigned_by_matid.items()):
        print(f"  matid {m}: {count}")

    print("Unassigned PJ terminals by material ID:")
    for m, count in sorted(unassigned_by_matid.items()):
        print(f"  matid {m}: {count}")

    print(f"Annulus/base terminals assigned to closest AHA segment: {annulus_to_aha_count}")
    print(f"Annulus/base terminals assigned to closest RV segment  : {annulus_to_rv_count}")

    if annulus_to_aha_count > 0:
        print("Annulus/base terminals redistributed to AHA segments:")
        for seg in range(1, 18):
            c = annulus_to_aha_segments[seg - 1]
            if c > 0:
                print(f"  AHA {seg:2d}: {c}")

    if annulus_to_rv_count > 0:
        print("Annulus/base terminals redistributed to RV segments:")
        for seg in range(1, 9):
            c = annulus_to_rv_segments[seg - 1]
            if c > 0:
                print(f"  RV {seg:2d}: {c}")

    assigned_total = int(np.sum(aha_counts) + np.sum(rv_counts))
    print(f"Assigned total        : {assigned_total}")
    print(f"Check total           : {assigned_total + outside_count + unassigned_count}")

    # ------------------------------------------------------------
    # Save CSV files
    # ------------------------------------------------------------
    outdir = os.path.join(IODet["outputfolder"], IODet["caseID"])
    os.makedirs(outdir, exist_ok=True)

    if write_csv:
        summary_path = os.path.join(outdir, "PJ_terminal_counts_by_segments.csv")

        with open(summary_path, "w") as f:
            f.write("type,segment,count\n")

            for seg in range(1, 18):
                f.write(f"AHA,{seg},{aha_counts[seg - 1]}\n")

            if isBiV:
                for seg in range(1, 9):
                    f.write(f"RV,{seg},{rv_counts[seg - 1]}\n")

            f.write(f"outside,0,{outside_count}\n")
            f.write(f"unassigned,0,{unassigned_count}\n")
            f.write(f"annulus_to_AHA,0,{annulus_to_aha_count}\n")
            f.write(f"annulus_to_RV,0,{annulus_to_rv_count}\n")

        detail_path = os.path.join(outdir, "PJ_terminal_segment_assignments.csv")

        with open(detail_path, "w") as f:
            f.write("terminal_id,x,y,z,cell_id,matid,aha_segment,rv_segment\n")

            for row in terminal_assignments:
                f.write(
                    f"{row[0]},{row[1]},{row[2]},{row[3]},"
                    f"{row[4]},{row[5]},{row[6]},{row[7]}\n"
                )

        print(f"Saved: {summary_path}")
        print(f"Saved: {detail_path}")

    return aha_counts, rv_counts, terminal_assignments

def plot_AHA_terminal_bullseye(aha_counts, outdir,
                               filename="PJ_terminals_AHA_bullseye.png"):
    """
    Bullseye plot for PJ terminal counts in AHA 1-17.
    """

    aha_counts = np.asarray(aha_counts, dtype=float)

    if aha_counts.size != 17:
        raise ValueError("aha_counts must have length 17.")

    os.makedirs(outdir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"aspect": "equal"})
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.axis("off")

    r_apex = 0.25
    r_apical = 0.50
    r_mid = 0.75
    r_basal = 1.00

    vmin = np.nanmin(aha_counts)
    vmax = np.nanmax(aha_counts)

    if vmax == vmin:
        vmax = vmin + 1.0

    cmap = plt.cm.viridis
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

    def add_wedge(seg_id, r_inner, r_outer, theta1, theta2):
        value = aha_counts[seg_id - 1]

        wedge = Wedge(
            center=(0, 0),
            r=r_outer,
            theta1=theta1,
            theta2=theta2,
            width=r_outer - r_inner,
            facecolor=cmap(norm(value)),
            edgecolor="black",
            linewidth=1.0
        )

        ax.add_patch(wedge)

        theta_mid = np.deg2rad(0.5 * (theta1 + theta2))
        r_text = 0.5 * (r_inner + r_outer)

        ax.text(
            r_text * np.cos(theta_mid),
            r_text * np.sin(theta_mid),
            f"{seg_id}\n{int(value)}",
            ha="center",
            va="center",
            fontsize=9
        )

    basal_angles = {
        1: (60, 120),
        2: (120, 180),
        3: (180, 240),
        4: (240, 300),
        5: (300, 360),
        6: (0, 60),
    }

    mid_angles = {
        7: (60, 120),
        8: (120, 180),
        9: (180, 240),
        10: (240, 300),
        11: (300, 360),
        12: (0, 60),
    }

    apical_angles = {
        13: (45, 135),
        14: (135, 225),
        15: (225, 315),
        16: (315, 405),
    }

    for seg_id, angles in basal_angles.items():
        add_wedge(seg_id, r_mid, r_basal, angles[0], angles[1])

    for seg_id, angles in mid_angles.items():
        add_wedge(seg_id, r_apical, r_mid, angles[0], angles[1])

    for seg_id, angles in apical_angles.items():
        add_wedge(seg_id, r_apex, r_apical, angles[0], angles[1])

    value = aha_counts[16]

    circle = plt.Circle(
        (0, 0),
        r_apex,
        facecolor=cmap(norm(value)),
        edgecolor="black",
        linewidth=1.0
    )

    ax.add_patch(circle)

    ax.text(
        0, 0,
        f"17\n{int(value)}",
        ha="center",
        va="center",
        fontsize=9
    )

    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("PJ terminal count")

    ax.set_title("PJ terminal nodes by LV/septum AHA segment")

    outpath = os.path.join(outdir, filename)
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {outpath}")

    return outpath

def plot_RV8_terminal_segments(rv_counts, outdir,
                               filename="PJ_terminals_RV8_segments.png"):
    """
    Schematic plot for PJ terminal counts in RV 1-8.
    """

    rv_counts = np.asarray(rv_counts, dtype=float)

    if rv_counts.size != 8:
        raise ValueError("rv_counts must have length 8.")

    os.makedirs(outdir, exist_ok=True)

    data = np.array([
        [rv_counts[0], rv_counts[1], rv_counts[2]],
        [rv_counts[3], rv_counts[4], rv_counts[5]],
        [rv_counts[6], np.nan,       rv_counts[7]],
    ])

    labels = np.array([
        ["1\nBasal inf.", "2\nBasal lat.", "3\nBasal ant."],
        ["4\nMid inf.",   "5\nMid lat.",   "6\nMid ant."],
        ["7\nApical inf.", "",             "8\nApical ant."],
    ])

    fig, ax = plt.subplots(figsize=(7, 5))

    # cmap = plt.cm.viridis.copy()
    cmap = mpl.cm.get_cmap("viridis")
    cmap.set_bad(color="white")

    im = ax.imshow(data, cmap=cmap)

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if np.isfinite(data[i, j]):
                ax.text(
                    j, i,
                    f"{labels[i, j]}\n{int(data[i, j])}",
                    ha="center",
                    va="center",
                    fontsize=9
                )

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("PJ terminal nodes by RV segment")

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("PJ terminal count")

    outpath = os.path.join(outdir, filename)
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {outpath}")

    return outpath        

def bullseye_plot_terminal_nodes(IODet, SimDet, R_apex=1.0,
                                 coord_cols=(0, 1, 2)):
    """
    Load PJ terminal nodes, split into LBB/RBB using LV/RV endocardial VTU surfaces,
    count LBB by AHA and RBB by RV segments, then plot.
    """
    terminal_points = load_PJ_terminal_points_from_SimDetails(
        IODet,
        SimDet,
        coord_cols=coord_cols
    )

    lv_endo_vtu = os.path.join(IODet["directory_ep"], "lv_endo.vtu")
    rv_endo_vtu = os.path.join(IODet["directory_ep"], "rv_endo.vtu")

    lv_endo_points = load_vtu_points(lv_endo_vtu)
    rv_endo_points = load_vtu_points(rv_endo_vtu)

    # First point in pj_fiber_biv.csv is His
    his_point = terminal_points[0, :]

    # For AHA, His should be between segments 2 and 3 = 180 deg
    terminal_points_rot, his_point_rot, theta_shift = rotate_points_xy_to_his_angle(
        terminal_points,
        his_point,
        target_angle_deg=180.0
    )

    lv_endo_points_rot, _, _ = rotate_points_xy_to_his_angle(
        lv_endo_points,
        his_point,
        target_angle_deg=180.0
    )

    rv_endo_points_rot, _, _ = rotate_points_xy_to_his_angle(
        rv_endo_points,
        his_point,
        target_angle_deg=180.0
    )    

    lbb_aha_counts, rbb_rv_counts, terminal_assignments = count_PJ_terminals_LBB_RBB_by_segments(
        IODet=IODet,
        SimDet=SimDet,
        terminal_points=terminal_points_rot,
        lv_endo_points=lv_endo_points_rot,
        rv_endo_points=rv_endo_points_rot,
        R_apex=R_apex,
        write_csv=True
    )

    outdir = os.path.join(IODet["outputfolder"], IODet["caseID"])
    os.makedirs(outdir, exist_ok=True)

    plot_LV_AHA_reference_style(
        lbb_aha_counts,
        outdir,
        filename="PJ_terminals_LBB_AHA_reference_style.png",
        title="LBB terminal distribution by LV/septum AHA segment"
    )

    plot_BiV_terminal_bullseye(
        lbb_aha_counts,
        rbb_rv_counts,
        outdir,
        filename="PJ_terminals_LBB_RBB_BiV_bullseye.png"
    )

    return lbb_aha_counts, rbb_rv_counts, terminal_assignments    

def plot_BiV_terminal_bullseye(aha_counts, rv_counts, outdir,
                               filename="PJ_terminals_BiV_bullseye.png"):
    """
    Combined LV AHA bullseye + fake RV bullseye plot for PJ terminal counts.

    LV/septum:
        AHA 1-17

    RV:
        1 = basal inferior
        2 = basal lateral
        3 = basal anterior
        4 = mid inferior
        5 = mid lateral
        6 = mid anterior
        7 = apical inferior
        8 = apical anterior
    """

    aha_counts = np.asarray(aha_counts, dtype=float)
    rv_counts = np.asarray(rv_counts, dtype=float)

    if aha_counts.size != 17:
        raise ValueError("aha_counts must have length 17.")

    if rv_counts.size != 8:
        raise ValueError("rv_counts must have length 8.")

    os.makedirs(outdir, exist_ok=True)

    # Use one shared color scale for LV and RV counts
    # Use separate color scales for LV/LBB and RV/RBB
    cmap_lv = mpl.cm.get_cmap("viridis")
    cmap_rv = mpl.cm.get_cmap("plasma")

    vmin_lv = np.nanmin(aha_counts)
    vmax_lv = np.nanmax(aha_counts)

    vmin_rv = np.nanmin(rv_counts)
    vmax_rv = np.nanmax(rv_counts)

    if vmax_lv == vmin_lv:
        vmax_lv = vmin_lv + 1.0

    if vmax_rv == vmin_rv:
        vmax_rv = vmin_rv + 1.0

    norm_lv = mpl.colors.Normalize(vmin=vmin_lv, vmax=vmax_lv)
    norm_rv = mpl.colors.Normalize(vmin=vmin_rv, vmax=vmax_rv)

    fig, ax = plt.subplots(figsize=(10, 7), subplot_kw={"aspect": "equal"})
    ax.axis("off")

    # ============================================================
    # LV AHA bullseye
    # ============================================================
    lv_center = np.array([0.0, 0.0])

    r_apex = 0.25
    r_apical = 0.50
    r_mid = 0.75
    r_basal = 1.00

    def add_lv_wedge(seg_id, r_inner, r_outer, theta1, theta2):
        value = aha_counts[seg_id - 1]

        wedge = Wedge(
            center=lv_center,
            r=r_outer,
            theta1=theta1,
            theta2=theta2,
            width=r_outer - r_inner,
            facecolor=cmap_lv(norm_lv(value)),
            edgecolor="black",
            linewidth=1.0
        )

        ax.add_patch(wedge)

        theta_mid = np.deg2rad(0.5 * (theta1 + theta2))
        r_text = 0.5 * (r_inner + r_outer)

        x = lv_center[0] + r_text * np.cos(theta_mid)
        y = lv_center[1] + r_text * np.sin(theta_mid)

        ax.text(
            x, y,
            f"{seg_id}\n{int(value)}",
            ha="center",
            va="center",
            fontsize=8
        )

    basal_angles = {
        1: (60, 120),
        2: (120, 180),
        3: (180, 240),
        4: (240, 300),
        5: (300, 360),
        6: (0, 60),
    }

    mid_angles = {
        7: (60, 120),
        8: (120, 180),
        9: (180, 240),
        10: (240, 300),
        11: (300, 360),
        12: (0, 60),
    }

    apical_angles = {
        13: (45, 135),
        14: (135, 225),
        15: (225, 315),
        16: (315, 405),
    }

    for seg_id, angles in basal_angles.items():
        add_lv_wedge(seg_id, r_mid, r_basal, angles[0], angles[1])

    for seg_id, angles in mid_angles.items():
        add_lv_wedge(seg_id, r_apical, r_mid, angles[0], angles[1])

    for seg_id, angles in apical_angles.items():
        add_lv_wedge(seg_id, r_apex, r_apical, angles[0], angles[1])

    # LV apex segment 17
    value = aha_counts[16]

    circle = plt.Circle(
        lv_center,
        r_apex,
        facecolor=cmap_lv(norm_lv(value)),
        edgecolor="black",
        linewidth=1.0
    )

    ax.add_patch(circle)

    ax.text(
        lv_center[0], lv_center[1],
        f"17\n{int(value)}",
        ha="center",
        va="center",
        fontsize=8
    )


    rv_center = lv_center
    rv_rotation = 30.0
    rv_theta_min = 60.0 + rv_rotation
    rv_theta_max = 240.0 + rv_rotation

    # Start exactly at the LV outer radius
    rv_r_inner = r_basal
    rv_r_apex = 1.18
    rv_r_mid = 1.38
    rv_r_basal = 1.58

    def add_rv_wedge(seg_id, r_inner, r_outer, theta1, theta2, label):
        value = rv_counts[seg_id - 1]

        wedge = Wedge(
            center=rv_center,
            r=r_outer,
            theta1=theta1,
            theta2=theta2,
            width=r_outer - r_inner,
            facecolor=cmap_rv(norm_rv(value)),
            edgecolor="black",
            linewidth=1.0
        )

        ax.add_patch(wedge)

        theta_mid = np.deg2rad(0.5 * (theta1 + theta2))
        r_text = 0.5 * (r_inner + r_outer)

        x = rv_center[0] + r_text * np.cos(theta_mid)
        y = rv_center[1] + r_text * np.sin(theta_mid)

        ax.text(
            x, y,
            f"{seg_id}\n{int(value)}",
            ha="center",
            va="center",
            fontsize=8
        )

    # Three sectors aligned with LV AHA 1, 2, 3
    th3 = np.linspace(rv_theta_min, rv_theta_max, 4)

    # Basal RV: outer crescent ring
    add_rv_wedge(1, rv_r_mid, rv_r_basal, th3[0], th3[1], "Basal inf.")
    add_rv_wedge(2, rv_r_mid, rv_r_basal, th3[1], th3[2], "Basal lat.")
    add_rv_wedge(3, rv_r_mid, rv_r_basal, th3[2], th3[3], "Basal ant.")

    # Mid RV: middle crescent ring
    add_rv_wedge(4, rv_r_apex, rv_r_mid, th3[0], th3[1], "Mid inf.")
    add_rv_wedge(5, rv_r_apex, rv_r_mid, th3[1], th3[2], "Mid lat.")
    add_rv_wedge(6, rv_r_apex, rv_r_mid, th3[2], th3[3], "Mid ant.")

    # Apical RV: inner crescent ring, split into two sectors
    th2 = np.linspace(rv_theta_min, rv_theta_max, 3)

    add_rv_wedge(7, rv_r_inner, rv_r_apex, th2[0], th2[1], "Apical inf.")
    add_rv_wedge(8, rv_r_inner, rv_r_apex, th2[1], th2[2], "Apical ant.")

    sm_lv = mpl.cm.ScalarMappable(cmap=cmap_lv, norm=norm_lv)
    sm_lv.set_array([])

    sm_rv = mpl.cm.ScalarMappable(cmap=cmap_rv, norm=norm_rv)
    sm_rv.set_array([])

    cbar_lv = plt.colorbar(sm_lv, ax=ax, fraction=0.030, pad=0.02)
    cbar_lv.set_label("LBB terminal count")

    cbar_rv = plt.colorbar(sm_rv, ax=ax, fraction=0.030, pad=0.02)
    cbar_rv.set_label("RBB terminal count")    

    ax.set_title("PJ terminal node distribution by BiV segment", fontsize=16)

    ax.set_xlim(-1.75, 1.35)
    ax.set_ylim(-1.75, 1.75)

    outpath = os.path.join(outdir, filename)
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {outpath}")

    return outpath    

def load_vtu_points(vtu_path):
    """
    Load points from a .vtu file as an N x 3 numpy array.
    """

    if not os.path.exists(vtu_path):
        raise RuntimeError(f"Could not find VTU file: {vtu_path}")

    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(vtu_path)
    reader.Update()

    grid = reader.GetOutput()
    pts = grid.GetPoints()

    if pts is None:
        raise RuntimeError(f"No points found in VTU file: {vtu_path}")

    npts = pts.GetNumberOfPoints()
    points = np.zeros((npts, 3))

    for i in range(npts):
        points[i, :] = pts.GetPoint(i)

    print(f"Loaded VTU points: {vtu_path}")
    print(f"  Number of points: {npts}")
    print("  First few points:")
    print(points[:5, :])

    return points    


def count_PJ_terminals_LBB_RBB_by_segments(IODet, SimDet,
                                           terminal_points,
                                           lv_endo_points,
                                           rv_endo_points,
                                           R_apex=1.0,
                                           write_csv=True):
    """
    Split PJ terminal nodes into LBB/RBB using closest LV/RV endocardial VTU surface,
    then count:
        LBB terminals -> LV/septum AHA 1-17
        RBB terminals -> RV 1-8
    """

    isBiV = SimDet.get("isBiV", False)
    if not isBiV:
        raise RuntimeError("This LBB/RBB split function is intended for BiV only.")

    mesh_params = {
        "directory": IODet["directory_me"],
        "casename": IODet["casename_me"],
        "casename_marked": IODet["casename_me"],
        "outputfolder": IODet["outputfolder"],
        "foldername": IODet["folderName"],
        "isBiV": isBiV,
    }

    Mesh_obj = biv_mechanics_mesh(mesh_params, SimDet)

    mesh = Mesh_obj.mesh
    mesh.init()

    matid = Mesh_obj.matid

    terminal_points = np.asarray(terminal_points, dtype=float)
    lv_endo_points = np.asarray(lv_endo_points, dtype=float)
    rv_endo_points = np.asarray(rv_endo_points, dtype=float)

    LV_ID = 0
    SEPTUM_ID = 1
    RV_ID = 2

    # ------------------------------------------------------------
    # AHA tags for LV + septum
    # ------------------------------------------------------------
    aha_raw = tag_AHA_segments(
        mesh,
        apex_axis="z",
        R_apex=R_apex,
        reference_point=SimDet["HIS"],
        rotate=True,
        plot_debug=False
    )

    aha_segments = df.MeshFunction("size_t", mesh, mesh.topology().dim())
    aha_segments.set_all(0)

    for cell in df.cells(mesh):
        cidx = cell.index()
        m = int(matid[cidx])
        tag = int(aha_raw[cidx])

        if m in [LV_ID, SEPTUM_ID] and 1 <= tag <= 17:
            aha_segments[cidx] = tag
        else:
            aha_segments[cidx] = 0

    # ------------------------------------------------------------
    # RV 8 tags
    # ------------------------------------------------------------
    rv_segments = tag_RV_8_segments(
        mesh,
        matid,
        rv_id=RV_ID,
        flip_order=False
    )

    # ------------------------------------------------------------
    # Precompute AHA and RV regional cell midpoints
    # ------------------------------------------------------------
    aha_cell_points = []
    aha_cell_tags = []

    rv_cell_points = []
    rv_cell_tags = []

    for cell in df.cells(mesh):
        cidx = cell.index()
        m = int(matid[cidx])
        mp = cell.midpoint()

        aha_tag = int(aha_segments[cidx])
        if m in [LV_ID, SEPTUM_ID] and 1 <= aha_tag <= 17:
            aha_cell_points.append([mp.x(), mp.y(), mp.z()])
            aha_cell_tags.append(aha_tag)

        rv_tag = int(rv_segments[cidx])
        if m == RV_ID and 1 <= rv_tag <= 8:
            rv_cell_points.append([mp.x(), mp.y(), mp.z()])
            rv_cell_tags.append(rv_tag)

    aha_cell_points = np.asarray(aha_cell_points, dtype=float)
    aha_cell_tags = np.asarray(aha_cell_tags, dtype=int)

    rv_cell_points = np.asarray(rv_cell_points, dtype=float)
    rv_cell_tags = np.asarray(rv_cell_tags, dtype=int)

    if len(aha_cell_points) == 0:
        raise RuntimeError("No valid LV/septum AHA cells found.")

    if len(rv_cell_points) == 0:
        raise RuntimeError("No valid RV segment cells found.")

    def closest_dist2_to_surface(p, surface_points):
        d2 = np.sum((surface_points - p)**2, axis=1)
        idx = int(np.argmin(d2))
        return float(d2[idx])

    def closest_AHA_segment(p):
        d2 = np.sum((aha_cell_points - p)**2, axis=1)
        idx = int(np.argmin(d2))
        return int(aha_cell_tags[idx]), float(d2[idx])

    def closest_RV_segment(p):
        d2 = np.sum((rv_cell_points - p)**2, axis=1)
        idx = int(np.argmin(d2))
        return int(rv_cell_tags[idx]), float(d2[idx])

    # ------------------------------------------------------------
    # Split terminals into LBB/RBB and count regions
    # ------------------------------------------------------------
    lbb_aha_counts = np.zeros(17, dtype=int)
    rbb_rv_counts = np.zeros(8, dtype=int)

    n_lbb = 0
    n_rbb = 0

    terminal_assignments = []

    for i, p in enumerate(terminal_points):
        p = np.asarray(p, dtype=float)

        d2_lv_endo = closest_dist2_to_surface(p, lv_endo_points)
        d2_rv_endo = closest_dist2_to_surface(p, rv_endo_points)

        if d2_lv_endo <= d2_rv_endo:
            branch = "LBB"

            aha_tag, d2_region = closest_AHA_segment(p)
            rv_tag = 0

            lbb_aha_counts[aha_tag - 1] += 1
            n_lbb += 1

        else:
            branch = "RBB"

            aha_tag = 0
            rv_tag, d2_region = closest_RV_segment(p)

            rbb_rv_counts[rv_tag - 1] += 1
            n_rbb += 1

        terminal_assignments.append([
            i,
            p[0], p[1], p[2],
            branch,
            np.sqrt(d2_lv_endo),
            np.sqrt(d2_rv_endo),
            aha_tag,
            rv_tag,
            np.sqrt(d2_region)
        ])

    # ------------------------------------------------------------
    # Print diagnostics
    # ------------------------------------------------------------
    print("PJ terminal split by closest endocardial surface:")
    print(f"  LBB terminals: {n_lbb}")
    print(f"  RBB terminals: {n_rbb}")
    print(f"  Total        : {n_lbb + n_rbb}")
    print(f"  Input total  : {len(terminal_points)}")

    print("LBB terminal counts by AHA segment:")
    for seg in range(1, 18):
        print(f"  AHA {seg:2d}: {lbb_aha_counts[seg - 1]}")

    print("RBB terminal counts by RV segment:")
    for seg in range(1, 9):
        print(f"  RV {seg:2d}: {rbb_rv_counts[seg - 1]}")

    # ------------------------------------------------------------
    # Save CSV
    # ------------------------------------------------------------
    outdir = os.path.join(IODet["outputfolder"], IODet["caseID"])
    os.makedirs(outdir, exist_ok=True)

    if write_csv:
        summary_path = os.path.join(outdir, "PJ_terminal_counts_LBB_RBB_by_segments.csv")

        with open(summary_path, "w") as f:
            f.write("branch,type,segment,count\n")

            for seg in range(1, 18):
                f.write(f"LBB,AHA,{seg},{lbb_aha_counts[seg - 1]}\n")

            for seg in range(1, 9):
                f.write(f"RBB,RV,{seg},{rbb_rv_counts[seg - 1]}\n")

        detail_path = os.path.join(outdir, "PJ_terminal_assignments_LBB_RBB.csv")

        with open(detail_path, "w") as f:
            f.write(
                "terminal_id,x,y,z,branch,"
                "dist_to_LV_endo,dist_to_RV_endo,"
                "aha_segment,rv_segment,dist_to_assigned_region\n"
            )

            for row in terminal_assignments:
                f.write(
                    f"{row[0]},{row[1]},{row[2]},{row[3]},"
                    f"{row[4]},{row[5]},{row[6]},"
                    f"{row[7]},{row[8]},{row[9]}\n"
                )

        print(f"Saved: {summary_path}")
        print(f"Saved: {detail_path}")

    return lbb_aha_counts, rbb_rv_counts, terminal_assignments

def plot_LV_AHA_reference_style(aha_counts, outdir,
                                filename="PJ_terminals_LBB_AHA_reference_style.png",
                                title="LBB terminal distribution by LV/septum AHA segment"):
    """
    LV/septum AHA 17-segment bullseye plot in a clinical/reference style.

    Uses the existing AHA 17 counts:
        basal:  1-6
        mid:    7-12
        apical: 13-16
        apex:   17

    Anatomical orientation:
        anterior = top
        septal   = right
        posterior= bottom
        lateral  = left
    """
    aha_counts = np.asarray(aha_counts, dtype=float)

    if aha_counts.size != 17:
        raise ValueError("aha_counts must have length 17.")

    os.makedirs(outdir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.5, 7.5), subplot_kw={"aspect": "equal"})
    ax.axis("off")

    # ------------------------------------------------------------
    # Radii
    # ------------------------------------------------------------
    r_apex = 0.26
    r_apical = 0.52
    r_mid = 0.76
    r_basal = 1.00

    # ------------------------------------------------------------
    # Color scale
    # ------------------------------------------------------------
    vmin = np.nanmin(aha_counts)
    vmax = np.nanmax(aha_counts)

    if vmax == vmin:
        vmax = vmin + 1.0

    cmap = mpl.cm.get_cmap("Reds")
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

    # ------------------------------------------------------------
    # Helper
    # ------------------------------------------------------------
    def add_wedge(seg_id, r_inner, r_outer, theta1, theta2):
        value = aha_counts[seg_id - 1]

        wedge = Wedge(
            center=(0, 0),
            r=r_outer,
            theta1=theta1,
            theta2=theta2,
            width=r_outer - r_inner,
            facecolor=cmap(norm(value)),
            edgecolor="black",
            linewidth=1.0
        )

        ax.add_patch(wedge)

        theta_mid = np.deg2rad(0.5 * (theta1 + theta2))
        r_text = 0.5 * (r_inner + r_outer)

        x = r_text * np.cos(theta_mid)
        y = r_text * np.sin(theta_mid)

        ax.text(
            x, y,
            f"{seg_id}\n{int(value)}",
            ha="center",
            va="center",
            fontsize=9,
            color="black"
        )

    # ------------------------------------------------------------
    # AHA geometry
    # Same as your current function:
    # anterior is top, septal is right, posterior bottom, lateral left.
    # ------------------------------------------------------------
    basal_angles = {
        1: (60, 120),
        2: (120, 180),
        3: (180, 240),
        4: (240, 300),
        5: (300, 360),
        6: (0, 60),
    }

    mid_angles = {
        7: (60, 120),
        8: (120, 180),
        9: (180, 240),
        10: (240, 300),
        11: (300, 360),
        12: (0, 60),
    }

    apical_angles = {
        13: (45, 135),
        14: (135, 225),
        15: (225, 315),
        16: (315, 405),
    }

    for seg_id, angles in basal_angles.items():
        add_wedge(seg_id, r_mid, r_basal, angles[0], angles[1])

    for seg_id, angles in mid_angles.items():
        add_wedge(seg_id, r_apical, r_mid, angles[0], angles[1])

    for seg_id, angles in apical_angles.items():
        add_wedge(seg_id, r_apex, r_apical, angles[0], angles[1])

    # Apex segment 17
    value = aha_counts[16]

    circle = plt.Circle(
        (0, 0),
        r_apex,
        facecolor=cmap(norm(value)),
        edgecolor="black",
        linewidth=1.0
    )

    ax.add_patch(circle)

    ax.text(
        0, 0,
        f"17\n{int(value)}",
        ha="center",
        va="center",
        fontsize=9,
        color="black"
    )

    # ------------------------------------------------------------
    # Anatomical labels like reference image
    # ------------------------------------------------------------
    ax.text(0.0, 1.18, "ANTERIOR",
            ha="center", va="center", fontsize=15, fontweight="bold")

    ax.text(1.25, 0.0, "SEPTAL",
            ha="center", va="center", fontsize=15, fontweight="bold")

    ax.text(0.0, -1.18, "POSTERIOR",
            ha="center", va="center", fontsize=15, fontweight="bold")

    ax.text(-1.25, 0.0, "LATERAL",
            ha="center", va="center", fontsize=15, fontweight="bold")

    # Ring labels
    ax.text(
        1.38, 1.08,
        "BASE: 1–6\nMIDDLE: 7–12\nAPEX: 13–17",
        ha="left",
        va="top",
        fontsize=11,
        fontweight="bold"
    )

    # ------------------------------------------------------------
    # Colorbar
    # ------------------------------------------------------------
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.08)
    cbar.set_label("LBB terminal count")

    ax.set_title(title, fontsize=14)

    ax.set_xlim(-1.55, 1.85)
    ax.set_ylim(-1.45, 1.35)

    outpath = os.path.join(outdir, filename)
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {outpath}")

    return outpath    

def tag_LV_His14_segments(mesh, matid=None, lv_ids=[0, 1],
                          reference_point=None,
                          apex_axis="z",
                          R_apex=1.0,
                          rotate=True,
                          plot_debug=False):
    """
    His-centered LV/septum 14-segment scheme.

    Segments:
        Base:   11-15, 5 sectors
        Middle: 16-20, 5 sectors
        Apex:   21-24, 4 sectors

    Orientation after rotation:
        His direction = septal center = 0 degrees
        Anterior      = 90 degrees
        Lateral       = 180 degrees
        Posterior     = 270 degrees

    If matid is provided, only cells with matid in lv_ids are tagged.
    Other cells are assigned 0.
    """

    cell_tags = df.MeshFunction("size_t", mesh, mesh.topology().dim())
    cell_tags.set_all(0)

    coords = []
    cell_indices = []

    for cell in df.cells(mesh):
        cidx = cell.index()

        if matid is not None:
            m = int(matid[cidx])
            if m not in lv_ids:
                continue

        mp = cell.midpoint()
        coords.append([mp.x(), mp.y(), mp.z()])
        cell_indices.append(cidx)

    coords = np.asarray(coords)

    if coords.shape[0] == 0:
        raise RuntimeError("No LV/septum cells found for His-centered LV14 segmentation.")

    x_all = coords[:, 0]
    y_all = coords[:, 1]
    z_all = coords[:, 2]

    if apex_axis == "z":
        long_coord = z_all
    elif apex_axis == "x":
        long_coord = x_all
    elif apex_axis == "y":
        long_coord = y_all
    else:
        raise ValueError("apex_axis must be 'x', 'y', or 'z'.")

    z_min = np.min(long_coord)
    z_max = np.max(long_coord)

    basal_th = z_min + 2.0 / 3.0 * (z_max - z_min)
    mid_th = z_min + 1.0 / 3.0 * (z_max - z_min)

    theta_shift = 0.0

    if reference_point is not None and rotate:
        his = np.asarray(reference_point).flatten()
        theta_his = np.degrees(np.arctan2(his[1], his[0])) % 360.0

        # His becomes septal center = 0 degrees
        theta_shift = (0.0 - theta_his) % 360.0

        print("His-centered LV14 segmentation:")
        print(f"  theta_his   = {theta_his:.2f} deg")
        print(f"  theta_shift = {theta_shift:.2f} deg")
        print("  His is placed at septal center = 0 deg")

    for ii, cidx in enumerate(cell_indices):
        x = x_all[ii]
        y = y_all[ii]
        z = long_coord[ii]

        theta = np.degrees(np.arctan2(y, x)) % 360.0
        theta_rot = (theta + theta_shift) % 360.0

        if z >= basal_th:
            # 5 basal sectors
            # 13: 18-90
            # 14: 90-162
            # 15: 162-234
            # 11: 234-306  posterior
            # 12: 306-378  septal/His side, contains 0 deg
            sector = int(((theta_rot - 18.0) % 360.0) // 72.0)
            tag_map = [13, 14, 15, 11, 12]
            tag = tag_map[sector]

        elif z >= mid_th:
            # 5 middle sectors
            # 18: 18-90
            # 19: 90-162
            # 20: 162-234
            # 16: 234-306  posterior
            # 17: 306-378  septal/His side, contains 0 deg
            sector = int(((theta_rot - 18.0) % 360.0) // 72.0)
            tag_map = [18, 19, 20, 16, 17]
            tag = tag_map[sector]

        else:
            # 4 apical sectors
            # 22: -45-45    septal/His
            # 23: 45-135    anterior
            # 24: 135-225   lateral
            # 21: 225-315   posterior
            sector = int(((theta_rot - 45.0) % 360.0) // 90.0)
            tag_map = [23, 24, 21, 22]
            tag = tag_map[sector]

    return cell_tags  

def count_PJ_terminals_LBB_RBB_LV14_by_segments(IODet, SimDet,
                                                terminal_points,
                                                lv_endo_points,
                                                rv_endo_points,
                                                R_apex=1.0,
                                                write_csv=True):
    """
    Split PJ terminal nodes into LBB/RBB using closest LV/RV endocardial VTU surface,
    then count:
        LBB terminals -> His-centered LV14 segments 11-24
        RBB terminals -> RV segments 1-8

    LV14 angular convention:
        0 deg   = septal / His direction
        90 deg  = anterior
        180 deg = lateral
        270 deg = posterior

    Base/mid 5-sector geometry:
        12 / 17 : 306-18   septal / His side
        13 / 18 : 18-90    anterior-septal
        14 / 19 : 90-162   anterior-lateral
        15 / 20 : 162-234  lateral
        11 / 16 : 234-306  posterior

    Apex 4-sector geometry:
        22 : 315-45      septal / His side
        23 : 45-135      anterior
        24 : 135-225     lateral
        21 : 225-315     posterior
    """

    isBiV = SimDet.get("isBiV", False)

    if not isBiV:
        raise RuntimeError("This function is intended for BiV only.")

    mesh_params = {
        "directory": IODet["directory_me"],
        "casename": IODet["casename_me"],
        "casename_marked": IODet["casename_me"],
        "outputfolder": IODet["outputfolder"],
        "foldername": IODet["folderName"],
        "isBiV": isBiV,
    }

    Mesh_obj = biv_mechanics_mesh(mesh_params, SimDet)

    mesh = Mesh_obj.mesh
    mesh.init()

    matid = Mesh_obj.matid

    terminal_points = np.asarray(terminal_points, dtype=float)
    lv_endo_points = np.asarray(lv_endo_points, dtype=float)
    rv_endo_points = np.asarray(rv_endo_points, dtype=float)

    RV_ID = 2

    # ------------------------------------------------------------
    # His rotation
    # ------------------------------------------------------------
    his = np.asarray(SimDet["HIS"], dtype=float).flatten()

    theta_his = np.degrees(np.arctan2(his[1], his[0])) % 360.0
    theta_shift = 0.0

    print("Direct LV14 assignment:")
    print(f"  theta_his   = {theta_his:.2f} deg")
    print(f"  theta_shift = {theta_shift:.2f} deg")
    print("  His is placed at 0 deg inside the septal sector.")

    # ------------------------------------------------------------
    # Long-axis split from LV endocardial points
    # ------------------------------------------------------------
    lv_z = lv_endo_points[:, 2]

    z_min = float(np.min(lv_z))
    z_max = float(np.max(lv_z))

    mid_th = z_min + (1.0 / 3.0) * (z_max - z_min)
    basal_th = z_min + (2.0 / 3.0) * (z_max - z_min)

    print("LV14 z thresholds from LV endocardial VTU:")
    print(f"  z_min    = {z_min:.4f}")
    print(f"  mid_th   = {mid_th:.4f}")
    print(f"  basal_th = {basal_th:.4f}")
    print(f"  z_max    = {z_max:.4f}")

    def assign_LV14_segment_from_point(p):
        """
        Assign one terminal point to His-centered LV14 segment.
        """

        x = float(p[0])
        y = float(p[1])
        z = float(p[2])

        theta = np.degrees(np.arctan2(y, x)) % 360.0
        theta_rot = (theta + theta_shift) % 360.0

        if z >= basal_th:
            # 5 basal sectors:
            # sector 0: 18-90
            # sector 1: 90-162
            # sector 2: 162-234
            # sector 3: 234-306 posterior
            # sector 4: 306-18 septal/His
            sector = int(((theta_rot - 18.0) % 360.0) // 72.0)
            tag_map = [13, 14, 15, 11, 12]
            tag = tag_map[sector]
            ring = "base"

        elif z >= mid_th:
            # 5 middle sectors.
            sector = int(((theta_rot - 18.0) % 360.0) // 72.0)
            tag_map = [18, 19, 20, 16, 17]
            tag = tag_map[sector]
            ring = "middle"

        else:
            # 4 apical sectors:
            # sector 0: 45-135 anterior
            # sector 1: 135-225 lateral
            # sector 2: 225-315 posterior
            # sector 3: 315-45 septal/His
            sector = int(((theta_rot - 45.0) % 360.0) // 90.0)
            tag_map = [23, 24, 21, 22]
            tag = tag_map[sector]
            ring = "apex"

        return tag, theta_rot, ring

    # ------------------------------------------------------------
    # RV 8 tags from mechanics mesh
    # ------------------------------------------------------------
    rv_segments = tag_RV_8_segments(
        mesh,
        matid,
        rv_id=RV_ID,
        flip_order=False
    )

    rv_cell_points = []
    rv_cell_tags = []

    mat_values = []
    rv_values = []

    for cell in df.cells(mesh):
        cidx = cell.index()
        m = int(matid[cidx])
        rv_tag = int(rv_segments[cidx])

        mat_values.append(m)
        rv_values.append(rv_tag)

        if m == RV_ID and 1 <= rv_tag <= 8:
            mp = cell.midpoint()
            rv_cell_points.append([mp.x(), mp.y(), mp.z()])
            rv_cell_tags.append(rv_tag)

    rv_cell_points = np.asarray(rv_cell_points, dtype=float)
    rv_cell_tags = np.asarray(rv_cell_tags, dtype=int)

    mat_values = np.asarray(mat_values, dtype=int)
    rv_values = np.asarray(rv_values, dtype=int)

    print("Material IDs found in mechanics mesh:")
    for val in np.unique(mat_values):
        print(f"  matid {val}: {np.sum(mat_values == val)} cells")

    print("RV tags found:")
    for val in np.unique(rv_values):
        print(f"  RV tag {val}: {np.sum(rv_values == val)} cells")

    if len(rv_cell_points) == 0:
        raise RuntimeError("No valid RV segment cells found.")

    def closest_dist2_to_surface(p, surface_points):
        d2 = np.sum((surface_points - p) ** 2, axis=1)
        idx = int(np.argmin(d2))
        return float(d2[idx])

    def closest_RV_segment(p):
        d2 = np.sum((rv_cell_points - p) ** 2, axis=1)
        idx = int(np.argmin(d2))
        return int(rv_cell_tags[idx]), float(d2[idx])

    # ------------------------------------------------------------
    # Split terminals into LBB/RBB and count regions
    # ------------------------------------------------------------
    lbb_lv14_counts = np.zeros(14, dtype=int)
    rbb_rv_counts = np.zeros(8, dtype=int)

    n_lbb = 0
    n_rbb = 0

    terminal_assignments = []

    for i, p in enumerate(terminal_points):
        p = np.asarray(p, dtype=float)

        d2_lv_endo = closest_dist2_to_surface(p, lv_endo_points)
        d2_rv_endo = closest_dist2_to_surface(p, rv_endo_points)

        if d2_lv_endo <= d2_rv_endo:
            branch = "LBB"

            lv14_tag, theta_rot, lv14_ring = assign_LV14_segment_from_point(p)
            rv_tag = 0
            d2_region = 0.0

            lbb_lv14_counts[lv14_tag - 11] += 1
            n_lbb += 1

        else:
            branch = "RBB"

            lv14_tag = 0
            theta_rot = np.nan
            lv14_ring = "none"

            rv_tag, d2_region = closest_RV_segment(p)

            rbb_rv_counts[rv_tag - 1] += 1
            n_rbb += 1

        terminal_assignments.append([
            i,
            p[0], p[1], p[2],
            branch,
            np.sqrt(d2_lv_endo),
            np.sqrt(d2_rv_endo),
            lv14_tag,
            rv_tag,
            theta_rot,
            lv14_ring,
            np.sqrt(d2_region)
        ])

    # ------------------------------------------------------------
    # Print diagnostics
    # ------------------------------------------------------------
    print("PJ terminal split by closest endocardial surface:")
    print(f"  LBB terminals: {n_lbb}")
    print(f"  RBB terminals: {n_rbb}")
    print(f"  Total        : {n_lbb + n_rbb}")
    print(f"  Input total  : {len(terminal_points)}")

    print("LBB terminal counts by His-centered LV14 segment:")
    for seg in range(11, 25):
        print(f"  LV14 {seg:2d}: {lbb_lv14_counts[seg - 11]}")

    print("RBB terminal counts by RV segment:")
    for seg in range(1, 9):
        print(f"  RV {seg:2d}: {rbb_rv_counts[seg - 1]}")

    assigned_total = int(np.sum(lbb_lv14_counts) + np.sum(rbb_rv_counts))
    print(f"Assigned total        : {assigned_total}")
    print(f"Check total           : {assigned_total}")

    # ------------------------------------------------------------
    # Save CSV
    # ------------------------------------------------------------
    outdir = os.path.join(IODet["outputfolder"], IODet["caseID"])
    os.makedirs(outdir, exist_ok=True)

    if write_csv:
        summary_path = os.path.join(
            outdir,
            "PJ_terminal_counts_LBB_RBB_LV14_by_segments.csv"
        )

        with open(summary_path, "w") as f:
            f.write("branch,type,segment,count\n")

            for seg in range(11, 25):
                f.write(f"LBB,LV14,{seg},{lbb_lv14_counts[seg - 11]}\n")

            for seg in range(1, 9):
                f.write(f"RBB,RV,{seg},{rbb_rv_counts[seg - 1]}\n")

        detail_path = os.path.join(
            outdir,
            "PJ_terminal_assignments_LBB_RBB_LV14.csv"
        )

        with open(detail_path, "w") as f:
            f.write(
                "terminal_id,x,y,z,branch,"
                "dist_to_LV_endo,dist_to_RV_endo,"
                "lv14_segment,rv_segment,theta_rot,lv14_ring,"
                "dist_to_assigned_region\n"
            )

            for row in terminal_assignments:
                f.write(
                    f"{row[0]},{row[1]},{row[2]},{row[3]},"
                    f"{row[4]},{row[5]},{row[6]},"
                    f"{row[7]},{row[8]},{row[9]},"
                    f"{row[10]},{row[11]}\n"
                )

        print(f"Saved: {summary_path}")
        print(f"Saved: {detail_path}")

    return lbb_lv14_counts, rbb_rv_counts, terminal_assignments

def plot_BiV_terminal_bullseye_LV14(lv14_counts, rv_counts, outdir,
                                    filename="PJ_terminals_LBB_RBB_LV14_BiV_bullseye.png"):
    """
    Combined His-centered LV14 bullseye + RV8 crescent.

    LV/septum:
        Base:   11-15
        Middle: 16-20
        Apex:   21-24

    RV:
        1-8
    """
    lv14_counts = np.asarray(lv14_counts, dtype=float)
    rv_counts = np.asarray(rv_counts, dtype=float)

    if lv14_counts.size != 14:
        raise ValueError("lv14_counts must have length 14.")

    if rv_counts.size != 8:
        raise ValueError("rv_counts must have length 8.")

    os.makedirs(outdir, exist_ok=True)

    # Separate color scales
    cmap_lv = mpl.cm.viridis
    cmap_rv = mpl.cm.get_cmap("plasma")

    vmin_lv = np.nanmin(lv14_counts)
    vmax_lv = np.nanmax(lv14_counts)
    if vmax_lv == vmin_lv:
        vmax_lv = vmin_lv + 1.0

    vmin_rv = np.nanmin(rv_counts)
    vmax_rv = np.nanmax(rv_counts)
    if vmax_rv == vmin_rv:
        vmax_rv = vmin_rv + 1.0

    norm_lv = mpl.colors.Normalize(vmin=vmin_lv, vmax=vmax_lv)
    norm_rv = mpl.colors.Normalize(vmin=vmin_rv, vmax=vmax_rv)

    fig, ax = plt.subplots(figsize=(10, 7), subplot_kw={"aspect": "equal"})
    ax.axis("off")

    lv_center = np.array([0.0, 0.0])

    r_apex_inner = 0.00
    r_apex_outer = 0.42
    r_mid_outer = 0.72
    r_basal_outer = 1.00

    def lv_value(seg_id):
        return lv14_counts[seg_id - 11]

    def add_lv14_wedge(seg_id, r_inner, r_outer, theta1, theta2):
        value = lv_value(seg_id)

        wedge = Wedge(
            center=lv_center,
            r=r_outer,
            theta1=theta1,
            theta2=theta2,
            width=r_outer - r_inner,
            facecolor=cmap_lv(norm_lv(value)),
            edgecolor="black",
            linewidth=1.0
        )

        ax.add_patch(wedge)

        theta_mid = np.deg2rad(0.5 * (theta1 + theta2))
        r_text = 0.5 * (r_inner + r_outer)

        x = lv_center[0] + r_text * np.cos(theta_mid)
        y = lv_center[1] + r_text * np.sin(theta_mid)

        ax.text(
            x, y,
            f"{seg_id}\n{int(value)}",
            ha="center",
            va="center",
            fontsize=8
        )

    # ------------------------------------------------------------
    # His marker
    # ------------------------------------------------------------
    his_theta = np.deg2rad(0.0)
    his_r = r_basal_outer + 0.08

    his_x = lv_center[0] + his_r * np.cos(his_theta)
    his_y = lv_center[1] + his_r * np.sin(his_theta)

    ax.plot(
        his_x,
        his_y,
        marker="*",
        markersize=16,
        markeredgecolor="black",
        markerfacecolor="yellow",
        zorder=20
    )

    ax.text(
        his_x + 0.10,
        his_y,
        "His",
        ha="left",
        va="center",
        fontsize=10,
        fontweight="bold",
        zorder=21
    )
    base_angles = {
        13: (18, 90),
        14: (90, 162),
        15: (162, 234),
        11: (234, 306),
        12: (306, 378),
    }

    middle_angles = {
        18: (18, 90),
        19: (90, 162),
        20: (162, 234),
        16: (234, 306),
        17: (306, 378),
    }

    apex_angles = {
        23: (45, 135),
        24: (135, 225),
        21: (225, 315),
        22: (315, 405),
    }

    for seg_id, angles in base_angles.items():
        add_lv14_wedge(seg_id, r_mid_outer, r_basal_outer, angles[0], angles[1])

    for seg_id, angles in middle_angles.items():
        add_lv14_wedge(seg_id, r_apex_outer, r_mid_outer, angles[0], angles[1])

    for seg_id, angles in apex_angles.items():
        add_lv14_wedge(seg_id, r_apex_inner, r_apex_outer, angles[0], angles[1])

    # ------------------------------------------------------------
    # Labels like reference plot
    # ------------------------------------------------------------
    ax.text(0.0, 1.78, "ANTERIOR", ha="center", va="center",
            fontsize=13, fontweight="bold")
    ax.text(1.82, 0.0, "SEPTAL / HIS", ha="center", va="center",
            fontsize=13, fontweight="bold", rotation=90)
    ax.text(0.0, -1.78, "POSTERIOR", ha="center", va="center",
            fontsize=13, fontweight="bold")
    ax.text(-1.82, 0.0, "LATERAL", ha="center", va="center",
            fontsize=13, fontweight="bold", rotation=90)

    ax.text(
        1.35, 1.45,
        "BASE: 11–15\nMIDDLE: 16–20\nAPEX: 21–24",
        ha="left",
        va="top",
        fontsize=10,
        fontweight="bold"
    )

    # ------------------------------------------------------------
    # Colorbars
    # ------------------------------------------------------------
    sm_lv = mpl.cm.ScalarMappable(cmap=cmap_lv, norm=norm_lv)
    sm_lv.set_array([])

    cbar_lv = plt.colorbar(sm_lv, ax=ax, fraction=0.030, pad=0.02)
    cbar_lv.set_label("LBB terminal count")

    ax.set_title("PJ terminal node distribution by LBB/RBB segment", fontsize=15)

    ax.set_xlim(-2.05, 2.35)
    ax.set_ylim(-2.00, 1.95)

    outpath = os.path.join(outdir, filename)
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {outpath}")

    return outpath   

def bullseye_plot_terminal_nodes_LV14(IODet, SimDet, R_apex=1.0,
                                      coord_cols=(0, 1, 2)):
    """
    LV14/RV8 workflow using terminal nodes already stored in SimDet['pj_tnodes'].

    1. Load PJ terminal nodes directly from SimDet['pj_tnodes'].
    2. Load LV/RV endocardial VTU surfaces.
    3. Split each terminal into LBB or RBB by closest endocardial surface.
    4. Count:
           LBB -> His-centered LV14 segments 11-24
           RBB -> RV segments 1-8
    5. Plot LV14 bullseye.
    """

    terminal_points = load_PJ_terminal_points_from_SimDetails(
        IODet,
        SimDet,
        coord_cols=coord_cols
    )

    lv_endo_vtu = os.path.join(IODet["directory_ep"], "lv_endo.vtu")
    rv_endo_vtu = os.path.join(IODet["directory_ep"], "rv_endo.vtu")

    lv_endo_points = load_vtu_points(lv_endo_vtu)
    rv_endo_points = load_vtu_points(rv_endo_vtu)


    # The first point in pj_fiber_biv.csv is the His point
    his_point = terminal_points[0, :]

    # Rotate terminal nodes so His is at 0 degrees
    terminal_points_rot, his_point_rot, theta_shift = rotate_points_xy_to_his_zero(
        terminal_points,
        his_point
    )

    # Rotate LV/RV endocardial points by the same rotation
    lv_endo_points_rot, _, _ = rotate_points_xy_to_his_zero(
        lv_endo_points,
        his_point
    )

    rv_endo_points_rot, _, _ = rotate_points_xy_to_his_zero(
        rv_endo_points,
        his_point
    )

    lbb_lv14_counts, rbb_rv_counts, terminal_assignments = (
        count_PJ_terminals_LBB_RBB_LV14_by_segments(
            IODet=IODet,
            SimDet=SimDet,
            terminal_points=terminal_points,
            lv_endo_points=lv_endo_points,
            rv_endo_points=rv_endo_points,
            R_apex=R_apex,
            write_csv=True
        )
    )

    outdir = os.path.join(IODet["outputfolder"], IODet["caseID"])
    os.makedirs(outdir, exist_ok=True)

    plot_BiV_terminal_bullseye_LV14(
        lbb_lv14_counts,
        rbb_rv_counts,
        outdir,
        filename="PJ_terminals_LBB_LV14_bullseye.png"
    )

    return lbb_lv14_counts, rbb_rv_counts, terminal_assignments   

def load_PJ_terminal_points_from_SimDetails(IODet, SimDet, coord_cols=(0, 1, 2)):
    """
    Load PJ terminal node coordinates directly from SimDet['pj_tnodes'].

    Expected:
        SimDet["pj_tnodes"] = "pj_fiber_biv.csv"

    coord_cols defines which CSV columns contain x,y,z.
    """
    pj_file = SimDet["pj_tnodes"]

    if os.path.isabs(pj_file):
        pj_path = pj_file
    else:
        pj_path = os.path.join(IODet["directory_pj"], pj_file)

    if not os.path.exists(pj_path):
        raise RuntimeError(f"Could not find PJ terminal node file: {pj_path}")

    print(f"Loading PJ terminal nodes directly from: {pj_path}")

    try:
        data = np.loadtxt(pj_path, delimiter=",")
    except Exception:
        data = np.loadtxt(pj_path)

    if data.ndim == 1:
        data = data.reshape(1, -1)

    terminal_points = data[:, coord_cols]

    print(f"Loaded PJ terminal nodes: {terminal_points.shape[0]}")
    print("First few terminal points:")
    print(terminal_points[:5, :])

    return terminal_points    

def rotate_points_xy_to_his_zero(points, his_point):
    """
    Rotate points in the x-y plane so that the His point lies at 0 degrees.

    This changes x/y coordinates but leaves z unchanged.

    After rotation:
        His direction = 0 deg
        anterior      = 90 deg
        lateral       = 180 deg
        posterior     = 270 deg
    """

    points = np.asarray(points, dtype=float)
    his_point = np.asarray(his_point, dtype=float).flatten()

    theta_his = np.arctan2(his_point[1], his_point[0])
    theta_shift = -theta_his

    c = np.cos(theta_shift)
    s = np.sin(theta_shift)

    R = np.array([
        [c, -s],
        [s,  c]
    ])

    points_rot = points.copy()
    points_rot[:, 0:2] = points[:, 0:2].dot(R.T)

    his_rot = his_point.copy()
    his_rot[0:2] = his_point[0:2].dot(R.T)

    print("Rotating coordinates so His is at 0 degrees:")
    print(f"  theta_his original = {np.degrees(theta_his) % 360.0:.2f} deg")
    print(f"  theta_shift        = {np.degrees(theta_shift):.2f} deg")
    print(f"  His original       = {his_point}")
    print(f"  His rotated        = {his_rot}")
    print(f"  His rotated angle  = {np.degrees(np.arctan2(his_rot[1], his_rot[0])) % 360.0:.2f} deg")

    return points_rot, his_rot, theta_shift    


def rotate_points_xy_to_his_angle(points, his_point, target_angle_deg=180.0):
    """
    Rotate points in the x-y plane so that the His point lies at target_angle_deg.

    For AHA:
        target_angle_deg = 180 deg
        His lies between AHA segment 2 and 3.

    For LV14:
        target_angle_deg = 0 deg
        His lies on the septal/right side.
    """

    points = np.asarray(points, dtype=float)
    his_point = np.asarray(his_point, dtype=float).flatten()

    theta_his = np.arctan2(his_point[1], his_point[0])
    theta_target = np.deg2rad(target_angle_deg)

    theta_shift = theta_target - theta_his

    c = np.cos(theta_shift)
    s = np.sin(theta_shift)

    R = np.array([
        [c, -s],
        [s,  c]
    ])

    points_rot = points.copy()
    points_rot[:, 0:2] = points[:, 0:2].dot(R.T)

    his_rot = his_point.copy()
    his_rot[0:2] = his_point[0:2].dot(R.T)

    print("Rotating coordinates:")
    print(f"  theta_his original = {np.degrees(theta_his) % 360.0:.2f} deg")
    print(f"  target angle       = {target_angle_deg:.2f} deg")
    print(f"  theta_shift        = {np.degrees(theta_shift):.2f} deg")
    print(f"  His original       = {his_point}")
    print(f"  His rotated        = {his_rot}")
    print(f"  His rotated angle  = {np.degrees(np.arctan2(his_rot[1], his_rot[0])) % 360.0:.2f} deg")

    return points_rot, his_rot, theta_shift    


def compute_strain_fast(IODet,SimDet,LVid=[0, 1],      # LV free wall + septum by default
    SEPid=1,RVid=2,cycle=None,):
    """
    Compute Ecc, Ell, Err for the last full displacement cycle.

    Reference frame:
        first displacement frame of the last full cycle

    Therefore:
        Ecc[0] ~ 0
        Ell[0] ~ 0
        Err[0] ~ 0
    """

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # -------------------------------------------------
    # Load displacement
    # -------------------------------------------------
    npz_path = os.path.join(
        IODet["outputfolder"],
        IODet["caseID"],
        "ME_u",
        "u_arr.npz"
    )

    if rank == 0:
        print("Reading:", npz_path)
        data = np.load(npz_path, allow_pickle=True)["data"]
        n_steps = len(data)
        print("[Rank 0] loaded {} steps".format(n_steps))
    else:
        data = None
        n_steps = None

    n_steps = comm.bcast(n_steps, root=0)

    # -------------------------------------------------
    # Mesh
    # -------------------------------------------------
    isBiV = SimDet.get("isBiV", False)
    isLV  = SimDet.get("isLV", False)

    mesh_params = {
        "directory": IODet["directory_me"],
        "casename": IODet["casename_me"],
        "casename_marked": IODet["casename_me"],
        "outputfolder": IODet["outputfolder"],
        "foldername": IODet["folderName"],
        "isBiV": isBiV,
        "isLV": isLV,
    }

    if isLV:
        Mesh_obj = lv_mechanics_mesh(mesh_params, SimDet)
    else:
        Mesh_obj = biv_mechanics_mesh(mesh_params, SimDet)

    # -------------------------------------------------
    # Fibers
    # -------------------------------------------------
    fiberfile = os.path.join(IODet["directory_me"], "BiV_fibers.h5")

    VQuadelem = df.VectorElement(
        "Quadrature",
        Mesh_obj.mesh.ufl_cell(),
        degree=4,
        quad_scheme="default"
    )

    fiberCLR = df.FunctionSpace(Mesh_obj.mesh, VQuadelem)

    eC0 = df.Function(fiberCLR)
    eL0 = df.Function(fiberCLR)
    eR0 = df.Function(fiberCLR)

    h5 = df.HDF5File(Mesh_obj.mesh.mpi_comm(), fiberfile, "r")
    h5.read(eC0, "BiV_fibers/eC0")
    h5.read(eL0, "BiV_fibers/eL0")
    h5.read(eR0, "BiV_fibers/eR0")
    h5.close()

    eC0 = eC0 / df.sqrt(df.inner(eC0, eC0))
    eL0 = eL0 / df.sqrt(df.inner(eL0, eL0))
    eR0 = eR0 / df.sqrt(df.inner(eR0, eR0))

    # -------------------------------------------------
    # Displacement space
    # -------------------------------------------------
    var_deg = 1 if SimDet["Mechanics Discretization"] == "P1P1" else 2
    V = df.VectorFunctionSpace(Mesh_obj.mesh, "CG", var_deg)

    udisp = df.Function(V)

    params = {
        "mesh": Mesh_obj.mesh,
        "displacement_variable": udisp,
        "material model": SimDet["GiccioneParams"]["Passive model"],
        "material params": SimDet["GiccioneParams"]["Passive params"],
        "incompressible": SimDet["GiccioneParams"]["incompressible"],
        "growth_tensor": None,
    }

    uflforms = Forms(params)
    Fmat = uflforms.Fmat()

    # -------------------------------------------------
    # Helper for one or multiple material regions
    # -------------------------------------------------
    def dx_sub(mesh_obj, sub_ids):
        if isinstance(sub_ids, list):
            dx_total = mesh_obj.dx(sub_ids[0])
            for mid in sub_ids[1:]:
                dx_total += mesh_obj.dx(mid)
            return dx_total
        return mesh_obj.dx(sub_ids)

    dx_LV = dx_sub(Mesh_obj, LVid)

    # -------------------------------------------------
    # Define last full displacement cycle
    # -------------------------------------------------
    dt_out = SimDet["dt"] * SimDet["writeStep"]

    steps_per_cycle = int(round(
        SimDet["HeartBeatLength"] / dt_out
    ))

    n_full_cycles = n_steps // steps_per_cycle

    if n_full_cycles < 1:
        raise RuntimeError("Not enough frames for one full cycle.")

    last_start = (n_full_cycles - 1) * steps_per_cycle
    last_end   = n_full_cycles * steps_per_cycle

    ED_index = last_start

    if rank == 0:
        print("Using reference: first frame of last full cycle")
        print("dt_out =", dt_out)
        print("steps_per_cycle =", steps_per_cycle)
        print("n_full_cycles =", n_full_cycles)
        print("last full cycle indices =", last_start, "to", last_end)
        print("reference ED_index =", ED_index)

    if rank == 0:
        u_ref_global = data[ED_index]
    else:
        u_ref_global = None

    u_ref_global = comm.bcast(u_ref_global, root=0)

    # -------------------------------------------------
    # Reference displacement function
    # -------------------------------------------------
    u_ref_fun = df.Function(V)

    vec_ref = u_ref_fun.vector()
    lo, hi = vec_ref.local_range()
    vec_ref.set_local(u_ref_global[lo:hi])
    vec_ref.apply("insert")

    # Also initialize current displacement with reference
    vec = udisp.vector()
    lo, hi = vec.local_range()
    vec.set_local(u_ref_global[lo:hi])
    vec.apply("insert")

    # -------------------------------------------------
    # Reference deformation gradient
    # Important: do NOT project Fref
    # -------------------------------------------------
    params_ref = dict(params)
    params_ref["displacement_variable"] = u_ref_fun

    uflforms_ref = Forms(params_ref)
    # Fref = uflforms_ref.Fmat()
    Fref = df.project(uflforms.Fmat(), df.TensorFunctionSpace(Mesh_obj.mesh, "DG", 0))
    # -------------------------------------------------
    # Volumes
    # -------------------------------------------------
    wall_vol_LV = df.assemble(df.Constant(1.0) * dx_LV)

    if isBiV:
        wall_vol_RV = df.assemble(df.Constant(1.0) * Mesh_obj.dx(RVid))

    if isBiV and SEPid is not None:
        wall_vol_SEP = df.assemble(df.Constant(1.0) * Mesh_obj.dx(SEPid))

    # -------------------------------------------------
    # Long-axis orientation diagnostic, once only
    # -------------------------------------------------
    eLong = df.Constant((0.0, 0.0, 1.0))

    dot_eL_long = df.inner(eL0, eLong)

    mean_dot = df.assemble(dot_eL_long * dx_LV) / wall_vol_LV
    mean_abs_dot = df.assemble(
        df.sqrt(dot_eL_long * dot_eL_long + df.Constant(1e-12)) * dx_LV
    ) / wall_vol_LV

    if rank == 0:
        print("LV eL0 alignment with visual long axis:")
        print("  mean dot(eL0, eLong) =", mean_dot)
        print("  mean abs dot(eL0, eLong) =", mean_abs_dot)

    # -------------------------------------------------
    # Storage
    # -------------------------------------------------
    Ecc_arr, Ell_arr, Err_arr = [], [], []
    Ecc_arr_RV, Ell_arr_RV, Err_arr_RV = [], [], []
    Ecc_arr_SEP, Ell_arr_SEP, Err_arr_SEP = [], [], []

    uL_mean_arr, uL_rms_arr = [], []
    uC_rms_arr, uR_rms_arr = [], []
    uZ_mean_arr, uZ_rms_arr = [], []

    # -------------------------------------------------
    # Loop only over last full cycle
    # -------------------------------------------------
    frame_indices = list(range(last_start, last_end))

    for local_i, step in enumerate(frame_indices):

        if rank == 0 and local_i % 20 == 0:
            print("Cycle step {}/{}  global step {}".format(
                local_i,
                len(frame_indices),
                step
            ))

        if rank == 0:
            u_global = data[step]
        else:
            u_global = None

        u_global = comm.bcast(u_global, root=0)

        vec = udisp.vector()
        lo, hi = vec.local_range()
        vec.set_local(u_global[lo:hi])
        vec.apply("insert")

        # Relative deformation gradient
        Frel = Fmat * df.inv(Fref)
        Crel = Frel.T * Frel

        Ccc = df.inner(eC0, Crel * eC0)
        Cll = df.inner(eL0, Crel * eL0)
        Crr = df.inner(eR0, Crel * eR0)

        Ecc = 0.5 * (1.0 - 1.0 / Ccc)
        Ell = 0.5 * (1.0 - 1.0 / Cll)
        Err = 0.5 * (1.0 - 1.0 / Crr)

        # -----------------------------
        # LV / LV+SEP average
        # -----------------------------
        ecc_lv = df.assemble(Ecc * dx_LV) / wall_vol_LV
        ell_lv = df.assemble(Ell * dx_LV) / wall_vol_LV
        err_lv = df.assemble(Err * dx_LV) / wall_vol_LV

        Ecc_arr.append(float(ecc_lv))
        Ell_arr.append(float(ell_lv))
        Err_arr.append(float(err_lv))

        # -----------------------------
        # Reference sanity check
        # -----------------------------
        if local_i == 0 and rank == 0:
            print("Reference-frame strain sanity check:")
            print("  global step =", step)
            print("  Ecc_ref =", ecc_lv)
            print("  Ell_ref =", ell_lv)
            print("  Err_ref =", err_lv)

        # -----------------------------
        # Displacement diagnostics
        # -----------------------------
        du = udisp - u_ref_fun

        uL = df.inner(du, eL0)
        uC = df.inner(du, eC0)
        uR = df.inner(du, eR0)

        uL_mean_arr.append(float(df.assemble(uL * dx_LV) / wall_vol_LV))
        uL_rms_arr.append(float(np.sqrt(df.assemble(uL * uL * dx_LV) / wall_vol_LV)))
        uC_rms_arr.append(float(np.sqrt(df.assemble(uC * uC * dx_LV) / wall_vol_LV)))
        uR_rms_arr.append(float(np.sqrt(df.assemble(uR * uR * dx_LV) / wall_vol_LV)))

        uLong = df.inner(du, eLong)

        uZ_mean_arr.append(float(df.assemble(uLong * dx_LV) / wall_vol_LV))
        uZ_rms_arr.append(float(np.sqrt(df.assemble(uLong * uLong * dx_LV) / wall_vol_LV)))

        # -----------------------------
        # RV average
        # -----------------------------
        if isBiV:
            Ecc_arr_RV.append(float(df.assemble(Ecc * Mesh_obj.dx(RVid)) / wall_vol_RV))
            Ell_arr_RV.append(float(df.assemble(Ell * Mesh_obj.dx(RVid)) / wall_vol_RV))
            Err_arr_RV.append(float(df.assemble(Err * Mesh_obj.dx(RVid)) / wall_vol_RV))

        # -----------------------------
        # Septum average
        # -----------------------------
        if isBiV and SEPid is not None:
            Ecc_arr_SEP.append(float(df.assemble(Ecc * Mesh_obj.dx(SEPid)) / wall_vol_SEP))
            Ell_arr_SEP.append(float(df.assemble(Ell * Mesh_obj.dx(SEPid)) / wall_vol_SEP))
            Err_arr_SEP.append(float(df.assemble(Err * Mesh_obj.dx(SEPid)) / wall_vol_SEP))

    # -------------------------------------------------
    # Save and print results
    # -------------------------------------------------
    if rank == 0:

        base = os.path.join(IODet["outputfolder"], IODet["caseID"])
        os.makedirs(base, exist_ok=True)

        for subdir in ["ECC", "ELL", "ERR"]:
            os.makedirs(os.path.join(base, subdir), exist_ok=True)

        t_cycle = np.arange(len(Ecc_arr)) * dt_out

        def save_plot(y, name):
            plt.figure()
            plt.plot(t_cycle, y)
            plt.xlabel("Time [ms]")
            plt.ylabel(name)
            plt.savefig(os.path.join(base, "{}.png".format(name)), dpi=300)
            plt.close()

        def save_region_outputs(prefix, Ecc_data, Ell_data, Err_data):

            Ecc_data = np.asarray(Ecc_data)
            Ell_data = np.asarray(Ell_data)
            Err_data = np.asarray(Err_data)

            save_plot(Ecc_data, "ECC/Ecc_{}_lastcycle".format(prefix))
            save_plot(Ell_data, "ELL/Ell_{}_lastcycle".format(prefix))
            save_plot(Err_data, "ERR/Err_{}_lastcycle".format(prefix))

            np.savez(
                os.path.join(base, "ECC", "Ecc_{}_lastcycle.npz".format(prefix)),
                Ecc_data
            )
            np.savez(
                os.path.join(base, "ELL", "Ell_{}_lastcycle.npz".format(prefix)),
                Ell_data
            )
            np.savez(
                os.path.join(base, "ERR", "Err_{}_lastcycle.npz".format(prefix)),
                Err_data
            )

            i_ecc = int(np.argmin(Ecc_data))
            i_ell = int(np.argmin(Ell_data))
            i_err = int(np.argmax(Err_data))

            print("-------------------------------------------------")
            print(prefix, "last full cycle strain")
            print("-------------------------------------------------")
            print("{} Ecc peak: {} at {} ms".format(
                prefix, Ecc_data[i_ecc], i_ecc * dt_out
            ))
            print("{} Ell peak: {} at {} ms".format(
                prefix, Ell_data[i_ell], i_ell * dt_out
            ))
            print("{} Err peak: {} at {} ms".format(
                prefix, Err_data[i_err], i_err * dt_out
            ))

            print("{} Ecc first point: {}".format(prefix, Ecc_data[0]))
            print("{} Ell first point: {}".format(prefix, Ell_data[0]))
            print("{} Err first point: {}".format(prefix, Err_data[0]))

        save_region_outputs("LV", Ecc_arr, Ell_arr, Err_arr)

        if isBiV:
            save_region_outputs("RV", Ecc_arr_RV, Ell_arr_RV, Err_arr_RV)

        if isBiV and SEPid is not None:
            save_region_outputs("SEP", Ecc_arr_SEP, Ell_arr_SEP, Err_arr_SEP)

        # -------------------------------------------------
        # Save LV displacement diagnostics
        # -------------------------------------------------
        Ecc_cycle = np.asarray(Ecc_arr)
        Ell_cycle = np.asarray(Ell_arr)
        Err_cycle = np.asarray(Err_arr)

        uL_mean_cycle = np.asarray(uL_mean_arr)
        uL_rms_cycle  = np.asarray(uL_rms_arr)
        uC_rms_cycle  = np.asarray(uC_rms_arr)
        uR_rms_cycle  = np.asarray(uR_rms_arr)
        uZ_mean_cycle = np.asarray(uZ_mean_arr)
        uZ_rms_cycle  = np.asarray(uZ_rms_arr)

        i_ecc = int(np.argmin(Ecc_cycle))
        i_ell = int(np.argmin(Ell_cycle))

        np.savez(
            os.path.join(base, "LV_displacement_diagnostics_lastcycle.npz"),
            time_ms=t_cycle,
            Ecc=Ecc_cycle,
            Ell=Ell_cycle,
            Err=Err_cycle,
            uL_mean=uL_mean_cycle,
            uL_rms=uL_rms_cycle,
            uC_rms=uC_rms_cycle,
            uR_rms=uR_rms_cycle,
            uLong_mean=uZ_mean_cycle,
            uLong_rms=uZ_rms_cycle,
            reference_index=ED_index,
            last_start=last_start,
            last_end=last_end,
        )
                

def compute_strain_AHA_fast(IODet,SimDet,LVid=[0, 1],RVid=2, cycle=None,ED_reference="max_volume_from_PV",ED_index_ref=None,PV_file=None,PV_LV_col=2,n_loading_rows=25,R_apex=1.0,debug_HIS_orientation=False,):
    """
    AHA strain calculation for BiV.

    Outputs:
        Ecc_AHA_lastcycle.npz
        Ell_AHA_lastcycle.npz
        Ecc_Ell_AHA_SDI_summary.txt
        AHA_segments_used_for_strain.pvd

    Important:
        - No dx_aha segment assembly is used.
        - AHA averaging is done with DG0 vector assembly + NumPy grouping.
        - SDI uses the paper definition:

          SDI = std(t_peak_strain_i - t_systole) / t_RR * 100

        - AHA segments 1-16 are used for SDI.
        - Segment 17 is saved but excluded from SDI.
        - SDI_Ecc is the value comparable to the paper table.
    """

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    ffc_fast = {
        "representation": "uflacs",
        "quadrature_degree": 4,
    }

    # ------------------------------------------------------------
    # Helper: MPI-safe AHA tagging
    # ------------------------------------------------------------
    def tag_AHA_segments_local(mesh, SimDet, R_apex=1.0):
        mesh.init()

        aha = df.MeshFunction("size_t", mesh, mesh.topology().dim())
        aha.set_all(0)

        coords = mesh.coordinates()

        if coords.shape[0] > 0:
            local_z_min = np.min(coords[:, 2])
            local_z_max = np.max(coords[:, 2])
        else:
            local_z_min = 1.0e30
            local_z_max = -1.0e30

        z_min = comm.allreduce(local_z_min, op=MPI.MIN)
        z_max = comm.allreduce(local_z_max, op=MPI.MAX)

        h = z_max - z_min

        basal_th = z_max - h / 3.0
        mid_th = z_max - 2.0 * h / 3.0

        theta_shift = 0.0

        if "HIS" in SimDet:
            his = np.asarray(SimDet["HIS"]).flatten()
            theta_his = np.degrees(np.arctan2(his[1], his[0])) % 360.0

            # HIS becomes theta_rotated = 180 deg.
            # With this sector formula, this is the boundary between AHA 2 and 3.
            theta_shift = (180.0 - theta_his) % 360.0

        for cell in df.cells(mesh):
            mp = cell.midpoint()

            x = mp.x()
            y = mp.y()
            z = mp.z()

            theta = np.degrees(np.arctan2(y, x)) % 360.0
            theta_rotated = (theta + theta_shift) % 360.0

            if z >= basal_th:
                tag = int((theta_rotated - 60.0) % 360.0 // 60.0) + 1

            elif z >= mid_th:
                tag = int((theta_rotated - 60.0) % 360.0 // 60.0) + 7

            else:
                tag = int((theta_rotated - 45.0) % 360.0 // 90.0) + 13

            if np.linalg.norm([x, y, z - z_min]) < R_apex:
                tag = 17

            aha[cell] = tag

        return aha

    # ------------------------------------------------------------
    # Helper: optional HIS debug text file
    # ------------------------------------------------------------
    def write_HIS_debug(mesh, SimDet, aha_segments, matid, outdir):
        if rank == 0:
            os.makedirs(outdir, exist_ok=True)

        comm.Barrier()

        his = np.asarray(SimDet["HIS"]).flatten()
        theta_his = np.degrees(np.arctan2(his[1], his[0])) % 360.0
        theta_shift = (180.0 - theta_his) % 360.0
        theta_his_rotated = (theta_his + theta_shift) % 360.0

        local_best_dist = 1.0e30
        local_best_cell = -1
        local_best_tag = -1

        for cell in df.cells(mesh):
            use_cell = True

            if matid is not None:
                m = int(matid[cell])
                if m not in [0, 1]:
                    use_cell = False

            if use_cell:
                mp = cell.midpoint()
                p = np.array([mp.x(), mp.y(), mp.z()])
                dist = np.linalg.norm(p - his)

                if dist < local_best_dist:
                    local_best_dist = dist
                    local_best_cell = cell.index()
                    local_best_tag = int(aha_segments[cell])

        gathered = comm.gather(
            (local_best_dist, rank, local_best_cell, local_best_tag),
            root=0,
        )

        if rank == 0:
            best_dist, best_rank, best_cell, best_tag = min(
                gathered,
                key=lambda x: x[0],
            )

            debug_path = os.path.join(outdir, "AHA_HIS_debug_summary.txt")

            with open(debug_path, "w") as f:
                f.write("HIS/AHA orientation debug\n")
                f.write("Expected: HIS should lie on AHA segment 2/3 boundary\n")
                f.write("Expected theta_his_rotated_deg = 180\n\n")
                f.write("HIS_x\t{:.8f}\n".format(his[0]))
                f.write("HIS_y\t{:.8f}\n".format(his[1]))
                f.write("HIS_z\t{:.8f}\n".format(his[2]))
                f.write("theta_his_raw_deg\t{:.8f}\n".format(theta_his))
                f.write("theta_shift_deg\t{:.8f}\n".format(theta_shift))
                f.write("theta_his_rotated_deg\t{:.8f}\n".format(theta_his_rotated))
                f.write("closest_rank\t{}\n".format(best_rank))
                f.write("closest_cell\t{}\n".format(best_cell))
                f.write("closest_AHA_tag\t{}\n".format(best_tag))
                f.write("closest_distance\t{:.8f}\n".format(best_dist))

            print("Saved HIS debug:", debug_path)

    # ------------------------------------------------------------
    # Helper: paper-style t_systole from PV
    # ------------------------------------------------------------
    def get_t_systole_from_PV_ED_to_ED(
        PV_file,
        PV_LV_col,
        n_steps_u,
        dt,
        writeStep,
        BCL,
        n_loading_rows=25,
    ):
        pv = np.genfromtxt(PV_file)

        if pv.ndim == 1:
            raise ValueError("PV file loaded as 1D. Check BiV_PV.txt format.")

        dt = float(dt)
        BCL = float(BCL)
        writeStep = int(round(writeStep))
        n_steps_u = int(n_steps_u)

        steps_per_cycle_u = int(round(BCL / (dt * writeStep)))
        steps_per_cycle_pv = int(round(BCL / dt))

        V_raw = pv[:, PV_LV_col]
        V_pv = V_raw[n_loading_rows:]

        n_u_complete_cycles = n_steps_u // steps_per_cycle_u
        n_pv_complete_cycles = len(V_pv) // steps_per_cycle_pv

        if n_u_complete_cycles < 1:
            raise ValueError("No complete displacement cycle found.")

        if n_pv_complete_cycles < 2:
            raise ValueError("Need at least two complete PV cycles.")

        pv_first_matching_cycle = n_pv_complete_cycles - n_u_complete_cycles
        pv_matching_cycle = pv_first_matching_cycle + (n_u_complete_cycles - 1)

        pv_cycle_start = pv_matching_cycle * steps_per_cycle_pv
        pv_cycle_end = pv_cycle_start + steps_per_cycle_pv

        V_phase_cycle = V_pv[pv_cycle_start:pv_cycle_end]

        ED_local_pv = int(np.argmax(V_phase_cycle))

        # -------------------------------------------------
        # Guard against selecting the final sample as ED
        # because it often belongs to the next-cycle ED phase.
        # -------------------------------------------------
        edge_tol_ms = 5.0
        edge_tol_pv = int(round(edge_tol_ms / dt))

        if ED_local_pv >= steps_per_cycle_pv - edge_tol_pv:
            print("WARNING: ED detected at end of PV cycle.")
            print("Original ED_local_pv =", ED_local_pv)
            print("Treating ED as beginning-of-cycle equivalent.")

            ED_local_pv = 0

        ed_to_ed_start = (pv_matching_cycle - 1) * steps_per_cycle_pv + ED_local_pv
        ed_to_ed_end = ed_to_ed_start + steps_per_cycle_pv

        if ed_to_ed_start < 0 or ed_to_ed_end > len(V_pv):
            raise ValueError("Could not construct complete ED-to-ED PV cycle.")

        V_ed_to_ed = V_pv[ed_to_ed_start:ed_to_ed_end]

        ES_local_pv = int(np.argmin(V_ed_to_ed))
        ES_time_ms = ES_local_pv * dt
        ES_volume = float(V_ed_to_ed[ES_local_pv])

        return ES_time_ms, ES_volume, ED_local_pv, ES_local_pv

    # ------------------------------------------------------------
    # Helper: segmental peak timing using systolic search window
    # ------------------------------------------------------------
    def get_segment_peak_times_paper_definition(
        strain_16,
        dt_out,
        BCL,
        t_systole_ms,
        min_fraction_of_BCL=0.20,
        half_window_ms=250.0,
    ):
        n_t = strain_16.shape[1]
        t_ms = np.arange(n_t) * dt_out

        search_start_ms = max(
            min_fraction_of_BCL * BCL,
            t_systole_ms - half_window_ms,
        )

        search_end_ms = min(
            BCL,
            t_systole_ms + half_window_ms,
        )

        mask = (t_ms >= search_start_ms) & (t_ms <= search_end_ms)

        if np.sum(mask) < 3:
            print("WARNING: systolic search window too small. Using full cycle.")
            mask = np.ones_like(t_ms, dtype=bool)

        valid_indices = np.where(mask)[0]

        peak_indices = np.zeros(16, dtype=int)
        peak_times_ms = np.zeros(16)

        for i in range(16):
            curve = strain_16[i, :]
            curve_win = curve[valid_indices]

            if np.all(np.isnan(curve_win)):
                peak_indices[i] = -1
                peak_times_ms[i] = np.nan
            else:
                local_idx = int(np.nanargmin(curve_win))
                global_idx = int(valid_indices[local_idx])

                peak_indices[i] = global_idx
                peak_times_ms[i] = global_idx * dt_out

        delay_ms = peak_times_ms - t_systole_ms

        return peak_times_ms, peak_indices, delay_ms, search_start_ms, search_end_ms

    # ------------------------------------------------------------
    # Load displacement
    # ------------------------------------------------------------
    npz_path = os.path.join(
        IODet["outputfolder"],
        IODet["caseID"],
        "ME_u",
        "u_arr.npz",
    )

    if rank == 0:
        print("Reading:", npz_path)
        data = np.load(npz_path, allow_pickle=True)["data"]
        n_steps = len(data)
        print("[Rank 0] loaded {} displacement steps".format(n_steps))
    else:
        data = None
        n_steps = None

    n_steps = comm.bcast(n_steps, root=0)

    # ------------------------------------------------------------
    # Load mechanics mesh
    # ------------------------------------------------------------
    isBiV = SimDet.get("isBiV", False)
    isLV = SimDet.get("isLV", False)

    mesh_params = {
        "directory": IODet["directory_me"],
        "casename": IODet["casename_me"],
        "casename_marked": IODet.get("casename_marked", IODet["casename_me"]),
        "outputfolder": IODet["outputfolder"],
        "foldername": IODet["folderName"],
        "isBiV": isBiV,
        "isLV": isLV,
    }

    Mesh_obj = lv_mechanics_mesh(mesh_params, SimDet) if isLV else biv_mechanics_mesh(mesh_params, SimDet)
    mesh = Mesh_obj.mesh

    # ------------------------------------------------------------
    # Load eC0/eL0 from BiV_fibers.h5
    # ------------------------------------------------------------
    fiberfile = os.path.join(IODet["directory_me"], "BiV_fibers.h5")
    quad_degree = 4

    VQuadelem = df.VectorElement(
        "Quadrature",
        mesh.ufl_cell(),
        degree=quad_degree,
        quad_scheme="default",
    )

    fiberCLR = df.FunctionSpace(mesh, VQuadelem)

    eC0 = df.Function(fiberCLR)
    eL0 = df.Function(fiberCLR)

    h5 = df.HDF5File(mesh.mpi_comm(), fiberfile, "r")
    h5.read(eC0, "BiV_fibers/eC0")
    h5.read(eL0, "BiV_fibers/eL0")
    h5.close()

    eC0 = eC0 / df.sqrt(df.inner(eC0, eC0))
    eL0 = eL0 / df.sqrt(df.inner(eL0, eL0))

    # ------------------------------------------------------------
    # Mechanics forms
    # ------------------------------------------------------------
    var_deg = 1 if SimDet["Mechanics Discretization"] == "P1P1" else 2

    V = df.VectorFunctionSpace(mesh, "CG", var_deg)
    udisp = df.Function(V)

    params = {
        "mesh": mesh,
        "displacement_variable": udisp,
        "material model": SimDet["GiccioneParams"]["Passive model"],
        "material params": SimDet["GiccioneParams"]["Passive params"],
        "incompressible": SimDet["GiccioneParams"]["incompressible"],
        "growth_tensor": None,
    }

    uflforms = Forms(params)
    Fmat = uflforms.Fmat()

    # ------------------------------------------------------------
    # ED reference and analyzed cycle
    # ------------------------------------------------------------
    if rank == 0:
        steps_per_cycle = int(
            SimDet["HeartBeatLength"] / (SimDet["dt"] * SimDet["writeStep"])
        )

        last_start = n_steps - steps_per_cycle
        last_end = n_steps

        if ED_reference == "cycle_start":
            ED_index = last_start
            print("Using ED reference: first frame of last cycle")

        elif ED_reference == "manual":
            if ED_index_ref is None:
                raise ValueError("ED_reference='manual' requires ED_index_ref.")

            ED_index = int(ED_index_ref)
            last_start = ED_index
            last_end = min(ED_index + steps_per_cycle, n_steps)

            print("Using ED reference: manual ED_index_ref")

        elif ED_reference == "max_volume_from_PV":
            if PV_file is None:
                PV_file = os.path.join(
                    IODet["outputfolder"],
                    IODet["caseID"],
                    "BiV_PV.txt",
                )

            ED_index, last_start, last_end, ED_time_ms, ED_volume = \
                find_ED_from_PV_cycle_index(
                    PV_file=PV_file,
                    PV_LV_col=PV_LV_col,
                    n_steps_u=n_steps,
                    dt=SimDet["dt"],
                    writeStep=SimDet["writeStep"],
                    BCL=SimDet["HeartBeatLength"],
                    n_loading_rows=n_loading_rows,
                )

            print("Using ED reference: PV-derived max LV volume phase")
            print("ED_index in u_arr =", ED_index)
            print("ED time within cycle [ms] =", ED_time_ms)
            print("ED volume =", ED_volume)
            print("Analyzed u cycle indices =", last_start, "to", last_end)

        elif ED_reference == "last_cycle_first_frame":
            # Reference = first displacement frame of the last complete cycle
            ED_index = last_start
            last_end = last_start + steps_per_cycle

            print("Using reference: first displacement frame of last cycle")
            print("ED_index in u_arr =", ED_index)
            print("Analyzed u cycle indices =", last_start, "to", last_end)

        else:
            raise ValueError(
                "Unknown ED_reference. Use 'max_volume_from_PV', 'cycle_start', or 'manual'."
            )

        u_ref_global = data[ED_index]

    else:
        ED_index = None
        last_start = None
        last_end = None
        u_ref_global = None

    ED_index = comm.bcast(ED_index, root=0)
    last_start = comm.bcast(last_start, root=0)
    last_end = comm.bcast(last_end, root=0)
    u_ref_global = comm.bcast(u_ref_global, root=0)

    vec = udisp.vector()
    lo, hi = vec.local_range()
    vec.set_local(u_ref_global[lo:hi])
    vec.apply("insert")

    Fref = df.project(Fmat, df.TensorFunctionSpace(mesh, "DG", 0))

    # ------------------------------------------------------------
    # AHA segmentation and LV/septum masking
    # ------------------------------------------------------------
    aha_raw = tag_AHA_segments_local(
        mesh=mesh,
        SimDet=SimDet,
        R_apex=R_apex,
    )

    aha_segments = df.MeshFunction(
        "size_t",
        mesh,
        mesh.topology().dim(),
    )
    aha_segments.set_all(0)

    if isBiV:
        matid = Mesh_obj.matid

        for cell in df.cells(mesh):
            m = int(matid[cell])
            tag = int(aha_raw[cell])

            if m in [0, 1] and 1 <= tag <= 17:
                aha_segments[cell] = tag
            else:
                aha_segments[cell] = 0

    else:
        matid = None

        for cell in df.cells(mesh):
            aha_segments[cell] = int(aha_raw[cell])

    # ------------------------------------------------------------
    # Output directory
    # ------------------------------------------------------------
    base = os.path.join(IODet["outputfolder"], IODet["caseID"])

    if rank == 0:
        os.makedirs(base, exist_ok=True)

    comm.Barrier()

    df.File(os.path.join(base, "AHA_segments_used_for_strain.pvd")) << aha_segments

    if rank == 0:
        plot_AHA_HIS_orientation(
            SimDet=SimDet,
            outdir=os.path.join(base, "AHA_debug"),
            filename="AHA_HIS_orientation.png",
        )

    comm.Barrier()

    if debug_HIS_orientation:
        write_HIS_debug(
            mesh=mesh,
            SimDet=SimDet,
            aha_segments=aha_segments,
            matid=matid,
            outdir=os.path.join(base, "AHA_debug"),
        )

    # ------------------------------------------------------------
    # FAST DG0 vector assembly setup
    # ------------------------------------------------------------
    FS0 = df.FunctionSpace(mesh, "DG", 0)
    v0 = df.TestFunction(FS0)

    def dx_subdomain(mesh_obj, ids):
        metadata = {"quadrature_degree": 4}

        if isinstance(ids, (list, tuple)):
            out = mesh_obj.dx(ids[0], metadata=metadata)
            for sid in ids[1:]:
                out = out + mesh_obj.dx(sid, metadata=metadata)
            return out
        else:
            return mesh_obj.dx(ids, metadata=metadata)

    dx_LV = dx_subdomain(Mesh_obj, LVid)

    mass_vec = df.assemble(
        v0 * dx_LV,
        form_compiler_parameters=ffc_fast,
    ).get_local()

    dofmap0 = FS0.dofmap()
    own_start, own_end = dofmap0.ownership_range()

    local_dof_to_aha = np.zeros(len(mass_vec), dtype=np.int64)

    for cell in df.cells(mesh):
        tag = int(aha_segments[cell])

        if tag < 1 or tag > 17:
            continue

        dof = int(dofmap0.cell_dofs(cell.index())[0])

        if dof < len(mass_vec):
            loc = dof
        elif own_start <= dof < own_end:
            loc = dof - own_start
        else:
            continue

        local_dof_to_aha[loc] = tag

    aha_vol_local = np.zeros(17)

    for loc, tag in enumerate(local_dof_to_aha):
        if 1 <= tag <= 17:
            aha_vol_local[tag - 1] += mass_vec[loc]

    aha_volumes = np.zeros(17)

    for i in range(17):
        aha_volumes[i] = comm.allreduce(aha_vol_local[i], op=MPI.SUM)

    if rank == 0:
        print("LV + septum AHA volumes from FAST DG0 mass vector:", flush=True)
        for i, vol in enumerate(aha_volumes, start=1):
            print("AHA {:2d}: {:.6e}".format(i, vol), flush=True)

    # ------------------------------------------------------------
    # Storage
    # ------------------------------------------------------------
    n_aha_seg = 17
    Ecc_AHA = [[] for _ in range(n_aha_seg)]
    Ell_AHA = [[] for _ in range(n_aha_seg)]

    dt_out = SimDet["dt"] * SimDet["writeStep"]

    # ------------------------------------------------------------
    # Main time loop
    # ------------------------------------------------------------
    for step in range(last_start, last_end):

        if rank == 0 and (step - last_start) % 5 == 0:
            print(
                "[AHA strain FAST] Step {}/{}".format(
                    step - last_start,
                    last_end - last_start,
                ),
                flush=True,
            )

        if rank == 0:
            u_global = data[step]
        else:
            u_global = None

        u_global = comm.bcast(u_global, root=0)

        vec = udisp.vector()
        lo, hi = vec.local_range()
        vec.set_local(u_global[lo:hi])
        vec.apply("insert")

        F = Fmat * df.inv(Fref)
        C = F.T * F

        Ccc = df.inner(eC0, C * eC0)
        Cll = df.inner(eL0, C * eL0)

        Ecc = 0.5 * (1.0 - 1.0 / Ccc)
        Ell = 0.5 * (1.0 - 1.0 / Cll)

        ecc_vec = df.assemble(
            v0 * Ecc * dx_LV,
            form_compiler_parameters=ffc_fast,
        ).get_local()

        ell_vec = df.assemble(
            v0 * Ell * dx_LV,
            form_compiler_parameters=ffc_fast,
        ).get_local()

        ecc_sum_local = np.zeros(17)
        ell_sum_local = np.zeros(17)

        for loc, tag in enumerate(local_dof_to_aha):
            if 1 <= tag <= 17:
                ecc_sum_local[tag - 1] += ecc_vec[loc]
                ell_sum_local[tag - 1] += ell_vec[loc]

        ecc_sum_global = np.zeros(17)
        ell_sum_global = np.zeros(17)

        for i in range(17):
            ecc_sum_global[i] = comm.allreduce(ecc_sum_local[i], op=MPI.SUM)
            ell_sum_global[i] = comm.allreduce(ell_sum_local[i], op=MPI.SUM)

        for i in range(17):
            if aha_volumes[i] > 0.0:
                Ecc_AHA[i].append(ecc_sum_global[i] / aha_volumes[i])
                Ell_AHA[i].append(ell_sum_global[i] / aha_volumes[i])
            else:
                Ecc_AHA[i].append(np.nan)
                Ell_AHA[i].append(np.nan)

    # ------------------------------------------------------------
    # Save and compute paper-style SDI
    # ------------------------------------------------------------
    SDI_Ecc = None
    SDI_Ell = None

    if rank == 0:
        Ecc_mat = np.vstack(Ecc_AHA)
        Ell_mat = np.vstack(Ell_AHA)

        np.savez(os.path.join(base, "Ecc_AHA_lastcycle.npz"), *Ecc_AHA)
        np.savez(os.path.join(base, "Ell_AHA_lastcycle.npz"), *Ell_AHA)

        Ecc_16 = Ecc_mat[:16, :]
        Ell_16 = Ell_mat[:16, :]

        if PV_file is None:
            PV_file = os.path.join(
                IODet["outputfolder"],
                IODet["caseID"],
                "BiV_PV.txt",
            )

        t_systole_ms, ES_volume, ED_local_pv, ES_local_pv = \
            get_t_systole_from_PV_ED_to_ED(
                PV_file=PV_file,
                PV_LV_col=PV_LV_col,
                n_steps_u=n_steps,
                dt=SimDet["dt"],
                writeStep=SimDet["writeStep"],
                BCL=SimDet["HeartBeatLength"],
                n_loading_rows=n_loading_rows,
            )

        ecc_peak_time_ms, ecc_peak_idx, ecc_delay_ms, ecc_w0, ecc_w1 = \
            get_segment_peak_times_paper_definition(
                strain_16=Ecc_16,
                dt_out=dt_out,
                BCL=SimDet["HeartBeatLength"],
                t_systole_ms=t_systole_ms,
                min_fraction_of_BCL=0.20,
                half_window_ms=250.0,
            )

        ell_peak_time_ms, ell_peak_idx, ell_delay_ms, ell_w0, ell_w1 = \
            get_segment_peak_times_paper_definition(
                strain_16=Ell_16,
                dt_out=dt_out,
                BCL=SimDet["HeartBeatLength"],
                t_systole_ms=t_systole_ms,
                min_fraction_of_BCL=0.20,
                half_window_ms=250.0,
            )

        SDI_Ecc = np.nanstd(ecc_delay_ms) / SimDet["HeartBeatLength"] * 100.0
        SDI_Ell = np.nanstd(ell_delay_ms) / SimDet["HeartBeatLength"] * 100.0

        summary_path = os.path.join(base, "Ecc_Ell_AHA_SDI_summary.txt")

        with open(summary_path, "w") as f:
            f.write("LV Ecc/Ell AHA-SDI from BiV model\n")
            f.write("Paper definition:\n")
            f.write("SDI = std(t_peak_strain_i - t_systole) / t_RR * 100\n\n")

            f.write("AHA segments used for SDI: 1-16\n")
            f.write("Segment 17 saved but excluded from SDI\n\n")

            f.write("ED_index_u\t{}\n".format(ED_index))
            f.write("cycle_start_u\t{}\n".format(last_start))
            f.write("cycle_end_u\t{}\n".format(last_end))
            f.write("dt_output_ms\t{:.8f}\n".format(dt_out))
            f.write("BCL_ms\t{:.8f}\n".format(SimDet["HeartBeatLength"]))
            f.write("t_systole_ms\t{:.8f}\n".format(t_systole_ms))
            f.write("ES_volume\t{:.8f}\n".format(ES_volume))
            f.write("ED_local_pv\t{}\n".format(ED_local_pv))
            f.write("ES_local_pv\t{}\n".format(ES_local_pv))
            f.write("Ecc_search_window_ms\t{:.8f}\t{:.8f}\n".format(ecc_w0, ecc_w1))
            f.write("Ell_search_window_ms\t{:.8f}\t{:.8f}\n\n".format(ell_w0, ell_w1))

            f.write(
                "AHA_segment\t"
                "Ecc_peak_time_ms\tEcc_delay_to_systole_ms\tEcc_min_value\t"
                "Ell_peak_time_ms\tEll_delay_to_systole_ms\tEll_min_value\n"
            )

            for i in range(16):
                ei = ecc_peak_idx[i]
                li = ell_peak_idx[i]

                ecc_val = Ecc_16[i, ei] if ei >= 0 else np.nan
                ell_val = Ell_16[i, li] if li >= 0 else np.nan

                f.write(
                    "{}\t{:.8f}\t{:.8f}\t{:.8f}\t{:.8f}\t{:.8f}\t{:.8f}\n".format(
                        i + 1,
                        ecc_peak_time_ms[i],
                        ecc_delay_ms[i],
                        ecc_val,
                        ell_peak_time_ms[i],
                        ell_delay_ms[i],
                        ell_val,
                    )
                )

            f.write("\n")
            f.write("SDI_Ecc_percent\t{:.8f}\n".format(SDI_Ecc))
            f.write("SDI_Ell_percent\t{:.8f}\n".format(SDI_Ell))

        print("--------------------------------------------------")
        print("LV AHA SDI using paper definition")
        print("t_systole_ms =", t_systole_ms)
        print("ES_volume =", ES_volume)
        print("Ecc search window = [{:.1f}, {:.1f}] ms".format(ecc_w0, ecc_w1))
        print("Ell search window = [{:.1f}, {:.1f}] ms".format(ell_w0, ell_w1))
        print("Ecc time-to-peak [ms]:", ecc_peak_time_ms)
        print("Ecc delay to systole [ms]:", ecc_delay_ms)
        print("Ell time-to-peak [ms]:", ell_peak_time_ms)
        print("Ell delay to systole [ms]:", ell_delay_ms)
        print("SDI_Ecc = {:.4f} %".format(SDI_Ecc))
        print("SDI_Ell = {:.4f} %".format(SDI_Ell))
        print("Saved:", os.path.join(base, "Ecc_AHA_lastcycle.npz"))
        print("Saved:", os.path.join(base, "Ell_AHA_lastcycle.npz"))
        print("Saved:", summary_path)
        print("--------------------------------------------------")

    SDI_Ecc = comm.bcast(SDI_Ecc, root=0)
    SDI_Ell = comm.bcast(SDI_Ell, root=0)

    return SDI_Ecc, SDI_Ell    

def plot_AHA_HIS_orientation(
    SimDet,
    outdir,
    filename="AHA_HIS_orientation.png",
):
    """
    Plot the AHA 17-segment bullseye and mark the HIS bundle location.

    This uses the same angular convention as compute_strain_AHA_mpi():

        theta_shift = 180 - theta_his

    Therefore, after rotation, the HIS bundle should appear at
    theta_rotated = 180 degrees, which is the boundary between
    AHA segment 2 and AHA segment 3.

    Output:
        AHA_HIS_orientation.png
    """

    import os
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Wedge, Circle

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    if "HIS" not in SimDet:
        print("WARNING: SimDet['HIS'] not found. Cannot plot HIS orientation.")
        return

    his = np.asarray(SimDet["HIS"]).flatten()

    theta_his_raw = np.degrees(np.arctan2(his[1], his[0])) % 360.0
    theta_shift = (180.0 - theta_his_raw) % 360.0
    theta_his_rotated = (theta_his_raw + theta_shift) % 360.0

    # ------------------------------------------------------------
    # AHA segment angular sectors in the rotated coordinate system.
    # These match your tag_AHA_segments_local() formula.
    # ------------------------------------------------------------

    # Basal and mid segments:
    # tag = int((theta_rotated - 60) % 360 // 60) + 1
    basal_mid_sectors = {
        1: (60, 120),
        2: (120, 180),
        3: (180, 240),
        4: (240, 300),
        5: (300, 360),
        6: (0, 60),
    }

    # Apical segments:
    # tag = int((theta_rotated - 45) % 360 // 90) + 13
    apical_sectors = {
        13: (45, 135),
        14: (135, 225),
        15: (225, 315),
        16: (315, 405),
    }

    # Ring radii
    r_center = 0.22
    r_apical_inner = r_center
    r_apical_outer = 0.48
    r_mid_inner = r_apical_outer
    r_mid_outer = 0.74
    r_basal_inner = r_mid_outer
    r_basal_outer = 1.00

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect("equal")
    ax.axis("off")

    cmap = plt.cm.tab20

    def add_segment(seg_id, theta1, theta2, r_inner, r_outer):
        color = cmap((seg_id - 1) % 20)

        wedge = Wedge(
            center=(0.0, 0.0),
            r=r_outer,
            theta1=theta1,
            theta2=theta2,
            width=(r_outer - r_inner),
            facecolor=color,
            edgecolor="black",
            linewidth=1.2,
            alpha=0.65,
        )
        ax.add_patch(wedge)

        theta_mid = np.deg2rad((theta1 + theta2) / 2.0)
        r_mid = 0.5 * (r_inner + r_outer)

        x = r_mid * np.cos(theta_mid)
        y = r_mid * np.sin(theta_mid)

        ax.text(
            x,
            y,
            str(seg_id),
            ha="center",
            va="center",
            fontsize=13,
            fontweight="bold",
            color="black",
        )

    # Basal ring: segments 1-6
    for seg in [1, 2, 3, 4, 5, 6]:
        t1, t2 = basal_mid_sectors[seg]
        add_segment(seg, t1, t2, r_basal_inner, r_basal_outer)

    # Mid ring: segments 7-12 use same angular sectors as 1-6
    for seg in [7, 8, 9, 10, 11, 12]:
        base_seg = seg - 6
        t1, t2 = basal_mid_sectors[base_seg]
        add_segment(seg, t1, t2, r_mid_inner, r_mid_outer)

    # Apical ring: segments 13-16
    for seg in [13, 14, 15, 16]:
        t1, t2 = apical_sectors[seg]
        add_segment(seg, t1, t2, r_apical_inner, r_apical_outer)

    # Segment 17 center
    center = Circle(
        (0.0, 0.0),
        r_center,
        facecolor=cmap(16),
        edgecolor="black",
        linewidth=1.2,
        alpha=0.65,
    )
    ax.add_patch(center)

    ax.text(
        0.0,
        0.0,
        "17",
        ha="center",
        va="center",
        fontsize=13,
        fontweight="bold",
        color="black",
    )

    # ------------------------------------------------------------
    # Plot HIS location after rotation.
    # In your convention, this should be at 180 degrees.
    # ------------------------------------------------------------
    theta_his_plot = np.deg2rad(theta_his_rotated)

    r_his = 1.13
    x_his = r_his * np.cos(theta_his_plot)
    y_his = r_his * np.sin(theta_his_plot)

    ax.plot(
        [0.0, 1.08 * np.cos(theta_his_plot)],
        [0.0, 1.08 * np.sin(theta_his_plot)],
        color="red",
        linewidth=2.5,
        linestyle="--",
    )

    ax.plot(
        x_his,
        y_his,
        marker="*",
        markersize=22,
        color="red",
        markeredgecolor="black",
        markeredgewidth=1.0,
    )

    ax.text(
        x_his,
        y_his + 0.08,
        "HIS",
        ha="center",
        va="bottom",
        fontsize=13,
        fontweight="bold",
        color="red",
    )

    # Mark the expected 2/3 boundary
    theta_boundary = np.deg2rad(180.0)
    ax.plot(
        [0.0, 1.03 * np.cos(theta_boundary)],
        [0.0, 1.03 * np.sin(theta_boundary)],
        color="black",
        linewidth=1.5,
        linestyle=":",
    )

    ax.text(
        -1.20,
        -0.06,
        "AHA 2/3 boundary",
        ha="right",
        va="center",
        fontsize=11,
        color="black",
    )

    # Orientation labels
    ax.text(0.0, 1.18, "90°", ha="center", va="center", fontsize=10)
    ax.text(-1.18, 0.0, "180°", ha="center", va="center", fontsize=10)
    ax.text(0.0, -1.18, "270°", ha="center", va="center", fontsize=10)
    ax.text(1.18, 0.0, "0°", ha="center", va="center", fontsize=10)

    title = (
        "AHA orientation check\n"
        "HIS raw angle = {:.2f}°, theta_shift = {:.2f}°, rotated HIS = {:.2f}°"
    ).format(theta_his_raw, theta_shift, theta_his_rotated)

    ax.set_title(title, fontsize=12)

    ax.set_xlim(-1.35, 1.35)
    ax.set_ylim(-1.35, 1.35)

    outpath = os.path.join(outdir, filename)
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("Saved AHA/HIS orientation plot:", outpath)


def plot_AHA_time_to_systole_comparison(
    case_dirs,
    case_labels,
    SimDet,
    outpath,
    PV_LV_col=2,
    n_loading_rows=25,
    ecc_npz_name="Ecc_AHA_lastcycle.npz",
    ell_npz_name="Ell_AHA_lastcycle.npz",
    vmin=-200.0,
    vmax=200.0,
):
    """
    Generate paper-style AHA bullseye plots of time-to-systole.

    Rows:
        1) Ecc time-to-systole
        2) Ell time-to-systole

    Columns:
        One column per case.

    For each case:
        delay_i = t_peak_strain_i - t_systole

    AHA segments:
        Segments 1-16 are plotted.
        Segment 17 is shown in black and excluded.

    Inputs
    ------
    case_dirs : list of str
        Full output folders for each case, e.g.
        "/mnt/scratch/hagersan/biv4/bell-baseline5-2"

    case_labels : list of str
        Labels shown above columns, e.g.
        ["Baseline", "LAFB", "LSAFB", "LPFB", "LBBB"]

    SimDet : dict
        Simulation details.

    outpath : str
        Full path for output PNG.

    Expected files in each case folder:
        Ecc_AHA_lastcycle.npz
        Ell_AHA_lastcycle.npz
        BiV_PV.txt
    """
    BCL = float(SimDet["HeartBeatLength"])
    dt = float(SimDet["dt"])
    writeStep = int(SimDet["writeStep"])
    dt_out = dt * writeStep

    # ------------------------------------------------------------
    # Helper: load AHA npz as matrix, shape = 17 x n_time
    # ------------------------------------------------------------
    def load_AHA_npz(path):
        data = np.load(path)
        arrs = []

        for i in range(len(data.files)):
            arrs.append(data["arr_{}".format(i)])

        return np.vstack(arrs)

    def get_t_systole_from_last_PV_cycle(pv_file):
        pv = np.genfromtxt(pv_file)

        if pv.ndim == 1:
            raise ValueError("PV file loaded as 1D. Check BiV_PV.txt format.")

        V_raw = pv[:, PV_LV_col]
        V_pv = V_raw[n_loading_rows:]

        steps_per_cycle_pv = int(round(BCL / dt))
        n_pv_complete_cycles = len(V_pv) // steps_per_cycle_pv

        if n_pv_complete_cycles < 1:
            raise ValueError("No complete PV cycle found.")

        start = (n_pv_complete_cycles - 1) * steps_per_cycle_pv
        end = start + steps_per_cycle_pv

        V_cycle = V_pv[start:end]

        ES_local_pv = int(np.argmin(V_cycle))
        t_systole_ms = ES_local_pv * dt
        ES_volume = float(V_cycle[ES_local_pv])

        ED_local_pv = int(np.argmax(V_cycle))
        t_ED_ms = ED_local_pv * dt

        print("PV cycle-start reference:")
        print("  ED phase time [ms] =", t_ED_ms)
        print("  ES phase time [ms] =", t_systole_ms)
        print("  ES volume =", ES_volume)

        return t_systole_ms, ES_volume


    # ------------------------------------------------------------
    # Helper: find regional peak strain times using systolic window
    # ------------------------------------------------------------
    def get_delay_to_systole(strain_mat, t_systole_ms):
        """
        strain_mat shape: 17 x n_time.
        Uses AHA 1-16 only.
        """

        strain_16 = strain_mat[:16, :]
        n_t = strain_16.shape[1]

        t_ms = np.arange(n_t) * dt_out

        search_start_ms = max(0.20 * BCL, t_systole_ms - 250.0)
        search_end_ms = min(BCL, t_systole_ms + 250.0)

        mask = (t_ms >= search_start_ms) & (t_ms <= search_end_ms)
        valid_idx = np.where(mask)[0]

        if len(valid_idx) < 3:
            print("WARNING: systolic search window too small. Using full cycle.")
            valid_idx = np.arange(n_t)

        peak_time = np.zeros(16)
        peak_val = np.zeros(16)

        for i in range(16):
            curve = strain_16[i, :]
            curve_win = curve[valid_idx]

            if np.all(np.isnan(curve_win)):
                peak_time[i] = np.nan
                peak_val[i] = np.nan
            else:
                local_idx = int(np.nanargmin(curve_win))
                global_idx = int(valid_idx[local_idx])

                peak_time[i] = global_idx * dt_out
                peak_val[i] = curve[global_idx]

        delay = peak_time - t_systole_ms

        return delay, peak_time, peak_val, search_start_ms, search_end_ms

    # ------------------------------------------------------------
    # Helper: AHA bullseye plot
    # ------------------------------------------------------------
    def bullseye_time_plot(ax, data17, cmap, norm):
        """
        data17 has length 17.
        Segments 1-16 are colored.
        Segment 17 is black.
        """

        # ax.set_theta_zero_location("E")
        # ax.set_theta_direction(1)
        # ax.set_xticks([])
        # ax.set_yticks([])
        # ax.spines["polar"].set_visible(False)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(-1.05, 1.05)
        ax.set_ylim(-1.05, 1.05)
        ax.axis("off")

        # Ring radii
        r_center = 0.18
        r_apical_outer = 0.44
        r_mid_outer = 0.70
        r_basal_outer = 1.00

        # These sectors match your AHA helper:
        # basal/mid: tag = int((theta - 60) % 360 // 60) + 1
        basal_mid_sectors = {
            1: (60, 120),
            2: (120, 180),
            3: (180, 240),
            4: (240, 300),
            5: (300, 360),
            6: (0, 60),
        }

        # apical: tag = int((theta - 45) % 360 // 90) + 13
        apical_sectors = {
            13: (45, 135),
            14: (135, 225),
            15: (225, 315),
            16: (315, 405),
        }

        def add_wedge(seg, theta1, theta2, r_inner, r_outer):
            val = data17[seg - 1]

            if np.isnan(val):
                facecolor = "white"
            else:
                facecolor = cmap(norm(val))

            wedge = Wedge(
                center=(0.0, 0.0),
                r=r_outer,
                theta1=theta1,
                theta2=theta2,
                width=r_outer - r_inner,
                facecolor=facecolor,
                edgecolor="black",
                linewidth=1.0,
            )

            ax.add_patch(wedge)

        # Basal: 1-6
        for seg in [1, 2, 3, 4, 5, 6]:
            theta1, theta2 = basal_mid_sectors[seg]
            add_wedge(seg, theta1, theta2, r_mid_outer, r_basal_outer)

        # Mid: 7-12
        for seg in [7, 8, 9, 10, 11, 12]:
            base_seg = seg - 6
            theta1, theta2 = basal_mid_sectors[base_seg]
            add_wedge(seg, theta1, theta2, r_apical_outer, r_mid_outer)

        # Apical: 13-16
        for seg in [13, 14, 15, 16]:
            theta1, theta2 = apical_sectors[seg]
            add_wedge(seg, theta1, theta2, r_center, r_apical_outer)

        # Segment 17 black center
        center = Circle(
            (0.0, 0.0),
            r_center,
            facecolor="black",
            edgecolor="black",
            linewidth=1.0,
        )
        ax.add_patch(center)

        # ax.set_ylim(0.0, 1.0)
        ax.set_xlim(-1.05, 1.05)
        ax.set_ylim(-1.05, 1.05)
        ax.set_aspect("equal", adjustable="box")
        ax.axis("off")

    # ------------------------------------------------------------
    # Compute delays for all cases
    # ------------------------------------------------------------
    ecc_delay_all = []
    ell_delay_all = []
    sdi_ecc_all = []
    sdi_ell_all = []

    for case_dir, label in zip(case_dirs, case_labels):
        ecc_path = os.path.join(case_dir, ecc_npz_name)
        ell_path = os.path.join(case_dir, ell_npz_name)
        pv_path = os.path.join(case_dir, "BiV_PV.txt")

        if not os.path.exists(ecc_path):
            raise FileNotFoundError(ecc_path)

        if not os.path.exists(ell_path):
            raise FileNotFoundError(ell_path)

        if not os.path.exists(pv_path):
            raise FileNotFoundError(pv_path)

        Ecc_mat = load_AHA_npz(ecc_path)
        Ell_mat = load_AHA_npz(ell_path)

        t_systole_ms, ES_volume = get_t_systole_from_last_PV_cycle(pv_path)

        ecc_delay16, ecc_peak_time, ecc_peak_val, ecc_w0, ecc_w1 = \
            get_delay_to_systole(Ecc_mat, t_systole_ms)

        ell_delay16, ell_peak_time, ell_peak_val, ell_w0, ell_w1 = \
            get_delay_to_systole(Ell_mat, t_systole_ms)

        ecc_delay17 = np.full(17, np.nan)
        ell_delay17 = np.full(17, np.nan)

        ecc_delay17[:16] = ecc_delay16
        ell_delay17[:16] = ell_delay16

        SDI_Ecc = np.nanstd(ecc_delay16) / BCL * 100.0
        SDI_Ell = np.nanstd(ell_delay16) / BCL * 100.0

        ecc_delay_all.append(ecc_delay17)
        ell_delay_all.append(ell_delay17)
        sdi_ecc_all.append(SDI_Ecc)
        sdi_ell_all.append(SDI_Ell)

        print("--------------------------------------------------")
        print(label)
        print("t_systole_ms =", t_systole_ms)
        print("ES_volume =", ES_volume)
        print("Ecc delay min/max =", np.nanmin(ecc_delay16), np.nanmax(ecc_delay16))
        print("Ell delay min/max =", np.nanmin(ell_delay16), np.nanmax(ell_delay16))
        print("SDI_Ecc = {:.4f} %".format(SDI_Ecc))
        print("SDI_Ell = {:.4f} %".format(SDI_Ell))

    # ------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------
    n_cases = len(case_dirs)

    fig, axes = plt.subplots(2,n_cases,figsize=(2.25 * n_cases, 5.0),)
    if n_cases == 1:
        axes = np.array([[axes[0]], [axes[1]]])

    cmap = plt.cm.coolwarm
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

    for j, label in enumerate(case_labels):
        bullseye_time_plot(axes[0, j], ecc_delay_all[j], cmap, norm)
        bullseye_time_plot(axes[1, j], ell_delay_all[j], cmap, norm)
        axes[0, j].set_title(label, fontsize=11, pad=10)

    axes[0, 0].text(-0.30, 0.5,"Maximum Ecc [-]",transform=axes[0, 0].transAxes,rotation=90,ha="center",va="center",fontsize=11,)
    axes[1, 0].text(-0.30,0.5,"Maximum Ell [-]",transform=axes[1, 0].transAxes,rotation=90,ha="center", va="center",fontsize=11,)

    cbar_ax = fig.add_axes([0.30, 0.08, 0.40, 0.035])
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    cbar = fig.colorbar(sm,cax=cbar_ax,orientation="horizontal",)
    cbar.set_label("Time to systole [ms]", fontsize=10)

    plt.subplots_adjust(left=0.06,right=0.98,top=0.88,bottom=0.18,wspace=0.20,hspace=0.30,)

    outdir = os.path.dirname(outpath)
    if outdir and not os.path.exists(outdir):
        os.makedirs(outdir)

    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("Saved:", outpath)

    return {"case_labels": case_labels,"SDI_Ecc": sdi_ecc_all,"SDI_Ell": sdi_ell_all,"Ecc_delay": ecc_delay_all,"Ell_delay": ell_delay_all,}              