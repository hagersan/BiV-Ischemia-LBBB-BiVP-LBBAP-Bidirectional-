import sys, pdb, vtk
from dolfin import *

sys.setrecursionlimit(5000)  # Increase to a higher value
sys.path.append("/mnt/Research")
sys.path.append("/mnt/Output")
sys.path.append("/mnt/Research/heArt")
sys.path.append("/home/fenicstools")

from heArt.src.sim_protocols.run_BiV_ClosedLoop_pctrl_bidirectional import (run_BiV_ClosedLoop as run_BiV_ClosedLoop,)

from heArt.src.postprocessing.postprocessdata2 import (compute_activation as compute_activation)
from heArt.src.postprocessing.postprocessdata2 import (postprocessdata as postprocessdata,)
from heArt.src.postprocessing.postprocessdata2 import (plothemodynamics as plothemodynamics,)
from heArt.src.postprocessing.postprocessdata2 import (plotpressure as plotpressure)
from heArt.src.postprocessing.postprocessdata2 import (dumpvtk as dumpvtk)
from heArt.src.postprocessing.postprocessdata2 import (extractdisplacementloading as extractdisplacementloading,)
from heArt.src.postprocessing.postprocessdata2 import (extractdisplacement as extractdisplacement,)
from heArt.src.postprocessing.postprocessdata2 import (compute_strain_fast as compute_strain,)
from heArt.src.postprocessing.postprocessdata2 import (compute_strain_AHA_fast as compute_strain_AHA_mpi)
from heArt.src.postprocessing.postprocessdata2 import (check_AHA_plot as check_AHA_plot)
from heArt.src.postprocessing.postprocessdata2 import (strain_plot_AHA as strain_plot_AHA)

from heArt.src.postprocessing.postprocessdata2 import (plot_PV_comparison as plot_PV_comparison)
from heArt.src.postprocessing.postprocessdata2 import (plot_PV_comparison_ind as plot_PV_comparison_ind)
from heArt.src.postprocessing.postprocessdata2 import (plothemodynamics_comparison as plothemodynamics_comparison)
from heArt.src.postprocessing.postprocessdata2 import (plothemodynamics_comparison_ind as plothemodynamics_comparison_ind)
from heArt.src.postprocessing.postprocessdata2 import (plot_AHA_time_to_systole_comparison as plot_AHA_time_to_systole_comparison)
from heArt.src.postprocessing.postprocessdata2 import (plot_strain_comparison_ind as plot_strain_comparison_ind)

#  - - - - - - - - - - - -- - - - - - - - - - - - - - - - - - - - - - -
IODetails = {
    "directory_me": "../BiVMesh/ischemia/",
    "casename_me": "biv_idealized3_generalizedv11",    
    "casename_me_marked": "biv_idealized3_generalizedv11",      
    "directory_ep": "../BiVMesh/ischemia/",
    "casename_ep": "biv_idealized3_generalizedv6", 
    "casename_ep_marked": "biv_idealized3_generalizedv6",          
    "directory_pj": "../Biv_pj_mesh_coarse/cv/",
    "casename_pj": "PJmarked_isch", # baseline/lbbb pj mesh with branch order and ischemic mask attributes
    "casename_pj_marked": "PJmarked_isch",
    "outputfolder": "./output", 
    "folderName": "/",
    "caseID": "baseline",
    "isLV": False,
    # for case comparison         
    "comparison_plot_mode":"disease", # disease, pacing, ischemic_pacing
    "directory_baseline":"baseline",
    "directory_block":"lbbb",
    "directory_bivp":"bivp",
    "directory_lbbap":"lbbap",
    "directory_ischemia":"ischemia",
    "directory_ischemia_block":"ilbbb",
    "directory_ischemia_bivp":"ibvp",
    "directory_ischemia_lbbap":"ilbbap",
}

contRactility = 500e3 
GuccioneParams = {
    "ParamsSpecified": True,
    "Passive model": {"Name": "Guccione"},
    "Passive params": {
        "Cparam":  Constant(50.0), 
        "bff": Constant(29.0),
        "bfx": Constant(13.3),
        "bxx": Constant(26.6),
    },
    "Active model": {"Name": "Time-varying"},
    "Active params": {
        "tau": 25,
        "t_trans": 300,
        "B": 4.75,
        "t0": 275,
        "l0": 1.58,
        "Tmax": Constant(contRactility),
        "Ca0": 4.35,
        "Ca0max": 4.35,
        "lr": 1.85,
    },
    "HomogenousActivation": False,
    "deg": 4,
    "Kappa": 1e6, 
    "incompressible": True,
}

# Systemic
Circparam = {
    "Ees_la": 350,
    "A_la": 58.67,
    "B_la": 0.049,
    "V0_la": 10,
    "Tmax_la": 120,
    "tau_la": 75,
    "tdelay_la": 160,
    "Csa": 0.0025, 
    "Cad": 0.0330,
    "Csv": 0.30, 
    "Vsa0": 700,
    "Vsv0": 2575.0,     
    "Vad0": 40,
    "Rav": 500.0,
    "Rsv": 100.0, 
    "Rsa": 30000,
    "Rad": 106000,  
    "Rmv": 1500.0, 
    # Pulmonary
    "Ees_ra": 81.33,
    "A_ra": 466.6,
    "B_ra": 0.033,
    "V0_ra": 20,
    "Tmax_ra": 120,
    "tau_ra": 25,
    "tdelay_ra": 100,
    "Cpa": 0.015, 
    "Cpv": 0.9, 
    "Vpa0": 360,
    "Vpv0": 15, 
    "Rpv": 500.0,
    "Rtv": 500.0, 
    "Rpa": 9000.0, 
    "Rpvv": 400, 
    # flow rate
    "Q_lvad": 0.0,
    "Q_sv": 0.0659387173129,
    "Q_av": 0.0,
    "Q_sa": 0.0113526347222,
    "Q_ad": 0.0797504135138,
    "Q_mv": 0.0,
    "Q_tv": 0.0,
    "Q_pa": 0.001213376269,
    "Q_pv": 0.0646105314407,
    "Q_pvv": 0.0,
    # volumes
    "V_sa":	717.4355397561658, 
    "V_ad":	267.79164160229897, 
    "V_sv":	2805.8358210377214, 
    "V_pa":	376.2608779393844, 
    "V_pv":	995.5661433005655, 
    "V_LA":	14.03152199785293, 
    "V_LV":	114.82581087906651, 
    "V_RV":	67.85457509081614, 
    "V_RA":	33.2976245144343, 
    "stop_iter": 0,
}

SimDetails = {
    "HeartBeatLength": 800.0,
    "dt": 0.5, 
    "EDP": 1.77139e+03*0.0075, 
    "RVEDPfactor": 	1.03761e+03/1.77139e+03, 
    "writeStep": 5.0, 
    "GiccioneParams": GuccioneParams,
    "nLoadSteps": 25, 
    "DTI_EP": False,
    "DTI_ME": False,
    # PJ geometry + conduction  
    "isPJ": True,  
    "pj_tnodes": "pj_fiber_biv.csv", # csv file for terminal nodes  
    "d_pj": 20, # factor can depend on complexity of pj network, suggestion: run a singular isolated ep cycle to confirm baseline (for LV: Total PF activation time (ms)=20.1 ± 7.0 - Dosdall et al. 2010)    
    "PJ_current_intensity": 6.6, # should be also used to modify pj total activation time
    "branch_scaling_mode": "bell", # function shape for branch scaling, higher conduction velocity in the mid region with slower conduction velocity at order 0 and order 7    
    "branch_scaling": 0.1, # scaling factor for branch order
    # Myocarial geometry + conduction
    "isLV": False,     
    "isBiV": True,  
    # diffusion factors were modified to match total BiV activation of approx 100 [ms] - Durrer et al. 1970
    "d_iso": 0.005, # isotropic conduction velocity 
    "d_ani_factor":  0.04, # anistotropic conduction velocity along the fibers 
    "ani_factor": 1000, # anistotropic conduction velocity along the sheet and sheet-normal direction 
    "current_intensity": 0.6,      
    # Initial stimulation from PJ
    "Native_PJ_pacing": True,      
    "ploc_mode": False,
    "ploc": [[0, 2.5, 1]], # starting point of pj network
    "HIS":[[0, 2.5, 0]],
    "pacing_timing": [[0.0, 20.0]], # native pacing timing/duration wrt cycle time
    # Bidirectional: Myocardium -> Purkinje
    "Myo_to_PJ": True, # True or False
    "Myo_PJ_apply_current": True,
    "Myo_PJ_candidate_mode": "all_pj_nodes",
    "Myo_PJ_phi_threshold": 1.1,
    "Myo_PJ_pj_activation_threshold": 1.1,
    "Myo_PJ_kernel_k": 16,
    "Myo_PJ_current_intensity": 1.5,
    "Myo_PJ_stim_duration": 2.0,
    "Myo_PJ_total_current_cap": 6.6,
    "Myo_PJ_normalize_by_active_sources": False,    
    "Myo_PJ_debug": True,
    "Myo_PJ_debug_interval": 1.0,
    "CRT_to_PJ": False,
    # Left Bundle Branch Block (LBBB)
    "lbbb": False, # True or False
    "lbbb_delay": 0.00,
    "lbbb_location" : 2, #cell location in pj network        
    # External Pacing
    "CRT": False, # True or False 
     # "CRT_pos": [[1.15068,2.28288,-5.91177],[-0.49253,-3.07426,-0.85193]], #BIVP: 1:RV lead, 2:LV lead  
    "CRT_pos": [[-0.14112,1.99502,0]], #LBBAP 1: LBBB lead         
    "CRT_pacing_timing": [[0.0, 20.0]], # pacing timing/duration wrt cycle time
    "CRT_current_intensity": 1.0, # pacing current
    "CRT_eps_factor": 1.0, # area/radius for pacing electrode
    # Ischemia 
    "Ischemia": False, # True or False
    "Ischemia_regions": [[-2.29651, -0.30801, -6.58009, 3.5]], #[x0, y0, z0, r0],   
    "Ischemia_matid": [0, 1, 2], #0:LV, 1:Septum, 2: RV 
    "Ischemic_stiffness_factor": 3.0, # increase of passive stiffness in ischemic substrat   
    "Ischemic_ep_factor": 0.01, # decrease of diffusion tensor compared to remote area
    # Mechanics
    "closedloopparam": Circparam,
    "springbc": True, # Robin boundary condition
    "Mechanics Discretization": "P1P1",
    "ispctrl": True,
    "spring_atbase": True,  
    "epiid_Kadj_coeff": [10, 30], # normal / transverse spring coefficient
    "dashpotparam": [20.0e1,1.0e2], # damping tensor   
    "active_region": [0,1,2], # matid 0:LV, 1:Septum, 2:RV
    "annulus_region": [3], # matid 3: Base 
    "annulus_stiffness_factor": 200, # base boundary constraing  
    "RVtopid": 6, 
    "LVtopid": 5, 
    "topid": 4, 
    "LVendoid": 2,
    "RVendoid": 3,
    "epiid": 1,
    "abs_tol": 5e-7,
    "rel_tol": 5e-8,
    "ploc_tol": 0.02,
    "p_a": 0.0,
    "p_v": 1300.0,
    "beta_a": 3.5e-5,
    "beta_v": 3.0e-5,    
}

# #  - - - - - - - - - - - -- - - - - - - - - - - - - - - - - - - - - - -
# # Simulation
# run_BiV_ClosedLoop(IODet=IODetails, SimDet=SimDetails)
# #  - - - - - - - - - - - -- - - - - - - - - - - - - - - - - - - - - - -

# #  - - - - - - - - - - - -- - - - - - - - - - - - - - - - - - - - - - -
# # Postprocessing: Individual
compute_activation(IODet=IODetails, SimDet=SimDetails)
# postprocessdata(IODet=IODetails, SimDet=SimDetails)
# plothemodynamics(IODet=IODetails, SimDet=SimDetails, cycle=Circparam["stop_iter"])
# plotpressure(IODet=IODetails, SimDet=SimDetails, cycle=Circparam["stop_iter"])
# dumpvtk(IODet=IODetails, SimDet=SimDetails, EP_var=[["phi", "CG", 1]], PJ_var=[["phi", "CG", 1]])
# extractdisplacement(IODet=IODetails, SimDet=SimDetails)
# compute_strain(IODet=IODetails, SimDet=SimDetails, LVid = [0,1], RVid = 2, cycle=Circparam["stop_iter"])
# compute_strain_AHA_mpi(IODet=IODetails,SimDet=SimDetails,LVid=[0, 1],RVid=2,ED_reference="cycle_start",PV_LV_col=2,n_loading_rows=25,R_apex=1.0,debug_HIS_orientation=False,)
#
## Debugging
# check_AHA_plot(IODet=IODetails, SimDet=SimDetails, target_segment=1, R_apex=1.0) # Just a check: AHA regions for LV and RV
# strain_plot_AHA(IODet=IODetails,SimDet=SimDetails,mode='all')
#
# # Postprocessing: Comparison
# plot_PV_comparison(IODet=IODetails, SimDet=SimDetails, cycle=None)
# plothemodynamics_comparison(IODet=IODetails, SimDet=SimDetails, cycle=None)
# plot_PV_comparison_ind(IODet=IODetails, SimDet=SimDetails, cycle=Circparam["stop_iter"])
# plothemodynamics_comparison_ind(IODet=IODetails, SimDet=SimDetails, cycle=Circparam["stop_iter"])
# plot_strain_comparison_ind(IODet=IODetails, SimDet=SimDetails, cycle=Circparam["stop_iter"])
#
# # AHA_time_to_systole.png comparison
# case_dirs=[ IODetails["outputfolder"]+IODetails["directory_baseline"], IODetails["outputfolder"]+IODetails["directory_block"], IODetails["outputfolder"]+IODetails["directory_ischemia_block"],
#             IODetails["outputfolder"]+IODetails["directory_lbbap"], IODetails["outputfolder"]+IODetails["directory_bivp"],
#             IODetails["outputfolder"]+IODetails["directory_ischemia_lbbap"], IODetails["outputfolder"]+IODetails["directory_ischemia_bivp"]]
#
# case_labels = [ "Baseline", "LBBB", "Ischemia+LBBB", 
#                 "lbbap", "bivp", 
#                 "ilbbap", "ibivp"]
# plot_AHA_time_to_systole_comparison(case_dirs=case_dirs, case_labels=case_labels, SimDet=SimDetails, outpath="/mnt/scratch/hagersan/biv4/AHA_time_to_systole_pacing.png", PV_LV_col=2, n_loading_rows=25, vmin=-200.0, vmax=200.0,)
# #  - - - - - - - - - - - -- - - - - - - - - - - - - - - - - - - - - - -