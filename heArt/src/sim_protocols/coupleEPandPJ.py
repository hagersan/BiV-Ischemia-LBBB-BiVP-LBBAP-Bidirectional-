from dolfin import *
import sys,pdb
sys.path.append("/mnt/Research")
from fenicstools import *
import numpy as np

class coupleEPandPJ(object):
    """
    Couple Purkinje electrophysiology to myocardial electrophysiology.

    This class implements forward electrical coupling from the
    Purkinje fiber network to the myocardium through prescribed
    Purkinje-myocardial junction (PMJ) locations.

    The Purkinje electrical potential is evaluated at each
    prescribed terminal location. When the potential exceeds
    the activation threshold, an electrical stimulus is applied
    to the corresponding myocardial stimulation site.

    The coupling uses a threshold-based stimulation approach
    rather than directly solving an electrical current exchange
    between the two domains.

    Parameters
    ----------
    EPmodel : object
        Myocardial electrophysiology model.

    PJmodel : object
        Purkinje electrophysiology model.

    EPparams : dict
        Myocardial EP parameters, including junction locations
        and stimulation intensity.

    PJparams : dict
        Purkinje electrophysiology parameters.

    state_obj : object
        Simulation state containing the time increment.

    Notes
    -----
    The first location is excluded from terminal stimulation.

    A normalized Purkinje potential threshold of 0.9 is used
    to detect activation.

    The variable tstart_arr tracks stimulation timing
    independently at each terminal location.
    """

    def __init__(self, EPmodel, PJmodel, EPparams, PJparams, state_obj):
        self.EPmodel = EPmodel
        self.EPparams = EPparams
        self.PJmodel = PJmodel
        self.PJparams = PJparams
        self.state_obj = state_obj
        self.pj_t_nodes = np.stack(EPparams["ploc"])
        self.pj_intensity = EPparams["current_intensity"]
        self.tstart_arr = [-10]*len(self.pj_t_nodes)

        try:
            self.probesPJ = Probes(self.pj_t_nodes.flatten(), PJmodel.w_ep.function_space().sub(0))
        except ValueError:
            self.probesPJ = Probes(self.pj_t_nodes.flatten(), PJmodel.w_ep.function_space())
        comms = PJmodel.mesh.mpi_comm()

    def UpdatePJandEP(self):
        """
        Couple Purkinje electrophysiology to myocardial electrophysiology.

        This class implements forward electrical coupling from the
        Purkinje fiber network to the myocardium through prescribed
        Purkinje-myocardial junction (PMJ) locations.

        The Purkinje electrical potential is evaluated at each
        prescribed terminal location. When the potential exceeds
        the activation threshold, an electrical stimulus is applied
        to the corresponding myocardial stimulation site.

        The coupling uses a threshold-based stimulation approach
        rather than directly solving an electrical current exchange
        between the two domains.

        Parameters
        ----------
        EPmodel : object
            Myocardial electrophysiology model.

        PJmodel : object
            Purkinje electrophysiology model.

        EPparams : dict
            Myocardial EP parameters, including junction locations
            and stimulation intensity.

        PJparams : dict
            Purkinje electrophysiology parameters.

        state_obj : object
            Simulation state containing the time increment.

        Notes
        -----
        The first location is excluded from terminal stimulation.

        A normalized Purkinje potential threshold of 0.9 is used
        to detect activation.

        The variable tstart_arr tracks stimulation timing
        independently at each terminal location.
        """
        probesPJ = self.probesPJ
        PJmodel = self.PJmodel
        EPmodel = self.EPmodel
        state_obj = self.state_obj
        pj_t_nodes = self.pj_t_nodes

        # Evaluate the current Purkinje electrical potential at each
        # prescribed junction location.
        # The evaluated potential is used to determine whether
        # myocardial stimulation should be initiated.
        probesPJ(PJmodel.getphivar())
        Nevals = probesPJ.number_of_evaluations()
        probes_val = probesPJ.array()

        # broadcast from proc 0 to other processes
        comms = EPmodel.mesh.mpi_comm()
        rank = MPI.rank(comms)
        probes_val_bcast = probesPJ.array(N=Nevals-1) # probe will only send to rank =0
        if(not rank == 0):
            probes_val_bcast = np.empty(len(pj_t_nodes))

        comms.Bcast(probes_val_bcast, root=0)

        for p in range(1, len(pj_t_nodes)): # initial node excluded from terminal nodes
            phi_pj_val = probes_val_bcast[p]   
            if(phi_pj_val > 0.9 and self.tstart_arr[p] < 1.0):
                if(self.tstart_arr[p] < 0):
                    self.tstart_arr[p] = 0
                    if("fstim_val_array" in dir(EPmodel)):
                        EPmodel.fstim_val_array[p] = self.pj_intensity
                    else:
                        EPmodel.fstim_array[p].iStim = self.pj_intensity
                else:
                    self.tstart_arr[p] += state_obj.dt.dt
            else:
                if("fstim_val_array" in dir(EPmodel)):
                    EPmodel.fstim_val_array[p] = 0.0
                else:
                    EPmodel.fstim_array[p].iStim = 0.0

        if("UpdateActivation" in dir(EPmodel)):
            EPmodel.UpdateActivation()

    def reset(self):
        # Per-terminal stimulation timer:
        #   -10 : Terminal has not yet initiated myocardial stimulation.
        #     0 : Stimulation has been initiated.
        #   > 0 : Elapsed time accumulated following initiation.
        #
        # The timer is used with a 1.0 time-unit threshold to
        # limit repeated stimulation during the cardiac cycle.
        # It is reset at the start of a new cycle.        
        self.tstart_arr = [-10]*len(self.pj_t_nodes)
        return