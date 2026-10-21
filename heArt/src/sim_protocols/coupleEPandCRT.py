import dolfin
from dolfin import *
import sys,pdb
sys.path.append("/mnt/Research")
from fenicstools import *
import numpy as np

class coupleEPandCRT(object):
    """
    Couple externally prescribed CRT pacing to myocardial EP.

    This class applies electrical stimulation to the myocardial
    electrophysiology model at predefined CRT pacing locations.

    CRT stimulation is implemented through the existing
    myocardial stimulation sources, using prescribed stimulus
    intensity and cardiac-cycle timing windows.

    The CRT sources are assigned stimulation indices following
    those reserved for Purkinje-myocardial junctions (PMJs).

    This class does not directly stimulate the Purkinje network.
    However, myocardial activation initiated by CRT may
    subsequently trigger retrograde Purkinje activation when
    the optional myocardium-to-Purkinje coupling is enabled.

    Parameters
    ----------
    EPmodel : object
        Myocardial electrophysiology model.

    EPparams : dict
        Electrophysiology parameters containing:
            CRT_pos
            CRT_current_intensity
            CRT_pacing_timing
            nPJ
            nCRT

    state_obj : object
        Simulation state containing the current cardiac-cycle
        time.
    """

    def __init__(self, EPmodel, EPparams, state_obj):
        self.EPmodel = EPmodel
        self.EPparams = EPparams
        self.state_obj = state_obj

        # CRT pacing node coordinates
        self.crt_nodes = np.array(EPparams["CRT_pos"])    
        self.crt_intensity = EPparams["CRT_current_intensity"]
        self.crt_timing = np.array(EPparams["CRT_pacing_timing"])  

        # --- Skip if CRT inactive ---
        if len(self.crt_nodes) == 0 or self.crt_intensity == 0.0:
            info("CRT inactive or empty; pacing will remain off.")
            self.fstim_idx = []
            return

        # Myocardial stimulation sources are organized sequentially:
        #
        #   [0, ..., nPJ-1]          : Purkinje-myocardial coupling sites
        #   [nPJ, ..., nPJ+nCRT-1]   : CRT pacing sites
        #
        # This prevents CRT stimulation from directly replacing
        # the stimulation coefficients assigned to PMJ sources.
        #
        # The indexing convention must remain consistent with
        # the source allocation in the myocardial EP model.

        nPJ  = EPparams.get("nPJ", 0)
        nCRT = EPparams.get("nCRT", len(self.crt_nodes))
        self.fstim_idx = list(range(nPJ, nPJ + nCRT))

        info(f"✅ CRT pacing will control fstim indices {self.fstim_idx}")
        self.comms = EPmodel.mesh.mpi_comm()


    # -------------------------------------------------------------------------
    def UpdateCRTandEP(self):
        """
        Activates CRT pacing sources (fstim * delta) during timing windows.
        """
        if not self.fstim_idx:
            return

        EPmodel = self.EPmodel
        t = self.state_obj.t

        # Evaluate the prescribed CRT pacing windows using
        # cardiac-cycle time rather than cumulative simulation time.
        #
        # Each timing entry contains:
        #   (stimulation onset, stimulation duration)
        #
        # CRT stimulation is active if the current time falls within
        # at least one prescribed pacing window.
        #
        # All CRT pacing sites share the same timing condition
        # and stimulation intensity in this implementation.
        active_now = any((t_on <= t <= t_on + dur) for (t_on, dur) in self.crt_timing if dur > 0.0)

        stim_val = self.crt_intensity if (active_now and self.crt_intensity > 0.0) else 0.0

        # Update CRT fstim entries
        for idx in self.fstim_idx:
            EPmodel.fstim_array[idx].iStim = stim_val

        # Optional hook for EP model
        if hasattr(EPmodel, "UpdateActivation"):
            EPmodel.UpdateActivation()

    # -------------------------------------------------------------------------
    def reset(self):
        """
        Resets CRT pacing states between cardiac cycles.
        """
        if not self.fstim_idx:
            return

        for idx in self.fstim_idx:
            self.EPmodel.fstim_array[idx].iStim = 0.0
        return