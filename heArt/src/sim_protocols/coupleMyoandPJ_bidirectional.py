from dolfin import *
import dolfin as dolfin
import numpy as np
from mpi4py import MPI as pyMPI
from scipy.spatial import cKDTree


class coupleMyoandPJ(object):
    """
    SAFE Myocardium -> Purkinje reverse coupling.

    This is the active version of the validated SAFE_SENSE coupler.

    What stays unchanged
    --------------------
    - existing CRT -> myocardium pathway
    - existing native His -> PJ pathway
    - existing PJ -> myocardium pathway
    - myocardial and PJ PDE solvers
    - ME/circulation driver order

    What this class adds
    --------------------
    1. Read the CURRENT solved myocardial phi after the myocardial solve.
    2. Compute the same validated local kernel-weighted myocardial signal
       around every PJ mesh vertex (all PJ nodes are candidates).
    3. If that signal crosses Myo_PJ_phi_threshold and the PJ node has not
       already activated this cycle, start ONE temporary PJ capture pulse.
    4. Add that pulse to the already-existing PJmodel.I_source_vec AFTER
       PJmodel.UpdateActivation() and BEFORE the normal PJ solve.
    5. Stop the pulse as soon as that PJ node reaches the PJ activation
       threshold, or when the maximum pulse duration is reached.
    6. Never trigger the same PJ node twice within one cardiac cycle.

    Important safety choices
    ------------------------
    - No fenicstools.Probes.
    - No direct overwrite of PJ phi/r.
    - No persistent source allocation for all ~5431 PJ nodes.
    - Reverse source vectors are assembled lazily ONLY for actual captures.
    - Reverse source shape is the same Lorentzian-like kernel used by
      EPmodel_cpp2 for PJ stimulation.
    - Each reverse source is conservatively normalized against the existing
      validated native-His source so that, per unit coefficient, neither its
      global L1 magnitude nor its global Linf magnitude exceeds the native
      source.
    - Existing native-His current is never cleared here.

    Physical interpretation
    -----------------------
    The reverse coupling represents retrograde electrical
    activation of the Purkinje fiber network originating
    from the surrounding myocardium.

    This mechanism may occur when myocardial activation,
    including activation initiated by conventional CRT,
    reaches regions electrically connected to the
    Purkinje network.

    Rather than directly solving electrical current
    exchange between the 3D myocardium and 1D Purkinje
    network, the coupling is implemented using a
    threshold-triggered stimulation approach.

    When the local myocardial activation signal exceeds
    the prescribed threshold, a temporary stimulus is
    applied to the corresponding Purkinje node.

    Subsequent propagation through the Purkinje network
    is governed by its electrophysiology model.    
    """

    def __init__(self, EPmodel, PJmodel, EPparams, PJparams, state_obj, SimDet):
        self.EPmodel = EPmodel
        self.PJmodel = PJmodel
        self.EPparams = EPparams
        self.PJparams = PJparams
        self.state_obj = state_obj
        self.SimDet = SimDet

        self.comm = PJmodel.mesh.mpi_comm()
        self.rank = MPI.rank(self.comm)

        self.apply_current = bool(SimDet.get("Myo_PJ_apply_current", True))

        mode = str(SimDet.get("Myo_PJ_candidate_mode", "all_pj_nodes")).lower()

        if mode not in ("all", "all_pj", "all_pj_nodes", "mesh_vertices"):
            raise RuntimeError(
                "Use Myo_PJ_candidate_mode='all_pj_nodes'."
            )

        self.myo_threshold = float(SimDet.get("Myo_PJ_phi_threshold", 1.1))

        self.pj_activation_threshold = float(SimDet.get("Myo_PJ_pj_activation_threshold", 0.9))

        self.kernel_k = max(1,int(SimDet.get("Myo_PJ_kernel_k", 16)),)

        self.intensity = float(
            SimDet.get(
                "Myo_PJ_current_intensity",
                SimDet.get("PJ_current_intensity", 0.0),
            )
        )

        self.stim_duration = float(
            SimDet.get("Myo_PJ_stim_duration", 2.0)
        )

        self.total_current_cap = float(
            SimDet.get("Myo_PJ_total_current_cap", 6.6)
        )

        self.normalize_by_active_sources = bool(
            SimDet.get(
                "Myo_PJ_normalize_by_active_sources",
                False,
            )
        )

        self.debug = bool(
            SimDet.get("Myo_PJ_debug", True)
        )

        self.debug_interval = float(
            SimDet.get("Myo_PJ_debug_interval", 1.0)
        )

        self._next_debug_time = 0.0

        # MPI diagnostics/safety.  Keep enabled for the first parallel test.
        self.mpi_debug = bool(
            SimDet.get("Myo_PJ_mpi_debug", True)
        )

        # --------------------------------------------------------------
        # Build deterministic global coordinate tables.
        # --------------------------------------------------------------

        (
            self.myo_coords,
            self._myo_local_to_global,
        ) = self._build_global_vertex_table(
            self.EPmodel.mesh
        )

        (
            self.pj_coords,
            self._pj_local_to_global,
        ) = self._build_global_vertex_table(
            self.PJmodel.mesh
        )

        self.candidate_nodes = self.pj_coords
        self.n_candidates = int(
            len(self.candidate_nodes)
        )

        # --------------------------------------------------------------
        # Reciprocal sensing map.
        # This is the same SAFE_SENSE implementation that already worked.
        # --------------------------------------------------------------

        if hasattr(self.EPmodel, "havg"):
            self.kernel_eps = float(
                self.EPmodel.havg
            )
        else:
            self.kernel_eps = 0.02 * (
                float(self.EPmodel.mesh.hmin())
                +
                float(self.EPmodel.mesh.hmax())
            )

        (self.kernel_indices,self.kernel_weights,self.nearest_distance,) = self._build_kernel_map()

        # Two independent activation locks are maintained:
        #
        # pj_activated_this_cycle:
        #   Records whether a Purkinje node has already exceeded
        #   the electrical activation threshold through any pathway.
        #
        # reverse_fired_this_cycle:
        #   Records whether a reverse-coupling stimulus has already
        #   been initiated at the node, even if the stimulus did
        #   not result in successful Purkinje activation.
        #
        # Together, these conditions prevent repeated reverse
        # stimulation within the same cardiac cycle.

        # Persistent latch after ACTUAL PJ phi > activation threshold.
        self.pj_activated_this_cycle = np.zeros(self.n_candidates,dtype=bool,)

        # Persistent latch as soon as reverse capture starts.
        # This prevents a failed/expired pulse from being retriggered again
        # in the same beat.
        self.reverse_fired_this_cycle = np.zeros(self.n_candidates,dtype=bool,)

        # For rising-threshold crossing detection.
        self.prev_myo_above_threshold = np.zeros(self.n_candidates,dtype=bool,)

        # candidate_index -> {"vector": PETScVector, "elapsed": float}
        self.active_reverse = {}

        # --------------------------------------------------------------
        # Prepare lazy reverse-current source assembly.
        # --------------------------------------------------------------
        self.havg_pj = 0.02 * (float(self.PJmodel.mesh.hmin())+float(self.PJmodel.mesh.hmax()))
        self._delta_module = None
        self.native_source_l1 = None
        self.native_source_linf = None

        if self.apply_current:
            self._prepare_reverse_source_assembly()

        if self.rank == 0:
            print("--------------------------------------------------",flush=True,)
            print("[Myo->PJ SAFE ACTIVE] initialized",flush=True,)
            print("MODE                        = {}".format("ACTIVE" if self.apply_current else "SENSE ONLY"),flush=True,)
            print("fenicstools.Probes          = NOT USED",flush=True,)
            print("PJ candidate count          = {}".format(self.n_candidates),flush=True,)
            print("global myocardial vertices  = {}".format(len(self.myo_coords)),flush=True,)
            print("myocardial kernel eps       = {:.6g}".format(self.kernel_eps),flush=True,)
            print("myocardial kernel k         = {}".format(self.kernel_indices.shape[1]if self.n_candidates > 0 else 0),flush=True,)
            print("Myo->PJ phi threshold       = {}".format(self.myo_threshold),flush=True,)
            print("PJ activation threshold     = {}".format(self.pj_activation_threshold),flush=True,)
            print("Myo->PJ current intensity   = {}".format(self.intensity),flush=True,)
            print("Myo->PJ max pulse duration  = {} ms".format(self.stim_duration),flush=True,)
            print("Myo->PJ total current cap   = {}".format(self.total_current_cap),flush=True,)

            if self.apply_current:
                print("native source L1            = {:.6g}".format(self.native_source_l1),flush=True,)
                print("native source Linf          = {:.6g}".format(self.native_source_linf),flush=True,)

            if self.n_candidates > 0:
                print("PJ-to-Myo nearest dist min = {:.6g}".format(float(np.min(self.nearest_distance))),flush=True,)
                print("PJ-to-Myo nearest dist med = {:.6g}".format(float(np.median(self.nearest_distance))),flush=True,)
                print("PJ-to-Myo nearest dist max = {:.6g}".format(float(np.max(self.nearest_distance))),flush=True,)

            print("PJ reactivation within beat = DISABLED",flush=True,)
            print( "--------------------------------------------------",flush=True,)

    # ==================================================================
    # Deterministic global coordinate tables
    # ==================================================================

    @staticmethod
    def _key(point, tol=1.0e-10):
        p = np.asarray(point,dtype=float,).reshape(-1)
        xyz = np.zeros(3,dtype=float,)
        xyz[:min(3, len(p))] = p[:min(3, len(p))]

        return tuple(np.round(xyz / tol).astype(np.int64))

    def _build_global_vertex_table(self, mesh):
        local_coords = np.asarray(mesh.coordinates(),dtype=float,)
        xyz_local = np.zeros((len(local_coords), 3),dtype=float,)

        if len(local_coords) > 0:
            ncol = min(3,local_coords.shape[1],)
            xyz_local[:, :ncol] = (local_coords[:, :ncol])

        gathered = self.comm.gather(xyz_local,root=0,)

        if self.rank == 0:
            coords = []
            key_to_global = {}

            for block in gathered:
                block = np.asarray(block,dtype=float,)

                for point in block:
                    key = self._key(point)

                    if key not in key_to_global:
                        key_to_global[key] = len(coords)
                        coords.append(point.tolist())

            global_coords = np.asarray(coords,dtype=float,)

            if global_coords.size == 0:
                global_coords = np.zeros((0, 3),dtype=float,)

        else:
            global_coords = None

        global_coords = self.comm.bcast(global_coords,root=0,)
        global_coords = np.asarray(global_coords,dtype=float,)
        global_map = {self._key(point): i for i, point in enumerate(global_coords)}
        local_to_global = np.asarray([global_map[self._key(point)]for point in xyz_local],dtype=np.int64,)

        return global_coords, local_to_global

    # --------------------------------------------------------------
    # Spatially weighted myocardial activation sensing
    # --------------------------------------------------------------
    #
    # For each candidate Purkinje vertex, identify the k nearest
    # myocardial vertices.
    #
    # The local myocardial activation signal is evaluated as:
    #
    #   phi_m_local(i) = sum_j w_ij * phi_m(j)
    #
    # where the normalized kernel weights are:
    #
    #   w_ij = K(d_ij) / sum_l K(d_il)
    #
    #   K(d) = eps / (pi * (d**2 + eps**2))
    #
    # The normalized weights sum to one for each candidate.
    #
    # This provides a spatially weighted electrical signal
    # rather than relying on a single nearest myocardial node.
    # --------------------------------------------------------------

    def _build_kernel_map(self):

        if self.n_candidates == 0:
            return (np.zeros((0, 1),dtype=np.int64,),np.zeros((0, 1),dtype=float,),np.zeros(0,dtype=float,),)

        if len(self.myo_coords) == 0:
            raise RuntimeError("No myocardial vertices were found.")

        k_use = min(self.kernel_k,len(self.myo_coords),)
        tree = cKDTree(self.myo_coords)
        distances, indices = tree.query(self.candidate_nodes,k=k_use,)
        distances = np.asarray(distances,dtype=float,)
        indices = np.asarray(indices,dtype=np.int64,)

        if k_use == 1:
            distances = distances.reshape((-1, 1))
            indices = indices.reshape((-1, 1))

        raw = (self.kernel_eps/np.pi/(distances * distances+self.kernel_eps*self.kernel_eps))
        row_sum = np.sum(raw,axis=1,)

        if np.any(row_sum <= 0.0):
            raise RuntimeError("Invalid reciprocal-kernel weight sum.")

        weights = (raw/row_sum[:, None])
        nearest = distances[:, 0].copy()

        return indices, weights, nearest

    # ==================================================================
    # Current scalar fields
    # ==================================================================

    def _current_myo_phi(self):
        # EPmodel_basic_test.getphivar() returns the PREVIOUS time level.
        # The driver calls this hook after the current myocardial solve and
        # before UpdateVar(), so read the current solved phi directly.
        try:
            phi, _ = self.EPmodel.w_ep.split(deepcopy=True)
            return phi

        except Exception:
            return self.EPmodel.w_ep

    def _current_pj_phi(self):
        return self.PJmodel.getphivar()

    # ==================================================================
    # MPI-safe global CG1 vertex values
    # ==================================================================

    def _global_vertex_values(self,mesh,scalar_function,local_to_global,n_global,label,):
        local_values = np.asarray(scalar_function.compute_vertex_values(mesh),dtype=float,).reshape(-1)

        if (local_values.size!=local_to_global.size):
            raise RuntimeError("{} vertex/value mismatch on rank {}: " "{} values for {} vertices".format(label,self.rank,local_values.size,local_to_global.size,))

        gathered_idx = self.comm.gather(local_to_global,root=0,)
        gathered_val = self.comm.gather(local_values,root=0,)

        if self.rank == 0:
            sums = np.zeros(n_global,dtype=float,)
            counts = np.zeros(n_global,dtype=np.int64,)

            for idx, val in zip(gathered_idx,gathered_val,):

                idx = np.asarray(idx,dtype=np.int64,)
                val = np.asarray(val,dtype=float,)
                np.add.at(sums,idx,val,)
                np.add.at(counts,idx,1,)

            if np.any(counts == 0):
                raise RuntimeError("{} global vertex table has unfilled entries.".format(label))

            values = (sums/counts.astype(float))

        else:
            values = np.empty(n_global,dtype=float,)

        self.comm.Bcast(values,root=0,)

        return values

    # ==================================================================
    # MPI safety helpers
    # ==================================================================

    def _sync_vector(self, vec):
        """
        Finalize/synchronize a DOLFIN/PETSc vector on every MPI rank.

        This is intentionally conservative.  The original PJ solver sees
        PJmodel.I_source_vec only after this method has completed on all ranks.
        """
        try:
            vec.apply("insert")
        except Exception:
            # Some PETSc-backed vectors are already assembled and do not
            # require apply().  Do not alter values in that case.
            pass

        self.comm.Barrier()

    def _assert_same_active_keys(self):
        """
        Fail fast if the Python-side active-source state ever differs by rank.
        A mismatch here would otherwise show up later as a PETSc collective hang.
        """
        keys = tuple(sorted(int(k) for k in self.active_reverse.keys()))
        all_keys = self.comm.allgather(keys)

        if any(k != all_keys[0] for k in all_keys):
            raise RuntimeError(
                "Myo->PJ MPI state mismatch: active_reverse keys differ by rank: {}"
                .format(all_keys))

    def _vector_global_stats(self, vec):
        local = np.asarray(vec.get_local(), dtype=float)

        local_finite = bool(np.all(np.isfinite(local)))
        local_min = float(np.min(local)) if local.size else np.inf
        local_max = float(np.max(local)) if local.size else -np.inf
        local_absmax = float(np.max(np.abs(local))) if local.size else 0.0

        finite_all = self.comm.allreduce(1 if local_finite else 0,op=pyMPI.MIN,)
        gmin = self.comm.allreduce(local_min, op=pyMPI.MIN)
        gmax = self.comm.allreduce(local_max, op=pyMPI.MAX)
        gabsmax = self.comm.allreduce(local_absmax, op=pyMPI.MAX)

        return bool(finite_all), float(gmin), float(gmax), float(gabsmax)

    def _pre_pj_solve_check(self):
        """
        Check the exact PJ state/source handed to the normal PJ solver.

        This does not alter PJ phi, r, or the source.  It gives us a clean
        failure before PETSc if one rank has NaN/Inf or an inconsistent source.
        """
        w_ok, w_min, w_max, w_abs = self._vector_global_stats(self.PJmodel.w_ep.vector())
        r_ok, r_min, r_max, r_abs = self._vector_global_stats(self.PJmodel.r_ep.vector())
        s_ok, s_min, s_max, s_abs = self._vector_global_stats(self.PJmodel.I_source_vec)

        # Compute the reaction term exactly as the PJ solver will, but into
        # a COPY so the model state is untouched.
        # Do NOT copy f_phi_.vector() here.  In parallel DOLFIN 2019.1,
        # that Function vector may still be in a PETSc state for which
        # VecCopy is illegal before the first PJ FHN update.
        #
        # w_ep.vector() is already initialized and has the same PJ scalar
        # layout expected by Update_fphi, so use it only as an allocated
        # scratch vector and immediately overwrite it through Update_fphi.
        f_preview = self.PJmodel.w_ep.vector().copy()
        f_preview.zero()
        f_preview.apply("insert")

        self.PJmodel.FHNmodel.Update_fphi(f_preview,self.PJmodel.w_ep.vector(),self.PJmodel.r_ep.vector(),)

        f_preview.apply("insert")
        f_ok, f_min, f_max, f_abs = self._vector_global_stats(f_preview)

        all_ok = w_ok and r_ok and s_ok and f_ok

        if self.mpi_debug and self.rank == 0:
            print(
                "[Myo->PJ-MPI-CHECK] "
                "t={:.3f} | "
                "phi=[{:.6g},{:.6g}] absmax={:.6g} finite={} | "
                "r=[{:.6g},{:.6g}] absmax={:.6g} finite={} | "
                "fphi=[{:.6g},{:.6g}] absmax={:.6g} finite={} | "
                "source=[{:.6g},{:.6g}] absmax={:.6g} finite={}"
                .format(
                    float(self.state_obj.t),
                    w_min, w_max, w_abs, w_ok,
                    r_min, r_max, r_abs, r_ok,
                    f_min, f_max, f_abs, f_ok,
                    s_min, s_max, s_abs, s_ok,
                ),
                flush=True,
            )

        if not all_ok:
            raise RuntimeError(
                "Non-finite PJ state/source detected before PJ solve at t={}"
                .format(float(self.state_obj.t)))

    # ==================================================================
    # Reverse PJ source assembly
    # ==================================================================

    def _global_vector_norms(self, vec):
        local = np.asarray(vec.get_local(),dtype=float,)
        local_l1 = float(np.sum(np.abs(local)))
        local_linf = (float(np.max(np.abs(local))) if local.size > 0 else 0.0)
        global_l1 = self.comm.allreduce(local_l1,op=pyMPI.SUM,)
        global_linf = self.comm.allreduce(local_linf,op=pyMPI.MAX,)

        return float(global_l1), float(global_linf)

    def _prepare_reverse_source_assembly(self):
        if (not hasattr(self.PJmodel,"fstim_array",) or len(self.PJmodel.fstim_array) == 0):
            raise RuntimeError("Active Myo->PJ requires the existing native ""PJ source PJmodel.fstim_array[0].")

        native_vec = (self.PJmodel.fstim_array[0])
        (self.native_source_l1,self.native_source_linf,) = self._global_vector_norms(native_vec)

        if (self.native_source_l1 <= 0.0 or self.native_source_linf <= 0.0):
            raise RuntimeError("Existing native PJ source is invalid.")

        cpp_code = r"""
        #include <pybind11/pybind11.h>
        #include <pybind11/eigen.h>
        #include <dolfin/function/Expression.h>
        #include <cmath>

        class reverse_delta_exp : public dolfin::Expression
        {
        public:
            Eigen::VectorXd x0;
            double eps;

            reverse_delta_exp()
                : dolfin::Expression()
            {}

            void eval(
                Eigen::Ref<Eigen::VectorXd> values,
                Eigen::Ref<const Eigen::VectorXd> x
            ) const
            {
                double norm_squared = 0.0;

                for (
                    int i = 0;
                    i < x0.size();
                    ++i
                )
                {
                    norm_squared +=
                        (x[i] - x0[i])
                        *
                        (x[i] - x0[i]);
                }

                values[0] =
                    eps
                    /
                    M_PI
                    /
                    (
                        norm_squared
                        +
                        eps * eps
                    );
            }
        };

        PYBIND11_MODULE(SIGNATURE, m)
        {
            pybind11::class_<
                reverse_delta_exp,
                std::shared_ptr<reverse_delta_exp>,
                dolfin::Expression
            >(m, "reverse_delta_exp")
                .def(pybind11::init<>())
                .def_readwrite(
                    "x0",
                    &reverse_delta_exp::x0
                )
                .def_readwrite(
                    "eps",
                    &reverse_delta_exp::eps
                );
        }
        """
        self._delta_module = (dolfin.compile_cpp_code(cpp_code))

    def _build_reverse_source_vector(self,point,):
        """
        Assemble ONE PJ source at ONE captured PJ coordinate.

        It is normalized conservatively against the validated native-His
        source.  The scale chosen is the smaller of the native/reverse L1
        and Linf ratios, so one unit of reverse-source coefficient cannot
        exceed the native unit source in either global measure.
        """
        delta = dolfin.CompiledExpression(self._delta_module.reverse_delta_exp(),degree=1,)
        delta.eps = float(self.havg_pj)
        delta.x0 = np.asarray(point,dtype=float,)
        reverse_vec = (self.PJmodel.I_source_vec.copy())
        reverse_vec[:] = 0.0
        form = (Constant(1.0) * delta * self.PJmodel.wtest_ep * dolfin.dx(domain=self.PJmodel.mesh))
        assemble(form,tensor=reverse_vec,)

        # Explicitly finalize the distributed source vector before any
        # collective norm or later AXPY.
        self._sync_vector(reverse_vec)
        reverse_l1, reverse_linf = (self._global_vector_norms(reverse_vec))

        if (reverse_l1 <= 0.0 or reverse_linf <= 0.0):
            raise RuntimeError("Invalid reverse PJ source at {}.".format(np.asarray(point).tolist()))
        
        scale_l1 = (self.native_source_l1/reverse_l1)
        scale_linf = (self.native_source_linf/reverse_linf)
        source_scale = min(scale_l1,scale_linf,)

        if (not np.isfinite(source_scale) or source_scale <= 0.0):
            raise RuntimeError("Invalid reverse-source normalization at {}.".format(np.asarray(point).tolist()))

        reverse_vec *= float(source_scale)

        # Scaling is local PETSc work; synchronize before storing/reusing
        # the distributed source on later timesteps.
        self._sync_vector(reverse_vec)

        return reverse_vec, float(source_scale)

    # ==================================================================
    # Main coupling update
    # ==================================================================

    def UpdateMyoandPJ(self):
        """
        Update reverse electrical coupling from myocardium to Purkinje.

        The method evaluates the current myocardial electrical state
        and determines whether new Purkinje stimulation events
        should be initiated.

        Algorithm
        ---------
        1. Identify Purkinje nodes that have already activated.

        2. Evaluate kernel-weighted myocardial electrical signals
           at all candidate Purkinje nodes.

        3. Detect upward crossings of the myocardial activation
           threshold.

        4. Exclude Purkinje nodes that have already activated
           or previously received reverse stimulation.

        5. Assemble temporary reverse stimulation sources for
           newly eligible nodes.

        6. Add active reverse sources to the existing Purkinje
           stimulation vector without removing native stimulation.

        7. Update stimulation timers and terminate pulses after
           successful activation or the maximum permitted duration.

        8. Verify the distributed Purkinje state and source vector
           before the subsequent Purkinje electrophysiology solve.

        Notes
        -----
        This method must be called after the myocardial EP solve
        and after PJmodel.UpdateActivation(), but before the
        Purkinje EP solve.

        The coupling is implemented through temporary stimulus
        currents rather than direct modification of Purkinje
        electrical state variables.
        """     
           
        if self.n_candidates == 0:
            return

        dt = float(self.state_obj.dt.dt)

        # --------------------------------------------------------------
        # 1. Read current PJ phi and permanently latch every PJ node that
        #    has already activated by ANY pathway in this cardiac cycle.
        # --------------------------------------------------------------
        pj_global = self._global_vertex_values(self.PJmodel.mesh,self._current_pj_phi(),self._pj_local_to_global,len(self.pj_coords),"PJ phi",)
        active_now = (pj_global>self.pj_activation_threshold)
        newly_locked_mask = np.logical_and(active_now,np.logical_not(self.pj_activated_this_cycle),)

        self.pj_activated_this_cycle[active_now] = True

        newly_locked = int(np.count_nonzero(newly_locked_mask))

        # Stop reverse current immediately on the timestep AFTER the PJ node
        # actually crossed the activation threshold in the previous PJ solve.
        stopped_on_capture = []

        for k in sorted(self.active_reverse.keys()):
            if active_now[int(k)]:
                stopped_on_capture.append(int(k))

        for k in stopped_on_capture:
            del self.active_reverse[k]

        # --------------------------------------------------------------
        # 2. Use the already-validated SAFE_SENSE myocardial mapping.
        # --------------------------------------------------------------
        myo_global = self._global_vertex_values(self.EPmodel.mesh,self._current_myo_phi(),self._myo_local_to_global,len(self.myo_coords),"myocardial phi",)
        myo_local = np.sum(myo_global[self.kernel_indices]*self.kernel_weights,axis=1,)
        myo_above = (myo_local>self.myo_threshold)
        myo_crossing = np.logical_and(myo_above,np.logical_not(self.prev_myo_above_threshold),)
        eligible = np.logical_and(myo_crossing,np.logical_and(np.logical_not(self.pj_activated_this_cycle),np.logical_not(self.reverse_fired_this_cycle),),)
        new_capture_idx_local = np.where(eligible)[0]

        # Root owns the event decision; every rank receives exactly the same
        # candidate list.  This prevents a one-rank floating-point/state
        # discrepancy from producing different PETSc source operations.
        if self.rank == 0:
            new_capture_idx = [int(k) for k in new_capture_idx_local.tolist()]
        else:
            new_capture_idx = None

        new_capture_idx = self.comm.bcast(new_capture_idx,root=0,)
        self.prev_myo_above_threshold[:] = (myo_above)

        # --------------------------------------------------------------
        # 3. Start each newly detected capture ONCE.
        # --------------------------------------------------------------
        started = []
        started_scales = []

        if self.apply_current:
            for k in new_capture_idx:
                k = int(k)

                # Latch immediately: no second reverse trigger this beat,
                # even if the pulse eventually times out.
                self.reverse_fired_this_cycle[k] = True
                vec, scale = (self._build_reverse_source_vector(self.candidate_nodes[k]))
                self.active_reverse[k] = {"vector": vec,"elapsed": 0.0,}
                started.append(k)
                started_scales.append(scale)

        # --------------------------------------------------------------
        # 4. ADD active reverse pulses to the source vector that
        #    PJmodel.UpdateActivation() already built.
        #
        #    Native His is preserved.  Nothing is zeroed here.
        # --------------------------------------------------------------

        active_keys = sorted(self.active_reverse.keys())
        amplitudes = np.zeros(len(active_keys),dtype=float,)

        if len(active_keys) > 0:
            amplitudes[:] = (self.intensity)

            if self.normalize_by_active_sources:
                amplitudes /= float(len(active_keys))

            if (self.total_current_cap > 0.0):
                requested_total = float(np.sum(np.abs(amplitudes)))

                if (requested_total>self.total_current_cap):
                    amplitudes *= (self.total_current_cap/requested_total)

            for amp, k in zip(amplitudes,active_keys,):
                self.PJmodel.I_source_vec.axpy(float(amp),self.active_reverse[k]["vector"],)

        # Finalize the exact source vector the normal PJ solver will use.
        # Do this even when there are zero reverse sources, because
        # PJmodel.UpdateActivation() rebuilt I_source_vec just before us.
        self._sync_vector(self.PJmodel.I_source_vec)

        # Python-side state must be identical on every rank.
        self._assert_same_active_keys()

        # --------------------------------------------------------------
        # 5. Advance pulse timers.  This is a maximum duration only.
        #    Actual capture stops the pulse earlier via active_now above.
        # --------------------------------------------------------------

        timed_out = []

        for k in active_keys:
            self.active_reverse[k]["elapsed"] += dt

            if (self.active_reverse[k]["elapsed"]>=self.stim_duration):
                timed_out.append(int(k))

        for k in timed_out:
            if k in self.active_reverse:
                del self.active_reverse[k]

        # --------------------------------------------------------------
        # 6. Verify the exact distributed PJ state/source before returning
        #    control to solver_FHN_pj.solvenonlinear().
        # --------------------------------------------------------------

        self._pre_pj_solve_check()

        # --------------------------------------------------------------
        # 7. Compact diagnostics.
        # --------------------------------------------------------------

        t_now = float(self.state_obj.t)
        regular_due = (self.debug_interval <= 0.0 or t_now + 1.0e-12>=self._next_debug_time)
        event_due = ( len(new_capture_idx) > 0 or newly_locked > 0 or len(stopped_on_capture) > 0 or len(timed_out) > 0 or len(active_keys) > 0)

        if (self.debug and self.rank == 0 and (regular_due or event_due)):
            print(
                "[Myo->PJ-ACTIVE] "
                "t={:.3f} ms | "
                "global_myo_max={:.6g} | "
                "kernel_myo_max={:.6g} | "
                "PJ_max={:.6g} | "
                "crossing={} | "
                "new_capture={} | "
                "new_PJ_locked={} | "
                "stopped_on_capture={} | "
                "timed_out={} | "
                "active_sources={} | "
                "locked={}/{} | "
                "reverse_fired={}/{}"
                .format(t_now,
                        float(np.max(myo_global)) if myo_global.size else 0.0,
                        float(np.max(myo_local)) if myo_local.size else 0.0,
                        float(np.max(pj_global)) if pj_global.size else 0.0,
                        int(np.count_nonzero(myo_crossing)),
                        int(len(new_capture_idx)),
                        int(newly_locked),
                        int(len(stopped_on_capture)),
                        int(len(timed_out)),
                        int(len(active_keys)),
                        int(np.count_nonzero(self.pj_activated_this_cycle)),
                        int(self.n_candidates),
                        int(np.count_nonzero(self.reverse_fired_this_cycle)),
                        int(self.n_candidates),),
                flush=True,)

            if len(started) > 0:
                preview = started[:5]
                print("[Myo->PJ-ACTIVE] START indices = {}".format(preview),flush=True,)
                print("[Myo->PJ-ACTIVE] START phi = {}".format([float(myo_local[int(k)])for k in preview]),flush=True,)
                print("[Myo->PJ-ACTIVE] START xyz = {}".format([self.candidate_nodes[int(k)].tolist()for k in preview]),flush=True,)
                print("[Myo->PJ-ACTIVE] source scales = {}".format([float(x)for x in started_scales[:5]]),flush=True,)

            if len(stopped_on_capture) > 0:
                print("[Myo->PJ-ACTIVE] CAPTURED/STOP indices = {}".format(stopped_on_capture[:10]),flush=True,)

            if len(timed_out) > 0:
                print("[Myo->PJ-ACTIVE] TIMEOUT indices = {}".format(timed_out[:10]),flush=True,)

            if (self.debug_interval > 0.0 and regular_due):
                while (self._next_debug_time <= t_now + 1.0e-12):
                    self._next_debug_time += (self.debug_interval)

    def reset(self):
        """
        Start of a new cardiac cycle.

        This is the ONLY place the once-per-cycle reverse-capture locks
        are cleared.
        """
        self.pj_activated_this_cycle[:] = False
        self.reverse_fired_this_cycle[:] = False
        self.prev_myo_above_threshold[:] = False
        self.active_reverse = {}
        self._next_debug_time = 0.0

        if (self.debug and self.rank == 0):
            print("[Myo->PJ-ACTIVE] reset for new cardiac cycle",flush=True,)