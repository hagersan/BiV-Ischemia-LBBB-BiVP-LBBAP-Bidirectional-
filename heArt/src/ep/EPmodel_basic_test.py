from dolfin import *
import numpy as np
from ..utils.nsolver import NSolver as NSolver

from ufl import indices
import dolfin as dolfin
from mpi4py import MPI as pyMPI
from ..utils.oops_objects_MRC2 import build_active_indicator
import pdb

class EPmodel(object):
    """
    Finite-element myocardial electrophysiology model.

    The model describes propagation of electrical excitation
    through the myocardium using a modified Aliev-Panfilov
    reaction-diffusion formulation.

    The electrophysiology system consists of:
        phi : Dimensionless excitation variable.
        r   : Recovery variable.

    Electrical propagation is controlled by an anisotropic
    diffusion tensor constructed from the local myocardial
    fiber, sheet, and sheet-normal directions.

    The model supports:
        - Isotropic and anisotropic electrical diffusion.
        - Ischemia-dependent diffusion modifications.
        - Purkinje-myocardial junction stimulation.
        - Direct CRT stimulation.
        - Spatially distributed stimulation sources.
        - Parallel finite-element simulations.

    Numerical formulation
    ---------------------
    The governing reaction-diffusion system is discretized
    using the finite-element method.

    The excitation variable is approximated using continuous
    first-order elements (CG1), while the recovery variable
    uses discontinuous piecewise-constant elements (DG0).

    Time integration uses an implicit formulation with
    nonlinear residual and Jacobian assembly.

    The nonlinear problem is solved using NSolver.

    Parameters
    ----------
    params : dict
        Model configuration containing the myocardial mesh,
        fiber architecture, diffusion coefficients, ischemia
        parameters, stimulation locations, and solver settings.

    Notes
    -----
    Stimulation sources are constructed using a regularized
    spatial kernel centered at prescribed pacing locations.

    The first stimulation entries may correspond to
    Purkinje-myocardial junctions, followed by CRT
    stimulation sites.
    """

    def __init__(self, params):
        self.parameters = self.default_parameters()
        self.parameters.update(params)
        self.mesh = self.parameters["EPmesh"]

        if "ploc" in list(self.parameters.keys()):
            self.max_pace_label = len(self.parameters["ploc"])
            self.ploc = self.parameters["ploc"]
        else:
            self.max_pace_label = 0
            self.ploc = None

        self.ischemia_mask = self.parameters.get("Ischemia_mask", None)
        
        # ============================================================
        # ISCHEMIC REGION INDICATOR
        # ============================================================
        #
        # Construct a spatial indicator identifying healthy
        # and ischemic myocardial regions.
        #
        # The resulting field is used to interpolate between
        # healthy and ischemic diffusion tensors.
        #
        # Healthy tissue:
        #     active_indicator = 1
        #
        # Ischemic tissue:
        #     active_indicator = 0
        #
        # The construction and MPI handling of this field
        # are implemented in build_active_indicator().
        # ============================================================

        self.active_indicator = build_active_indicator(mesh=self.mesh,ischemia_mask=self.ischemia_mask,enable_ischemia=bool(self.parameters.get("Ischemia", False)),label="EP")

        P1_ep = FiniteElement("CG", self.mesh.ufl_cell(), 1, quad_scheme="default")
        P1_ep._quad_scheme = "default"
        P2_ep = FiniteElement("DG", self.mesh.ufl_cell(), 0, quad_scheme="default")
        P2_ep._quad_scheme = "default"

        self.W_ep = FunctionSpace(self.mesh, MixedElement([P1_ep, P2_ep]))
        self.w_ep = Function(self.W_ep)
        self.dw_ep = TrialFunction(self.W_ep)
        self.wtest_ep = TestFunction(self.W_ep)
        self.w_n_ep = Function(self.W_ep)

        self.F_FHN, self.J_FHN, self.fstim_array = self.Problem()

    def default_parameters(self):
        return {"ploc": [[0.0, 0.0, 0.0, 100.0, 1]], "pacing_timing": [[1, 20.0]],}

    def calculateDmat(self, f0, s0, n0, mesh, mId):
        """
        Construct the anisotropic myocardial diffusion tensor.

        Electrical diffusion is defined using the local
        orthonormal myocardial coordinate system:

            f0 : Fiber direction
            s0 : Sheet direction
            n0 : Sheet-normal direction

        The diffusion tensor is:

            D = d_iso*I
                + d_ani*(f0 ⊗ f0)
                + (d_ani/ani_factor)*(s0 ⊗ s0)
                + (d_ani/ani_factor)*(n0 ⊗ n0)

        The isotropic contribution provides baseline diffusion,
        while the anisotropic terms introduce preferential
        electrical propagation along myocardial fibers.

        Ischemic tissue
        ---------------
        Within the ischemic region, the complete diffusion
        tensor is multiplied by Ischemic_ep_factor.

        D_ischemic = Ischemic_ep_factor * D_healthy

        Returns
        -------
        D_effective : UFL tensor
            Spatially varying myocardial diffusion tensor.
        """

        d_iso = self.parameters["d_iso"]  
        d_ani = self.parameters["d_ani"] 
        ani_factor = self.parameters["ani_factor"]
        ischemic_factor = self.parameters["Ischemic_ep_factor"]
        d_Ani = d_ani
        dParam = {}

        D_iso = Constant(((d_iso, "0.0", "0.0"), ("0.0", d_iso, "0.0"), ("0.0", "0.0", d_iso)))

        i, j = indices(2)
        Dij = (d_Ani)*f0[i]*f0[j] + (d_Ani*1/ani_factor)*s0[i]*s0[j] + (d_Ani*1/ani_factor)*n0[i]*n0[j]+ D_iso[i,j]
        D_tensor = as_tensor(Dij, (i, j)) 

        healthy_D = D_tensor
        ischemic_D = ischemic_factor * D_tensor 

        if self.parameters["Ischemia"]:
            # print("Debug ischemia D")          
            D_effective = self.active_indicator * healthy_D + (1 - self.active_indicator) * ischemic_D
        else:
            D_effective = healthy_D

        return D_effective        


    def Problem(self):
        state_obj = self.parameters["state_obj"]

        if "f0" in list(self.parameters.keys()):
            f0_ep = self.parameters["f0"]

        if "s0" in list(self.parameters.keys()):
            s0_ep = self.parameters["s0"]

        if "n0" in list(self.parameters.keys()):
            n0_ep = self.parameters["n0"]

        AHAid_ep = self.parameters["AHAid"]
        matid_ep = self.parameters["matid"]

        if "facetboundaries" in list(self.parameters.keys()):
            facetboundaries_ep = self.parameters["facetboundaries"]
        else:
            facetboundaries_ep = None

        mesh = self.mesh
        W_ep = self.W_ep
        w_n_ep = self.w_n_ep
        w_ep = self.w_ep
        dw_ep = self.dw_ep
        wtest_ep = self.wtest_ep

        phi0 = interpolate(Expression("0.0", degree=0), W_ep.sub(0).collapse())
        r0 = interpolate(Expression("0.0", degree=0), W_ep.sub(1).collapse())

        assign(w_n_ep, [phi0, r0])

        phi_n, r_n = split(w_n_ep)
        phi, r = split(w_ep)

        bcs_ep = []

        phi_test, r_test = split(wtest_ep)

        # ============================================================
        # MODIFIED ALIEV-PANFILOV ELECTROPHYSIOLOGY MODEL
        # ============================================================
        #
        # The myocardial electrical state is described by:
        #
        #   phi : Dimensionless excitation variable
        #   r   : Recovery variable
        #
        # The reaction-diffusion equations are:
        #
        #   dphi/dtau = div(D grad(phi))
        #              + f_phi(phi, r) + I_stim
        #
        #   dr/dtau = f_r(phi, r)
        #
        # with:
        #
        #   f_phi = -c*phi*(phi-alpha)*(phi-1) - r*phi
        #
        #   f_r = eps(phi,r)*[-r-c*phi*(phi-b-1)]
        #
        #   eps(phi,r) = g + mu1*r/(mu2+phi)
        #
        # The normalized time increment is:
        #
        #   dtau = dt / fhn_timeNormalizer
        #
        # The reaction parameters control excitation,
        # repolarization, and recovery dynamics.
        #
        # ============================================================

        alpha = Constant(0.01)
        g = Constant(0.002)
        b = Constant(0.15)
        c = Constant(8)
        mu1 = Constant(0.2)
        mu2 = Constant(0.3)

        def eps_FHN(phi, r):
            return g + (mu1 * r) / (mu2 + phi)

        def f_phi(phi, r):
            return -c * phi * (phi - alpha) * (phi - 1) - r * phi

        def f_r(phi, r):
            return eps_FHN(phi, r) * (-r - c * phi * (phi - b - 1.0))

        fhn_timeNormalizer = 12.9
        k = state_obj.dt / fhn_timeNormalizer
        f_phi = f_phi(phi, r)
        f_r = f_r(phi, r)
        comm = mesh.mpi_comm()  # MPI Communicator 

        if 'interval' in mesh.ufl_cell()._cellname:
            # ========================================================
            # BRANCH-DEPENDENT PURKINJE DIFFUSION
            # ========================================================
            #
            # For a one-dimensional EP mesh, electrical diffusion
            # may be modified according to the local branch order.
            #
            # Three scaling modes are available:
            #
            # terminal_slow:
            #     Progressively reduces diffusion toward terminal
            #     branches using a quadratic relationship.
            #
            # linear:
            #     Applies a linear branch-order-dependent scaling.
            #
            # bell:
            #     Uses a bell-shaped relationship with maximum
            #     diffusion scaling near normalized branch order 0.6.
            #
            # Branch scaling is bounded below by 0.2.
            #
            # For LBBB simulations, an additional regional
            # conductivity factor is applied to elements
            # identified by matid == 1.
            #
            # The effective diffusion coefficient is:
            #
            # D = d_iso * branch_scaling * lbbb_scaling
            #
            # ========================================================

            branch_id =  self.parameters["branchid"]
            branch_scaling = self.parameters["branch_scaling"]

            V0 = FunctionSpace(mesh, "DG", 0)
            branch_scaling_func = Function(V0)
            scale_local = branch_scaling_func.vector().get_local()

            dofmap = V0.dofmap()
            branch = branch_id.array()  
            kmin = branch.min()
            kmax = branch.max()      

            r = self.parameters["branch_scaling"]  
            mode = self.parameters["branch_scaling_mode"]      # 'terminal_slow', 'bell', 'linear'

            for cell in cells(mesh):
                dof = dofmap.cell_dofs(cell.index())[0]      # DG0 has 1 dof per cell
                k_branch= int(branch[cell.index()])

                if mode == "terminal_slow": # slow down toward terminal branches
                    x = (k_branch- kmin) / (kmax - kmin + 1e-12)
                    p = 2   # or 3 for stronger terminal drop
                    scale = 1 - r * x ** p

                elif mode == "linear": # simple linear slowdown from proximal
                    k_branch= int(branch[cell.index()])                     
                    scale = 1.0 - (k_branch- kmin) * r  

                elif mode == "bell": # fastest mid branches 
                    x = (k_branch- kmin)/(kmax - kmin + 1e-12)
                    raw = np.exp(-((x - 0.6)**2)/(2*0.25**2))   # skewed bell (peak shifted right)
                    scale = 1 - r*(1 - raw)   

                else :     
                    scale = 1.0                     

                scale_local[dof] = max(0.2, scale)  # clamp to avoid very small diffusion

            branch_scaling_func.vector().set_local(scale_local)
            branch_scaling_func.vector().apply("insert")

            if "lbbb" in list(self.parameters.keys()):           
                if self.parameters["lbbb"]:
                    # Create a piecewise scaling Function
                    lbbb_scaling = Function(FunctionSpace(mesh, "DG", 0))
                    lbbb_values = lbbb_scaling.vector()
                    lbbb_values[:] = np.where(matid_ep.array() == 1, self.parameters["lbbb_delay"], 1.0)

                    # Set the final `Dmat` variable
                    Dmat = Constant(self.parameters["d_iso"]) * branch_scaling_func * lbbb_scaling 
                else:
                    Dmat = Constant(self.parameters["d_iso"]) * branch_scaling_func
            else:
                Dmat = Constant(self.parameters["d_iso"]) * branch_scaling_func
        else:
            D_tensor = self.calculateDmat(f0_ep, s0_ep, n0_ep, mesh=mesh, mId=AHAid_ep)
            Dmat = D_tensor


        self.fstim_array = []
        self.delta_array = []
        hmin = mesh.hmin()
        hmax = mesh.hmax()
        ploc_fct = 0.02
        self.havg = ploc_fct*(hmin + hmax)

        # Factor to enlarge the kernel for CRT electrodes
        CRT_eps_factor = self.parameters["CRT_eps_factor"] #2.0   # e.g. 3× wider kernel for CRT
        if self.parameters["CRT"]:
            n_CRT_electrodes = len(self.parameters["CRT_pos"])   # you said: exactly two electrodes        
        else:
            n_CRT_electrodes = 0


        # ============================================================
        # SPATIALLY DISTRIBUTED ELECTRICAL STIMULATION
        # ============================================================
        #
        # Electrical pacing is represented using regularized
        # spatial stimulation kernels centered at prescribed
        # stimulation locations.
        #
        # The spatial kernel is:
        #
        #   delta_eps(x) = eps / [pi*(|x-x0|^2 + eps^2)]
        #
        # The characteristic width is defined as:
        #
        #   eps = 0.02*(hmin + hmax)
        #
        # CRT electrodes use:
        #
        #   eps_CRT = CRT_eps_factor * eps
        #
        # The total stimulation source is:
        #
        #   I_stim(x,t) = sum_p I_p(t)*delta_p(x)
        #
        # Stimulation amplitudes are controlled externally
        # by the Purkinje-myocardial and CRT coupling classes.
        #
        # ============================================================

        first_crt_idx = max(0, self.max_pace_label - n_CRT_electrodes)        

        self.cpp_code0 = """
            
            #include <pybind11/pybind11.h>
            #include <pybind11/eigen.h>
            #include <dolfin/function/Expression.h>
            #include <iostream>
            
            class test_exp : public dolfin::Expression {
              public:
                
                Eigen::VectorXd x0;
                double eps;
                
                test_exp() : dolfin::Expression() { }
        
                void eval(Eigen::Ref<Eigen::VectorXd> values, Eigen::Ref<const Eigen::VectorXd> x) const {
                    double norm_squared = 0.0;
                    for (int i = 0; i < x0.size(); ++i)
                    {
                        norm_squared += (x[i] - x0[i]) * (x[i] - x0[i]);
                    }
                    //values[0] = eps /  std::sqrt(norm_squared + eps * eps);
                    values[0] = eps / M_PI / (norm_squared + eps * eps);
                    //std::cout << values[0] << " " <<  x[0] << " " <<  x[1] << " " << x[2] << std::endl;
                    //std::cout << values[0] << " " <<  x[0] << " " <<  x[1] << " " << x[2] << std::endl;
                }
        
            };
            
            PYBIND11_MODULE(SIGNATURE, m) {
                pybind11::class_<test_exp, std::shared_ptr<test_exp>, dolfin::Expression>
                (m, "test_exp")
                .def(pybind11::init<>())
                .def_readwrite("x0", &test_exp::x0)
                .def_readwrite("eps", &test_exp::eps)
                ;
            }
        """

        for p in np.arange(0, self.max_pace_label):
            fstim = Expression("iStim", iStim=0.000, degree=1)
            self.fstim_array.append(Expression("iStim", iStim=0.000, degree=1))
            x0 = self.ploc[p]
            delta = dolfin.CompiledExpression(dolfin.compile_cpp_code(self.cpp_code0).test_exp(), degree=1)

            # use larger eps ONLY for the last two pacing sites (CRT electrodes)
            if p >= first_crt_idx:
                delta.eps = CRT_eps_factor * self.havg
            else:
                delta.eps = self.havg
     
            delta.x0 = np.array(x0, dtype=float)
            self.delta_array.append(delta)

        dx_ep = dolfin.dx(mesh)

        # ============================================================
        # FINITE-ELEMENT RESIDUAL
        # ============================================================
        #
        # The coupled excitation-recovery system is discretized
        # using an implicit time-integration formulation.
        #
        # The excitation residual contains:
        #
        #   1. Temporal evolution of electrical excitation.
        #   2. Anisotropic electrical diffusion.
        #   3. Nonlinear excitation reaction.
        #   4. Externally applied stimulation currents.
        #
        # The recovery residual contains:
        #
        #   1. Temporal evolution of the recovery variable.
        #   2. Nonlinear recovery kinetics.
        #
        # The nonlinear Jacobian is calculated automatically
        # by differentiating the residual with respect to
        # the mixed electrophysiology solution.
        #
        # No essential Dirichlet boundary conditions are
        # prescribed in the current formulation.
        # ============================================================

        self.F_FHN = (((phi - phi_n) / k) * phi_test * dx_ep + dot(Dmat * grad(phi), grad(phi_test)) * dx_ep - f_phi * phi_test * dx_ep + ((r - r_n) / k) * r_test * dx_ep - f_r * r_test * dx_ep)

        for fstim, delta in zip(self.fstim_array, self.delta_array):
            self.F_FHN -=  fstim * delta * phi_test * dx_ep


        self.J_FHN = derivative(self.F_FHN, w_ep, dw_ep)
        return (self.F_FHN, self.J_FHN, self.fstim_array,)

    def Solver(self):
        solverparams_FHN = {
            "Jacobian": self.J_FHN,
            "F": self.F_FHN,
            "w": self.w_ep,
            "boundary_conditions": [],
            "Type": 0,
            "mesh": self.mesh,
            "mode": 1,
            # --- Add tolerance forwarding here ---
            "abs_tol": self.parameters.get("abs_tol"),
            "rel_tol": self.parameters.get("rel_tol"),              
        }

        solver_FHN = NSolver(solverparams_FHN)
        return solver_FHN

    def UpdateVar(self):
        # Update EP variable
        self.w_n_ep.assign(self.w_ep)
        # Update Stimulus variable
        pace_time_array = self.parameters["pacing_timing"]

    def getphivar(self):
        phi_, r_ = self.w_n_ep.split(deepcopy=True)
        phi_.rename("phi_", "phi_")
        return phi_

    def getrvar(self):
        phi_, r_ = self.w_n_ep.split(deepcopy=True)
        r_.rename("r_", "r_")
        return r_

    def interpolate_potential_ep2me_phi(self, V_me):
        LagrangeInterpolator.interpolate(V_me, self.getphivar())
        return V_me

    def clamp_variables(self, lower_bound=-0.05):
        """Clamp phi and r components of the mixed Function w_ep."""
        phi, r = self.w_ep.split(deepcopy=True)

        # Clamp phi
        phi_vals = phi.vector().get_local()
        phi_vals = np.maximum(phi_vals, lower_bound)
        phi.vector().set_local(phi_vals)
        phi.vector().apply("insert")

        # Clamp r
        r_vals = r.vector().get_local()
        r_vals = np.maximum(r_vals, lower_bound)
        r.vector().set_local(r_vals)
        r.vector().apply("insert")

        # Re-assign to mixed function
        assign(self.w_ep, [phi, r])


    def reset(self):
        self.w_n_ep.vector().zero()
        self.w_ep.vector().zero()

        # sync across MPI
        self.w_n_ep.vector().apply("insert")
        self.w_ep.vector().apply("insert")

        # Only reset r if it exists
        if hasattr(self, "r_ep"):
            r_vec = self.r_ep.vector()
            self.FHNmodel.Zero_r(r_vec)
            self.r_ep.vector().apply("insert")

        return

class Delta(UserExpression):
    def __init__(self, eps, x0, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.x0 = x0

    def eval(self, values, x):
        eps = self.eps
        x0 = self.x0
        values[0] = eps/pi/(np.linalg.norm(x-x0)**2 + eps**2)

    def value_shape(self):
        return ()