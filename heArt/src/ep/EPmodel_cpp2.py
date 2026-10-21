from dolfin import *
import numpy as np
import os
import inspect

from ufl import indices
import dolfin as dolfin
from mpi4py import MPI as pyMPI
import pdb
from petsc4py import PETSc

class EPmodel(object):
    """
    Finite-element electrophysiology model for the Purkinje fiber network.

    The model describes electrical excitation propagation through a
    one-dimensional Purkinje network using an Aliev-Panfilov-type
    reaction-diffusion formulation.

    The electrophysiology model consists of two state variables:

        phi : Dimensionless electrical excitation variable.
        r   : Electrical recovery variable.

    Electrical propagation is controlled by a spatially varying
    diffusion coefficient accounting for:

        1. Baseline Purkinje conductivity.
        2. Branch-order-dependent conductivity variations.
        3. Left bundle branch block (LBBB).
        4. Ischemia-related conductivity reductions.

    Electrical stimulation is represented using spatially distributed
    source functions centered at prescribed pacing locations.

    The model supports stimulation originating from native Purkinje
    pacing sites and additional stimulation introduced through
    myocardium-to-Purkinje reverse coupling.

    Numerical formulation
    ---------------------
    Spatial discretization:
        Continuous first-order Lagrange finite elements (CG1).

    Time integration:
        Semi-implicit treatment of the reaction-diffusion system.

    Diffusion:
        Implicit finite-element solution using a lumped mass matrix.

    Reaction kinetics:
        Evaluated through the compiled C++ implementation in fhn.cpp.

    Linear solver:
        PETSc LU solver.

    Notes
    -----
    The electrophysiology model is coupled to the myocardial model
    through separate Purkinje-to-myocardium and myocardium-to-Purkinje
    coupling operators.

    The coupling operators control stimulation sources but do not
    directly overwrite the Purkinje electrical state variables.
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

        # ============================================================
        # ELECTROPHYSIOLOGY PARAMETERS AND TIME NORMALIZATION
        # ============================================================
        #
        # The electrophysiology model uses normalized time to relate
        # the simulation timestep to the excitation-recovery kinetics.
        #
        # The normalized timestep is:
        #
        #     k = dt / fhn_timeNormalizer
        #
        # where:
        #
        #     dt                 : Simulation timestep
        #     fhn_timeNormalizer : Time scaling parameter
        #
        # The default normalization factor is 12.9.
        #
        # The electrophysiology reaction parameters are:
        #
        #     g, mu1, mu2 : Recovery kinetics parameters
        #     c           : Excitation/recovery scaling coefficient
        #     alpha       : Excitation threshold parameter
        #     b           : Recovery parameter
        #
        # These parameters are passed to the compiled
        # Aliev-Panfilov reaction model in fhn.cpp.
        # ============================================================
        if "dt" in list(self.parameters.keys()):
            dt = self.parameters["dt"]
        else:
            dt = Constant(1.0) 

        if "fhn_timeNormalizer" in list(self.parameters.keys()):
            fhn_timeNormalizer = self.parameters["fhn_timeNormalizer"]
        else:
            fhn_timeNormalizer = 12.9 
 
        self.k = dt/fhn_timeNormalizer
        self.EPparams = {
            "g": 0.002,
            "mu1": 0.2,
            "mu2": 0.3,
            "c": 8,
            "alpha": 0.01,
            "b": 0.15,
            "k": self.k,
        }

        if "d_iso" in list(self.parameters.keys()):
            self.d_iso = self.parameters["d_iso"]
        else:
            self.d_iso = 1.0

        if "d_ani" in list(self.parameters.keys()):
            self.d_ani = self.parameters["d_ani"]
        else:
            self.d_ani = 0.0

        if "ani_factor" in list(self.parameters.keys()):
            self.ani_factor = self.parameters["ani_factor"]
        else:
            self.ani_factor = 1.0

        # ============================================================
        # FINITE-ELEMENT DISCRETIZATION
        # ============================================================
        #
        # The electrical excitation variable phi is discretized
        # using continuous first-order Lagrange elements (CG1).
        #
        # The recovery variable r is stored in the same function
        # space but updated separately through the compiled
        # reaction model.
        #
        # State variables:
        #
        #     w_ep    : Current excitation solution
        #     w_n_ep  : Previous excitation solution
        #     r_ep    : Recovery variable
        #     f_phi_  : Excitation reaction term
        #
        # Additional PETSc vectors:
        #
        #     t1_ep        : Right-hand-side workspace
        #     I_source_vec : Assembled electrical stimulation source
        #
        # These variables are used to construct and solve the
        # discretized electrophysiology equations.
        # ============================================================
        P1_ep = FiniteElement("CG", self.mesh.ufl_cell(), 1, quad_scheme="default")
        P1_ep._quad_scheme = "default"

        self.W_ep = FunctionSpace(self.mesh, P1_ep)
        self.w_ep = Function(self.W_ep)
        self.r_ep = Function(self.W_ep)
        self.f_phi_ = Function(self.W_ep)
        self.dw_ep = TrialFunction(self.W_ep)
        self.wtest_ep = TestFunction(self.W_ep)
        self.w_n_ep = Function(self.W_ep)
        self.t1_ep = PETScVector()
        self.t1_ep = self.w_n_ep.vector().copy()
        self.I_source = Function(self.W_ep)

        self.I_source_vec = PETScVector()
        self.I_source_vec = self.w_n_ep.vector().copy()


        self.Mass_P, self.MD_matrix, self.fstim_val_array, self.FHNmodel = self.Problem()

    def Solver(self):
        return self.SolverEP(self)

    def default_parameters(self):
        return {"ploc": [[0.0, 0.0, 0.0, 100.0, 1]], "pacing_timing": [[1, 20.0]],}

    def calculateDmat(self, f0, s0, n0, mesh):
        d_iso = self.d_iso 
        d_ani = self.d_ani  
        ani_factor = self.ani_factor

        d_Ani = d_ani

        dParam = {}

        D_iso = Constant(((d_iso, "0.0", "0.0"), ("0.0", d_iso, "0.0"), ("0.0", "0.0", d_iso)))

        i, j = indices(2)
        Dij = (d_Ani)*f0[i]*f0[j] + (d_Ani*1/ani_factor)*s0[i]*s0[j] + (d_Ani*1/ani_factor)*n0[i]*n0[j]+ D_iso[i,j]
        D_tensor = as_tensor(Dij, (i, j))
        return D_tensor

    def MarkStimulus(self):
        mesh = self.mesh
        ploc = self.parameters["ploc"]
        EpiBCid_ep = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
        EpiBCid_ep.set_all(0)

        for ploc_ in ploc:
            class Omega_0(SubDomain):
                def inside(self, x, on_boundary):
                    return (
                        (x[0] - ploc_[0]) ** 2
                        + (x[1] - ploc_[1]) ** 2
                        + (x[2] - ploc_[2]) ** 2
                    ) <= ploc_[3] ** 2

            subdomain_0 = Omega_0()
            subdomain_0.mark(EpiBCid_ep, int(ploc_[4]))

        # Find maximum number of pace label
        self.max_pace_label = int(max(np.array(ploc)[:, 4]))
        return EpiBCid_ep

    def Problem(self):
        state_obj = self.parameters["state_obj"]

        if "f0" in list(self.parameters.keys()):
            f0_ep = self.parameters["f0"]

        if "s0" in list(self.parameters.keys()):
            s0_ep = self.parameters["s0"]

        if "n0" in list(self.parameters.keys()):
            n0_ep = self.parameters["n0"]

        if "AHAid_ep" in list(self.parameters.keys()):
            AHAid_ep = self.parameters["AHAid"]

        if "matid" in list(self.parameters.keys()):
            matid_ep = self.parameters["matid"]

        if "ischemia_mask" in list(self.parameters.keys()):
            ischemia_ep = self.parameters["ischemia_mask"]            

        if "facetboundaries" in list(self.parameters.keys()):
            self.facetboundaries_ep = self.parameters["facetboundaries"]
        else:
            self.facetboundaries_ep = None

        mesh = self.mesh

        W_ep = self.W_ep
        w_n_ep = self.w_n_ep
        w_ep = self.w_ep
        dw_ep = self.dw_ep
        wtest_ep = self.wtest_ep

        # ============================================================
        # LUMPED FINITE-ELEMENT MASS MATRIX
        # ============================================================
        #
        # The consistent finite-element mass matrix is:
        #
        #     M_ij = integral_Omega N_i * N_j dx
        #
        # where N_i and N_j are the finite-element basis functions.
        #
        # A lumped mass matrix is constructed by replacing the
        # consistent matrix with a diagonal approximation.
        #
        # The diagonal entries are obtained from:
        #
        #     M_lumped(i,i) = sum_j M_ij
        #
        # Mass lumping simplifies the reaction-diffusion update
        # and reduces the computational cost of the time-dependent
        # electrophysiology formulation.
        # ============================================================
        mass_form = dw_ep*wtest_ep*dx
        mass_action_form = action(mass_form, Constant(1))
        M_lumped = assemble(mass_form)
        M_lumped.zero()
        M_lumped.set_diagonal(assemble(mass_action_form))
        M_lumped_P = as_backend_type(M_lumped).mat()
        M_lumped_P = PETScMatrix(M_lumped_P)
        self.Mass_P = M_lumped_P.copy()
        pv = PETSc.Viewer()

        #-----------------------------------------------------------------------------

        # Create Conductivity matrix (A) ---------------------------------------------
        comm = mesh.mpi_comm()  # MPI Communicator
        if 'interval' in mesh.ufl_cell()._cellname:
            # ========================================================
            # PURKINJE ELECTRICAL DIFFUSION
            # ========================================================
            #
            # Electrical propagation through the one-dimensional
            # Purkinje network is controlled by a spatially varying
            # diffusion coefficient.
            #
            # The effective diffusion coefficient is:
            #
            #   D_PJ(x) = d_iso
            #             * s_branch(x)
            #             * s_LBBB(x)
            #             * s_ischemia(x)
            #
            # where:
            #
            #   d_iso       : Baseline diffusion coefficient
            #   s_branch    : Branch-order-dependent scaling
            #   s_LBBB      : Regional LBBB modification
            #   s_ischemia  : Regional ischemia modification
            #
            # Each scaling factor modifies the existing
            # diffusion coefficient multiplicatively.
            #
            # Therefore, LBBB and ischemia can be applied
            # simultaneously to the Purkinje network.
            # ========================================================
            V0 = FunctionSpace(mesh, "DG", 0)

            # -------------------------
            # 1. BRANCH SCALING
            # -------------------------
            branch_id = self.parameters["branchid"]
            branch = branch_id.array()

            branch_scaling_func = Function(V0)
            scale_local = branch_scaling_func.vector().get_local()

            dofmap = V0.dofmap()

            kmin = branch.min()
            kmax = branch.max()

            r = self.parameters["branch_scaling"]
            mode = self.parameters["branch_scaling_mode"]

            for cell in cells(mesh):
                dof = dofmap.cell_dofs(cell.index())[0]
                k_branch = int(branch[cell.index()])

                # Branch-dependent diffusion supports three modes:
                #
                # terminal_slow:
                #     Quadratic reduction toward higher branch orders.
                #
                # linear:
                #     Linear reduction with increasing branch order.
                #
                # bell:
                #     Piecewise branch-order scaling with maximum
                #     diffusion near 60% of the branch-order range.
                #     The proximal and distal sides use different
                #     normalized distances, with additional proximal
                #     scaling.
                #
                # The final branch scaling is bounded below by 0.2
                # to avoid excessively small diffusion coefficients.
                if mode == "terminal_slow":
                    x = (k_branch - kmin) / (kmax - kmin + 1e-12)
                    p = 2
                    scale = 1 - r * x**p

                elif mode == "linear":
                    scale = 1.0 - (k_branch - kmin) * r

                elif mode == "bell":
                    kmid = int(round(0.6 * (kmax - kmin) + kmin))
                    dist = abs(k_branch - kmid)

                    dmax_left  = kmid - kmin
                    dmax_right = kmax - kmid

                    if k_branch <= kmid:
                        dnorm = dist / (dmax_left + 1e-12)
                    else:
                        dnorm = dist / (dmax_right + 1e-12)

                    skew = 1.2 if k_branch < kmid else 1.0
                    scale = 1 - r * skew * dnorm

                else:
                    scale = 1.0

                scale_local[dof] = max(0.2, scale)

            branch_scaling_func.vector().set_local(scale_local)
            branch_scaling_func.vector().apply("insert")

            Dmat = Constant(self.d_iso) * branch_scaling_func

            # ========================================================
            # LEFT BUNDLE BRANCH BLOCK
            # ========================================================
            #
            # LBBB is modeled by modifying electrical diffusion
            # within the prescribed Purkinje network region.
            #
            # Network segments with:
            #
            #     matid == 1
            #
            # receive the conductivity scaling:
            #
            #     s_LBBB = lbbb_delay
            #
            # All other segments retain:
            #
            #     s_LBBB = 1.0
            #
            # The LBBB scaling multiplies the branch-dependent
            # diffusion coefficient.
            #
            # Setting lbbb_delay = 0 removes the diffusion
            # contribution within the affected network region,
            # while intermediate values represent partial
            # conductivity reduction.
            #
            # Note that lbbb_delay acts as a diffusion scaling
            # parameter rather than an explicitly imposed
            # activation-time delay.
            # ========================================================
            if self.parameters.get("lbbb", False):
                lbbb_scaling = Function(FunctionSpace(mesh, "DG", 0))
                lbbb_values = lbbb_scaling.vector()
                lbbb_values[:] = np.where(matid_ep.array() == 1, self.parameters["lbbb_delay"],1.0)
                Dmat = Dmat * lbbb_scaling

            # ========================================================
            # ISCHEMIA-DEPENDENT PURKINJE DIFFUSION
            # ========================================================
            #
            # Ischemia is represented through a spatial mask
            # identifying the affected Purkinje segments.
            #
            # For healthy segments:
            #
            #     s_ischemia = 1.0
            #
            # For ischemic segments:
            #
            #     s_ischemia = Ischemic_pj_factor
            #
            # The resulting diffusion coefficient is:
            #
            #     D_PJ = D_baseline * s_ischemia
            #
            # The ischemic factor is applied in addition to
            # branch-order and LBBB modifications.
            #
            # This permits simulation of combined ischemia
            # and conduction block within the Purkinje network.
            # ========================================================
            if self.parameters.get("Ischemia", False):
                ischemia_scaling = Function(FunctionSpace(mesh, "DG", 0))
                ischemia_values = ischemia_scaling.vector()
                ischemia_mask = self.parameters["ischemia_mask"]
                pj_isch_factor = float(self.parameters.get("Ischemic_pj_factor", 0.1))                
                ischemia_values[:] = np.where(ischemia_mask.array() == 1,pj_isch_factor,1.0)
                Dmat = Dmat * ischemia_scaling

            # -------------------------
            # 5. FINAL ASSEMBLY
            # -------------------------
            A = PETScMatrix()
            Fconduct = inner(Dmat * grad(dw_ep), grad(wtest_ep)) * dx
            assemble(Fconduct, tensor=A)

        # ============================================================
        # PURKINJE ELECTRICAL STIMULATION SOURCES
        # ============================================================
        #
        # Electrical stimulation is represented using spatially
        # distributed source functions centered at prescribed
        # stimulation locations.
        #
        # The spatial kernel is:
        #
        #     delta_eps(x) =
        #         eps / [pi*(|x-x0|^2 + eps^2)]
        #
        # where:
        #
        #     x0  : Stimulation location
        #     eps : Kernel width
        #
        # The kernel width is defined using the mesh size:
        #
        #     eps = 0.02*(hmin + hmax)
        #
        # Each stimulation location has an associated
        # preassembled finite-element source vector.
        #
        # The total source is constructed by combining
        # these vectors with their stimulation coefficients:
        #
        #     I_source = sum_p I_p(t)*S_p
        #
        # where S_p is the spatial source vector associated
        # with stimulation location p.
        #
        # Reverse myocardium-to-Purkinje coupling can add
        # additional temporary source vectors to I_source
        # before the Purkinje electrophysiology solve.
        # ============================================================
        self.fstim_array = []
        self.fstim_val_array = []
        self.delta_array = []
        hmin = mesh.hmin()
        hmax = mesh.hmax()
        havg = 0.02*(hmin + hmax)

        cpp_code0 = """
            
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
            self.fstim_val_array.append(0)
            x0 = self.ploc[p]
            delta = dolfin.CompiledExpression(dolfin.compile_cpp_code(cpp_code0).test_exp(), degree=1)
            delta.eps = havg
            delta.x0 = np.array(x0, dtype=float)
            self.delta_array.append(delta)

        dx_ep = dolfin.dx(mesh)

        for delta in self.delta_array:
            fstim_vector = self.I_source_vec.copy()
            fstim_form = Constant(1.0) * delta * wtest_ep * dx_ep
            assemble(fstim_form, tensor=fstim_vector)
            self.fstim_array.append(fstim_vector)

        # Create fphi (that will be updated in cpp) ----------------------------------
        F_phi = Constant(0.0)*wtest_ep*dx
        assemble(F_phi, tensor=self.f_phi_.vector())
        #-----------------------------------------------------------------------------

        # ============================================================
        # COMPILED EXCITATION-RECOVERY MODEL
        # ============================================================
        #
        # The nonlinear electrophysiology reaction kinetics are
        # evaluated using the compiled C++ implementation in
        # fhn.cpp.
        #
        # The C++ model receives the electrophysiology
        # parameters defined during initialization.
        #
        # It provides operations for:
        #
        #     Update_fphi:
        #         Evaluate the excitation reaction term.
        #
        #     Update_r:
        #         Advance the recovery variable.
        #
        #     Zero_r:
        #         Reset the recovery variable.
        #
        # The excitation reaction term is evaluated before
        # solving the diffusion equation.
        #
        # The recovery variable is updated separately during
        # the electrophysiology state update.
        #
        # The exact reaction equations and recovery update
        # should be documented alongside fhn.cpp.
        # ============================================================
        directory = os.path.dirname(inspect.getfile(self.__class__))
        cpp_file = open(os.path.join(directory,"fhn.cpp"),"r")
        code = cpp_file.read()
        cpp_file.close()
        ext_module = compile_cpp_code(code)
        self.FHNmodel = ext_module.FHNmodel(self.EPparams)
        #-----------------------------------------------------------------------------

        # ============================================================
        # SEMI-IMPLICIT TIME INTEGRATION
        # ============================================================
        #
        # The discretized excitation equation is solved using:
        #
        #     (M + k*A)*phi_(n+1)
        #         = M*[phi_n + k*f_phi(phi_n,r_n)]
        #           + k*I_source
        #
        # where:
        #
        #     M        : Lumped mass matrix
        #     A        : Electrical diffusion stiffness matrix
        #     k        : Normalized timestep
        #     f_phi    : Excitation reaction term
        #     I_source : Electrical stimulation source vector
        #
        # Diffusion is treated implicitly through the matrix
        # M + k*A, while the excitation reaction contribution
        # is evaluated explicitly from the current state.
        #
        # The resulting linear system is solved using
        # a PETSc LU solver.
        # ============================================================
        self.MD_matrix = self.Mass_P.copy()
        self.MD_matrix.mat().axpy(float(self.k), A.mat())
        #-----------------------------------------------------------------------------
         
        return (self.Mass_P, self.MD_matrix, self.fstim_val_array, self.FHNmodel)

    class SolverEP:
        """
        Linear solver for the Purkinje excitation equation.

        The solver advances the electrical excitation variable
        by evaluating the nonlinear reaction term and solving
        the semi-implicit reaction-diffusion system.

        The recovery variable is updated separately through
        the compiled reaction model.

        The solver uses a PETSc LU factorization for the
        linear system defined by M + k*A.
        """

        def __init__(self, EPmodel):

            self.EPmodel = EPmodel
            self.solver = PETScLUSolver()  # Use direct LU solver if preferred
            self.r_ep = EPmodel.r_ep
            self.w_ep = EPmodel.w_ep
            self.w_n_ep = EPmodel.w_n_ep
            self.f_phi_ = EPmodel.f_phi_
            self.FHNmodel = EPmodel.FHNmodel
            self.Mass_P = EPmodel.Mass_P
            self.t1_ep = EPmodel.t1_ep
            self.k = EPmodel.k
            self.I_source_vec = EPmodel.I_source_vec
            self.MD_matrix = EPmodel.MD_matrix

        def solvenonlinear(self):
            # --------------------------------------------------------
            # PURKINJE EXCITATION UPDATE
            # --------------------------------------------------------
            #
            # 1. Evaluate the nonlinear excitation reaction term.
            # 2. Add the reaction contribution to the previous
            #    excitation state.
            # 3. Multiply by the lumped mass matrix.
            # 4. Add the electrical stimulation source.
            # 5. Solve the implicit diffusion system.
            #
            # The recovery variable is updated separately
            # during UpdateVar().
            # --------------------------------------------------------    
            r_vec = self.r_ep.vector()
            w_vec = self.w_ep.vector()
            f_phi_vec = self.f_phi_.vector()
    
            # Update fphi using cpp module
            self.FHNmodel.Update_fphi(f_phi_vec, w_vec, r_vec)
            self.f_phi_.vector().set_local(f_phi_vec)
        
            # Solve for u for (M + kA)*u = M*(un + k*fphi + I)
            self.w_n_ep.vector().axpy(float(self.k), self.f_phi_.vector())
            self.Mass_P.mult(self.w_n_ep.vector(), self.t1_ep)
            self.t1_ep.axpy(float(self.k), self.I_source_vec)
            self.solver.solve(self.MD_matrix, self.w_ep.vector(), self.t1_ep)
            if np.isnan(np.min(self.w_ep.vector().get_local())):
                print("Nan encountered")
                exit(1)

    def UpdateActivation(self):
        """
        Assemble the prescribed Purkinje stimulation sources.

        The existing stimulation coefficients are multiplied
        by their associated preassembled spatial source vectors.

        The resulting source is:

            I_source = sum_p I_p(t)*S_p

        This method resets and reconstructs I_source_vec from
        the prescribed Purkinje stimulation sources.

        The optional myocardium-to-Purkinje coupling subsequently
        adds temporary reverse-capture sources to this vector.

        Therefore, the intended update order is:

            1. UpdateActivation()
            2. UpdateMyoandPJ()
            3. Purkinje electrophysiology solve

        This ordering preserves native stimulation while
        allowing additional reverse-coupling currents.
        """

        self.I_source_vec[:] = 0
        for p in np.arange(0, self.max_pace_label):
            self.I_source_vec.axpy(self.fstim_val_array[p], self.fstim_array[p])


    def UpdateVar(self):
        """
        Advance the Purkinje electrophysiology state variables.

        The current excitation solution is copied to the
        previous-state variable.

        The recovery variable is then advanced using the
        compiled reaction model in fhn.cpp.

        The temporary right-hand-side vector is reset for
        the next electrophysiology timestep.
        """
        self.w_n_ep.assign(self.w_ep)

        # Update Stimulus variable
        pace_time_array = self.parameters["pacing_timing"]

        # Update internal variable r using cpp module
        r_vec = self.r_ep.vector()
        w_vec = self.w_ep.vector()
        f_phi_vec = self.f_phi_.vector()

        self.FHNmodel.Update_r(w_vec, r_vec, self.EPparams)
        self.r_ep.vector().set_local(r_vec)
        self.t1_ep[:] = 0

    def Reset(self):

        self.w_ep.assign(self.w_n_ep)

    def getphivar(self):
        phi_ = self.w_ep
        phi_.rename("phi_", "phi_")
       
        return phi_

    def getrvar(self):
        r_ = self.r_ep
        r_.rename("r_", "r_")

        return r_

    def interpolate_potential_ep2me_phi(self, V_me):
        LagrangeInterpolator.interpolate(V_me, self.getphivar())
        return V_me

    def reset(self):
        """
        Reset Purkinje excitation and recovery variables.

        The excitation state is initialized to zero, and
        distributed PETSc vectors are synchronized.

        The recovery variable is reset using the compiled
        FHNmodel.Zero_r() method.

        This method is called at the beginning of a new
        cardiac cycle when the simulation requires
        electrical-state reinitialization.
        """        
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