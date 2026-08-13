"""
nlmevi_core: shared infrastructure for the Omega-shrinkage / IW-VI project
============================================================================

Everything in this file is stable, validated machinery that every phase
script (phase0, phase1, phase2's baseline/deltaOFV/realdata scripts, phase3)
imports rather than reimplements. Nothing in here is phase-specific --
phase-specific content (which scenarios to run, what to plot, CLI flags,
narrative print statements) lives in the phase scripts themselves.

Contents:
    - Generative models: OneCmtOral (linear, analytic), OneCmtIVBolusMM
      (nonlinear, RK4)
    - Variational posteriors: FreePosterior, AmortizedPosterior (Gaussian),
      FlowPosterior (conditional normalizing flow), make_posterior (dispatch)
    - Objectives: iw_elbo (with mc_reps variance reduction), is_marginal_loglik
      (post-hoc IS marginal likelihood + PSIS-style diagnostics)
    - Training: train_to_convergence (adaptive, parameter-based stopping),
      fit_model (the one trainer every phase script calls)

Design rule carried through every script that imports this module: ONE SEED
GENERATES ONE DATASET. Nothing in this module's training code ever touches a
data-generating RNG -- that's the caller's job, and it's what keeps every
method arm in an experiment comparable.
"""

import math
import time

import numpy as np
import torch
import torch.nn as nn

# float64 everywhere. Importance weights span many orders of magnitude and
# float32 will silently destroy high-K arms.
torch.set_default_dtype(torch.float64)

DEVICE = torch.device("cpu")   # default; call set_device() to change
# The default has stayed CPU deliberately for the small linear-tier models
# (tiny per-subject tensors, GPU kernel-launch overhead would dominate).
# The nonlinear (RK4) tier is a different story: many small SEQUENTIAL
# elementwise ops per forward pass, over potentially large (K, N) batches --
# whether GPU actually helps there is an empirical question, not something
# to assume either way. set_device() makes it a one-line, reversible choice
# rather than something baked into the model code.


def set_device(device_str="auto"):
    """
    Resolves 'auto' / 'cpu' / 'cuda' / 'mps' to an actual torch.device and
    updates the module-level DEVICE used throughout fitting (to_tensors,
    fit_model's theta creation, make_posterior's .to(DEVICE) call, etc. all
    read this at CALL time, not import time, so calling this before fitting
    is sufficient -- no need to touch anything else).

    'auto' prefers CUDA, then MPS (Apple Silicon GPU backend -- NOT CUDA;
    Apple Silicon has no CUDA support, this is PyTorch's separate Metal
    backend), falling back to CPU if neither is available.

    IMPORTANT: MPS does NOT support float64, categorically -- this is a
    permanent Metal/MPS backend limitation, not a version-specific gap that
    will be fixed. This project uses float64 everywhere on purpose
    (importance weights at high K span many orders of magnitude; float32
    would silently corrupt exactly the K=64 arm that matters most for the
    corrected results). Requesting 'mps' or getting it via 'auto' therefore
    raises immediately here, with an explanation, rather than failing deep
    inside model construction with a confusing partial traceback the first
    time a tensor actually hits the unsupported path. There is no float64
    workaround for MPS -- if you need GPU acceleration for this codebase,
    it would require a CUDA machine (and even then, verify the
    numerical-stability comments in iw_elbo/is_marginal_loglik still hold
    at whatever precision you end up using).

    Call this ONCE near the top of a script, before any fit_model() calls.
    Returns the resolved device for confirmation/logging.
    """
    global DEVICE
    if device_str == "auto":
        if torch.cuda.is_available():
            DEVICE = torch.device("cuda")
        elif torch.backends.mps.is_available():
            raise RuntimeError(
                "auto-detected MPS (Apple Silicon GPU), but MPS does NOT "
                "support float64 (a permanent backend limitation, not a "
                "bug) and this project requires float64 throughout for "
                "numerical stability at high K. Falling back silently to "
                "CPU would hide that 'auto' picked something incompatible; "
                "explicitly pass --device cpu instead -- that's the only "
                "option this codebase supports on Apple Silicon."
            )
        else:
            DEVICE = torch.device("cpu")
    elif device_str == "mps":
        raise RuntimeError(
            "MPS does NOT support float64 (a permanent Apple/Metal backend "
            "limitation, not a bug or version gap) and this project "
            "requires float64 throughout -- importance weights at high K "
            "span many orders of magnitude, and float32 would silently "
            "corrupt exactly the K=64 arm the corrected results depend on. "
            "There is no float64 workaround for MPS. Use --device cpu."
        )
    else:
        DEVICE = torch.device(device_str)
    return DEVICE


# %% =================================================================
#  Generative model 1: 1-cmt oral PK, analytic solution (linear tier)
# =====================================================================
from dataclasses import dataclass


@dataclass
class TrueParams:
    """Ground truth for the linear-PK model. Everything downstream is
    measured against these numbers."""
    CL: float = 3.0        # L/h
    V: float = 30.0        # L
    ka: float = 1.2        # 1/h
    om_CL: float = 0.30    # SD on log scale (~30% CV)
    om_V: float = 0.20
    om_ka: float = 0.40
    sigma: float = 0.20    # residual SD on log concentration
    dose: float = 100.0    # mg

    def eta_sd(self):
        return np.array([self.om_CL, self.om_V, self.om_ka])


def conc_analytic(t, CL, V, ka, dose):
    """
    1-compartment, first-order absorption, single bolus dose into depot.

        C(t) = (D*ka) / (V*(ka - ke)) * (exp(-ke*t) - exp(-ka*t)),   ke = CL/V

    Closed form, so no ODE solver is needed. This is the single biggest speed
    decision in the linear tier: it makes a full ablation grid cheap enough
    to run at high replicate counts on a laptop CPU.

    Shapes broadcast: t is (..., T), CL/V/ka are (..., 1).
    """
    ke = CL / V
    # Guard the removable singularity at ka == ke (flip-flop boundary).
    denom = ka - ke
    denom = torch.where(denom.abs() < 1e-8, torch.full_like(denom, 1e-8), denom)
    return (dose * ka) / (V * denom) * (torch.exp(-ke * t) - torch.exp(-ka * t))


class OneCmtOral:
    """
    The linear-tier population model. Knows how to evaluate log p(eta|theta)
    and log p(y|eta,theta) and NOTHING about how inference is done.

    That separation is the load-bearing abstraction of the whole project: any
    inference arm (free/amortized posterior, gaussian/flow family, any K)
    talks to this object only through `log_joint`. This is also what lets
    OneCmtIVBolusMM (nonlinear, RK4-integrated) below be a drop-in
    replacement for this model everywhere else in the codebase.

    theta (all unconstrained, log scale):
        [log CL, log V, log ka, log om_CL, log om_V, log om_ka, log sigma]
    """
    n_eta = 3
    n_theta = 7

    def __init__(self, dose):
        self.dose = dose

    @staticmethod
    def unpack(theta):
        tv = theta[:3]                    # log typical values
        om = torch.exp(theta[3:6])        # random-effect SDs
        sigma = torch.exp(theta[6])       # residual SD
        return tv, om, sigma

    def log_prior(self, eta, theta):
        """log p(eta | theta), summed over the 3 random effects. eta: (..., N, 3)"""
        _, om, _ = self.unpack(theta)
        return (-0.5 * (eta / om) ** 2 - torch.log(om) - 0.5 * math.log(2 * math.pi)).sum(-1)

    def predict(self, eta, theta, t):
        """Individual predicted concentrations. eta (...,N,3), t (N,T) -> (...,N,T)"""
        tv, _, _ = self.unpack(theta)
        CL = torch.exp(tv[0] + eta[..., 0:1])
        V = torch.exp(tv[1] + eta[..., 1:2])
        ka = torch.exp(tv[2] + eta[..., 2:3])
        return conc_analytic(t, CL, V, ka, self.dose)

    def log_lik(self, logy, eta, theta, t, mask):
        """
        log p(y | eta, theta) with exponential residual error, i.e. Gaussian on
        log concentration. Summed over observations. mask handles ragged designs.
        """
        _, _, sigma = self.unpack(theta)
        pred = self.predict(eta, theta, t).clamp_min(1e-12)
        resid = logy - torch.log(pred)
        ll = -0.5 * (resid / sigma) ** 2 - torch.log(sigma) - 0.5 * math.log(2 * math.pi)
        return (ll * mask).sum(-1)

    def log_joint(self, logy, eta, theta, t, mask):
        """The ONLY interface the inference code uses."""
        return self.log_prior(eta, theta) + self.log_lik(logy, eta, theta, t, mask)


def simulate(tp: TrueParams, n_subj, times, seed):
    """
    Generate one replicate of linear-tier data. `seed` fully determines the
    output. Returns numpy arrays so the dataset is a plain, hashable artifact
    that every inference arm receives identically.
    """
    rng = np.random.default_rng(seed)
    eta = rng.normal(0.0, 1.0, size=(n_subj, 3)) * tp.eta_sd()

    CL = tp.CL * np.exp(eta[:, 0:1])
    V = tp.V * np.exp(eta[:, 1:2])
    ka = tp.ka * np.exp(eta[:, 2:3])
    ke = CL / V

    t = np.tile(np.asarray(times, dtype=float), (n_subj, 1))
    C = (tp.dose * ka) / (V * (ka - ke)) * (np.exp(-ke * t) - np.exp(-ka * t))
    logy = np.log(np.clip(C, 1e-12, None)) + rng.normal(0.0, tp.sigma, size=C.shape)

    return dict(
        logy=logy, t=t,
        mask=np.ones_like(logy),
        eta_true=eta, seed=seed,
    )


def to_tensors(data):
    return {k: torch.as_tensor(v, device=DEVICE) for k, v in data.items()
            if k in ("logy", "t", "mask")}


LINEAR_PARAM_NAMES = ["CL", "V", "ka", "om_CL", "om_V", "om_ka", "sigma"]


def true_vector_linear(tp):
    return np.array([tp.CL, tp.V, tp.ka, tp.om_CL, tp.om_V, tp.om_ka, tp.sigma])


# %% =================================================================
#  Generative model 2: 1-cmt IV bolus, Michaelis-Menten elimination
#  (nonlinear tier -- no closed form, fixed-step RK4)
# =====================================================================
@dataclass
class MMTrueParams:
    """IV bolus into a single compartment, saturable (MM) elimination."""
    Vmax: float = 8.0       # amount/h
    Km: float = 2.0         # concentration units
    V: float = 30.0         # L
    om_Vmax: float = 0.30
    om_Km: float = 0.30
    om_V: float = 0.20
    sigma: float = 0.20
    dose: float = 100.0

    def eta_sd(self):
        return np.array([self.om_Vmax, self.om_Km, self.om_V])


def build_grid(times_obs, dt_max=0.1):
    """
    Fine RK4 grid that includes the observation times EXACTLY, so predictions
    at those times require no interpolation -- just an index lookup.
    """
    t_end = max(times_obs)
    fine = np.arange(0.0, t_end + 1e-9, dt_max)
    grid = np.unique(np.concatenate([fine, times_obs, [0.0]]))
    grid.sort()
    obs_idx = np.searchsorted(grid, times_obs)
    return grid, obs_idx


class OneCmtIVBolusMM:
    """
    dA/dt = -Vmax * (A/V) / (Km + A/V),   A(0) = dose

    No closed form. Integrated with fixed(-ish)-step RK4 through a grid that
    hits the observation times exactly. Same log_joint interface as
    OneCmtOral, so all the inference code in this module is agnostic to
    which one it's given.

    theta (log scale): [log Vmax, log Km, log V, log om_Vmax, log om_Km, log om_V, log sigma]
    """
    n_eta = 3
    n_theta = 7

    def __init__(self, dose, grid_np, obs_idx_np):
        self.dose = dose
        self.t_grid = torch.tensor(grid_np, device=DEVICE)
        self.dts = self.t_grid[1:] - self.t_grid[:-1]
        self.obs_idx = obs_idx_np

    @staticmethod
    def unpack(theta):
        tv = theta[:3]
        om = torch.exp(theta[3:6])
        sigma = torch.exp(theta[6])
        return tv, om, sigma

    def log_prior(self, eta, theta):
        _, om, _ = self.unpack(theta)
        return (-0.5 * (eta / om) ** 2 - torch.log(om) - 0.5 * math.log(2 * math.pi)).sum(-1)

    def _integrate(self, A0, Vmax, Km, V):
        A = A0
        outs = [A]
        for i in range(self.dts.shape[0]):
            dt = self.dts[i]
            def f(a):
                C = a / V
                return -Vmax * C / (Km + C + 1e-8)
            k1 = f(A)
            k2 = f(A + dt / 2 * k1)
            k3 = f(A + dt / 2 * k2)
            k4 = f(A + dt * k3)
            A = (A + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)).clamp_min(0.0)
            outs.append(A)
        return torch.stack(outs, dim=0)   # (n_grid, ..., N)

    def predict(self, eta, theta, t):
        tv, _, _ = self.unpack(theta)
        Vmax = torch.exp(tv[0] + eta[..., 0])
        Km = torch.exp(tv[1] + eta[..., 1])
        V = torch.exp(tv[2] + eta[..., 2])
        A0 = torch.full_like(Vmax, float(self.dose))
        traj = self._integrate(A0, Vmax, Km, V)          # (n_grid, ..., N)
        A_obs = traj[self.obs_idx].movedim(0, -1)         # (..., N, T_obs)
        return A_obs / V.unsqueeze(-1)

    def log_lik(self, logy, eta, theta, t, mask):
        _, _, sigma = self.unpack(theta)
        pred = self.predict(eta, theta, t).clamp_min(1e-12)
        resid = logy - torch.log(pred)
        ll = -0.5 * (resid / sigma) ** 2 - torch.log(sigma) - 0.5 * math.log(2 * math.pi)
        return (ll * mask).sum(-1)

    def log_joint(self, logy, eta, theta, t, mask):
        return self.log_prior(eta, theta) + self.log_lik(logy, eta, theta, t, mask)


class OneCmtIVBolusMMNoKmRE:
    """
    Identical dynamics to OneCmtIVBolusMM, but Km has NO random effect
    (n_eta=2: only Vmax and V vary between subjects). Km is a pure fixed
    effect.

    This is the RECOMMENDED nonlinear-tier model for headline Q3 results
    and for any FOCE/SAEM comparison, not a fallback. Putting IIV on both
    Vmax and Km simultaneously in a Michaelis-Menten model is a known
    cross-method identifiability problem (Vmax/Km are structurally
    correlated in the MM equation), not something specific to VI -- this
    project's own data confirms it empirically: across four independent
    conditions (two sampling designs, two step budgets), Omega_Vmax's bias
    stayed in a reasonable -10% to -16% range every time, while
    Omega_Km's stayed catastrophic (-77% to -93%) regardless of training
    duration or design changes -- consistent with a genuine structural
    instability, not undertraining or a fixable design flaw. Fixing IIV to
    Vmax+V (dropping Km's) matches standard pharmacometric practice for MM
    models and is also what a FOCE/SAEM comparison should use, so that
    baseline_nlmixr2.R and this model specify the same, actually-stable
    structure rather than comparing methods on something neither would be
    deployed with in practice.

    theta (log scale, 6 params instead of 7):
        [log Vmax, log Km, log V, log om_Vmax, log om_V, log sigma]
    """
    n_eta = 2
    n_theta = 6

    def __init__(self, dose, grid_np, obs_idx_np):
        self.dose = dose
        self.t_grid = torch.tensor(grid_np, device=DEVICE)
        self.dts = self.t_grid[1:] - self.t_grid[:-1]
        self.obs_idx = obs_idx_np

    @staticmethod
    def unpack(theta):
        tv = theta[:3]              # log Vmax, log Km, log V (Km has no eta)
        om = torch.exp(theta[3:5])  # om_Vmax, om_V only
        sigma = torch.exp(theta[5])
        return tv, om, sigma

    def log_prior(self, eta, theta):
        _, om, _ = self.unpack(theta)
        return (-0.5 * (eta / om) ** 2 - torch.log(om) - 0.5 * math.log(2 * math.pi)).sum(-1)

    def _integrate(self, A0, Vmax, Km, V):
        A = A0
        outs = [A]
        for i in range(self.dts.shape[0]):
            dt = self.dts[i]
            def f(a):
                C = a / V
                return -Vmax * C / (Km + C + 1e-8)
            k1 = f(A)
            k2 = f(A + dt / 2 * k1)
            k3 = f(A + dt / 2 * k2)
            k4 = f(A + dt * k3)
            A = (A + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)).clamp_min(0.0)
            outs.append(A)
        return torch.stack(outs, dim=0)

    def predict(self, eta, theta, t):
        tv, _, _ = self.unpack(theta)
        Vmax = torch.exp(tv[0] + eta[..., 0])
        Km = torch.exp(tv[1])   # no eta -- identical for every subject/sample
        V = torch.exp(tv[2] + eta[..., 1])
        A0 = torch.full_like(Vmax, float(self.dose))
        traj = self._integrate(A0, Vmax, Km, V)
        A_obs = traj[self.obs_idx].movedim(0, -1)
        return A_obs / V.unsqueeze(-1)

    def log_lik(self, logy, eta, theta, t, mask):
        _, _, sigma = self.unpack(theta)
        pred = self.predict(eta, theta, t).clamp_min(1e-12)
        resid = logy - torch.log(pred)
        ll = -0.5 * (resid / sigma) ** 2 - torch.log(sigma) - 0.5 * math.log(2 * math.pi)
        return (ll * mask).sum(-1)

    def log_joint(self, logy, eta, theta, t, mask):
        return self.log_prior(eta, theta) + self.log_lik(logy, eta, theta, t, mask)


def simulate_mm(mp: MMTrueParams, n_subj, times_obs, dt_max, seed):
    """Ground-truth simulation for the nonlinear tier, via the same
    fixed-step RK4 used by the model (so 'truth' and 'inference' use
    consistent numerics -- using an independent high-accuracy solver here
    would conflate RK4 discretization error with the VI bias being
    measured)."""
    rng = np.random.default_rng(seed)
    eta = rng.normal(0.0, 1.0, size=(n_subj, 3)) * mp.eta_sd()
    Vmax = mp.Vmax * np.exp(eta[:, 0])
    Km = mp.Km * np.exp(eta[:, 1])
    V = mp.V * np.exp(eta[:, 2])

    grid, obs_idx = build_grid(times_obs, dt_max)
    A = np.full(n_subj, mp.dose)
    dts = np.diff(grid)
    traj = [A.copy()]                       # grid[0] = t=0 state, matches len(grid)
    for dt in dts:
        C = A / V
        k1 = -Vmax * C / (Km + C + 1e-8)
        C2 = (A + dt / 2 * k1) / V
        k2 = -Vmax * C2 / (Km + C2 + 1e-8)
        C3 = (A + dt / 2 * k2) / V
        k3 = -Vmax * C3 / (Km + C3 + 1e-8)
        C4 = (A + dt * k3) / V
        k4 = -Vmax * C4 / (Km + C4 + 1e-8)
        A = np.clip(A + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4), 0.0, None)
        traj.append(A.copy())
    traj = np.stack(traj, axis=0)          # (n_grid, N)
    A_obs = traj[obs_idx].T                 # (N, T_obs)
    C_obs = A_obs / V[:, None]
    logy = np.log(np.clip(C_obs, 1e-12, None)) + rng.normal(0.0, mp.sigma, size=C_obs.shape)

    t_tiled = np.tile(np.asarray(times_obs, dtype=float), (n_subj, 1))
    return dict(logy=logy, t=t_tiled, mask=np.ones_like(logy), eta_true=eta, seed=seed)


def mm_true_vector(mp):
    return np.array([mp.Vmax, mp.Km, mp.V, mp.om_Vmax, mp.om_Km, mp.om_V, mp.sigma])


MM_PARAM_NAMES = ["Vmax", "Km", "V", "om_Vmax", "om_Km", "om_V", "sigma"]

# Matching versions for OneCmtIVBolusMMNoKmRE (the recommended nonlinear-tier
# model -- see its docstring). Note om_Km is absent; Km itself (the fixed
# effect) is still reported.
MM_NOKM_PARAM_NAMES = ["Vmax", "Km", "V", "om_Vmax", "om_V", "sigma"]


def mm_true_vector_nokm(mp):
    return np.array([mp.Vmax, mp.Km, mp.V, mp.om_Vmax, mp.om_V, mp.sigma])


# %% =================================================================
#  Variational posteriors
# =====================================================================
class FreePosterior(nn.Module):
    """
    Non-amortized q: each subject gets its own (mu_i, log s_i). Diagonal Gaussian.

    This is the Pumas / Tarek & Afonso setup. It has ZERO amortization gap by
    construction, which is precisely why it belongs in every experiment:
    contrasting it against AmortizedPosterior below ISOLATES the
    amortization term experimentally instead of estimating it.
    """
    amortized = False

    def __init__(self, n_subj, n_eta):
        super().__init__()
        self.mu = nn.Parameter(torch.zeros(n_subj, n_eta))
        self.log_s = nn.Parameter(torch.full((n_subj, n_eta), -1.0))

    def params(self, batch):
        return self.mu, self.log_s

    def rsample_and_logq(self, batch, K, drep=False):
        """
        Reparameterized sample plus log q. Returns eta (K,N,r), log_q (K,N).

        drep=True implements the doubly-reparameterized gradient estimator
        (Tucker et al. 2019): the density is evaluated with the variational
        parameters detached, which removes the score-function term that causes
        IWAE's signal-to-noise collapse as K grows (Rainforth et al. 2018).
        """
        mu, log_s = self.params(batch)
        s = torch.exp(log_s)
        eps = torch.randn((K,) + mu.shape, device=mu.device)
        eta = mu + s * eps

        if drep:
            mu_d, s_d = mu.detach(), s.detach()
            z = (eta - mu_d) / s_d
            log_q = (-0.5 * z ** 2 - torch.log(s_d) - 0.5 * math.log(2 * math.pi)).sum(-1)
        else:
            log_q = (-0.5 * eps ** 2 - torch.log(s) - 0.5 * math.log(2 * math.pi)).sum(-1)
        return eta, log_q


class AmortizedPosterior(nn.Module):
    """
    Amortized q: a shared encoder maps each subject's observations to (mu, log s).
    This is the VAE-NLME setup (Rohleff, Li, Baaz).

    Deliberately small. Li et al. keep hidden dims under 32 to avoid overfitting
    and non-identifiability, and this is a faithful version of that choice.
    """
    amortized = True

    def __init__(self, n_obs, n_eta, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * n_obs, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 2 * n_eta),
        )
        self.n_eta = n_eta

    def params(self, batch):
        x = torch.cat([batch["logy"], batch["t"] / batch["t"].max()], dim=-1)
        out = self.net(x)
        mu, log_s = out[..., :self.n_eta], out[..., self.n_eta:]
        return mu, log_s.clamp(-5.0, 2.0)   # keep the scale numerically sane

    rsample_and_logq = FreePosterior.rsample_and_logq   # identical given params()


class AffineCoupling(nn.Module):
    """
    One RealNVP-style affine coupling layer, conditioned on a context vector.

    Splits eta into two halves; one half passes through unchanged and is used
    (together with `cond`) to predict a scale/shift applied to the other half.
    `mask_first_half` alternates which half is held fixed across layers.
    """
    def __init__(self, dim, cond_dim, hidden=16, mask_first_half=True):
        super().__init__()
        self.d1 = dim // 2 if dim > 1 else 1
        self.d2 = dim - self.d1
        self.mask_first_half = mask_first_half
        in_dim = (self.d1 if mask_first_half else self.d2) + cond_dim
        out_dim = 2 * (self.d2 if mask_first_half else self.d1)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, out_dim),
        )
        # Zero-init the last layer -> this layer starts as the identity map
        # (standard RealNVP/Glow practice). Without this, combined with an
        # untrained amortized encoder, training can push samples into a bad
        # basin early on that Adam never recovers from (this happened in
        # practice: K=1 amortized+flow blow-ups, om estimates of 1.3+
        # against a true 0.3, before this fix).
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, z, cond, strength=1.0):
        z1, z2 = z[..., :self.d1], z[..., self.d1:]
        if self.mask_first_half:
            h = self.net(torch.cat([z1, cond], dim=-1))
            log_s, t = h.chunk(2, dim=-1)
            # Tighter bound (0.5 instead of 1.0): caps the max per-layer
            # multiplicative distortion at ~1.65x instead of ~2.7x.
            log_s = 0.5 * torch.tanh(log_s)
            # `strength` ramps 0->1 over a warmup window (set by the training
            # loop). At strength=0 this layer is EXACTLY the identity
            # regardless of what the net has learned -- gives an (also
            # untrained) amortized encoder time to stabilize before the flow
            # starts reshaping its output.
            log_s = strength * log_s
            t = strength * t
            z2 = z2 * torch.exp(log_s) + t
            log_det = log_s.sum(-1)
            out = torch.cat([z1, z2], dim=-1)
        else:
            h = self.net(torch.cat([z2, cond], dim=-1))
            log_s, t = h.chunk(2, dim=-1)
            log_s = 0.5 * torch.tanh(log_s)
            log_s = strength * log_s
            t = strength * t
            z1 = z1 * torch.exp(log_s) + t
            log_det = log_s.sum(-1)
            out = torch.cat([z1, z2], dim=-1)
        return out, log_det


class ConditionalFlow(nn.Module):
    """Stack of alternating-mask coupling layers. dim=3 -> 2 layers is enough
    for every eta to see every other eta at least once."""
    def __init__(self, dim, cond_dim, n_layers=2, hidden=16):
        super().__init__()
        self.layers = nn.ModuleList([
            AffineCoupling(dim, cond_dim, hidden, mask_first_half=(i % 2 == 0))
            for i in range(n_layers)
        ])

    def forward(self, z0, cond, strength=1.0):
        z = z0
        log_det_total = torch.zeros(z.shape[:-1], device=z.device, dtype=z.dtype)
        for layer in self.layers:
            z, ld = layer(z, cond, strength=strength)
            log_det_total = log_det_total + ld
        return z, log_det_total


class FlowPosterior(nn.Module):
    """
    Wraps a base posterior (FreePosterior or AmortizedPosterior) with a
    conditional flow. The base module still supplies a per-subject Gaussian
    (mu, log_s) -- used both as the flow's reparameterized base sample AND
    (detached) as the conditioning context.

    log q(eta) = log q0(z0) - log_det,   eta = T(z0; cond)

    This is the "richer variational family" arm: same K, same bound, richer
    q. If bias closes here more than it did from increasing K alone with a
    Gaussian family, the residual bias is a FAMILY problem, not a BOUND
    problem.
    """
    def __init__(self, base, n_eta, n_layers=2, hidden=16):
        super().__init__()
        self.base = base
        self.amortized = base.amortized
        self.flow = ConditionalFlow(n_eta, cond_dim=2 * n_eta,
                                    n_layers=n_layers, hidden=hidden)
        # Set by the training loop each step; 0 at start of warmup, 1 after.
        self.strength = 1.0

    def rsample_and_logq(self, batch, K, drep=False):
        mu, log_s = self.base.params(batch)
        s = torch.exp(log_s)
        eps = torch.randn((K,) + mu.shape, device=mu.device)
        z0 = mu + s * eps

        if drep:
            mu_d, s_d = mu.detach(), s.detach()
            zz = (z0 - mu_d) / s_d
            log_q0 = (-0.5 * zz ** 2 - torch.log(s_d) - 0.5 * math.log(2 * math.pi)).sum(-1)
        else:
            log_q0 = (-0.5 * eps ** 2 - torch.log(s) - 0.5 * math.log(2 * math.pi)).sum(-1)

        # Context is detached: the flow treats "this subject's Gaussian fit"
        # as a fixed embedding, not a path for extra gradient flow into the
        # base params. Keeps the two components' optimization decoupled.
        cond = torch.cat([mu.detach(), log_s.detach()], dim=-1)
        cond = cond.unsqueeze(0).expand(K, *cond.shape)

        eta, log_det = self.flow(z0, cond, strength=self.strength)
        log_q = log_q0 - log_det
        return eta, log_q


class FlowPosteriorFixedBase(nn.Module):
    """
    FUTURE-WORK EXPERIMENT, not used by any headline result -- additive
    only, does not replace FlowPosterior above. Tests whether decoupling
    the flow's base distribution from the amortized encoder (as in Arruda
    et al. 2024, "An amortized approach to non-linear mixed-effects
    modeling", ICML) fixes amortized+flow's instability (see README Key
    Finding 3).

    Mechanism difference from FlowPosterior: there, z0 = mu + s*eps, so the
    sample fed through the flow is STILL tied to the encoder's own
    still-training output, even though the conditioning input is detached
    -- a moving target. Here, z0 ~ N(0, I) is fixed from the start; the
    encoder's (mu, log_s) is used ONLY as flow conditioning, and is NOT
    detached this time, since it's now the sole gradient pathway back to
    the encoder (or free per-subject parameters). This mirrors Arruda et
    al.'s design: a fixed latent base, encoder/summary-net output feeding
    the flow's conditioning path only.

    NOT validated against any real result yet -- exists to be cheaply
    smoke-tested, not to be trusted or used in any reported number.
    """
    def __init__(self, base, n_eta, n_layers=2, hidden=16):
        super().__init__()
        self.base = base
        self.amortized = base.amortized
        self.flow = ConditionalFlow(n_eta, cond_dim=2 * n_eta,
                                    n_layers=n_layers, hidden=hidden)
        self.strength = 1.0

    def rsample_and_logq(self, batch, K, drep=False):
        mu, log_s = self.base.params(batch)   # NOT detached -- sole gradient path now
        n_eta = mu.shape[-1]
        z0 = torch.randn((K,) + mu.shape, device=mu.device)   # fixed base, no mu/log_s dependence
        log_q0 = (-0.5 * z0 ** 2 - 0.5 * math.log(2 * math.pi)).sum(-1)

        cond = torch.cat([mu, log_s], dim=-1)   # differentiable this time
        cond = cond.unsqueeze(0).expand(K, *cond.shape)

        eta, log_det = self.flow(z0, cond, strength=self.strength)
        log_q = log_q0 - log_det
        return eta, log_q


def make_posterior(kind, family, n_subj, n_obs, n_eta):
    """kind: 'free' | 'amortized'.   family: 'gaussian' | 'flow' | 'flow_fixedbase'."""
    base = FreePosterior(n_subj, n_eta) if kind == "free" else AmortizedPosterior(n_obs, n_eta)
    if family == "gaussian":
        return base.to(DEVICE)
    elif family == "flow":
        return FlowPosterior(base, n_eta).to(DEVICE)
    elif family == "flow_fixedbase":
        # Future-work experiment -- see FlowPosteriorFixedBase docstring.
        return FlowPosteriorFixedBase(base, n_eta).to(DEVICE)
    raise ValueError(family)


# %% =================================================================
#  Objectives
# =====================================================================
def iw_elbo(model, q, theta, batch, K, drep=False, mc_reps=1):
    """
    Importance-weighted ELBO, summed over subjects.

        IW-ELBO = E[ log (1/K) sum_k w_k ],    w_k = p(y, eta_k) / q(eta_k)

    K = 1 recovers the plain ELBO exactly. As K -> inf this converges to the log
    marginal likelihood, with bias O(1/K) (Burda et al. 2016; Nowozin 2018).

    THE WHOLE ABLATION AXIS IS THIS ONE INTEGER. That is the point: if Omega
    shrinkage is caused by the looseness of the bound, it must decrease in K.

    Two non-negotiables:
      - logsumexp, never a product of raw weights (they overflow immediately)
      - float64 (set at module import)

    mc_reps > 1 draws `mc_reps` INDEPENDENT single-batch estimates of this same
    K-sample bound and averages them. This is NOT the same as increasing K:
    increasing K logsumexp's across more samples and genuinely tightens the
    bound (changes what's being measured). Averaging over mc_reps repeats
    leaves the expectation, and therefore the bias-vs-K curve, untouched --
    it only reduces the VARIANCE of the gradient estimate used to train on
    it. This matters specifically at K=1: a single-sample gradient is the
    noisiest possible training signal, and for a posterior with several
    jointly-untrained nonlinear components (e.g. an amortized encoder feeding
    a flow) that noise was enough to occasionally derail optimization
    entirely. mc_reps=1 (the default) reproduces the original behavior
    exactly.
    """
    total = 0.0
    for _ in range(mc_reps):
        eta, log_q = q.rsample_and_logq(batch, K, drep=drep)
        log_p = model.log_joint(batch["logy"], eta, theta, batch["t"], batch["mask"])
        log_w = log_p - log_q                               # (K, N)
        total = total + (torch.logsumexp(log_w, dim=0) - math.log(K)).sum()
    return total / mc_reps


@torch.no_grad()
def is_marginal_loglik(model, q, theta, batch, K=4000, chunk=500):
    """
    Post-hoc marginal log-likelihood by importance sampling, using the FITTED
    variational distribution as the proposal.

    Doing it this way puts the objective function on NONMEM's OFV scale, which
    is what makes dOFV likelihood-ratio tests and BIC valid again.

    Also returns the PSIS-style tail index proxy per subject (a crude version
    here; use arviz.psislw for the real thing in Phase 2). A subject whose
    proxy is near 1 has a variational posterior that is NOT a trustworthy IS
    proposal -- a per-subject diagnostic with no counterpart in classical
    pharmacometrics.
    """
    parts = []
    for start in range(0, K, chunk):
        k = min(chunk, K - start)
        eta, log_q = q.rsample_and_logq(batch, k)
        log_p = model.log_joint(batch["logy"], eta, theta, batch["t"], batch["mask"])
        parts.append(log_p - log_q)
    log_w = torch.cat(parts, dim=0)                          # (K, N)

    ll = torch.logsumexp(log_w, dim=0) - math.log(log_w.shape[0])

    w_norm = torch.softmax(log_w, dim=0)
    top_share = w_norm.max(dim=0).values
    ess = 1.0 / (w_norm ** 2).sum(dim=0)                     # effective sample size

    return ll.sum().item(), ess.cpu().numpy(), top_share.cpu().numpy()


# %% =================================================================
#  Training: adaptive convergence detection + the unified trainer
# =====================================================================
def train_to_convergence(step_fn, get_omega_fn, min_steps=2000, check_every=250,
                         patience=4, tol=0.01, max_steps=25000, verbose=False):
    """
    Adaptive replacement for a fixed step count. Runs step_fn() (one full
    optimizer step; all side effects, no return value) repeatedly, checking
    every `check_every` steps whether the OMEGA ESTIMATES THEMSELVES (not the
    loss) have stopped moving.

    Why omega and not the loss: a long convergence-check run on this project
    found the ELBO staying essentially flat for the entire run while two of
    three omega components kept drifting hard for most of it -- the
    likelihood surface was flat in exactly the direction the parameters were
    still moving. Monitoring the loss would never have caught this.

    Converged when: across the last `patience` checkpoints (i.e. the last
    patience*check_every steps), every omega component's relative change is
    below `tol`. Never stops before `min_steps`. Always stops by `max_steps`
    (safety cap) even if not converged.

    Returns (n_steps_run, converged: bool, history: list of (step, omega_np)).
    """
    history = []
    step = 0
    while step < max_steps:
        step_fn()
        step += 1
        if step % check_every == 0:
            om = get_omega_fn()
            history.append((step, om))
            if step >= min_steps and len(history) > patience:
                _, old_om = history[-(patience + 1)]
                rel = np.abs((om - old_om) / np.clip(np.abs(old_om), 1e-8, None))
                if np.all(rel < tol):
                    if verbose:
                        print(f"    converged at step {step} (max rel change "
                              f"{rel.max():.4f} < tol {tol})")
                    return step, True, history
    if verbose:
        print(f"    *** did not converge within max_steps={max_steps} -- "
              f"last omega={history[-1][1] if history else 'n/a'}")
    return step, False, history


def fit_model(model, data, kind, family, K, theta_init, n_eta, n_steps=4000,
             lr=0.05, drep=True, seed=0, mc_reps=1, adaptive=True,
             min_steps=2000, tol=0.01, max_steps=25000, eval_k=4000):
    """
    THE trainer. Every phase script fits models by calling this function --
    do not reimplement a training loop elsewhere.

    Generalized over:
      (a) structural model (linear analytic or MM/RK4, or any future model
          exposing the same log_joint interface -- e.g. a Phase 3 addition),
      (b) variational family (gaussian or flow),
      (c) posterior structure (free or amortized).
    The model and posterior only ever communicate through model.log_joint and
    q.rsample_and_logq -- neither knows the other's internals, which is why
    adding a new model or family never requires touching this function.

    eval_k: passed through to is_marginal_loglik's post-hoc K (default 4000).
    This is DISTINCT from the training K argument above -- training K
    controls bound tightness during optimization; eval_k controls only the
    precision of the marginal-likelihood ESTIMATE computed once, after
    training, from the already-fitted q. Raising eval_k is the relevant
    lever if a downstream likelihood-ratio comparison (e.g. dOFV) looks
    miscalibrated in a way consistent with IS-estimation noise (e.g.
    ll_full < ll_reduced for nested models, which is impossible under exact
    MLE) -- it does not require retraining, only a more precise evaluation.

    adaptive=True (default): train until the omega estimates plateau (see
    train_to_convergence above), capped at max_steps. `n_steps` is then only
    a fallback used when adaptive=False, kept for exact reproduction of
    older fixed-step results if ever needed.

    mc_reps: passed through to iw_elbo. Averages independent single-K
    estimates per gradient step to cut gradient variance WITHOUT tightening
    the bound -- useful specifically for flow+K=1, where gradient noise on
    two jointly-untrained nonlinear components (encoder + flow) can derail
    training.

    Best-checkpoint tracking (always on): the plateau test alone only
    detects "stopped moving," not "stopped moving somewhere correct." A run
    can wander into a bad basin, get numerically stuck there, and satisfy
    the plateau test at a garbage fixed point -- this happened in practice
    (amortized posterior, full-scale check: a diverged fit registered as
    "converged" with 1000%+ bias, and a final loss catastrophically worse
    than every sane fit in the same run). Every `check_every`-th step, if
    the loss improved, the (theta, q) state is snapshotted; the best
    snapshot is restored before returning, regardless of what the final
    step's parameters look like. Ordinary early stopping.

    Returns (theta_np, q, ll, ess, top_share, n_steps_run, converged, cpu_secs).

    cpu_secs is measured with time.process_time(), NOT time.time(). This
    matters: wall-clock time includes OS sleep (a machine sleeping for hours
    mid-run inflates a 10-second fit to 900+ seconds with zero relationship
    to actual compute cost) and other-process contention. process_time()
    only counts CPU cycles this process actually consumed -- it does not
    advance during sleep at all, and is far less sensitive to competing
    processes. This is the number a VI-vs-FOCEI/SAEM speed claim should be
    built on, not time.time(). Callers that also want wall-clock (e.g. to
    report "how long did I wait") can still wrap the call with their own
    time.time() -- both are informative, but cpu_secs is the one that's
    portable across machines/interruptions and comparable to a similarly
    CPU-timed R-side measurement (proc.time() in R gives the equivalent).
    """
    _t0 = time.process_time()
    torch.manual_seed(seed)   # controls ONLY inference; never the data
    batch = to_tensors(data)
    n_subj, n_obs = batch["logy"].shape

    q = make_posterior(kind, family, n_subj, n_obs, n_eta)
    theta = torch.tensor(theta_init, device=DEVICE, requires_grad=True)

    is_flow = isinstance(q, FlowPosterior)
    if is_flow:
        # The flow's coupling nets get a lower LR than theta/base-q -- even
        # with zero-init they're a higher-variance component than a plain
        # Gaussian's (mu, log_s).
        param_groups = [
            {"params": [theta], "lr": lr},
            {"params": q.base.parameters(), "lr": lr},
            {"params": q.flow.parameters(), "lr": lr * 0.3},
        ]
    else:
        param_groups = [{"params": [theta], "lr": lr},
                        {"params": q.parameters(), "lr": lr}]
    opt = torch.optim.Adam(param_groups)
    # StepLR, not CosineAnnealingLR: the latter needs a known T_max, which
    # adaptive (unknown-length) training doesn't have.
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=2000, gamma=0.7)

    # Fixed absolute warmup window (not a % of total steps, since total is
    # unknown ahead of time under adaptive stopping).
    warmup_steps = 500

    _counter = {"n": 0}
    best = {"loss": float("inf"), "theta": None, "q_state": None, "step": -1}

    def step_fn():
        if is_flow:
            q.strength = min(1.0, _counter["n"] / warmup_steps)
        opt.zero_grad()
        loss = -iw_elbo(model, q, theta, batch, K, drep=drep, mc_reps=mc_reps) / n_subj
        loss.backward()
        if is_flow:
            torch.nn.utils.clip_grad_norm_([theta] + list(q.base.parameters()), 10.0)
            torch.nn.utils.clip_grad_norm_(list(q.flow.parameters()), 2.0)
        else:
            torch.nn.utils.clip_grad_norm_([theta] + list(q.parameters()), 10.0)
        opt.step()
        sched.step()
        _counter["n"] += 1
        if _counter["n"] % 250 == 0:
            with torch.no_grad():
                cur = loss.item()
            if cur < best["loss"]:
                best["loss"] = cur
                best["theta"] = theta.detach().clone()
                best["q_state"] = {k: v.clone() for k, v in q.state_dict().items()}
                best["step"] = _counter["n"]

    def get_omega_fn():
        with torch.no_grad():
            return torch.exp(theta[3:6]).cpu().numpy()

    if adaptive:
        n_run, converged, _ = train_to_convergence(
            step_fn, get_omega_fn, min_steps=min_steps, tol=tol,
            max_steps=max_steps,
        )
    else:
        for _ in range(n_steps):
            step_fn()
        n_run, converged = n_steps, True   # not tracked in legacy fixed-step mode

    # Restore the best checkpoint seen, not whatever the trajectory ended on.
    if best["theta"] is not None:
        with torch.no_grad():
            theta.copy_(best["theta"])
        q.load_state_dict(best["q_state"])

    ll, ess, top_share = is_marginal_loglik(model, q, theta.detach(), batch, K=eval_k)
    cpu_secs = time.process_time() - _t0
    return theta.detach().cpu().numpy(), q, ll, ess, top_share, n_run, converged, cpu_secs