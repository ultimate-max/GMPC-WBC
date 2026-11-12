"""Two locally condensed QPs exactly matching SharedUpper's frozen MPC.

The only duplicated quantities are installation motion and its physical wrench.
No task weight, physical bound, hierarchy level, horizon, or predictor changes.
Rows retain their centralized identifiers so equivalence can be audited before
any optimizer is involved.  All times include the work actually performed.
"""
from dataclasses import dataclass
import time
import numpy as np
import pinocchio as pin
from geometric_wbc import pose, rolling_contact, cone_matrix
from body_arm_interface import BodyArmInterface, full_frozen_maps


def _inf(x):
    return float(np.max(np.abs(x))) if np.size(x) else 0.


@dataclass
class LocalProblem:
    name: str
    n: int
    levels: list
    G: np.ndarray
    lo: np.ndarray
    hi: np.ndarray
    E: np.ndarray
    e: np.ndarray
    row_ids: np.ndarray
    task_ids: list
    nodes: list
    controls: list
    torque: np.ndarray
    wrench: np.ndarray
    idx: np.ndarray
    initial_p: np.ndarray
    initial_v: np.ndarray

    def recover(self, x):
        ps, vs = [self.initial_p.copy()], [self.initial_v.copy()]
        accs, forces = [], []
        for P, p, V, v, A, b, B, f in self.nodes:
            ps.append(P@x+p); vs.append(V@x+v)
            accs.append(A@x+b); forces.append(B@x+f)
        return dict(p=np.asarray(ps), v=np.asarray(vs), a=np.asarray(accs),
                    ground_force=np.asarray(forces),
                    tau=(self.torque@x).reshape(len(self.nodes), -1),
                    wrench=(self.wrench@x).reshape(len(self.nodes), 6))


@dataclass
class BodyArmProblem:
    body: LocalProblem
    arm: LocalProblem
    split: object
    full_maps: dict
    initial_p: np.ndarray
    initial_v: np.ndarray
    scale: np.ndarray
    wrench_scale: np.ndarray
    N: int
    dt: float
    timing: dict
    lift_body: np.ndarray
    offset_body: np.ndarray
    lift_arm: np.ndarray
    offset_arm: np.ndarray

    def to_local(self, global_u):
        u = np.asarray(global_u).reshape(-1)
        return self.lift_body@u+self.offset_body, self.lift_arm@u+self.offset_arm

    def to_global(self, xb, xa):
        ub = np.asarray(xb).reshape(self.N, 18)[:, :12]
        ua = np.asarray(xa).reshape(self.N, 12)[:, :6]
        return np.column_stack((ub, ua)).reshape(-1)

    def central_from_local(self):
        """Pull every local row back through the exact consensus parametrization."""
        levels = []
        for level, count in enumerate((12*self.N, 12*self.N, 90*self.N)):
            C, d = np.zeros((count, 18*self.N)), np.zeros(count)
            for local, lift, off in ((self.body, self.lift_body, self.offset_body),
                                     (self.arm, self.lift_arm, self.offset_arm)):
                ids = local.task_ids[level]; lc, ld = local.levels[level]
                C[ids] = lc@lift; d[ids] = lc@off+ld
            levels.append((C, d))
        G = np.zeros((134*self.N, 18*self.N))
        lo, hi = np.zeros(134*self.N), np.zeros(134*self.N)
        for local, lift, off in ((self.body, self.lift_body, self.offset_body),
                                 (self.arm, self.lift_arm, self.offset_arm)):
            G[local.row_ids] = local.G@lift
            lo[local.row_ids] = local.lo-local.G@off
            hi[local.row_ids] = local.hi-local.G@off
        return levels, G, lo, hi

    def audit_equivalence(self, levels, G, lo, hi, global_u):
        xb, xa = self.to_local(global_u)
        local_x = (xb, xa)
        task_errors = []
        for i, (C, d) in enumerate(levels):
            assembled = np.zeros(len(d))
            for local, x in zip((self.body, self.arm), local_x):
                lc, ld = local.levels[i]
                assembled[local.task_ids[i]] = lc@x+ld
            task_errors.append(_inf(assembled-(C@global_u+d)))
        lower, upper = np.zeros(len(lo)), np.zeros(len(hi))
        for local, x in zip((self.body, self.arm), local_x):
            lower[local.row_ids] = local.lo-local.G@x
            upper[local.row_ids] = local.hi-local.G@x
        fm, fu = np.isfinite(lo), np.isfinite(hi)
        result = dict(task_residual_errors=task_errors,
                      lower_residual_error=_inf(lower[fm]-(lo-G@global_u)[fm]),
                      upper_residual_error=_inf(upper[fu]-(hi-G@global_u)[fu]),
                      interface_residual=_inf(self.body.E@xb+self.body.e-
                                               self.arm.E@xa-self.arm.e),
                      torque_roundtrip_error=_inf(self.to_global(xb, xa)-global_u))
        cl, cg, ll, hh = self.central_from_local()
        result['task_matrix_errors'] = [_inf(a[0]-b[0]) for a, b in zip(cl, levels)]
        result['task_bias_errors'] = [_inf(a[1]-b[1]) for a, b in zip(cl, levels)]
        result['constraint_matrix_error'] = _inf(cg-G)
        result['constraint_lower_error'] = _inf(ll[fm]-lo[fm])
        result['constraint_upper_error'] = _inf(hh[fu]-hi[fu])
        result['bound_infinity_pattern_mismatch'] = int(
            np.count_nonzero(np.isfinite(ll) != fm)+np.count_nonzero(np.isfinite(hh) != fu))
        return result

    def recover(self, xb, xa):
        b, a = self.body.recover(xb), self.arm.recover(xa)
        global_u = self.to_global(xb, xa)
        tau = global_u.reshape(self.N, 18)*self.scale
        central = self.split.rollout_global(tau, self.initial_p, self.initial_v,
                                           self.dt, self.full_maps)
        J = self.split.interface_motion
        residuals = dict(
            interface_acceleration=(b['a'][:, :6]-a['a'][:, :6])@J.T,
            interface_velocity=(b['v'][:, :6]-a['v'][:, :6])@J.T,
            interface_position_tangent=(b['p'][:, :6]-a['p'][:, :6])@J.T,
            interface_wrench=b['wrench']-a['wrench'],
            body_prediction=b['p']-central['p'][:, self.body.idx],
            arm_prediction=a['p']-central['p'][:, self.arm.idx],
            body_velocity_prediction=b['v']-central['v'][:, self.body.idx],
            arm_velocity_prediction=a['v']-central['v'][:, self.arm.idx],
            body_acceleration_prediction=b['a']-central['a'][:, self.body.idx],
            arm_acceleration_prediction=a['a']-central['a'][:, self.arm.idx],
            body_contact_force_prediction=b['ground_force']-central['ground_force'])
        return dict(global_u=global_u, tau=tau, body=b, arm=a, full=central,
                    residuals=residuals,
                    residual_max={key: _inf(value) for key, value in residuals.items()})


class BodyArmBuilder:
    def __init__(self, plant, root, target, N=10, dt=.04):
        start = time.perf_counter()
        self.plant, self.target, self.N, self.dt = plant, target, N, dt
        self.interface = BodyArmInterface(root)
        self.scale = np.r_[np.full(12, 100.), np.full(6, 10.)]
        self.wrench_scale = np.r_[np.full(3, 100.), np.full(3, 10.)]
        self.cone = cone_matrix()
        self.setup_ms = 1000*(time.perf_counter()-start)

    def build(self, state, t, prev_tau, anchor, first_slew_time=.02):
        start = time.perf_counter()
        state = rolling_contact(state)
        split = self.interface.linearize(state)
        full = full_frozen_maps(state, split.full_C)
        p0 = self.plant.difference(state['q'], anchor[0])
        v0 = anchor[1].copy()
        linearization_ms = 1000*(time.perf_counter()-start)
        st = time.perf_counter()
        targets = []
        current = (pose(state['q']), pin.SE3(state['pose']))
        task_j = (np.eye(24)[:6], state['J'])
        for k in range(self.N):
            target = self.target.at(t+(k+1)*self.dt)
            values = []
            for i, name in enumerate(('base', 'ee')):
                E = pin.SE3(target[name+'_pose']).inverse()*current[i]
                values.append((pin.Jlog6(E)@task_j[i], pin.log6(E).vector,
                               task_j[i], E.inverse().action@target[name+'_twist']))
            targets.append((values, self.plant.difference(state['q'], target['q'])[6:],
                            target['a'][6:], target['tau']))
        target_ms = 1000*(time.perf_counter()-st)
        st = time.perf_counter()
        body = self._local('body', split.body, state, targets, p0, v0,
                           prev_tau, anchor[2], first_slew_time)
        body_ms = 1000*(time.perf_counter()-st)
        st = time.perf_counter()
        arm = self._local('arm', split.arm, state, targets, p0, v0,
                          prev_tau, anchor[2], first_slew_time)
        arm_ms = 1000*(time.perf_counter()-st)
        st = time.perf_counter()
        lb, ob, la, oa = self._lift(split, full, p0, v0)
        lift_ms = 1000*(time.perf_counter()-st)
        timing = dict(linearization_ms=linearization_ms, target_ms=target_ms,
                      body_assembly_ms=body_ms, arm_assembly_ms=arm_ms,
                      audit_lift_ms=lift_ms, setup_ms=self.setup_ms,
                      construction_ms=1000*(time.perf_counter()-start))
        return BodyArmProblem(body, arm, split, full, p0, v0, self.scale.copy(),
                              self.wrench_scale.copy(), self.N, self.dt, timing,
                              lb, ob, la, oa)

    def _local(self, name, dyn, state, targets, p0, v0, previous, initial_a, first):
        body = name == 'body'
        jglobal = np.arange(12) if body else np.arange(12, 18)
        nj, nv = len(jglobal), len(dyn.idx)
        nu, n = nj+6, (nj+6)*self.N
        scale = self.scale[jglobal]
        P, V = np.zeros((nv, n)), np.zeros((nv, n))
        p, v = p0[dyn.idx].copy(), v0[dyn.idx].copy()
        cost_m, cost_b, tids = [[], [], []], [[], [], []], [[], [], []]
        blocks, lower, upper, row_ids, nodes, controls = [], [], [], [], [], []
        es, eb, torques, wrenches = [], [], [], []
        old, old_a, old_b = None, None, None

        def bound(A, b, lo, hi, ids):
            blocks.append(A); lower.append(np.broadcast_to(lo, len(b))-b)
            upper.append(np.broadcast_to(hi, len(b))-b); row_ids.extend(ids)

        def cost(level, C, d, weight, ids):
            w = np.sqrt(np.broadcast_to(weight, len(d)))
            cost_m[level].append(w[:, None]*C); cost_b[level].append(w*d)
            tids[level].extend(ids)

        for k in range(self.N):
            U = np.zeros((nj, n)); U[:, nu*k:nu*k+nj] = np.eye(nj)
            W = np.zeros((6, n)); W[:, nu*k+nj:nu*(k+1)] = np.diag(self.wrench_scale)
            tm = scale[:, None]*U
            A = dyn.T@tm+dyn.W@W+dyn.Aq@P+dyn.Av@V
            b = dyn.a0+dyn.Aq@p+dyn.Av@(v-state['v'][dyn.idx])
            if body:
                B = dyn.F@tm+dyn.FW@W+dyn.Fq@P+dyn.Fv@V
                f = dyn.f0+dyn.Fq@p+dyn.Fv@(v-state['v'][dyn.idx])
            else:
                B, f = np.zeros((0, n)), np.zeros(0)
            es.append(np.vstack((dyn.Jinterface@A, W)))
            eb.append(np.r_[dyn.Jinterface@b, np.zeros(6)])
            off = 134*k
            bound(tm, np.zeros(nj), -.85*state['tau_max'][jglobal],
                  .85*state['tau_max'][jglobal], off+jglobal)
            if body:
                bound(self.cone@B, self.cone@f, np.tile([-np.inf]*4+[10.], 4),
                      np.tile([0.]*4+[800.], 4), off+np.arange(18, 38))
            step = first if k == 0 else self.dt
            rate = 1000. if body else 100.
            bound(tm if old is None else tm-old,
                  -previous[jglobal] if old is None else np.zeros(nj),
                  -step*rate, step*rate, off+38+jglobal)
            limit = 50. if body else 30.
            bound(A[6:], b[6:], -limit, limit, off+56+jglobal)
            P = P+self.dt*V+.5*self.dt**2*A
            p = p+self.dt*v+.5*self.dt**2*b
            V = V+self.dt*A; v = v+self.dt*b
            bound(P[6:], state['q'][7:][jglobal]+p[6:],
                  state['q_min'][jglobal]+.02, state['q_max'][jglobal]-.02,
                  off+74+jglobal)
            bound(P[6:], p[6:], -.15, .15, off+92+jglobal)
            vmax = np.minimum(state['v_max'][jglobal], 4.)
            bound(V[6:], v[6:], -vmax, vmax, off+110+jglobal)
            if body:
                limit = np.r_[np.full(3, .1), np.full(3, .3)]
                bound(P[:6], p[:6], -limit, limit, off+np.arange(128, 134))
            values, target_q, target_a, target_tau = targets[k]
            level = 0 if body else 1
            G, e, J, vd = values[level]
            G, J = G[:, dyn.idx], J[:, dyn.idx]
            terminal = 3. if k == self.N-1 else 1.
            cost(level, G@P, e+G@p,
                 terminal*np.r_[np.full(3, 4000.), np.full(3, 1600. if body else 1000.)],
                 12*k+np.arange(6))
            cost(level, J@V, J@v-vd, np.r_[np.full(3, 20.), np.full(3, 8.)],
                 12*k+np.arange(6, 12))
            smooth = 90*k
            cost(2, P[6:], p[6:]-target_q[jglobal], 5. if body else 40., smooth+jglobal)
            cost(2, A[6:], b[6:]-target_a[jglobal], .02, smooth+18+jglobal)
            cost(2, A[6:] if old_a is None else A[6:]-old_a[6:],
                 b[6:]-initial_a[dyn.idx][6:] if old_b is None else b[6:]-old_b[6:],
                 .02, smooth+36+jglobal)
            cost(2, U, -target_tau[jglobal]/scale, .002, smooth+54+jglobal)
            cost(2, U if old is None else (tm-old)/scale[:, None],
                 -previous[jglobal]/scale if old is None else np.zeros(nj),
                 .03, smooth+72+jglobal)
            nodes.append((P.copy(), p.copy(), V.copy(), v.copy(), A, b, B, f))
            controls.append(U); torques.append(tm); wrenches.append(W)
            old, old_a, old_b = tm, A, b
        levels = [(np.vstack(c) if c else np.zeros((0, n)),
                   np.concatenate(d) if d else np.zeros(0)) for c, d in zip(cost_m, cost_b)]
        return LocalProblem(name, n, levels, np.vstack(blocks), np.concatenate(lower),
                            np.concatenate(upper), np.vstack(es), np.concatenate(eb),
                            np.asarray(row_ids, dtype=int),
                            [np.asarray(ids, dtype=int) for ids in tids], nodes, controls,
                            np.vstack(torques), np.vstack(wrenches), dyn.idx.copy(),
                            p0[dyn.idx].copy(), v0[dyn.idx].copy())

    def _lift(self, split, full, p0, v0):
        n = 18*self.N
        lb, ob = np.zeros((18*self.N, n)), np.zeros(18*self.N)
        la, oa = np.zeros((12*self.N, n)), np.zeros(12*self.N)
        wm = split.wrench_map(full)
        P, V = np.zeros((24, n)), np.zeros((24, n))
        p, v = p0.copy(), v0.copy()
        for k in range(self.N):
            U = np.zeros((18, n)); U[:, 18*k:18*(k+1)] = np.diag(self.scale)
            W = (wm['Wtau']@U+wm['Wq']@P+wm['Wv']@V)/self.wrench_scale[:, None]
            w = (wm['w0']+wm['Wq']@p+wm['Wv']@(v-split.v0))/self.wrench_scale
            lb[18*k:18*k+12, 18*k:18*k+12] = np.eye(12)
            la[12*k:12*k+6, 18*k+12:18*k+18] = np.eye(6)
            lb[18*k+12:18*(k+1)] = W; ob[18*k+12:18*(k+1)] = w
            la[12*k+6:12*(k+1)] = W; oa[12*k+6:12*(k+1)] = w
            A = full['T']@U+full['Aq']@P+full['Av']@V
            b = full['a0']+full['Aq']@p+full['Av']@(v-split.v0)
            P = P+self.dt*V+.5*self.dt**2*A; p = p+self.dt*v+.5*self.dt**2*b
            V = V+self.dt*A; v = v+self.dt*b
        return lb, ob, la, oa
