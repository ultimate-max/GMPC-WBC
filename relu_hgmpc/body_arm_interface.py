"""Exact inertia partition of the *existing frozen-affine* B2+Z1 MPC.

This is not a new nonlinear predictor.  Both subsystems use the same frozen
M, h + C (v-v0), contact restoration, and local integration as SharedUpper.
The installation wrench is force ON BODY, expressed at z1_link00 in its LOCAL
axes, in Pinocchio [linear force; moment] order.  It is not an ADMM multiplier.

All URDF links below z1_mount belong to the arm; every other inertia belongs
to the body.  Keeping the common kinematic tree preserves coordinate order.
Zero-inertia branches are removed by coordinate selection, not duplicated.
"""
from dataclasses import dataclass
from pathlib import Path
import copy
import xml.etree.ElementTree as ET
import numpy as np
import scipy.linalg as la
import pinocchio as pin

BODY = np.arange(18)
ARM = np.r_[np.arange(6), np.arange(18, 24)]


def _inf(x):
    return float(np.max(np.abs(x))) if np.size(x) else 0.


def _model_with_inertias(tree, keep):
    tree = copy.deepcopy(tree)
    for link in tree.findall('link'):
        if link.get('name') not in keep:
            inertial = link.find('inertial')
            if inertial is not None:
                link.remove(inertial)
    return pin.buildModelFromXML(ET.tostring(tree, encoding='unicode'),
                                 pin.JointModelFreeFlyer())


@dataclass
class LocalDynamics:
    idx: np.ndarray
    M: np.ndarray
    h: np.ndarray
    C: np.ndarray
    Jinterface: np.ndarray
    Jcontact: np.ndarray
    T: np.ndarray
    W: np.ndarray
    a0: np.ndarray
    Aq: np.ndarray
    Av: np.ndarray
    F: np.ndarray = None
    FW: np.ndarray = None
    f0: np.ndarray = None
    Fq: np.ndarray = None
    Fv: np.ndarray = None


@dataclass
class FrozenSplit:
    body: LocalDynamics
    arm: LocalDynamics
    v0: np.ndarray
    interface_pose: np.ndarray
    interface_motion: np.ndarray
    exactness: dict
    full_C: np.ndarray

    def recombine(self):
        """Eliminate interface wrench using equality of base acceleration.

        Returns physical-torque maps with the same keys as SharedUpper:
          a = T tau + a0 + Aq p + Av (v-v0)
          f = F tau + f0 + Fq p + Fv (v-v0).
        Wrench maps are for the common force ON BODY.  No extra penalty,
        regularization, or physical approximation is introduced here.
        """
        b, a = self.body, self.arm
        E_b = np.eye(24)[b.idx]
        E_a = np.eye(24)[a.idx]
        U_b = np.eye(18)[:12]
        U_a = np.eye(18)[12:]
        compliance = b.W[:6] - a.W[:6]
        rhs = np.column_stack((a.T[:6] @ U_a - b.T[:6] @ U_b,
                               a.a0[:6] - b.a0[:6],
                               a.Aq[:6] @ E_a - b.Aq[:6] @ E_b,
                               a.Av[:6] @ E_a - b.Av[:6] @ E_b))
        wm = la.solve(compliance, rhs)
        wt, w0, wq, wv = wm[:, :18], wm[:, 18], wm[:, 19:43], wm[:, 43:]
        maps = {}
        for key, base_b, base_a, w in (
                ('T', b.T @ U_b, a.T @ U_a, wt),
                ('a0', b.a0, a.a0, w0),
                ('Aq', b.Aq @ E_b, a.Aq @ E_a, wq),
                ('Av', b.Av @ E_b, a.Av @ E_a, wv)):
            value_b, value_a = base_b + b.W @ w, base_a + a.W @ w
            value = np.zeros((24,) + value_b.shape[1:])
            value[b.idx] = value_b
            value[18:] = value_a[6:]
            maps[key] = value
        maps.update(F=b.F @ U_b + b.FW @ wt, f0=b.f0 + b.FW @ w0,
                    Fq=b.Fq @ E_b + b.FW @ wq,
                    Fv=b.Fv @ E_b + b.FW @ wv,
                    Wtau=wt, w0=w0, Wq=wq, Wv=wv,
                    interface_compliance=compliance)
        return maps

    def wrench_map(self, maps):
        """Recover physical installation wrench from arm base balance.

        ``maps['T']`` MUST map physical torque, not scaled QP variables.
        Independent of recombine(), and suitable for auditing a condensed QP.
        """
        a = self.arm
        R = -la.solve(a.Jinterface[:, :6].T, a.M[:6])
        c = -la.solve(a.Jinterface[:, :6].T, a.C[:6])
        E = np.eye(24)[a.idx]
        return dict(Wtau=R @ maps['T'][a.idx],
                    w0=R @ maps['a0'][a.idx] -
                       la.solve(a.Jinterface[:, :6].T, a.h[:6]),
                    Wq=R @ maps['Aq'][a.idx],
                    Wv=R @ maps['Av'][a.idx] + c @ E)

    def residuals(self, p, v, tau, acc, ground_force, wrench):
        """Unscaled local equations; a shared state does not hide force errors."""
        b, a = self.body, self.arm
        vb, va = (v-self.v0)[b.idx], (v-self.v0)[a.idx]
        body = b.M @ acc[b.idx] + b.h + b.C @ vb
        body -= np.r_[np.zeros(6), tau[:12]] + b.Jcontact.T @ ground_force
        body -= b.Jinterface.T @ wrench
        arm = a.M @ acc[a.idx] + a.h + a.C @ va
        arm -= np.r_[np.zeros(6), tau[12:]] - a.Jinterface.T @ wrench
        return dict(body_dynamics=_inf(body), arm_dynamics=_inf(arm))

    def rollout_global(self, torques, initial_p, initial_v, dt=.04, maps=None):
        """Recover full and interface trajectories from physical joint torques.

        ``p`` and ``v`` have N+1 rows; acceleration/contact force/wrench have
        N rows, evaluated before each integration step exactly as SharedUpper.
        Pass independently assembled full maps to audit the interface against
        the centralized predictor rather than calling recombine implicitly.
        """
        maps = self.recombine() if maps is None else maps
        wm = self.wrench_map(maps)
        ps, vs = [np.array(initial_p, copy=True)], [np.array(initial_v, copy=True)]
        accs, forces, wrenches, interface_accs = [], [], [], []
        for tau in np.asarray(torques):
            p, v = ps[-1], vs[-1]
            acc = maps['T']@tau+maps['a0']+maps['Aq']@p+maps['Av']@(v-self.v0)
            force = maps['F']@tau+maps['f0']+maps['Fq']@p+maps['Fv']@(v-self.v0)
            wrench = wm['Wtau']@tau+wm['w0']+wm['Wq']@p+wm['Wv']@(v-self.v0)
            accs.append(acc); forces.append(force); wrenches.append(wrench)
            interface_accs.append(self.interface_motion@acc[:6])
            ps.append(p+dt*v+.5*dt*dt*acc)
            vs.append(v+dt*acc)
        return dict(p=np.asarray(ps), v=np.asarray(vs), a=np.asarray(accs),
                    ground_force=np.asarray(forces), wrench=np.asarray(wrenches),
                    interface_a=np.asarray(interface_accs))


class BodyArmInterface:
    def __init__(self, root, mount_joint='z1_mount'):
        tree = ET.parse(Path(root)/'models/b2_z1.urdf').getroot()
        joints = {j.get('name'): j for j in tree.findall('joint')}
        mount = joints[mount_joint]
        if mount.get('type') != 'fixed':
            raise ValueError('The selected interface must be a fixed mount')
        self.frame_name = mount.find('child').get('link')
        arm_links = {self.frame_name}
        changed = True
        while changed:
            old = len(arm_links)
            for j in joints.values():
                if j.find('parent').get('link') in arm_links:
                    arm_links.add(j.find('child').get('link'))
            changed = len(arm_links) != old
        all_links = {e.get('name') for e in tree.findall('link')}
        self.arm_links = tuple(sorted(arm_links))
        self.body_links = tuple(sorted(all_links-arm_links))
        self.full = pin.buildModelFromXML(ET.tostring(tree, encoding='unicode'),
                                          pin.JointModelFreeFlyer())
        self.body = _model_with_inertias(tree, set(self.body_links))
        self.arm = _model_with_inertias(tree, set(self.arm_links))
        self.data = {name: model.createData() for name, model in
                     [('full', self.full), ('body', self.body), ('arm', self.arm)]}
        if self.full.nv != 24 or self.full.nq != 25:
            raise ValueError('This matched prototype requires the existing B2+Z1 24-DoF model')
        if list(self.full.names) != list(self.body.names) or list(self.full.names) != list(self.arm.names):
            raise ValueError('Inertia partition changed the kinematic coordinate tree')

    def linearize(self, state):
        """Use the exact frozen coefficients of SharedUpper, before condensation."""
        q, v = np.asarray(state['q']), np.asarray(state['v'])
        values = {}
        for name, model in [('full', self.full), ('body', self.body), ('arm', self.arm)]:
            data = self.data[name]
            M = np.array(pin.crba(model, data, q))
            h = np.array(pin.nonLinearEffects(model, data, q, v))
            C = np.array(pin.computeCoriolisMatrix(model, data, q, v))
            values[name] = (M, h, C)
        data = self.data['full']
        fid = self.full.getFrameId(self.frame_name)
        pin.computeJointJacobians(self.full, data, q)
        pin.updateFramePlacements(self.full, data)
        Ji = np.array(pin.getFrameJacobian(self.full, data, fid, pin.LOCAL))
        pose = np.array(data.oMf[fid].homogeneous)
        if _inf(Ji[:, 6:]) > 1e-12:
            raise ValueError('Mount is not rigidly attached to the floating base')
        M, h, C = values['full']
        MB, hB, CB = values['body']
        MA, hA, CA = values['arm']
        errors = dict(mass_partition=_inf(M-MB-MA), bias_partition=_inf(h-hB-hA),
                      coriolis_partition=_inf(C-CB-CA),
                      source_mass=_inf(M-state['M']), source_bias=_inf(h-state['h']),
                      body_arm_rows=_inf(MB[18:]), arm_leg_rows=_inf(MA[6:18]))
        if max(errors.values()) > 1e-9:
            raise ValueError(f'Inertia partition is not equivalent: {errors}')
        J = np.array(state['Jc'])
        if _inf(J[:, 18:]) > 1e-12:
            raise ValueError('Foot contacts unexpectedly depend on arm joints')
        ce = np.array(state['ce']).copy().reshape(4, 3)
        ce[:, :2] = 0.  # Same rolling_contact convention as SharedUpper.
        ce = ce.reshape(12)
        Z = np.diag(np.tile([0., 0., 1.], 4))
        mb, hb, cb = MB[np.ix_(BODY, BODY)], hB[BODY], CB[np.ix_(BODY, BODY)]
        jb, jib = J[:, BODY], Ji[:, BODY]
        K = np.block([[mb, -jb.T], [jb, np.zeros((12, 12))]])
        S = np.vstack((np.zeros((6, 12)), np.eye(12)))
        rhs = np.column_stack((np.vstack((S, np.zeros((12, 12)))),
                               np.vstack((jib.T, np.zeros((12, 6)))),
                               np.r_[-hb, -state['cdotv']-100*ce-20*J@v],
                               np.vstack((np.zeros((18, 18)), -100*Z@jb)),
                               np.vstack((-cb, -20*jb))))
        out = la.solve(K, rhs)
        b = LocalDynamics(BODY.copy(), mb, hb, cb, jib, jb,
                          out[:18, :12], out[:18, 12:18], out[:18, 18],
                          out[:18, 19:37], out[:18, 37:],
                          out[18:, :12], out[18:, 12:18], out[18:, 18],
                          out[18:, 19:37], out[18:, 37:])
        ma, ha, ca = MA[np.ix_(ARM, ARM)], hA[ARM], CA[np.ix_(ARM, ARM)]
        jia = Ji[:, ARM]
        sa = np.vstack((np.zeros((6, 6)), np.eye(6)))
        out = la.solve(ma, np.column_stack((sa, -jia.T, -ha, -ca)), assume_a='pos')
        a = LocalDynamics(ARM.copy(), ma, ha, ca, jia, np.zeros((0, 12)),
                          out[:, :6], out[:, 6:12], out[:, 12],
                          np.zeros((12, 12)), out[:, 13:])
        return FrozenSplit(b, a, v.copy(), pose, Ji[:, :6].copy(), errors, C.copy())


def full_frozen_maps(state, C):
    """Independent reproduction of the existing SharedUpper elimination."""
    M, J, v = state['M'], state['Jc'], state['v']
    ce = np.array(state['ce']).copy().reshape(4, 3)
    ce[:, :2] = 0.
    S = np.vstack((np.zeros((6, 18)), np.eye(18)))
    K = np.block([[M, -J.T], [J, np.zeros((12, 12))]])
    Z = np.diag(np.tile([0., 0., 1.], 4))
    rhs = np.column_stack((np.vstack((S, np.zeros((12, 18)))),
                           np.r_[-state['h'], -state['cdotv']-100*ce.reshape(12)-20*J@v],
                           np.vstack((np.zeros((24, 24)), -100*Z@J)),
                           np.vstack((-C, -20*J))))
    out = la.solve(K, rhs)
    return dict(T=out[:24, :18], a0=out[:24, 18], Aq=out[:24, 19:43],
                Av=out[:24, 43:], F=out[24:, :18], f0=out[24:, 18],
                Fq=out[24:, 19:43], Fv=out[24:, 43:])
