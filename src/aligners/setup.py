"""Goals-driven target setup: ideal arch placement, space balance, leveling, bite goals."""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np
from scipy.optimize import brentq

from aligners import fdi
from aligners.analyze import analyze_bite, arch_to_bite, contact_pairs, posed_teeth
from aligners.anatomy import ArchCurve, JawAnatomy, Tooth, ideal_arch
from aligners.case import Case, CaseError, write_json
from aligners.fdi import Jaw
from aligners.geometry import (
    Array,
    MeshSDF,
    rotation_about,
    transform_points,
    translation,
)
from aligners.motion import Movement, compose, decompose
from aligners.scan import patient_right_sign

ANCHOR_WEIGHT = 10.0  # leveling fit weight of held teeth
IPR_STEP_MM = 0.05
CONTACT_TOLERANCE_MM = 0.05
WARN_TRANSLATION_MM = 4.0
WARN_ROTATION_DEG = 30.0
WARN_TIP_TORQUE_DEG = 15.0


@dataclass
class Adjust:
    """Bite-driven adjustments to one jaw's solve."""

    delta: float | None = None  # fixed incisor advancement (mm)
    midline_x: float | None = None  # fixed midline x in this arch frame
    anterior_dz: float = 0.0  # vertical incisor change along arch z (+ = occlusal)
    tooth_dz: dict[int, float] = field(default_factory=dict)  # per-tooth vertical correction


@dataclass
class JawSetup:
    jaw: Jaw
    poses: dict[int, Array]
    ipr: dict[str, float]
    curve: ArchCurve
    delta: float
    midline_x: float
    residual_mm: dict[str, float]
    warnings: list[str] = field(default_factory=list)


def _side(n: int, jaw: Jaw) -> int:
    right = patient_right_sign(jaw)
    return int(right if fdi.is_patient_right(n) else -right)


def _expansion(goals: dict[str, Any], pos: int) -> float:
    if pos == 3:
        return goals["expand_canine_mm"] / 2
    if pos in (4, 5):
        return goals["expand_premolar_mm"] / 2
    if pos >= 6:
        return goals["expand_molar_mm"] / 2
    return 0.0


def _signed_angle_z(a: Array, b: Array) -> float:
    return float(np.arctan2(a[0] * b[1] - a[1] * b[0], a[0] * b[0] + a[1] * b[1]))


class JawSolver:
    def __init__(self, anatomy: JawAnatomy, goals: dict[str, Any], overrides: dict[str, Any]):
        self.a = anatomy
        self.jaw = anatomy.jaw
        self.goals = goals
        self.overrides = overrides
        self.teeth = anatomy.teeth
        self.order = anatomy.order()
        self.fixed = {n for n in self.teeth if overrides.get(str(n), {}).get("fixed")}
        anchors = {int(a) for a in goals["anchors"]} & set(self.teeth)
        # every tooth distal to a side's most mesial anchor is an anchor too
        self.anchors = set(self.fixed)
        for side in (1, -1):
            on_side = [n for n in self.teeth if _side(n, self.jaw) == side]
            side_anchor_pos = [fdi.position(n) for n in anchors if n in on_side]
            if side_anchor_pos:
                limit = min(side_anchor_pos)
                self.anchors |= {n for n in on_side if fdi.position(n) >= limit}
        self.movable = [n for n in self.order if n not in self.anchors]
        self.samples = {n: anatomy.enamel_samples(n, 800, seed=3) for n in self.order}
        self.sdfs = {n: MeshSDF.of(anatomy.contact_solids[n]) for n in self.order}
        self._space_samples: dict[float, list[dict[int, float]]] = {}

    # -- geometry helpers ---------------------------------------------------------------------
    def mid(self, t: Tooth) -> Array:
        return (t.mesial_contact + t.distal_contact) / 2

    def anchor_pose(self, n: int) -> Array:
        if n in self.fixed:
            return np.eye(4)
        dx = _side(n, self.jaw) * _expansion(self.goals, fdi.position(n))
        return translation(np.array([dx, 0.0, 0.0]))

    def base_half(self) -> tuple[Array, list[int]]:
        """Ideal arch control points (contact chain) with expansion applied."""
        widen = {pos: _expansion(self.goals, pos) for pos in range(1, 9)}
        fronts = [fdi.position(n) for n in self._front_anchors().values()]
        return ideal_arch(self.teeth, widen, min(fronts) if fronts else 6)

    def curve_for(self, half: Array, positions: list[int], delta: float) -> ArchCurve:
        """Symmetric target arch; the incisors move by delta, tapering to 0 at the anchors.

        `positions` are contact indices (0 = midline contact, k = contact k|k+1)."""
        fronts = [fdi.position(n) for n in self._front_anchors().values()]
        ref = min(fronts) if fronts else max(positions) + 2
        w = np.clip(1 - np.asarray(positions) / max(ref - 1, 1), 0.0, 1.0)
        return ArchCurve(half + np.stack([np.zeros_like(w), delta * w], axis=1))

    def midline_s(self, curve: ArchCurve, x_m: float) -> float:
        return float(curve.project(np.array([x_m, curve.point(0.0)[1]]))[0])

    def _front_anchors(self) -> dict[int, int]:
        """Most mesial anchor per side."""
        out = {}
        for side in (1, -1):
            cands = [n for n in self.anchors if _side(n, self.jaw) == side]
            if cands:
                out[side] = min(cands, key=fdi.position)
        return out

    def key(self, a: int, b: int) -> str:
        """Contact name with teeth in arch order (matches analyze.contact_pairs)."""
        first, second = sorted((a, b), key=self.order.index)
        return f"{first}-{second}"

    def side_teeth(self, side: int) -> list[int]:
        return sorted((n for n in self.movable if _side(n, self.jaw) == side), key=fdi.position)

    def side_contacts(self, side: int) -> list[tuple[int, int]]:
        """Contacts from the midline outward on one side: (mesial tooth, distal tooth)."""
        teeth = self.side_teeth(side)
        out = []
        for a, b in zip(teeth[:-1], teeth[1:], strict=True):
            out.append((a, b))
        front = self._front_anchors().get(side)
        if teeth and front is not None:
            out.append((teeth[-1], front))
        return out

    def midline_contact(self) -> tuple[int, int] | None:
        r, l_ = self.side_teeth(1), self.side_teeth(-1)
        if r and l_ and fdi.position(r[0]) == 1 and fdi.position(l_[0]) == 1:
            return (r[0], l_[0])
        return None

    def ipr_cap(self, a: int, b: int) -> float:
        anterior = fdi.position(a) <= 3 and fdi.position(b) <= 3
        return self.goals["ipr_max_anterior_mm"] if anterior else self.goals["ipr_max_posterior_mm"]

    # -- space balance ------------------------------------------------------------------------
    def allocate_ipr(self, need: dict[int, float]) -> tuple[dict[str, float], dict[int, float]]:
        """IPR per contact (rounded up to 0.05 mm) and the amount credited to each side.

        Each side fills its own contacts, anterior first; the midline contact takes up to half
        its cap from each side. The arch total is capped by ipr_max_total_mm.
        """
        total_cap = self.goals["ipr_max_total_mm"]
        given: dict[tuple[int, str], float] = {}
        total = 0.0
        mid = self.midline_contact()
        for side in (1, -1):
            remaining = max(0.0, need.get(side, 0.0))
            slots = [(self.key(*mid), self.ipr_cap(*mid) / 2, True)] if mid else []
            for a, b in self.side_contacts(side):
                slots.append(
                    (
                        self.key(a, b),
                        self.ipr_cap(a, b),
                        fdi.position(a) <= 3 and fdi.position(b) <= 3,
                    )
                )
            for anterior in (True, False):
                group = [sl for sl in slots if sl[2] == anterior]
                while remaining > 1e-6 and group and total < total_cap - 1e-6:
                    share = min(remaining, total_cap - total) / len(group)
                    keep = []
                    for sl in group:
                        room = sl[1] - given.get((side, sl[0]), 0.0)
                        give = min(share, room)
                        given[(side, sl[0])] = given.get((side, sl[0]), 0.0) + give
                        remaining -= give
                        total += give
                        if room - give > 1e-6:
                            keep.append(sl)
                    group = keep
        ipr: dict[str, float] = {}
        per_side = {1: 0.0, -1: 0.0}
        for (side, k), v in given.items():
            ipr[k] = ipr.get(k, 0.0) + v
            per_side[side] += v
        rounded = {
            k: float(np.ceil(v / IPR_STEP_MM - 1e-6) * IPR_STEP_MM)
            for k, v in ipr.items()
            if v > 1e-6
        }
        return rounded, per_side

    # -- placement ----------------------------------------------------------------------------
    def solve(self, adjust: Adjust | None = None) -> JawSetup:
        adjust = adjust or Adjust()
        g = self.goals
        right = patient_right_sign(self.jaw)
        shift = g["midline_shift_mm"][self.jaw]
        x_m = adjust.midline_x if adjust.midline_x is not None else -right * shift
        half, positions = self.base_half()
        warnings: list[str] = []
        poses = {n: self.anchor_pose(n) for n in self.anchors}
        if not g["align"]:
            poses.update({n: np.eye(4) for n in self.movable})
            base = self.curve_for(half, positions, 0.0)
            return self._finish(JawSetup(self.jaw, poses, {}, base, 0.0, x_m, {}, warnings))

        level = self._level_target(self.curve_for(half, positions, 0.0))

        def discrepancy(delta: float) -> dict[int, float]:
            _, ends = self._walk(self.curve_for(half, positions, delta), x_m, level, adjust, {}, {})
            return {side: -end for side, end in ends.items()}

        def residual_of(D: dict[int, float]) -> tuple[dict[str, float], dict[int, float]]:
            ipr, credited = self.allocate_ipr(D)
            return ipr, {side: D[side] - credited[side] for side in (1, -1)}

        lo = -g["max_incisor_retract_mm"] if g["close_spaces"] else 0.0
        hi = g["max_incisor_advance_mm"]
        if adjust.delta is not None:
            delta = float(np.clip(adjust.delta, -g["max_incisor_retract_mm"], hi))
            if abs(delta - adjust.delta) > 0.05:
                warnings.append(
                    f"bite goal needs {adjust.delta:+.1f} mm incisor change; limited to {delta:+.1f}"
                )
        elif hi <= lo:
            delta = float(lo)
        else:
            # The discrepancy is smooth in delta: sample it coarsely, interpolate the rest.
            # Vertical bite corrections barely change arch length, so the samples are cached
            # per midline position across the bite loop's re-solves.
            grid = np.linspace(lo, hi, 5)
            key = round(x_m, 3)
            if key not in self._space_samples:
                self._space_samples[key] = [discrepancy(d) for d in grid]
            samples = self._space_samples[key]

            def cost(d: float) -> float:
                D = {
                    side: float(np.interp(d, grid, [x[side] for x in samples])) for side in (1, -1)
                }
                _, res = residual_of(D)
                c = sum(r * r for r in res.values() if r > 0 or g["close_spaces"])
                return c + 0.01 * abs(d)

            fine = np.linspace(lo, hi, 161)
            delta = float(fine[np.argmin([cost(d) for d in fine])])

        curve = self.curve_for(half, positions, delta)
        ipr, residual = residual_of(discrepancy(delta))
        spacing = {
            side: -r / (len(self.side_teeth(side)) + 0.5)
            if r < 0 and self.side_teeth(side)
            else 0.0
            for side, r in residual.items()
        }
        placed, ends = self._walk(curve, x_m, level, adjust, ipr, spacing)
        poses.update(placed)
        for side, end in ends.items():
            name = "right" if side == right else "left"
            if end < -0.1:
                warnings.append(
                    f"{name} side: {-end:.1f} mm crowding left unresolved (increase expansion, "
                    "advancement or IPR limits; extraction is out of scope)"
                )
            if spacing[side] > 0.05:
                what = (
                    "left as spaces" if not g["close_spaces"] else "left (retraction limit reached)"
                )
                warnings.append(f"{name} side: {spacing[side]:.2f} mm space per contact {what}")
        setup = JawSetup(
            self.jaw,
            poses,
            ipr,
            curve,
            delta,
            x_m,
            {("right" if sd == right else "left"): r for sd, r in residual.items()},
            warnings,
        )
        return self._finish(setup)

    def _gap(self, a: int, Ta: Array, b: int, Tb: Array) -> float:
        """Signed clearance between two posed crowns: + = space, - = overlap depth."""
        pa = transform_points(np.linalg.inv(Tb) @ Ta, self.samples[a])
        pb = transform_points(np.linalg.inv(Ta) @ Tb, self.samples[b])
        return min(
            float(self.sdfs[b](pa, bound=3.0).min()), float(self.sdfs[a](pb, bound=3.0).min())
        )

    def _excess(self, a: int, b: int, poses: dict[int, Array], ipr: dict[str, float]) -> float:
        """Overlap beyond what the contact's IPR and tolerance allow."""
        allowed = CONTACT_TOLERANCE_MM + ipr.get(self.key(a, b), 0.0)
        gap = self._gap(a, poses.get(a, np.eye(4)), b, poses.get(b, np.eye(4)))
        return -gap - allowed

    def _clearance(
        self,
        prev: int,
        prev_pose: Array,
        n: int,
        place: Callable[[float], Array],
        s0: float,
        side: int,
        target: float,
    ) -> Callable[[float], float]:
        """Clearance minus target as a function of how far tooth n slides past s0."""
        return lambda u: self._gap(prev, prev_pose, n, place(s0 + side * u)) - target

    def _walk(
        self,
        curve: ArchCurve,
        x_m: float,
        level: Callable[[int, float], float],
        adjust: Adjust,
        ipr: dict[str, float],
        spacing: dict[int, float],
    ) -> tuple[dict[int, Array], dict[int, float]]:
        """Place teeth outward from the midline, each touching its mesial neighbor.

        Contact targets are -IPR (planned overlap) + spacing. Returns the poses and, per side,
        the clearance left against the front anchor (+ space, - crowding)."""
        s_mid = self.midline_s(curve, x_m)
        poses: dict[int, Array] = {}
        mesial_s: dict[int, float] = {}

        def placer(n: int, side: int) -> Callable[[float], Array]:
            t = self.teeth[n]

            def place(s_mesial: float) -> Array:
                s_distal = curve.chord_step(s_mesial, side, t.width)
                arc = abs((s_mesial + s_distal) / 2 - s_mid)
                return self._place(t, side, curve, s_mesial, s_distal, level(n, arc), adjust)

            return place

        def root(fn: Callable[[float], float], lo: float = -3.0, hi: float = 3.0) -> float:
            f_lo, f_hi = fn(lo), fn(hi)
            if f_lo >= 0:
                return lo
            if f_hi <= 0:
                return hi
            return float(cast(float, brentq(fn, lo, hi, xtol=0.01)))

        mid = self.midline_contact()
        if mid:
            r, l_ = mid
            target = (
                -ipr.get(self.key(r, l_), 0.0) + (spacing.get(1, 0.0) + spacing.get(-1, 0.0)) / 2
            )
            pr, pl = placer(r, 1), placer(l_, -1)
            u = root(lambda u: self._gap(r, pr(s_mid + u / 2), l_, pl(s_mid - u / 2)) - target)
            mesial_s[r], mesial_s[l_] = s_mid + u / 2, s_mid - u / 2
            poses[r], poses[l_] = pr(mesial_s[r]), pl(mesial_s[l_])
        ends = {1: 0.0, -1: 0.0}
        for side in (1, -1):
            teeth = self.side_teeth(side)
            if not teeth:
                continue
            if teeth[0] not in poses:
                mesial_s[teeth[0]] = s_mid
                poses[teeth[0]] = placer(teeth[0], side)(s_mid)
            gap = spacing.get(side, 0.0)
            prev = teeth[0]
            for n in teeth[1:]:
                target = -ipr.get(self.key(prev, n), 0.0) + gap
                s0 = curve.chord_step(mesial_s[prev], side, self.teeth[prev].width)
                place = placer(n, side)
                u = root(self._clearance(prev, poses[prev], n, place, s0, side, target))
                mesial_s[n] = s0 + side * u
                poses[n] = place(mesial_s[n])
                prev = n
            front = self._front_anchors().get(side)
            if front is not None:
                target = -ipr.get(self.key(prev, front), 0.0) + gap
                ends[side] = self._gap(prev, poses[prev], front, self.anchor_pose(front)) - target
        return poses, ends

    def _level_target(self, curve: ArchCurve) -> Callable[[int, float], float]:
        """Target cusp height as a function of |arc position| from the midline.

        A quadratic along the arch is fitted to the current cusp tips (held teeth weighted);
        spee_flatten pulls it toward the straight line from the incisors to the anchors.
        Upper lateral incisors stay 0.5 mm shorter than the curve.
        """
        g = self.goals
        s_mid = self.midline_s(curve, 0.0)
        pts = []
        for n, t in self.teeth.items():
            a = abs(float(curve.project(self.mid(t)[:2])[0]) - s_mid)
            pts.append((a, t.cusp[2], ANCHOR_WEIGHT if n in self.anchors else 1.0))
        a, z, w = (np.array(v) for v in zip(*pts, strict=True))
        A = np.stack([np.ones_like(a), a, a**2], axis=1) * w[:, None]
        c0, c1, c2 = np.linalg.lstsq(A, z * w, rcond=None)[0]
        a_end = float(a.max())

        def fit(x: float) -> float:
            return float(c0 + c1 * x + c2 * x * x)

        def target(number: int, arc: float) -> float:
            line = fit(0.0) + (fit(a_end) - fit(0.0)) * arc / a_end
            z_t = fit(arc) - g["spee_flatten"] * (fit(arc) - line)
            if self.jaw == "upper" and fdi.position(number) == 2:
                z_t -= 0.5
            return z_t

        return target

    def _place(
        self,
        t: Tooth,
        side: int,
        curve: ArchCurve,
        s_mesial: float,
        s_distal: float,
        z_cusp: float,
        adjust: Adjust,
    ) -> Array:
        g = self.goals
        p = self.mid(t)
        target_xy = (curve.point(s_mesial) + curve.point(s_distal)) / 2
        target_dir = curve.chord(s_mesial, s_distal)[:2]
        if not g["derotate"]:
            # keep the current rotation relative to the arch: aim at the rotated target chord
            s_now = float(curve.project(p[:2])[0])
            cur_dir = curve.chord(s_now - side * t.width / 2, s_now + side * t.width / 2)[:2]
            keep = _signed_angle_z(cur_dir, t.m[:2])
            c, sn = np.cos(keep), np.sin(keep)
            target_dir = np.array(
                [c * target_dir[0] - sn * target_dir[1], sn * target_dir[0] + c * target_dir[1]]
            )
        # turn about the tooth's own long axis (pure rotation, no added tip/torque) until the
        # occlusal projection of its mesiodistal axis follows the target direction
        R = np.eye(4)
        for _ in range(4):
            m_now = R[:3, :3] @ t.m
            err = _signed_angle_z(m_now[:2], target_dir)
            if abs(err) < 1e-4:
                break
            o_now = R[:3, :3] @ t.o
            R = rotation_about(o_now, err / max(abs(o_now[2]), 0.5), t.cr) @ R
        p = transform_points(R, p[None])[0]
        cusp = transform_points(R, t.cusp[None])[0]
        dz = z_cusp - cusp[2] if g["level"] else 0.0
        dz += adjust.tooth_dz.get(t.fdi, 0.0)
        pos = fdi.position(t.fdi)
        if pos <= 2:
            dz += adjust.anterior_dz
        elif pos == 3:
            dz += adjust.anterior_dz / 2
        move = np.array([target_xy[0] - p[0], target_xy[1] - p[1], dz])
        return translation(move) @ R

    # -- finishing ----------------------------------------------------------------------------
    def _finish(self, setup: JawSetup) -> JawSetup:
        self._apply_overrides(setup)
        self._check_contacts(setup)
        for n in self.order:
            mv = decompose(self.teeth[n], setup.poses[n])
            trans = float(np.linalg.norm([mv.mesial, mv.buccal, mv.extrusion]))
            if trans > WARN_TRANSLATION_MM:
                setup.warnings.append(f"tooth {n}: {trans:.1f} mm translation is large")
            if abs(mv.rotation) > WARN_ROTATION_DEG:
                setup.warnings.append(f"tooth {n}: {mv.rotation:.0f} deg rotation is large")
            if max(abs(mv.tip), abs(mv.torque)) > WARN_TIP_TORQUE_DEG:
                setup.warnings.append(
                    f"tooth {n}: tip/torque {mv.tip:.0f}/{mv.torque:.0f} deg is large"
                )
        return setup

    def _apply_overrides(self, setup: JawSetup) -> None:
        for key, ov in self.overrides.items():
            n = int(key)
            if n not in self.teeth:
                if fdi.jaw_of(n) == self.jaw:
                    setup.warnings.append(f"override for missing tooth {n} ignored")
                continue
            if ov.get("fixed"):
                setup.poses[n] = np.eye(4)
                continue
            mv = Movement(
                ov["mesial_mm"],
                ov["buccal_mm"],
                ov["extrusion_mm"],
                ov["rotation_deg"],
                ov["tip_deg"],
                ov["torque_deg"],
            )
            if any(v != 0 for v in mv.as_dict().values()):
                target = posed_teeth(self.a, setup.poses)[n]
                setup.poses[n] = compose(target, mv) @ setup.poses[n]

    def _check_contacts(self, setup: JawSetup) -> None:
        for a, b in contact_pairs(self.order):
            if a in self.anchors and b in self.anchors:
                continue
            excess = self._excess(a, b, setup.poses, setup.ipr)
            if excess > 0.05:
                setup.warnings.append(f"contact {a}-{b}: {excess:.2f} mm overlap remains")


def _jaw_report(anatomy: JawAnatomy, setup: JawSetup) -> dict[str, Any]:
    teeth = {}
    for n in anatomy.order():
        teeth[n] = {
            "pose": setup.poses[n],
            "movement": decompose(anatomy.teeth[n], setup.poses[n]).as_dict(),
        }
    return {
        "teeth": teeth,
        "ipr_mm": setup.ipr,
        "curve": setup.curve.as_dict(),
        "incisor_change_mm": setup.delta,
        "midline_x": setup.midline_x,
        "residual_mm": setup.residual_mm,
    }


def run_setup(case: Case) -> dict[str, Any]:
    plan = case.plan()
    goals = plan["goals"]
    for key in plan["teeth"]:
        if not fdi.is_valid(int(key)):
            raise CaseError(f'[teeth."{key}"] is not an FDI tooth number')
    anatomies = {
        jaw: JawAnatomy.load(case.require(case.anatomy_dir(jaw), "segment"), jaw)
        for jaw in case.jaws()
    }
    solvers = {jaw: JawSolver(a, goals, plan["teeth"]) for jaw, a in anatomies.items()}
    setups = {jaw: s.solve() for jaw, s in solvers.items()}
    result: dict[str, Any] = {"warnings": []}
    bite = goals["bite"]
    to_bite = arch_to_bite(case)
    if to_bite is None:
        raise CaseError("frames.json has no bite; run `aligners segment` again")
    adjust_u, adjust_l = Adjust(), Adjust()
    initial = analyze_bite(anatomies["upper"], anatomies["lower"], to_bite)
    baseline = {(i["lower"], i["upper"]): i["depth_mm"] for i in initial["interferences"]}
    for _ in range(4):
        metrics = analyze_bite(
            anatomies["upper"], anatomies["lower"], to_bite, {j: s.poses for j, s in setups.items()}
        )
        if "overjet_mm" not in metrics:
            break
        changed = False
        if bite["enforce"]:
            err = metrics["overbite_mm"] - bite["overbite_mm"]
            if abs(err) > 0.1:
                adjust_u.anterior_dz -= err * bite["overbite_split"]
                adjust_l.anterior_dz -= err * (1 - bite["overbite_split"])
                changed = True
            if abs(bite["overjet_mm"] - metrics["overjet_mm"]) > 0.1:
                adjust_u.delta = setups["upper"].delta + bite["overjet_mm"] - metrics["overjet_mm"]
                changed = True
            if (
                bite["align_midlines"]
                and abs(metrics.get("midline_upper_vs_lower_left_mm", 0.0)) > 0.1
            ):
                adjust_u.midline_x = (
                    setups["upper"].midline_x + metrics["midline_upper_vs_lower_left_mm"]
                )
                changed = True
        elif metrics["overjet_mm"] < bite["min_overjet_mm"] - 0.05:
            adjust_u.delta = setups["upper"].delta + bite["min_overjet_mm"] - metrics["overjet_mm"]
            changed = True
        # new occlusal interferences: intrude both teeth of the pair by half the excess each
        for hit in metrics["interferences"]:
            excess = hit["depth_mm"] - baseline.get((hit["lower"], hit["upper"]), 0.0) - 0.05
            if excess > 0.05:
                for adj, n in ((adjust_l, hit["lower"]), (adjust_u, hit["upper"])):
                    adj.tooth_dz[n] = adj.tooth_dz.get(n, 0.0) - (excess / 2 + 0.05)
                changed = True
        if not changed:
            break
        setups["lower"] = solvers["lower"].solve(adjust_l)
        setups["upper"] = solvers["upper"].solve(adjust_u)
    result["jaws"] = {jaw: _jaw_report(anatomies[jaw], s) for jaw, s in setups.items()}
    for jaw, s in setups.items():
        result["warnings"] += [f"{jaw}: {w}" for w in s.warnings]
    result["bite_after"] = analyze_bite(
        anatomies["upper"], anatomies["lower"], to_bite, {j: s.poses for j, s in setups.items()}
    )
    write_json(case.work / "setup.json", result)
    return result
