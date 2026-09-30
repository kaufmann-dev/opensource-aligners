"""Goals-driven target setup: ideal arch placement, space balance, leveling, bite goals."""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np
from scipy.optimize import brentq

from aligners import fdi
from aligners.analyze import analyze_bite, arch_to_bite, contact_pairs, posed_teeth
from aligners.anatomy import ArchCurve, ArchForm, JawAnatomy, Tooth, ideal_arch_form, incisal_edge
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
IPR_COST = 0.5  # per mm of enamel reduction
INCISOR_COST = 0.1  # per mm of incisor advancement/retraction
MAX_TIPPING_DEG = 12.0  # bucco-lingual crown movement done by tipping, the rest bodily
MAX_BITE_DZ_MM = 0.6  # most a tooth is intruded to clear an occlusal interference
BITE_ITERATIONS = 6  # setup re-solves for bite corrections
SPACING_REFINEMENTS = 3  # extra placement walks to distribute leftover space exactly
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


def _signed_tip(m: Array, axis: Array, angle: float, up: Array) -> float:
    """Rotation angle about `axis` that changes the slope of m against `up` by `angle`."""
    probe = rotation_about(axis, 1e-3)[:3, :3] @ m
    grows = probe @ up > m @ up
    return angle if grows else -angle


def _signed_angle_z(a: Array, b: Array) -> float:
    return float(np.arctan2(a[0] * b[1] - a[1] * b[0], a[0] * b[0] + a[1] * b[1]))


class JawSolver:
    def __init__(
        self,
        anatomy: JawAnatomy,
        goals: dict[str, Any],
        overrides: dict[str, Any],
        vertical: Array | None = None,
    ):
        """`vertical`: the shared frontal-view vertical in this arch frame (the lower arch's
        occlusal normal), against which incisal edges are leveled in both arches."""
        self.a = anatomy
        self.vertical = np.array([0.0, 0.0, 1.0]) if vertical is None else vertical
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
        self._anchor_poses: dict[int, Array] = {}
        self.inclination_target = self._inclination_targets()
        self.edges = {
            n: incisal_edge(np.asarray(anatomy.patches[n].vertices), t.m, t.o)
            for n, t in self.teeth.items()
            if fdi.position(n) <= 2
        }

    # -- geometry helpers ---------------------------------------------------------------------
    def mid(self, t: Tooth) -> Array:
        return (t.mesial_contact + t.distal_contact) / 2

    def anchor_pose(self, n: int) -> Array:
        """Held teeth keep their place along the arch (anchorage, molar relationship) and their
        occlusion, but follow the arch transversely (expansion); `fixed` teeth do not move."""
        if n in self.fixed:
            return np.eye(4)
        return self._anchor_poses.get(n, np.eye(4))

    def _place_anchors(
        self, curve: ArchCurve, level: Callable[[int, float], float], adjust: Adjust
    ) -> None:
        self._anchor_poses = {}
        s_mid = self.midline_s(curve, 0.0)
        for n in self.anchors - self.fixed:
            t = self.teeth[n]
            side = _side(n, self.jaw)
            s_mesial = float(curve.project(t.mesial_contact[:2])[0])
            s_distal = curve.chord_step(s_mesial, side, t.width)
            arc = abs(0.5 * (s_mesial + s_distal) - s_mid)
            self._anchor_poses[n] = self._place(
                t, side, curve, s_mesial, s_distal, level(n, arc), adjust, anchor=True
            )

    def _inclination_targets(self) -> dict[int, float]:
        """Bucco-lingual inclination (radians, + = crown outward) shared by homologous teeth:
        the mean of the left and right tooth, measured against the ideal arch."""
        curve = self.base_form().curve()
        incl = {}
        for n, t in self.teeth.items():
            s = float(curve.project(self.mid(t)[:2])[0])
            out = curve.outward(s)
            incl[n] = float(np.arctan2(t.o @ out, t.o[2]))
        targets = {}
        for n in self.teeth:
            if n in self.fixed:
                continue
            other = fdi.fdi_number(self.jaw, not fdi.is_patient_right(n), fdi.position(n))
            if other in incl:
                targets[n] = 0.5 * (incl[n] + incl[other])
        return targets

    def base_form(self) -> ArchForm:
        """Ideal arch through the contacts, with expansion, meeting the held teeth."""
        widen = {pos: _expansion(self.goals, pos) for pos in range(1, 9)}
        return ideal_arch_form(self.teeth, widen, self.anchors)

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
        form = self.base_form()
        warnings: list[str] = []
        if not g["align"]:
            poses = {n: np.eye(4) for n in self.order}
            base = form.curve(0.0)
            return self._finish(JawSetup(self.jaw, poses, {}, base, 0.0, x_m, {}, warnings))

        level = self._level_target(form.curve(0.0))
        self._place_anchors(form.curve(0.0), level, adjust)
        poses = {n: self.anchor_pose(n) for n in self.anchors}

        def discrepancy(delta: float) -> dict[int, float]:
            _, ends = self._walk(form.curve(delta), x_m, level, adjust, {}, {})
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
                """Unresolved crowding/spacing, then IPR, then incisor movement: space comes
                from arch development first and enamel reduction only as the last resort."""
                D = {
                    side: float(np.interp(d, grid, [x[side] for x in samples])) for side in (1, -1)
                }
                ipr, res = residual_of(D)
                c = sum(r * r for r in res.values() if r > 0 or g["close_spaces"])
                return c + IPR_COST * sum(ipr.values()) + INCISOR_COST * abs(d)

            fine = np.linspace(lo, hi, 161)
            delta = float(fine[np.argmin([cost(d) for d in fine])])

        curve = form.curve(delta)
        ipr, residual = residual_of(discrepancy(delta))
        spacing = {
            side: -r / (len(self.side_teeth(side)) + 0.5)
            if r < 0 and self.side_teeth(side)
            else 0.0
            for side, r in residual.items()
        }
        placed, ends = self._walk(curve, x_m, level, adjust, ipr, spacing)

        # Clearance does not grow 1:1 with the per-contact spacing (neighbors meet at an angle
        # to the arch, contacts settle discretely): refine the spacing per side by bracketed
        # regula falsi and keep the walk that leaves the least crowding, then the least space.
        def badness(e: dict[int, float]) -> float:
            return sum(10.0 * max(0.0, -v) + max(0.0, v) for v in e.values())

        walks = [(badness(ends), dict(spacing), placed, ends)]
        fits = {side: (0.0, -residual[side]) for side in (1, -1)}  # largest spacing, clearance >= 0
        over: dict[int, tuple[float, float]] = {}  # smallest spacing that crowds

        def record() -> None:
            for side in (1, -1):
                point = (spacing[side], ends[side])
                if ends[side] >= 0:
                    fits[side] = max(fits[side], point)
                else:
                    over[side] = min(over.get(side, point), point)

        record()
        for _ in range(SPACING_REFINEMENTS):
            todo = [side for side in (1, -1) if spacing[side] > 0 and not -0.05 < ends[side] < 0.15]
            if not todo:
                break
            for side in todo:
                s0, e0 = fits[side]
                if side in over:
                    s1, e1 = over[side]
                    guess = s0 + e0 * (s1 - s0) / (e0 - e1)
                    spacing[side] = float(
                        np.clip(guess, s0 + 0.2 * (s1 - s0), s1 - 0.2 * (s1 - s0))
                    )
                else:
                    spacing[side] = 1.5 * spacing[side]
            placed, ends = self._walk(curve, x_m, level, adjust, ipr, spacing)
            record()
            walks.append((badness(ends), dict(spacing), placed, ends))
        _, spacing, placed, ends = min(walks, key=lambda w: w[0])
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
        spee_flatten pulls it toward the straight line from the incisors to the anchors. Each
        tooth keeps its tooth type's height above or below that curve, taken as the mean of
        the left and right tooth: leveling evens out left/right differences and outliers
        without shortening naturally long teeth (canines) or lengthening short ones."""
        g = self.goals
        s_mid = self.midline_s(curve, 0.0)
        arcs: dict[int, float] = {}
        for n, t in self.teeth.items():
            arcs[n] = abs(float(curve.project(self.mid(t)[:2])[0]) - s_mid)
        a = np.array([arcs[n] for n in self.teeth])
        z = np.array([t.cusp[2] for t in self.teeth.values()])
        w = np.array([ANCHOR_WEIGHT if n in self.anchors else 1.0 for n in self.teeth])
        A = np.stack([np.ones_like(a), a, a**2], axis=1) * w[:, None]
        c0, c1, c2 = np.linalg.lstsq(A, z * w, rcond=None)[0]
        a_end = float(a.max())

        def fit(x: float) -> float:
            return float(c0 + c1 * x + c2 * x * x)

        residual = {n: float(t.cusp[2]) - fit(arcs[n]) for n, t in self.teeth.items()}
        offset = {}
        for n in self.teeth:
            other = fdi.fdi_number(self.jaw, not fdi.is_patient_right(n), fdi.position(n))
            offset[n] = 0.5 * (residual[n] + residual[other]) if other in residual else residual[n]

        def target(number: int, arc: float) -> float:
            line = fit(0.0) + (fit(a_end) - fit(0.0)) * arc / a_end
            return fit(arc) - g["spee_flatten"] * (fit(arc) - line) + offset[number]

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
        anchor: bool = False,
    ) -> Array:
        """Target pose of one tooth: derotated, contacts on the arch, inclination and height
        adjusted. Held teeth (`anchor`) keep their occlusion: they are only moved bodily onto
        the arch (e.g. with expansion), keeping their rotation, height and inclination."""
        g = self.goals
        p = self.mid(t)
        target_xy = (curve.point(s_mesial) + curve.point(s_distal)) / 2
        target_dir = curve.chord(s_mesial, s_distal)[:2]
        # The last tooth of each side has no distal neighbor, so its measured rotation is only
        # an estimate from the arch direction: keep it rather than "correct" it.
        if not g["derotate"] or anchor or t.fdi in (self.order[0], self.order[-1]):
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
        s_mid = 0.5 * (s_mesial + s_distal)
        out = curve.outward(s_mid)
        across = np.cross([0.0, 0.0, 1.0], out)  # rotating +z about this moves it toward `out`
        if g["harmonize_inclination"] and not anchor and t.fdi in self.inclination_target:
            o_now = R[:3, :3] @ t.o
            now = float(np.arctan2(o_now @ out, o_now[2]))
            R = rotation_about(across, self.inclination_target[t.fdi] - now, t.cr) @ R
        if g["level_incisal_edges"] and not anchor and fdi.position(t.fdi) <= 2:
            # tip about the labial axis until the incisal edge is horizontal in frontal view
            e_now = R[:3, :3] @ self.edges[t.fdi]
            slope = float(np.arcsin(np.clip(e_now @ self.vertical, -1.0, 1.0)))
            labial = R[:3, :3] @ t.b
            tip = _signed_tip(e_now, labial, -slope, self.vertical)
            R = rotation_about(labial, tip, t.cr) @ R
        p = transform_points(R, p[None])[0]
        if g["tipping"] and not anchor:
            # move the crown bucco-lingually by tipping about a point in the root
            o_now = R[:3, :3] @ t.o
            pivot = (
                transform_points(R, t.margin_center[None])[0]
                - g["tipping_pivot"] * t.root_length * o_now
            )
            lever = float((p - pivot) @ o_now)
            need = float((target_xy - p[:2]) @ out[:2])
            angle = float(
                np.clip(
                    np.arctan2(need, max(lever, 3.0)),
                    -np.radians(MAX_TIPPING_DEG),
                    np.radians(MAX_TIPPING_DEG),
                )
            )
            R = rotation_about(across, angle, pivot) @ R
            p = transform_points(R, self.mid(t)[None])[0]
        cusp = transform_points(R, t.cusp[None])[0]
        dz = z_cusp - cusp[2] if g["level"] and not anchor else 0.0
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
    to_bite = arch_to_bite(case)
    if to_bite is None:
        raise CaseError("frames.json has no bite; run `aligners segment` again")
    up_to_low = (np.linalg.inv(to_bite["lower"]) @ to_bite["upper"])[:3, :3]
    vertical = {"lower": np.array([0.0, 0.0, 1.0]), "upper": up_to_low.T @ [0.0, 0.0, 1.0]}
    solvers = {
        jaw: JawSolver(a, goals, plan["teeth"], vertical[jaw]) for jaw, a in anatomies.items()
    }
    setups = {jaw: s.solve() for jaw, s in solvers.items()}
    result: dict[str, Any] = {"warnings": []}
    bite = goals["bite"]
    adjust_u, adjust_l = Adjust(), Adjust()
    initial = analyze_bite(anatomies["upper"], anatomies["lower"], to_bite)
    baseline = {(i["lower"], i["upper"]): i["depth_mm"] for i in initial["interferences"]}
    for _ in range(BITE_ITERATIONS):
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
        else:
            if metrics["overjet_mm"] < bite["min_overjet_mm"] - 0.05:
                adjust_u.delta = (
                    setups["upper"].delta + bite["min_overjet_mm"] - metrics["overjet_mm"]
                )
                changed = True
            if metrics["overbite_mm"] < bite["min_overbite_mm"] - 0.05:
                err = metrics["overbite_mm"] - bite["min_overbite_mm"]
                adjust_u.anterior_dz -= err * bite["overbite_split"]
                adjust_l.anterior_dz -= err * (1 - bite["overbite_split"])
                changed = True
        if (
            bite["align_midlines"]
            and abs(metrics.get("midline_upper_vs_lower_left_mm", 0.0)) > 0.05
        ):
            # + means the upper midline lies to the patient's left; the upper arch frame's +x
            # points to the patient's left, so the upper segment moves by the negative amount
            adjust_u.midline_x = (
                setups["upper"].midline_x - metrics["midline_upper_vs_lower_left_mm"]
            )
            changed = True
        # new occlusal interferences: intrude both teeth of the pair (and their left/right
        # counterparts, so the arch stays symmetric) by half the excess each, within a cap
        front_excess = 0.0
        for hit in metrics["interferences"]:
            excess = hit["depth_mm"] - baseline.get((hit["lower"], hit["upper"]), 0.0) - 0.05
            if (
                excess > 0.05
                and fdi.position(hit["lower"]) <= 3
                and fdi.position(hit["upper"]) <= 3
            ):
                # lower incisors against the upper ones: bring the upper front along
                front_excess = max(front_excess, excess)
            elif excess > 0.05:
                for adj, n, jaw in (
                    (adjust_l, hit["lower"], "lower"),
                    (adjust_u, hit["upper"], "upper"),
                ):
                    twin = fdi.fdi_number(jaw, not fdi.is_patient_right(n), fdi.position(n))
                    for m in {n, twin}:
                        adj.tooth_dz[m] = max(
                            adj.tooth_dz.get(m, 0.0) - (excess / 2 + 0.05), -MAX_BITE_DZ_MM
                        )
                changed = True
        if front_excess > 0:
            # bring the upper front forward as far as allowed, then open the bite (intrude the
            # incisors, keeping the minimum overbite): lower incisal edges that sit lower meet
            # the upper palatal surfaces further labially
            need = front_excess + 0.05
            current = adjust_u.delta if adjust_u.delta is not None else setups["upper"].delta
            advance = min(need, max(0.0, goals["max_incisor_advance_mm"] - current))
            if advance > 0.01:
                adjust_u.delta = current + advance
                changed = True
            spare = metrics["overbite_mm"] - bite["min_overbite_mm"]
            if need - advance > 0.01 and spare > 0.05:
                dz = min(2.0 * (need - advance), spare)
                adjust_u.anterior_dz -= dz * bite["overbite_split"]
                adjust_l.anterior_dz -= dz * (1 - bite["overbite_split"])
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
