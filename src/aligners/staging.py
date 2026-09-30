"""Stage the setup: per-stage movement limits and collision-free scheduling by start delays."""

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import trimesh

from aligners import fdi
from aligners.analyze import contact_pairs, posed_tooth
from aligners.anatomy import JawAnatomy, Tooth
from aligners.case import Case, read_json, write_json
from aligners.geometry import Array, MeshSDF, transform_points
from aligners.motion import decompose, interpolate, interpolate_about, rotation_last_split

MAX_DELAY_ITERATIONS = 200
MAX_DELAYS_PER_PAIR = 12
SAMPLES = 1500
PAIR_RADIUS_MM = 2.5


@dataclass
class ToothPlan:
    fdi: int
    target: Array
    stages: int
    start: int = 0
    mode: str = "linear"  # or "rotation_last": translation/tip/torque first, then derotation
    split: float = 1.0  # fraction of the stages spent in the first phase (rotation_last)

    def progress(self, s: float) -> float:
        if self.stages == 0:
            return 1.0
        return float(np.clip((s - self.start) / self.stages, 0.0, 1.0))


def required_stages(
    tooth: Tooth, crown: trimesh.Trimesh, T: Array, limits: dict[str, float]
) -> tuple[int, str]:
    """Stages a tooth needs, and the limiting movement."""
    mv = decompose(tooth, T)
    rot_limit = (
        limits["rotation_canine_premolar_deg"]
        if fdi.is_canine_or_premolar(tooth.fdi)
        else limits["rotation_deg"]
    )
    pts = np.asarray(crown.vertices)
    disp = float(np.linalg.norm(transform_points(T, pts) - pts, axis=1).max())
    ratios = {
        "translation": math.hypot(mv.mesial, mv.buccal) / limits["translation_mm"],
        "extrusion": max(mv.extrusion, 0.0) / limits["extrusion_mm"],
        "intrusion": max(-mv.extrusion, 0.0) / limits["intrusion_mm"],
        "rotation": abs(mv.rotation) / rot_limit,
        "tip": abs(mv.tip) / limits["tip_deg"],
        "torque": abs(mv.torque) / limits["torque_deg"],
        "crown_point": disp / limits["crown_point_mm"],
    }
    limiting = max(ratios, key=lambda k: ratios[k])
    worst = ratios[limiting]
    return (0 if worst < 0.05 else math.ceil(worst - 1e-6)), limiting


class Stager:
    def __init__(
        self,
        anatomy: JawAnatomy,
        targets: dict[int, Array],
        limits: dict[str, float],
        ipr: dict[str, float] | None = None,
        ipr_caps: tuple[float, float] = (0.0, 0.0),
    ):
        """`ipr` is the setup's planned IPR per contact; `ipr_caps` (anterior, posterior) bound
        extra IPR the stager may add where no sequencing avoids a collision."""
        self.a = anatomy
        self.ipr = dict(ipr or {})
        self.ipr_caps = ipr_caps
        self.extra_ipr: dict[str, float] = {}
        self.order = anatomy.order()
        self.tol = limits["collision_tolerance_mm"]
        self.plans: dict[int, ToothPlan] = {}
        self.limiting: dict[int, str] = {}
        self.phases: dict[int, tuple[Array, Array, int, int]] = {}
        self.linear_stages: dict[int, int] = {}
        for n in self.order:
            count, why = required_stages(anatomy.teeth[n], anatomy.crowns[n], targets[n], limits)
            A, B = rotation_last_split(anatomy.teeth[n], targets[n])
            n_a, _ = required_stages(anatomy.teeth[n], anatomy.crowns[n], A, limits)
            moved = posed_tooth(anatomy.teeth[n], A)
            crown_a = trimesh.Trimesh(
                transform_points(A, np.asarray(anatomy.crowns[n].vertices)),
                anatomy.crowns[n].faces,
                process=False,
            )
            n_b, _ = required_stages(moved, crown_a, B, limits)
            if n_a > 0 and n_b > 0:
                self.phases[n] = (A, B, n_a, n_b)
            self.plans[n] = ToothPlan(n, targets[n], count)
            self.linear_stages[n] = count
            self.limiting[n] = why
        samples = {n: anatomy.enamel_samples(n, SAMPLES, seed=4) for n in self.order}
        self.sdf = {n: MeshSDF.of(anatomy.contact_solids[n]) for n in self.order}
        self.pairs = contact_pairs(self.order)
        self.ignored: set[tuple[int, int]] = set()
        # Only the part of each crown facing its neighbor can collide with it.
        self.near: dict[tuple[int, int], Array] = {}
        for a, b in self.pairs:
            for x, y in ((a, b), (b, a)):
                d = np.abs(self.sdf[y](samples[x], bound=PAIR_RADIUS_MM))
                self.near[(x, y)] = samples[x][d < PAIR_RADIUS_MM]
        self.allowed = {
            (a, b): max(self.overlap(a, b, 0.0, final=False), self.overlap(a, b, 0.0, final=True))
            + self.tol
            for a, b in self.pairs
        }

    def pose(self, n: int, s: float) -> Array:
        plan = self.plans[n]
        p = plan.progress(s)
        if plan.mode == "linear":
            return interpolate(self.a.teeth[n], plan.target, p)
        A, B, _, _ = self.phases[n]
        if p <= plan.split:
            return interpolate(self.a.teeth[n], A, p / plan.split)
        center = transform_points(A, self.a.teeth[n].cr[None])[0]
        return interpolate_about(center, B, (p - plan.split) / (1.0 - plan.split)) @ A

    def set_mode(self, n: int, mode: str) -> None:
        plan = self.plans[n]
        if mode == "linear":
            plan.stages = self.linear_stages[n]
            plan.mode, plan.split = "linear", 1.0
        else:
            _, _, n_a, n_b = self.phases[n]
            plan.stages = n_a + n_b
            plan.mode, plan.split = "rotation_last", n_a / (n_a + n_b)

    def total(self) -> int:
        return max((p.start + p.stages for p in self.plans.values()), default=0)

    def overlap(self, a: int, b: int, s: float, final: bool | None = None) -> float:
        if final is None:
            Ta, Tb = self.pose(a, s), self.pose(b, s)
        elif final:
            Ta, Tb = self.plans[a].target, self.plans[b].target
        else:
            Ta, Tb = np.eye(4), np.eye(4)
        worst = 0.0
        for x, y, Tx, Ty in ((a, b, Ta, Tb), (b, a, Tb, Ta)):
            pts = self.near[(x, y)]
            if len(pts):
                local = transform_points(np.linalg.inv(Ty) @ Tx, pts)
                worst = max(worst, -float(self.sdf[y](local, bound=1.0).min()))
        return worst

    def _moving(self, n: int, s: float) -> bool:
        plan = self.plans[n]
        return plan.stages > 0 and plan.start < s < plan.start + plan.stages + 0.5

    def first_collision(self, begin: float) -> tuple[int, int, float, float] | None:
        s = begin
        while s <= self.total() + 1e-9:
            for a, b in self.pairs:
                if (a, b) in self.ignored or not (self._moving(a, s) or self._moving(b, s)):
                    continue
                ov = self.overlap(a, b, s)
                if ov > self.allowed[(a, b)] + 1e-3:
                    return a, b, s, ov
            s += 0.5
        return None

    def _approach(self, n: int, other: int, s: float) -> float:
        c = self.a.teeth[n].center
        now = transform_points(self.pose(n, s), c[None])[0]
        before = transform_points(self.pose(n, s - 0.5), c[None])[0]
        toward = transform_points(self.pose(other, s), self.a.teeth[other].center[None])[0] - now
        return float(np.dot(now - before, toward / np.linalg.norm(toward)))

    def _pair_clear(self, a: int, b: int) -> bool:
        s = 0.5
        while s <= self.total() + 1e-9:
            if self.overlap(a, b, s) > self.allowed[(a, b)] + 1e-3:
                return False
            s += 0.5
        return True

    def _add_ipr(self, a: int, b: int) -> bool:
        worst = 0.0
        s = 0.5
        while s <= self.total() + 1e-9:
            worst = max(worst, self.overlap(a, b, s) - self.allowed[(a, b)])
            s += 0.5
        need = float(np.ceil((worst + 0.02) / 0.05) * 0.05)
        key = f"{a}-{b}"
        anterior = fdi.position(a) <= 3 and fdi.position(b) <= 3
        cap = self.ipr_caps[0] if anterior else self.ipr_caps[1]
        have = self.ipr.get(key, 0.0) + self.extra_ipr.get(key, 0.0)
        if need <= 0 or have + need > cap + 1e-9:
            return False
        self.extra_ipr[key] = self.extra_ipr.get(key, 0.0) + need
        self.allowed[(a, b)] += need
        return True

    def _best_fix(self, a: int, b: int) -> tuple[int, int, str, int] | None:
        """Cheapest (extra stages) change of one tooth that clears the pair: a later start,
        and/or doing its derotation after its other movements."""
        best: tuple[int, int, str, int] | None = None
        for n in (a, b):
            plan = self.plans[n]
            if plan.stages == 0:
                continue
            saved = (plan.start, plan.mode, plan.stages, plan.split)
            modes = [plan.mode] + (
                ["rotation_last"] if plan.mode == "linear" and n in self.phases else []
            )
            for mode in modes:
                self.set_mode(n, mode)
                extra = plan.stages - saved[2]
                for d in range(0, MAX_DELAYS_PER_PAIR + 1):
                    if mode == saved[1] and d == 0:
                        continue
                    cost = d + extra
                    if best is not None and cost >= best[0]:
                        break
                    plan.start = saved[0] + d
                    if self._pair_clear(a, b):
                        best = (cost, n, mode, d)
                        break
                plan.start, plan.mode, plan.stages, plan.split = saved
        return best

    def schedule(self) -> list[dict[str, Any]]:
        """Resolve intermediate collisions pair by pair.

        For each colliding pair the cheapest fix in extra stages is applied to one of the two
        teeth: a later start (its neighbor moves out of the way first) and/or moving it in two
        phases (translation, tip and torque first, then the derotation). Pairs no such change
        can fix are reported."""
        for _ in range(MAX_DELAY_ITERATIONS):
            hit = self.first_collision(0.5)
            if hit is None:
                break
            a, b, _, _ = hit
            fix = self._best_fix(a, b)
            if fix is None:
                # The path itself sweeps through the neighbor (e.g. a derotating edge): make
                # room with IPR at this contact if the plan's limits allow, else report it.
                if not self._add_ipr(a, b):
                    self.ignored.add((a, b))
                continue
            _, n, mode, d = fix
            self.set_mode(n, mode)
            self.plans[n].start += d
        remaining = []
        s = 0.5
        while s <= self.total():
            for a, b in self.pairs:
                ov = self.overlap(a, b, s)
                if ov > self.allowed[(a, b)] + 1e-3:
                    remaining.append({"stage": s, "pair": f"{a}-{b}", "overlap_mm": ov})
            s += 0.5
        return remaining


def run_stage(case: Case) -> dict[str, Any]:
    setup = read_json(case.require(case.work / "setup.json", "setup"))
    limits = case.config()["staging"]
    goals = case.plan()["goals"]
    caps = (goals["ipr_max_anterior_mm"], goals["ipr_max_posterior_mm"])
    result: dict[str, Any] = {"jaws": {}, "warnings": []}
    for jaw, info in setup["jaws"].items():
        anatomy = JawAnatomy.load(case.anatomy_dir(jaw), jaw)
        targets = {int(n): np.asarray(t["pose"], float) for n, t in info["teeth"].items()}
        stager = Stager(anatomy, targets, limits, info["ipr_mm"], caps)
        unresolved = stager.schedule()
        total = stager.total()
        stages = [
            {str(n): stager.pose(n, float(s)) for n in stager.order} for s in range(total + 1)
        ]
        result["jaws"][jaw] = {
            "stages": total,
            "teeth": {
                n: {
                    "start": p.start,
                    "stages": p.stages,
                    "mode": p.mode,
                    "limiting": stager.limiting[n],
                }
                for n, p in stager.plans.items()
            },
            "unresolved_collisions": unresolved,
            "poses": stages,
            "ipr_mm": {
                k: round(info["ipr_mm"].get(k, 0.0) + stager.extra_ipr.get(k, 0.0), 2)
                for k in sorted(set(info["ipr_mm"]) | set(stager.extra_ipr))
            },
            "ipr_for_staging_mm": stager.extra_ipr,
        }
        for k, v in stager.extra_ipr.items():
            result["warnings"].append(
                f"{jaw}: {v:.2f} mm IPR at {k} so the teeth can pass each other"
            )
        if unresolved:
            result["warnings"].append(
                f"{jaw}: {len(unresolved)} collisions could not be staged away"
            )
    write_json(case.work / "stages.json", result)
    return result
