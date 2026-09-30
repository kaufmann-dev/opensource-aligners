"""Stage the setup: per-stage movement limits and collision-free scheduling by start delays."""

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import trimesh

from aligners import fdi
from aligners.analyze import contact_pairs
from aligners.anatomy import JawAnatomy, Tooth
from aligners.case import Case, read_json, write_json
from aligners.geometry import Array, MeshSDF, transform_points
from aligners.motion import decompose, interpolate

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
    def __init__(self, anatomy: JawAnatomy, targets: dict[int, Array], limits: dict[str, float]):
        self.a = anatomy
        self.order = anatomy.order()
        self.tol = limits["collision_tolerance_mm"]
        self.plans: dict[int, ToothPlan] = {}
        self.limiting: dict[int, str] = {}
        for n in self.order:
            count, why = required_stages(anatomy.teeth[n], anatomy.crowns[n], targets[n], limits)
            self.plans[n] = ToothPlan(n, targets[n], count)
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
        return interpolate(self.a.teeth[n], plan.target, plan.progress(s))

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

    def _min_delay(self, victim: int, a: int, b: int) -> int | None:
        """Smallest extra start delay of `victim` that makes the pair collision-free."""
        start = self.plans[victim].start
        try:
            for d in range(1, MAX_DELAYS_PER_PAIR + 1):
                self.plans[victim].start = start + d
                if self._pair_clear(a, b):
                    return d
            return None
        finally:
            self.plans[victim].start = start

    def schedule(self) -> list[dict[str, Any]]:
        """Delay teeth until no intermediate stage collides.

        For each colliding pair, the smallest delay that clears the pair is searched for either
        tooth (the one moving in, or its neighbor, whose final position may be in the way) and
        the cheaper one is applied. Pairs no delay can fix are reported."""
        for _ in range(MAX_DELAY_ITERATIONS):
            hit = self.first_collision(0.5)
            if hit is None:
                break
            a, b, _, _ = hit
            options = []
            for n in (a, b):
                if self.plans[n].stages > 0:
                    d = self._min_delay(n, a, b)
                    if d is not None:
                        options.append((d, n))
            if not options:
                self.ignored.add((a, b))  # timing cannot fix it: the paths themselves collide
                continue
            d, n = min(options)
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
    result: dict[str, Any] = {"jaws": {}, "warnings": []}
    for jaw, info in setup["jaws"].items():
        anatomy = JawAnatomy.load(case.anatomy_dir(jaw), jaw)
        targets = {int(n): np.asarray(t["pose"], float) for n, t in info["teeth"].items()}
        stager = Stager(anatomy, targets, limits)
        unresolved = stager.schedule()
        total = stager.total()
        stages = [
            {str(n): stager.pose(n, float(s)) for n in stager.order} for s in range(total + 1)
        ]
        result["jaws"][jaw] = {
            "stages": total,
            "teeth": {
                n: {"start": p.start, "stages": p.stages, "limiting": stager.limiting[n]}
                for n, p in stager.plans.items()
            },
            "unresolved_collisions": unresolved,
            "poses": stages,
            "ipr_mm": info["ipr_mm"],
        }
        if unresolved:
            result["warnings"].append(
                f"{jaw}: {len(unresolved)} collisions could not be staged away"
            )
    write_json(case.work / "stages.json", result)
    return result
