from types import SimpleNamespace

from aligners.setup import IPR_STEP_MM, JawSolver


def test_rounded_ipr_stays_within_the_arch_total():
    goals = {"ipr_max_total_mm": 1.0, "ipr_max_anterior_mm": 0.3, "ipr_max_posterior_mm": 0.5}
    contacts = {1: [(41, 42), (42, 43), (43, 44)], -1: [(31, 32), (32, 33), (33, 34)]}
    solver = SimpleNamespace(
        goals=goals,
        midline_contact=lambda: (41, 31),
        side_contacts=lambda side: contacts[side],
        key=lambda a, b: f"{a}-{b}",
    )
    solver.ipr_cap = lambda a, b: JawSolver.ipr_cap(solver, a, b)  # type: ignore[arg-type]
    ipr, _ = JawSolver.allocate_ipr(solver, {1: 0.62, -1: 0.62})  # type: ignore[arg-type]
    assert sum(ipr.values()) <= goals["ipr_max_total_mm"] + 1e-9
    assert all(abs(v / IPR_STEP_MM - round(v / IPR_STEP_MM)) < 1e-9 for v in ipr.values())
