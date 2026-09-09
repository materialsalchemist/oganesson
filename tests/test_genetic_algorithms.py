"""Regression test for the ase.ga -> ase_ga migration.

ASE 3.27+ removed the bundled `ase.ga` package in favor of the standalone
`ase-ga` PyPI package; `ase.ga.*` now only works as a deprecated bridge (and
only when `ase-ga` happens to be installed anyway), with a FutureWarning
saying the placeholders will be removed entirely in a future release. This
test locks in that oganesson imports directly from `ase_ga` and that GA
population setup (which needs no ML potential) still works end-to-end.
"""
import os
import warnings


def test_genetic_algorithms_imports_without_warnings(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        import oganesson.genetic_algorithms as ga_mod
    assert not any(issubclass(w.category, FutureWarning) for w in caught)
    assert hasattr(ga_mod, "GA")


def test_ga_population_setup(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from oganesson.genetic_algorithms import GA

    ga = GA(
        species=["Na"] * 4 + ["H"] * 4,
        population_size=4,
        box_volume=100,
        rmax=10,
        steps=10,
        model="diep",
    )
    assert ga.N == 4
    assert os.path.isdir(ga.path)
    assert os.path.isfile(os.path.join(ga.path, "ga.db"))
