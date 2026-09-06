"""Browser saving preserves pyramids and only updates registration metadata."""

import numpy as np
import pytest

from multiview_stitcher import msi_utils, ngff_utils, param_utils
from multiview_stitcher import spatial_image_utils as si_utils
from multiview_stitcher.browser import saving, store
from multiview_stitcher._tests.test_ngff_v06 import make_sim


@pytest.mark.parametrize("ndim", [2, 3])
def test_save_and_update_through_browser_store(tmp_path, monkeypatch, ndim):
    from multiview_stitcher import transformation

    def no_interpolation(*args, **kwargs):
        raise AssertionError("Single-view export must not interpolate pixels")

    monkeypatch.setattr(transformation, "affine_transform", no_interpolation)
    monkeypatch.setattr(
        transformation, "dask_image_affine_transform", no_interpolation
    )
    sim, affine = make_sim(ndim=ndim)
    sim.data = sim.data.astype("uint32") + 2**25
    msim = msi_utils.get_msim_from_sim(sim, scale_factors=[2])
    fetch, write = store.directory_fetch(tmp_path), store.directory_write(
        tmp_path
    )
    url = "/__mvs__/fs/out/image.ome.zarr"
    plans = saving.prepare([msim], ["input.czi"], "registered", "registered")
    saving.save(
        plans[0], url, "registered", "registered", fetch=fetch, write=write
    )
    with pytest.raises(ValueError, match="already exists"):
        saving.save(
            plans[0], url, "registered", "registered", fetch=fetch, write=write
        )
    path = tmp_path / url.lstrip("/")
    restored = ngff_utils.read_msim_from_ome_zarr(path)
    for key in msi_utils.get_sorted_scale_keys(msim):
        expected = msi_utils.get_sim_from_msim(msim, scale=key)
        actual = msi_utils.get_sim_from_msim(restored, scale=key)
        np.testing.assert_array_equal(actual.data, expected.data)
        import zarr

        assert zarr.open_array(
            path / key / "image", mode="r"
        ).metadata.dimension_names == tuple(expected.dims)
    before = {
        p.relative_to(path): p.read_bytes()
        for p in path.rglob("*")
        if p.is_file()
    }
    moved = affine.copy()
    moved[0, -1] += 10
    msi_utils.set_affine_transform(
        msim, param_utils.affine_to_xaffine(moved), "registered"
    )
    plans = saving.prepare(
        [msim],
        [url],
        "registered",
        "registered",
        metadata_only=True,
        fetch=fetch,
    )
    assert before == {
        p.relative_to(path): p.read_bytes()
        for p in path.rglob("*")
        if p.is_file()
    }
    saving.save(
        plans[0],
        None,
        "registered",
        "registered",
        metadata_only=True,
        fetch=fetch,
        write=write,
    )
    after = {
        p.relative_to(path): p.read_bytes()
        for p in path.rglob("*")
        if p.is_file()
    }
    assert [str(p) for p in before if before[p] != after[p]] == ["zarr.json"]
    actual = ngff_utils.read_sim_from_ome_zarr(path)
    np.testing.assert_allclose(
        si_utils.get_affine_from_sim(
            actual, si_utils.DEFAULT_TRANSFORM_KEY
        ).values.squeeze(),
        moved,
    )


def test_save_preflight_rejects_varying_transforms():
    sim, affine = make_sim(n_t=2)
    varying = si_utils.get_affine_from_sim(sim, "registered").copy(deep=True)
    varying.values[1, 0, -1] += 5
    si_utils.set_sim_affine(sim, varying, "registered")
    msim = msi_utils.get_msim_from_sim(sim, scale_factors=[])
    with pytest.raises(NotImplementedError, match="static"):
        saving.prepare([msim], ["input.czi"], "registered", "registered")


def test_worker_rejects_stale_save_plan():
    from multiview_stitcher.browser import Session, SourceSpec, WorkerRuntime

    sim, _ = make_sim()
    runtime = WorkerRuntime()
    runtime.session = Session()
    runtime.session.msims = [
        msi_utils.get_msim_from_sim(sim, scale_factors=[])
    ]
    runtime.session.sources = [SourceSpec(url="input.czi")]
    assert runtime.handle(
        "prepare_save",
        {
            "indices": [0],
            "transform_key": "registered",
            "target": "registered",
        },
    ) == {"count": 1}
    runtime.session.bump_generation()
    with pytest.raises(ValueError, match="changed"):
        runtime.handle("save_image", {"index": 0, "destination": "unused"})


def test_parallel_save_selects_one_view(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from multiview_stitcher.browser import LocalBridge, Session, WorkerRuntime
    from multiview_stitcher.browser.executors import RemoteFusionExecutor

    sim, _ = make_sim()
    source = tmp_path / "source.zarr"
    ngff_utils.write_sim_to_ome_zarr(
        sim,
        source,
        ngff_version="0.6",
        transform_key="registered",
        show_progressbar=False,
    )
    decoy = tmp_path / "decoy.zarr"
    ngff_utils.write_sim_to_ome_zarr(
        sim.copy(data=sim.data * 0),
        decoy,
        ngff_version="0.6",
        transform_key="registered",
        show_progressbar=False,
    )
    session = Session()
    # The exported view is not the first view in the workers' session.
    session.load([str(decoy), str(source)])
    msim = session.msims[1]
    key = session.default_transform_key()
    plan = saving.prepare([msim], [str(source)], key, "registered")[0]
    runtime = WorkerRuntime()
    with ThreadPoolExecutor(max_workers=2) as pool:
        bridge = LocalBridge(
            runner=runtime.run_task,
            map_func=lambda func, items: list(pool.map(func, items)),
        )
        executor = RemoteFusionExecutor(session.spec(), bridge=bridge)
        saving.save(
            plan,
            str(tmp_path / "parallel.zarr"),
            key,
            "registered",
            executor=executor,
            copy_view=1,
        )
    restored = ngff_utils.read_sim_from_ome_zarr(tmp_path / "parallel.zarr")
    np.testing.assert_array_equal(restored.data, sim.data)
