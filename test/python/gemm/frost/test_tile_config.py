# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The tile-config catalog's name grammar: the cluster's K term (``cluster1x1x{S}``, the K-cluster split-K).  NO GPU.

The K axis of a cluster exists in ONE form -- ``cga_size_k == split_k_slices``, the split-K reduced through distributed shared
memory inside the GEMM kernel -- so the grammar spells ``x{k}`` after ``cluster{m}x{n}`` only where it is that axis, every name
without the term is the 1-deep cluster it always was (every catalog entry round-trips unchanged), and a K term that is not the
slice count is the typed decline.
"""

import pytest

from cudnn.gemm.frost.tile_config import CATALOG, DEFAULT_CONFIG, TileConfig, by_name

pytestmark = pytest.mark.L0

_K_CLUSTER_NAMES = [
    "CONFIG_sm100_128x8x128_128x8x32_cluster1x1x8_1ctamma_splitK8_swapAB",
    "CONFIG_sm100_128x16x128_128x16x32_cluster1x1x8_1ctamma_splitK8_swapAB",
    "CONFIG_sm100_128x32x128_128x32x32_cluster1x1x8_1ctamma_splitK8_swapAB",
    "CONFIG_sm100_128x128x128_128x128x32_cluster1x1x8_1ctamma_splitK8_swapAB",
    "CONFIG_sm100_128x8x128_128x8x32_cluster1x1x4_1ctamma_splitK4_swapAB",
    "CONFIG_sm100_64x8x128_64x8x32_cluster1x1x4_1ctamma_splitK4_swapAB",
    "CONFIG_sm100_128x8x128_128x8x32_cluster1x1x6_1ctamma_splitK6_swapAB",
    "CONFIG_sm100_128x128x128_128x128x32_cluster1x1x2_1ctamma_splitK2",
]


@pytest.mark.parametrize("name", _K_CLUSTER_NAMES)
def test_the_cluster_k_term_round_trips_as_the_split_k_slice_count(name):
    cfg = by_name(name)
    assert cfg.name == name
    assert cfg.cga_size_k == cfg.split_k_slices > 1
    assert cfg.cga_size_mnk == (1, 1, cfg.split_k_slices) == cfg.cluster_shape
    assert cfg.cga_size_m == cfg.cga_size_n == 1
    # the same geometry without the term is the two-kernel split: a DIFFERENT config, 1 deep
    two_kernel = by_name(name.replace(f"x{cfg.split_k_slices}_1ctamma", "_1ctamma"))
    assert two_kernel.cga_size_k == 1 and two_kernel.split_k_slices == cfg.split_k_slices and two_kernel.cluster_shape == (1, 1, 1)
    assert two_kernel != cfg


def test_every_catalog_name_is_one_deep_and_round_trips_unchanged():
    """No catalog entry carries the term (the K-cluster form is named explicitly, never enumerated), and every entry's name
    still resolves to itself."""
    for cfg in (DEFAULT_CONFIG, *CATALOG):
        assert cfg.cga_size_k == 1 and "x1_1ctamma" not in cfg.name.replace("_cluster", "@", 1).split("@", 1)[0], cfg.name
        assert "cluster" in cfg.name and cfg.cluster_shape[2] == 1
        assert by_name(cfg.name) == cfg, cfg.name
        assert by_name(cfg.name).name == cfg.name


@pytest.mark.parametrize(
    "cga_k, slices",
    [(2, 1), (4, 8), (8, 4), (3, 6)],
    ids=["k2_unsplit", "k4_split8", "k8_split4", "k3_split6"],
)
def test_a_k_term_that_is_not_the_slice_count_is_the_typed_decline(cga_k, slices):
    from dataclasses import replace

    base = by_name("CONFIG_sm100_128x8x128_128x8x32_cluster1x1_1ctamma_splitK8_swapAB")
    with pytest.raises(NotImplementedError, match="cross-CTA K reduction otherwise is not implemented"):
        replace(base, cga_size_k=cga_k, split_k_slices=slices)
    # the same through the grammar: the name spells a K term the slice count does not match
    with pytest.raises((NotImplementedError, KeyError)):
        by_name(f"CONFIG_sm100_128x8x128_128x8x32_cluster1x1x{cga_k}_1ctamma{'_splitK' + str(slices) if slices > 1 else ''}_swapAB")


def test_the_k_term_counts_toward_the_cga_limit():
    """A 1x1x16 cluster is 16 CTAs (the architecture limit) and spells its K term in the limit message; 1x1x17 is over it."""
    from dataclasses import replace

    base = by_name("CONFIG_sm100_128x8x128_128x8x32_cluster1x1_1ctamma_splitK8_swapAB")
    sixteen = replace(base, cga_size_k=16, split_k_slices=16)
    assert sixteen.name == "CONFIG_sm100_128x8x128_128x8x32_cluster1x1x16_1ctamma_splitK16_swapAB"
    with pytest.raises(NotImplementedError, match="1x1x17 is 17 CTAs"):
        replace(base, cga_size_k=17, split_k_slices=17)


def test_the_knob_record_declines_the_cluster_k_axis_typed():
    """The shared knob vocabulary has no cluster-K axis: a K-cluster config is named by its catalog name only, never snapped
    to the two-kernel split under the same knobs."""
    from cudnn.gemm.frost.knobs import GemmKnobs

    cfg = by_name(_K_CLUSTER_NAMES[0])
    with pytest.raises(NotImplementedError, match="not in the knob vocabulary"):
        GemmKnobs.from_config(cfg)
    assert GemmKnobs.from_config(by_name(_K_CLUSTER_NAMES[0].replace("x8_1ctamma", "_1ctamma"))).split_k_slices == 8


def test_as_pipeline_carries_the_k_term():
    from cudnn.gemm.frost.tile_config import as_pipeline

    cfg = by_name("CONFIG_sm100_128x128x128_128x128x32_cluster1x1x2_1ctamma_splitK2")
    assert as_pipeline(cfg, "sm100") is cfg
    assert isinstance(cfg, TileConfig) and cfg.fallback_cga_size_mnk == (1, 1, 1)
