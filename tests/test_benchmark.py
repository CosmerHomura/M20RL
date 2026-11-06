from scripts.benchmark import bench_stepping


def test_bench_stepping_single_env_positive():
    fps = bench_stepping(1, steps=20)
    assert fps > 0.0
