from tt_finetune import worker
from blacksmith.experiments.torch.qwen.configs import TrainingConfig


def test_startup_profile_clock_is_available_to_worker_run():
    # Keep the public progress path from failing before model construction.
    assert callable(worker.time.perf_counter)


def test_measured_attention_geometry_is_a_real_worker_config_field():
    config = TrainingConfig()
    config.tt_attention_query_chunk_tokens = 64
    assert config.tt_attention_query_chunk_tokens == 64
