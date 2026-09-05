import numpy as np
import torch

from motion_prediction.baselines import constant_velocity
from motion_prediction.data import load_trajectory_samples, make_synthetic_scenario, save_trajectory_samples, scenario_to_sample
from motion_prediction.metrics import ade, best_of_k_ade, fde
from motion_prediction.models import HistoryGRU, SocialMapPredictor


def test_synthetic_preprocessing_is_target_centric():
    sample = scenario_to_sample(make_synthetic_scenario(seed=3), history_steps=10, future_steps=20)
    assert sample.history.shape == (10, 2)
    assert sample.future.shape == (20, 2)
    assert np.allclose(sample.history[-1], 0.0)
    assert sample.neighbors.shape[1:] == (10, 2)
    assert sample.neighbor_is_ego.any()
    assert sample.target_length > 0
    assert sample.traffic_light_states[0] == 6


def test_constant_velocity_has_expected_shape_and_metric():
    history = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]], dtype=np.float32)
    prediction = constant_velocity(history, 2)
    target = np.array([[3.0, 0.0], [4.0, 0.0]], dtype=np.float32)
    assert prediction.shape == target.shape
    assert ade(prediction, target) == 0.0
    assert fde(prediction, target) == 0.0


def test_best_of_k_metric():
    target = np.zeros((3, 2), dtype=np.float32)
    predictions = np.stack([np.ones((3, 2)), target])
    assert best_of_k_ade(predictions, target) == 0.0


def test_models_emit_expected_shapes():
    history = torch.randn(4, 10, 2)
    model = HistoryGRU(hidden_size=16, future_steps=20)
    assert model(history).shape == (4, 20, 2)
    social = SocialMapPredictor(hidden_size=16, future_steps=20, modes=3)
    trajectories, logits = social(
        history,
        torch.randn(4, 8, 10, 2),
        torch.ones(4, 8, 10, dtype=torch.bool),
        torch.randn(4, 32, 32, 2),
        torch.ones(4, 32, 32, dtype=torch.bool),
    )
    assert trajectories.shape == (4, 3, 20, 2)
    assert logits.shape == (4, 3)


def test_social_model_can_take_a_preprocessed_sample():
    sample = scenario_to_sample(make_synthetic_scenario(seed=7), history_steps=10, future_steps=20)
    model = SocialMapPredictor(hidden_size=8, future_steps=20, modes=2)
    trajectories, logits = model(
        torch.from_numpy(sample.history)[None],
        torch.from_numpy(sample.neighbors)[None],
        torch.from_numpy(sample.neighbor_mask)[None],
        torch.from_numpy(sample.map_polylines)[None],
        torch.from_numpy(sample.map_mask)[None],
    )
    assert trajectories.shape == (1, 2, 20, 2)
    assert logits.shape == (1, 2)


def test_preprocessing_uses_scenario_current_time_index():
    scenario = make_synthetic_scenario(seed=8, history=15, future=20)
    sample = scenario_to_sample(scenario, history_steps=10, future_steps=20)
    target = scenario.agents[0]
    expected_origin = target.positions[scenario.current_time_index]
    assert np.allclose(sample.history[-1], 0.0)
    assert np.allclose(sample.future[0], (target.positions[scenario.current_time_index + 1] - expected_origin) @ np.array(
        [
            [np.cos(-target.heading[scenario.current_time_index]), -np.sin(-target.heading[scenario.current_time_index])],
            [np.sin(-target.heading[scenario.current_time_index]), np.cos(-target.heading[scenario.current_time_index])],
        ],
        dtype=np.float32,
    ).T)


def test_trajectory_cache_round_trip(tmp_path):
    original = scenario_to_sample(make_synthetic_scenario(seed=9), history_steps=10, future_steps=20)
    path = tmp_path / "samples.npz"
    save_trajectory_samples([original], path)
    restored = load_trajectory_samples(path)[0]
    assert restored.scenario_id == original.scenario_id
    assert restored.target_id == original.target_id
    assert restored.map_types == original.map_types
    assert np.array_equal(restored.history, original.history)
    assert np.array_equal(restored.neighbor_mask, original.neighbor_mask)
