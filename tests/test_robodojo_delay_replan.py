from concurrent.futures import ThreadPoolExecutor

from scripts.robodojo_delay_replan_deploy import _run_one_episode


class _FakeEnv:
    eval_num = 4

    def __init__(self):
        self.steps = 0
        self.reset_calls = 0
        self.actions = []

    def reset(self):
        self.reset_calls += 1

    def get_obs(self):
        return {"step": self.steps}

    def is_episode_end(self):
        return self.steps >= 4

    def take_action(self, action):
        self.actions.append(action)
        self.steps += 1


class _FakeClient:
    def __init__(self):
        self.calls = []

    def call(self, *, func_name, obs=None):
        self.calls.append((func_name, obs))
        if func_name == "get_action":
            return [{"step": 1}, {"step": 2}, {"step": 3}]
        return None


def test_async_smoke_loop_does_not_reset_task_or_duplicate_bootstrap(monkeypatch):
    monkeypatch.setenv("ROBODOJO_DELAY_MS", "0")
    monkeypatch.setenv("ROBODOJO_REPLAN_STEPS", "2")
    monkeypatch.delenv("ROBODOJO_SWEEP_METRICS", raising=False)
    env = _FakeEnv()
    client = _FakeClient()

    with ThreadPoolExecutor(max_workers=1) as executor:
        result = _run_one_episode(env, client, executor)

    assert env.reset_calls == 0
    assert result["steps"] == 4
    assert result["inference_requests"] >= 2
    assert [name for name, _ in client.calls].count("reset") == 1
    assert [name for name, _ in client.calls].count("update_obs") >= 2
