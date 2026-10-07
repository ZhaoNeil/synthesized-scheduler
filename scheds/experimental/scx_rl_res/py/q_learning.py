import scxbus
import math
import os
import random
import time
from collections import deque, namedtuple

# Cap every numerical thread pool to 1 BEFORE importing numpy/torch. The agent is
# pinned to a single core (--agent-cpu) and any pool threads torch/BLAS spawn
# inherit that pin, so an uncapped pool (torch inter-op defaults to #cores) piles
# dozens of threads onto the one agent core and starves the main thread mid-call
# (measured: ~64 OS threads, ~99% of each inference call spent off-CPU waiting).
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

torch.set_num_threads(1)
# set_num_interop_threads must be called before the first inter-op op; cap it too
# (set_num_threads only bounds intra-op). Guarded in case it was already started.
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass

# --- Overhead accounting (whole-call, per decision kind) ---------------------
# Measured Python-side per call: wall time (perf_counter) AND this thread's CPU
# time (thread_time). The Rust agent separately times the full Rust->Python->Rust
# round-trip around the call; (round-trip - wall) is the pyo3/GIL boundary glue,
# and (wall - cpu) is time the agent thread was descheduled while "in" the call.
_OVHD = {
    "action": {"count": 0, "wall_us": 0.0, "cpu_us": 0.0},
    "slice": {"count": 0, "wall_us": 0.0, "cpu_us": 0.0},
}


def _record_ovhd(kind, w0, c0):
    o = _OVHD[kind]
    o["count"] += 1
    o["wall_us"] += (time.perf_counter() - w0) * 1e6
    o["cpu_us"] += (time.thread_time() - c0) * 1e6


def get_overhead():
    """Return (action_count, action_wall_us, action_cpu_us,
    slice_count, slice_wall_us, slice_cpu_us) accumulated this episode."""
    a, s = _OVHD["action"], _OVHD["slice"]
    return (a["count"], a["wall_us"], a["cpu_us"], s["count"], s["wall_us"], s["cpu_us"])

# Directory the model checkpoints (res.pth / res_best.pth) are read from and
# written to. The Rust agent sets this via set_model_dir() so the scheduler can
# keep its weights alongside the crate instead of a hard-coded ghOSt path.
_MODEL_DIR = "."


def set_model_dir(path):
    """Module-level API expected by Rust: where to load/save checkpoints."""
    global _MODEL_DIR
    _MODEL_DIR = path
    os.makedirs(_MODEL_DIR, exist_ok=True)


def _model_path(name):
    return os.path.join(_MODEL_DIR, name)


# --------- Snapshot helpers ----------
def read_snapshot():
    snap = scxbus.get_latest()
    if snap is None:
        print("No snapshot yet")
        return None
    return snap


def train_step(batch_size=1, reward=None, done=False):
    """
    One scheduler tick:
      - Pull latest snapshot.
      - If 'reward' not provided, compute it from the previous snapshot.
      - Train exactly one step and return the chosen action [0..19].
    """
    w0, c0 = time.perf_counter(), time.thread_time()
    try:
        return step_from_snapshot(batch_size, reward=reward, done=done)
    finally:
        # Exclude the terminal (done=True) finish_episode call from the per-episode
        # action overhead.
        if not done:
            _record_ovhd("action", w0, c0)


N_CORES = 50
# Core excluded from the action space. -1 = exclude nothing (all N_CORES cores
# are valid placement targets): CPU 0 is a worker and the agent runs on a core
# outside [0, N_CORES).
GLOBAL_CPU_ID = -1


class DQN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)  # [B, output_dim]


class SliceDQN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


class DuelingDQN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.value = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.adv = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        f = self.feature(x)
        v = self.value(f)  # [B,1]
        a = self.adv(f)  # [B,A]
        q = v + a - a.mean(dim=1, keepdim=True)
        return q


Transition = namedtuple(
    "Transition", ["state", "action", "reward", "next_state", "done"]
)


class ReplayBuffer:
    def __init__(self, capacity=10_000):
        self.buf = deque(maxlen=capacity)

    def push(self, *args):
        self.buf.append(Transition(*args))

    def state_dict(self):
        transitions = list(self.buf)
        if not transitions:
            return {
                "capacity": self.buf.maxlen,
                "states": torch.empty((0, 0), dtype=torch.float32),
                "actions": torch.empty((0,), dtype=torch.int64),
                "rewards": torch.empty((0,), dtype=torch.float32),
                "next_states": torch.empty((0, 0), dtype=torch.float32),
                "dones": torch.empty((0,), dtype=torch.bool),
            }
        return {
            "capacity": self.buf.maxlen,
            "states": torch.tensor(
                np.stack([t.state for t in transitions]), dtype=torch.float32
            ),
            "actions": torch.tensor([t.action for t in transitions], dtype=torch.int64),
            "rewards": torch.tensor(
                [t.reward for t in transitions], dtype=torch.float32
            ),
            "next_states": torch.tensor(
                np.stack([t.next_state for t in transitions]), dtype=torch.float32
            ),
            "dones": torch.tensor([t.done for t in transitions], dtype=torch.bool),
        }

    def load_state_dict(self, data):
        self.buf = deque(maxlen=int(data["capacity"]))
        states = data["states"].cpu().numpy()
        actions = data["actions"].cpu().numpy()
        rewards = data["rewards"].cpu().numpy()
        next_states = data["next_states"].cpu().numpy()
        dones = data["dones"].cpu().numpy()
        for i in range(len(actions)):
            self.buf.append(
                Transition(
                    states[i].astype(np.float32, copy=False),
                    int(actions[i]),
                    float(rewards[i]),
                    next_states[i].astype(np.float32, copy=False),
                    bool(dones[i]),
                )
            )

    def sample(self, batch_size):
        if batch_size is None:
            batch_size = 32

        batch = random.sample(self.buf, batch_size)
        states = torch.tensor(np.stack([t.state for t in batch]), dtype=torch.float32)
        actions = torch.tensor([t.action for t in batch], dtype=torch.int64).unsqueeze(
            1
        )
        rewards = torch.tensor(
            [t.reward for t in batch], dtype=torch.float32
        ).unsqueeze(1)
        next_states = torch.tensor(
            np.stack([t.next_state for t in batch]), dtype=torch.float32
        )
        dones = torch.tensor([t.done for t in batch], dtype=torch.float32).unsqueeze(1)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buf)


# --------- Online, one-step-at-a-time agent ----------
class OnlineDQNAgent:
    """
    Call agent.act_train(current_metrics, reward=..., done=...) each time you have fresh metrics.
    The agent will:
      1) train one step from the previous transition (if any),
      2) return an action for the *current* metrics.
    """

    def __init__(
        self,
        input_dim=56,
        hidden_dim=128,
        output_dim=N_CORES,
        lr=3e-4,
        gamma=0.99,
        batch_size=64,
        replay_capacity=150_000,
        start_epsilon=1.0,
        end_epsilon=0.02,
        # ~236 placement steps/episode, so ε is still ~0.04 at episode 50. The
        # extra late exploration helps placement reach the low-latency regime;
        # a shorter schedule (8_000) converged early to a worse one.
        epsilon_decay_steps=15_000,
        target_sync_every=8285,
        slice_target_sync_every=700,
        device=None,
    ):
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_sync_every = target_sync_every
        self.slice_target_sync_every = slice_target_sync_every

        self.policy_net = DuelingDQN(input_dim, hidden_dim, output_dim).to(self.device)
        self.target_net = DuelingDQN(input_dim, hidden_dim, output_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.buffer = ReplayBuffer(replay_capacity)

        # The 8 ms slice matters most for the response-time objective: aggressive
        # preemption is how a waiting task gets its first run fastest.
        self.slice_candidates = [8, 32, 174, 390, 1633]
        if os.environ.get("NO_8MS"):
            # Ablation (NO_8MS=1): drop the 8 ms slice. This resizes the slice
            # net, so it needs a fresh model (cannot load a 5-action checkpoint).
            self.slice_candidates = [c for c in self.slice_candidates if c != 8]
        self.initial_slice_idx = self.slice_candidates.index(174)
        self.current_slice_idx = None
        self.n_slice_actions = len(self.slice_candidates)  # target slice index

        slice_input_dim = input_dim + len(self.slice_candidates) + 1
        self.slice_policy_net = SliceDQN(slice_input_dim, 64, self.n_slice_actions).to(
            self.device
        )
        self.slice_target_net = SliceDQN(slice_input_dim, 64, self.n_slice_actions).to(
            self.device
        )
        self.slice_target_net.load_state_dict(self.slice_policy_net.state_dict())
        self.slice_target_net.eval()

        self.slice_optimizer = optim.Adam(self.slice_policy_net.parameters(), lr=1e-4)
        self.slice_buffer = ReplayBuffer(10_000)

        # ε-greedy scheduling
        self.start_epsilon = start_epsilon
        self.end_epsilon = end_epsilon
        self.epsilon_decay_steps = max(1, epsilon_decay_steps)

        # rolling episode/step state
        self.steps_done = 0
        self.slice_steps_done = 0
        self.last_state = None  # np.ndarray shape (6,)
        self.last_action = None  # int

        self.last_slice_state = None
        self.last_slice_action = None
        self.accumulated_slice_reward = 0.0

        # action space size
        self.n_actions = output_dim

        self.training = True
        self.eval_epsilon = 0.0  # greedy in eval

        self.episode_return = 0.0
        self.best_return = -float("inf")

    def _epsilon(self):
        if not self.training:
            return self.eval_epsilon
        t = min(1.0, self.steps_done / self.epsilon_decay_steps)
        return self.end_epsilon + (self.start_epsilon - self.end_epsilon) * math.exp(
            -5 * t
        )

    def _slice_epsilon(self):
        if not self.training:
            return self.eval_epsilon
        # The slice head fires far less often than the placement head, so it
        # uses its own fixed ~5000-step window (~23 episodes) and a 0.02 floor:
        # near-greedy by mid-training so the slice policy can settle, but not so
        # early that it locks into a large-slice regime.
        slice_end_epsilon = 0.02
        macro_decay_steps = 5000
        t = min(1.0, self.slice_steps_done / macro_decay_steps)
        return slice_end_epsilon + (self.start_epsilon - slice_end_epsilon) * math.exp(
            -5 * t
        )

    @torch.no_grad()
    def _select_action(self, state_np):
        eps = self._epsilon()
        if random.random() < eps:
            return random.randrange(self.n_actions), eps
        s = torch.tensor(state_np, dtype=torch.float32, device=self.device).unsqueeze(0)
        q = self.policy_net(s)
        return int(q.argmax(dim=1).item()), eps

    def _optimize_one_step(self):
        bs = getattr(self, "batch_size", 32)
        if bs is None:
            bs = 32

        if not self.training or len(self.buffer) < bs:
            return 0.0

        states, actions, rewards, next_states, dones = self.buffer.sample(bs)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        q_sa = self.policy_net(states).gather(1, actions)
        with torch.no_grad():
            next_q_max = self.target_net(next_states).max(dim=1, keepdim=True).values
            target = rewards + self.gamma * (1.0 - dones) * next_q_max

        loss = nn.functional.smooth_l1_loss(q_sa, target)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10.0)
        self.optimizer.step()
        return float(loss.item())

    def _optimize_slice_step(self):
        bs = getattr(self, "batch_size", 32)
        if bs is None:
            bs = 32
        if not self.training or len(self.slice_buffer) < bs:
            return 0.0

        states, actions, rewards, next_states, dones = self.slice_buffer.sample(bs)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        q_sa = self.slice_policy_net(states).gather(1, actions)
        with torch.no_grad():
            next_actions = self.slice_policy_net(next_states).argmax(
                dim=1, keepdim=True
            )
            next_q = self.slice_target_net(next_states).gather(1, next_actions)
            target = rewards + self.gamma * (1.0 - dones) * next_q

        loss = nn.functional.smooth_l1_loss(q_sa, target)
        self.slice_optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.slice_policy_net.parameters(), 10.0)
        self.slice_optimizer.step()
        return float(loss.item())

    @staticmethod
    def _soft_update(target_net, policy_net, tau):
        """Polyak-average the policy weights into the target net. Smooths the
        bootstrap target so the slice policy converges instead of oscillating
        between regimes the way a hard periodic sync (which jumps the target) did."""
        with torch.no_grad():
            for tp, pp in zip(target_net.parameters(), policy_net.parameters()):
                tp.mul_(1.0 - tau).add_(pp, alpha=tau)

    def _ensure_current_slice_idx(self):
        if self.current_slice_idx is None:
            self.current_slice_idx = self.initial_slice_idx

    def _slice_state(self, metrics):
        self._ensure_current_slice_idx()
        one_hot = np.zeros(len(self.slice_candidates), dtype=np.float32)
        one_hot[self.current_slice_idx] = 1.0
        current_ms = float(self.slice_candidates[self.current_slice_idx])
        max_ms = float(max(self.slice_candidates))
        slice_scale = np.array([math.log(current_ms) / math.log(max_ms)], dtype=np.float32)
        return np.concatenate([metrics.astype(np.float32, copy=False), one_hot, slice_scale])

    def _shape_slice_reward(self, score, target_slice_idx, next_metrics):
        target_ms = float(self.slice_candidates[target_slice_idx])
        baseline_ms = 174.0
        max_ms = float(max(self.slice_candidates))
        target_scale = math.log(target_ms) / math.log(max_ms)
        baseline_scale = math.log(baseline_ms) / math.log(max_ms)

        rq_features = next_metrics[6:] if len(next_metrics) > 6 else []
        max_rq_norm = float(np.max(rq_features)) if len(rq_features) else 0.0
        pressure = (
            0.8 * max(0.0, float(next_metrics[0]))
            + 0.6 * max_rq_norm
            + 0.4 * max(0.0, float(next_metrics[1]))
            + 0.4 * max(0.0, float(next_metrics[2]))
        )
        pressure = math.tanh(pressure / 6.0)

        # Mild prior only: under pressure, lean toward slices at or below the
        # baseline. Kept small so it does not override the learned values.
        large_target_penalty = 2.0 * pressure * max(0.0, target_scale - baseline_scale)
        small_target_bonus = 0.5 * pressure * max(0.0, baseline_scale - target_scale)
        return float(score - large_target_penalty + small_target_bonus)

    # ---- main API ----
    def act_train(self, metrics, batch_size=1, reward=0.0, done=False):
        if self.last_state is not None and self.last_action is not None:
            actions = (
                self.last_action
                if isinstance(self.last_action, list)
                else [self.last_action]
            )
            for a in actions:
                if a is not None:
                    self.buffer.push(self.last_state, a, reward, metrics, done)

            self.episode_return += reward
            self.accumulated_slice_reward += reward

        if self.training:
            self.steps_done += 1
            if self.steps_done % 10 == 0:
                loss = self._optimize_one_step()

            if self.steps_done % self.target_sync_every == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())

        eps = self._epsilon()
        cpu_actions = []

        if self.training and random.random() < eps:
            valid_cores = [c for c in range(self.n_actions) if c != GLOBAL_CPU_ID]
            cpu_actions = [random.choice(valid_cores) for _ in range(batch_size)]
        else:
            state_t = torch.tensor(
                metrics, dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            with torch.no_grad():
                q_values = self.policy_net(state_t)

                if torch.isnan(q_values).any() or torch.isinf(q_values).any():
                    print("[WARNING] Q-values exploded! Resetting to zeros.")
                    q_values = torch.zeros_like(q_values)

                temperature = 0.5
                q_values_stable = q_values - q_values.max(dim=1, keepdim=True).values
                logits = q_values_stable / temperature
                if 0 <= GLOBAL_CPU_ID < self.n_actions:
                    logits[0, GLOBAL_CPU_ID] = -float("inf")

            penalty_weight = 0.5
            assigned_counts = torch.zeros(self.n_actions, device=self.device)

            if self.training:
                for _ in range(batch_size):
                    adjusted_logits = (
                        logits - assigned_counts.unsqueeze(0) * penalty_weight
                    )
                    probs = torch.softmax(adjusted_logits, dim=1)
                    action = torch.multinomial(probs, num_samples=1).item()
                    cpu_actions.append(action)
                    assigned_counts[action] += 1
            else:
                for i in range(batch_size):
                    adjusted_logits = (
                        logits - assigned_counts.unsqueeze(0) * penalty_weight
                    )
                    probs = torch.softmax(adjusted_logits.squeeze(0), dim=0)
                    if (
                        not torch.isfinite(probs).all()
                        or float(probs.sum().item()) <= 0.0
                    ):
                        valid_cores = [
                            c for c in range(self.n_actions) if c != GLOBAL_CPU_ID
                        ]
                        action = valid_cores[i % len(valid_cores)]
                    else:
                        cdf = torch.cumsum(probs, dim=0)
                        u = ((i + 0.5) * 0.6180339887498949) % 1.0
                        threshold = torch.tensor(u, device=self.device)
                        action = int(torch.searchsorted(cdf, threshold).item())
                        action = min(action, self.n_actions - 1)
                    cpu_actions.append(action)
                    assigned_counts[action] += 1

        self.last_state = metrics
        self.last_action = cpu_actions

        if done:
            self.last_slice_state = None
            self.last_slice_action = None
            self.accumulated_slice_reward = 0.0

        return cpu_actions

    def infer_time_slice(self, metrics, slice_reward=None):
        self.slice_steps_done += 1
        slice_state = self._slice_state(metrics)

        if self.last_slice_state is None or self.last_slice_action is None:
            self.last_slice_state = slice_state
            self.last_slice_action = self.current_slice_idx
            self.accumulated_slice_reward = 0.0
            return self.slice_candidates[self.current_slice_idx]

        if self.last_slice_state is not None and self.last_slice_action is not None:
            score = (
                float(slice_reward)
                if slice_reward is not None
                else self.accumulated_slice_reward
            )
            score = self._shape_slice_reward(score, self.last_slice_action, metrics)
            # Fold the slice reward into the episode return: the placement
            # reward is potential-based (telescopes to ~0 over an episode), so
            # this per-tick backlog cost is what makes the reported return a
            # time integral that tracks the latency objective across regimes.
            self.episode_return += score
            self.slice_buffer.push(
                self.last_slice_state, self.last_slice_action, score, slice_state, False
            )

            if self.training:
                if self.slice_steps_done % 4 == 0:
                    loss = self._optimize_slice_step()
                    # Soft target update each optimize step (tau=0.005) to
                    # avoid the oscillation of a periodic hard sync.
                    self._soft_update(
                        self.slice_target_net, self.slice_policy_net, tau=0.005
                    )

        eps = self._slice_epsilon()

        if random.random() < eps:
            target_slice_idx = random.randrange(self.n_slice_actions)
        else:
            state_t = torch.tensor(
                slice_state, dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            with torch.no_grad():
                slice_q_values = self.slice_policy_net(state_t)
            target_slice_idx = int(slice_q_values.argmax(dim=1).item())

        if target_slice_idx > self.current_slice_idx:
            self.current_slice_idx += 1
        elif target_slice_idx < self.current_slice_idx:
            self.current_slice_idx -= 1

        chosen_slice = self.slice_candidates[self.current_slice_idx]

        self.last_slice_state = slice_state
        self.last_slice_action = target_slice_idx
        self.accumulated_slice_reward = 0.0

        return chosen_slice

    def set_training(self, flag: bool):
        self.training = bool(flag)

    def reset_episode(self):
        self.last_state = None
        self.last_action = None
        self.episode_return = 0.0

        self.last_slice_state = None
        self.last_slice_action = None
        self.accumulated_slice_reward = 0.0
        self.current_slice_idx = None

    def get_and_reset_episode_return(self):
        ret = float(self.episode_return)
        if not math.isfinite(ret):
            ret = 0.0  # never print nan/inf -> avoids NA in train_metrics.csv
        self.episode_return = 0.0
        return ret

    def save(self, path="schedulers/rl/res.pth"):
        torch.save(
            {
                "policy": self.policy_net.state_dict(),
                "target": self.target_net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "steps_done": self.steps_done,
                "slice_policy": self.slice_policy_net.state_dict(),
                "slice_target": self.slice_target_net.state_dict(),
                "slice_optimizer": self.slice_optimizer.state_dict(),
                "slice_steps_done": self.slice_steps_done,
                "slice_buffer": self.slice_buffer.state_dict(),
            },
            path,
        )

    def load(self, path="schedulers/rl/res.pth", strict=True):
        data = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(data["policy"], strict=strict)
        self.target_net.load_state_dict(data["target"], strict=strict)
        self.optimizer.load_state_dict(data["optimizer"])
        self.steps_done = data.get("steps_done", 0)
        if "slice_policy" in data:
            try:
                self.slice_policy_net.load_state_dict(
                    data["slice_policy"], strict=strict
                )
                self.slice_target_net.load_state_dict(
                    data["slice_target"], strict=strict
                )
                self.slice_optimizer.load_state_dict(data["slice_optimizer"])
                self.slice_steps_done = data.get("slice_steps_done", 0)
                self.slice_buffer.load_state_dict(data["slice_buffer"])
            except RuntimeError as e:
                print(f"[Python] Skipping incompatible slice weights: {e}")


_global_agent = OnlineDQNAgent()
_prev_snapshot = None
_prev_slice_snapshot = None


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def set_training(flag: bool):
    """Module-level API expected by Rust: toggle training/eval mode."""
    _global_agent.set_training(flag)


def reset_agent():
    global _global_agent, _prev_snapshot, _prev_slice_snapshot
    _global_agent = OnlineDQNAgent()
    _prev_snapshot = None
    _prev_slice_snapshot = None


def reset_episode():
    """Module-level API expected by Rust: clear per-episode state."""
    global _prev_snapshot, _prev_slice_snapshot
    _global_agent.reset_episode()
    _prev_snapshot = None
    _prev_slice_snapshot = None


def get_and_reset_episode_return():
    ret = _global_agent.get_and_reset_episode_return()
    if _global_agent.training:
        # Always persist the latest weights. res_best.pth is promoted by the
        # training harness on the objective metric (lowest first-run latency),
        # NOT by episodic return: the reward is only a proxy and here it
        # anti-correlates with latency (return falls while latency improves), so
        # a return-based "best" freezes an early, high-latency checkpoint. The
        # agent can't see the episode's latency (it's computed post-hoc from the
        # task log), so best-checkpoint selection lives in rl_res.sh.
        _global_agent.save(_model_path("res.pth"))
    return ret


def load_trained_model(path="schedulers/rl/res.pth"):
    """Load previously saved weights into the global agent."""
    try:
        _global_agent.load(path)
        print(f"[Python] Successfully loaded model weights from {path}")
    except Exception as e:
        print(f"[Python] Failed to load model: {e}")


def metrics_from_snapshot(snap):
    # 1. Basic task state
    log_unstarted = np.log1p(snap["unstarted_tasks"])

    # 2. Per-CPU queue lengths
    rq_sizes = np.array([float(size) for size in snap["rq_sizes"]], dtype=np.float32)

    # 3. Load balance metrics, log-scaled to match the per-core features below.
    avg_rq = np.log1p(np.mean(rq_sizes))
    std_rq = np.log1p(np.std(rq_sizes))

    # 4. Global features
    global_features = [
        log_unstarted,
        avg_rq,
        std_rq,
        0.0,  # placeholder to preserve input_dim
        0.0,
        0.0,
    ]

    # 5. Per-queue features: log1p. During the arrival burst per-core queues run
    #    into the 1000s; log1p does not saturate there, so the placement net can
    #    still tell shorter cores from longer ones.
    rq_features = np.log1p(rq_sizes).tolist()

    state = np.array(global_features + rq_features, dtype=np.float32)
    return state


def step_from_snapshot(batch_size=1, reward=None, done=False):
    global _prev_snapshot
    snap = scxbus.get_latest()
    if snap is None:
        bs = batch_size if batch_size is not None else 1
        valid_cores = [c for c in range(N_CORES) if c != GLOBAL_CPU_ID]
        if _global_agent.training:
            return [random.choice(valid_cores) for _ in range(bs)]
        return [valid_cores[i % len(valid_cores)] for i in range(bs)]

    state = metrics_from_snapshot(snap)

    if reward is None and _prev_snapshot is not None:
        reward_to_use = compute_reward(_prev_snapshot, snap)
    elif reward is None:
        reward_to_use = 0.0
    else:
        reward_to_use = float(reward)

    actions = _global_agent.act_train(
        state, batch_size=batch_size, reward=reward_to_use, done=done
    )

    _prev_snapshot = snap
    return actions


def infer_time_slice():
    w0, c0 = time.perf_counter(), time.thread_time()
    try:
        return _infer_time_slice_body()
    finally:
        _record_ovhd("slice", w0, c0)


def _infer_time_slice_body():
    """Module-level API expected by Rust for slow-loop macro tuning."""
    global _prev_slice_snapshot
    snap = scxbus.get_latest()
    if snap is None:
        return 3000

    state = metrics_from_snapshot(snap)
    slice_reward = None
    if _prev_slice_snapshot is not None:
        slice_reward = compute_slice_reward(_prev_slice_snapshot, snap)
    chosen_slice = _global_agent.infer_time_slice(state, slice_reward=slice_reward)
    _prev_slice_snapshot = dict(snap)
    return chosen_slice


def _response_pressure(snap):
    rq_sizes = np.array([float(x) for x in snap["rq_sizes"]], dtype=np.float32)
    not_started = float(snap["unstarted_tasks"])
    total_rq = float(np.sum(rq_sizes))
    mean_rq = max(1.0, float(np.mean(rq_sizes)))
    max_rq = float(np.max(rq_sizes))
    std_rq = float(np.std(rq_sizes))

    queue_per_core = total_rq / max(1.0, float(len(rq_sizes)))
    imbalance = std_rq / mean_rq
    bottleneck = max_rq / mean_rq

    # Weights for the res objective (first-run latency). not_started is the BPF
    # nr_waiting_first_run -- the never-run backlog, excluding requeued tasks.
    # By Little's law, integrating it over the episode is the accumulated
    # first-run latency, so it leads the reward. queue_per_core (total queue,
    # including preempted tasks) is weighted lower; max_rq and
    # imbalance/bottleneck give placement its load-balance signal.
    pressure = (
        2.0 * np.log1p(not_started)
        + 0.5 * np.log1p(queue_per_core)
        + 1.0 * np.log1p(max_rq)
        + 0.5 * imbalance
        + 0.3 * bottleneck
    )
    # Defensive: a malformed snapshot must never turn the reward -> the replay
    # buffer -> the logged episode_return into nan (which prints unparseably and
    # lands as NA in train_metrics.csv, and would corrupt the Q-update).
    return float(pressure) if np.isfinite(pressure) else 0.0


def compute_slice_reward(prev_snap, curr_snap):
    # Per-tick cost of the standing waiting backlog. The slow loop ticks at a
    # fixed 500 ms, so summing this over an episode is a time integral of the
    # backlog -- by Little's law that is proportional to what the accumulated
    # first-run latency metric measures, regardless of how many scheduling
    # events the regime generates. Draining the backlog is already credited
    # through the value function (lower future pressure), so only the level is
    # scored.
    del prev_snap  # level-based: the previous snapshot is not used
    response_pressure = _response_pressure(curr_snap)
    return float(np.tanh(-response_pressure / 8.0) * 8.0)


# --- Reward policy ---
def compute_reward(prev_snap, curr_snap):
    # Potential-based shaping on the pressure level: r = gamma*Phi(s') - Phi(s)
    # with Phi = -log1p(pressure). Each step still rewards placements that lower
    # the pressure, but over an episode the sum telescopes to ~Phi(end) -
    # Phi(start) instead of scaling with the number of placement batches, so
    # high-churn regimes (small slices => many batches) are not penalized just
    # for making more decisions.
    gamma = 0.99
    prev_phi = -float(np.log1p(_response_pressure(prev_snap)))
    curr_phi = -float(np.log1p(_response_pressure(curr_snap)))
    return float(gamma * curr_phi - prev_phi)
