"""Pure potential-based reward transformation."""


def potential(state):
    xc, yc, xt, yt, _ = state
    return -(abs(xc - xt) + abs(yc - yt))


def learning_reward(base_reward, state, next_state, terminated, mode, gamma, scale):
    if mode == "base":
        return base_reward, 0.0
    if mode != "potential":
        raise ValueError(f"Unknown reward mode: {mode}")
    phi_next = 0 if terminated else potential(next_state)
    shaping = scale * (gamma * phi_next - potential(state))
    return base_reward + shaping, shaping
