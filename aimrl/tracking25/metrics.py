"""Tick-weighted sustained tracking and explicitly capped recovery metrics."""
import hashlib


def recovery(distances, events):
    eligible = [i for i, event in enumerate(events) if event and i + 1 <= 280 and i + 20 < len(distances)]
    lags, recovered = [], 0
    for index in eligible:
        lag = next((j for j in range(21) if distances[index + j] <= 1), None)
        recovered += int(lag is not None)
        lags.append(20 if lag is None else lag)
    return len(eligible), recovered, sum(lags)


class EpisodeMetrics:
    def __init__(self, initial):
        self.distance = []
        self.events = []
        self.hits = self.shots = self.misses = self.close = 0
        self.slow = self.fast = self.displacement = 0
        self.target_speed = self.target_path = self.target_net = 0
        self.turns = self.toggles = self.reflections = 0
        self.base_return = 0.0
        self.first = None
        self.bins = [[0, 0] for _ in range(3)]
        self.target_hash = hashlib.sha256(bytes(initial[2:]))

    def add(self, action, state, reward, info):
        self.distance.append(info['distance'])
        self.events.append(info['random_turn'] or info['speed_toggle'] or info['reflections'] > 0)
        self.hits += int(info['hit']); self.misses += int(info['miss']); self.shots += int(action == 5)
        self.base_return += reward
        self.close += int(info['distance'] <= 1)
        if self.first is None and info['distance'] <= 1:
            self.first = len(self.distance)
        slow, fast = action <= 3, action >= 6
        self.slow += int(slow); self.fast += int(fast)
        group = 0 if info['pre_distance'] <= 3 else 1 if info['pre_distance'] <= 11 else 2
        self.bins[group][0] += int(slow); self.bins[group][1] += int(fast)
        self.displacement += info['actual_displacement']
        self.target_speed += info['target_nominal_speed']; self.target_path += info['target_path_distance']
        self.target_net += info['target_net_displacement']
        self.turns += int(info['random_turn']); self.toggles += int(info['speed_toggle']); self.reflections += info['reflections']
        self.target_hash.update(bytes(state[2:]))

    def finish(self):
        count, recovered, lag = recovery(self.distance, self.events)
        moves = self.slow + self.fast
        result = {'steps': len(self.distance), 'distance_sum': sum(self.distance), 'close_ticks': self.close,
                  'hit_count': self.hits, 'shots': self.shots, 'miss_count': self.misses, 'base_return': self.base_return,
                  'shot_accuracy': self.hits / self.shots if self.shots else None,
                  'acquired': self.first is not None, 'first_acquisition_tick': self.first if self.first is not None else 300,
                  'slow_actions': self.slow, 'fast_actions': self.fast,
                  'fast_movement_fraction': self.fast / moves if moves else None,
                  'actual_displacement_sum': self.displacement, 'target_nominal_speed_sum': self.target_speed,
                  'target_path_distance_sum': self.target_path, 'target_net_displacement_sum': self.target_net,
                  'random_turns': self.turns, 'speed_toggles': self.toggles, 'reflections': self.reflections,
                  'eligible_events': count, 'recovered_events': recovered, 'recovery_capped_lag_sum': lag,
                  'target_trajectory_sha256': self.target_hash.hexdigest()}
        for i, label in enumerate(('0_3', '4_11', '12_plus')):
            result[f'slow_bin_{label}'], result[f'fast_bin_{label}'] = self.bins[i]
        return result


def aggregate(rows):
    def total(name): return sum(float(r[name]) for r in rows)
    ticks, n = total('steps'), len(rows)
    shots, hits, fast, slow = total('shots'), total('hit_count'), total('fast_actions'), total('slow_actions')
    events = total('eligible_events')
    result = {'episodes': n, 'ticks': int(ticks), 'mean_distance': total('distance_sum') / ticks,
              'close_fraction': total('close_ticks') / ticks, 'hits_per_episode': hits / n,
              'shots_per_episode': shots / n, 'misses_per_episode': total('miss_count') / n,
              'mean_base_return': total('base_return') / n, 'pooled_shot_accuracy': hits / shots if shots else None,
              'mean_first_acquisition_tick': total('first_acquisition_tick') / n,
              'acquired_fraction': sum(r['acquired'] in (True, 1, 'True', '1') for r in rows) / n,
              'episodes_with_hit_fraction': sum(float(r['hit_count']) > 0 for r in rows) / n,
              'slow_actions': int(slow), 'fast_actions': int(fast),
              'slow_movement_fraction': slow / (slow + fast) if slow + fast else None,
              'fast_movement_fraction': fast / (slow + fast) if slow + fast else None,
              'mean_crosshair_displacement': total('actual_displacement_sum') / ticks,
              'mean_target_nominal_speed': total('target_nominal_speed_sum') / ticks,
              'mean_target_path_distance': total('target_path_distance_sum') / ticks,
              'mean_target_net_displacement': total('target_net_displacement_sum') / ticks,
              'eligible_events': int(events), 'recovery_fraction': total('recovered_events') / events if events else None,
              'mean_capped_recovery_lag': total('recovery_capped_lag_sum') / events if events else None,
              'random_turns': int(total('random_turns')), 'speed_toggles': int(total('speed_toggles')), 'reflections': int(total('reflections'))}
    for label in ('0_3', '4_11', '12_plus'):
        a, b = total(f'slow_bin_{label}'), total(f'fast_bin_{label}')
        result[f'slow_bin_{label}'], result[f'fast_bin_{label}'] = int(a), int(b)
        result[f'fast_fraction_bin_{label}'] = b / (a + b) if a + b else None
    return result
