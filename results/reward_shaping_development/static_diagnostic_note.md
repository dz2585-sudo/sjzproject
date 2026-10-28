# Q-learning development Stationary-Target Diagnostic (Read-Only)

The saved run_config, metadata, and final validation records were checked: target movement and turning probabilities were both 0, the root seed was 42, training lasted 10,000 steps, and epsilon decayed over 8,000 steps. Evaluation used the same stationary-target configuration. Final validation success was 13.5%. No additional training was performed.

The first 5 failed scenarios were replayed in record order. The traces identify the first repeated state; Q-row entries follow the action order left/right/up/down/stay/fire.

## Scenario 100000
Replay length: 50 steps; misses: 0; steps with tied maximum Q values: 50; indices of the first repeated state: (0, 1).
```text
t=0: state=(0, 0, 1, 2, -1), action=0, Q=[0.0, -0.0075, 0.0, -0.0075, 0.0, 0.0]
t=1: state=(0, 0, 1, 2, -1), action=0, Q=[0.0, -0.0075, 0.0, -0.0075, 0.0, 0.0]
```

## Scenario 100001
Replay length: 50 steps; misses: 0; steps with tied maximum Q values: 0; indices of the first repeated state: (0, 1).
```text
t=0: state=(2, 4, 1, 4, 1), action=3, Q=[-0.0075, -0.0075, -0.0075, 0.0, -0.0075, -0.1575]
t=1: state=(2, 4, 1, 4, 1), action=3, Q=[-0.0075, -0.0075, -0.0075, 0.0, -0.0075, -0.1575]
```

## Scenario 100002
Replay length: 50 steps; misses: 0; steps with tied maximum Q values: 1; indices of the first repeated state: (1, 2).
```text
t=0: state=(3, 4, 0, 3, -1), action=1, Q=[-0.01929, -0.01494, -0.02218, -0.02233, -0.01494, -0.29244]
t=1: state=(4, 4, 0, 3, -1), action=3, Q=[-0.02127, -0.02127, -0.02218, -0.01494, -0.02233, -0.1575]
t=2: state=(4, 4, 0, 3, -1), action=3, Q=[-0.02127, -0.02127, -0.02218, -0.01494, -0.02233, -0.1575]
```

## Scenario 100003
Replay length: 50 steps; misses: 0; steps with tied maximum Q values: 50; indices of the first repeated state: (1, 2).
```text
t=0: state=(1, 0, 3, 3, 1), action=0, Q=[-0.01388, -0.01494, -0.01494, -0.01388, -0.01494, -0.29244]
t=1: state=(0, 0, 3, 3, 1), action=0, Q=[-0.01494, -0.01494, -0.0239, -0.02588, -0.02665, -0.50387]
t=2: state=(0, 0, 3, 3, 1), action=0, Q=[-0.01494, -0.01494, -0.0239, -0.02588, -0.02665, -0.50387]
```

## Scenario 100004
Replay length: 50 steps; misses: 0; steps with tied maximum Q values: 0; indices of the first repeated state: (0, 1).
```text
t=0: state=(3, 0, 1, 3, 1), action=4, Q=[-0.02036, -0.02142, -0.02233, -0.02142, -0.01494, -0.1575]
t=1: state=(3, 0, 1, 3, 1), action=4, Q=[-0.02036, -0.02142, -0.02233, -0.02142, -0.01494, -0.1575]
```

With a stationary target and a fixed greedy policy, repeated states lead to repeated behavior. Ties are resolved in a fixed action order. Limited training coverage and Q values that have not yet formed an effective path may contribute, but this inspection covers only 5 failed scenarios and cannot establish the overall cause of low success or compare stationary- and moving-target difficulty. No evaluation-configuration or model-loading error was found within this inspection.
