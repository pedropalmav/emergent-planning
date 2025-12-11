## Training probes

1. Generate a probing train and test dataset

```bash
python create_probe_dataset.py --num_episodes 3000 --name train
python create_probe_dataset.py --num_episodes 1000 --name test --env_name "valid-"
```

2. Train a $K \times K$ (e.g. $1 \times 1$ or $3 \times 3$) probe to predict feature $FEATURE$ (e.g. either `agent_onto_after` for $C_A$ or `tracked_box_next_push_onto_with` for $C_B$)

```bash
python3 train_conv_probe.py --feature FEATURE --kernel K --num_epochs 10
```

These probes can be trained for the following square-level concepts/features from the paper:

- `agent_onto_after`: direction which agent steps onto squares from ($C_A$ / AgentApproachDirection)
- `tracked_box_next_push_onto_with`: direction which box is pushed off of squares ($C_B$ / BoxPushDirection)
- `agent_onto`: squares agent will step onto (AgentApproach)
- `tracked_box_next_push_from`: squares that boxes will be pushed off of (BoxPush)
- `agent_onto_with`: direction which agent steps off of squares from (AgentExitDirection)
- `tracked_box_next_push_onto_after`: direction which box is pushed onto squares from (BoxApproachDirection)
