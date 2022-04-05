# StableBaselines3 - HuggingFace Hub

You can load a model from hugging face hub and run multiple trials. A few sample run
configurations `simple_sb3_cartpole` and `simple_sb3_lander` are defined in `run_params.yaml`.
They load a model from hugging face hub based as specified by
`repo_id` and `file_name`.
Note that the `class_name: data_pb2.PlayRunConfig` is
defined in `data.proto`.

You can start the run by:

```RUN_PARAMS=simple_sb3_lander cogment run start_run```

It uses the `play` run implementation as defined in
`environment/cogment_verse_environment/base_agent_adapter.py`.
The actor implementation for Hugging Face - Stable Baselines 3
agent adapter is defined in `torch_agents/cogment_verse_torch_agents/hf_sb3/sb3_adapter.py`

We also introduced
```
message HFHubModel {
    string repo_id = 1;
    string filename = 2;
}
```
that is added to `AgentConfig` in `data.proto`. the `repo_id` and
`filename` are used to load the corresponding huggingface model using
the `load_from_hub` function.
