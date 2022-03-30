# StableBaselines3 - HuggingFace Hub

You can load a model from hugging face hub and run multiple trials. A sample run
configuration `simple_sb3_lander` is defined in `run_params.yaml`.
It loads a model from hugging face hub based as specified by
`repo_id` and `file_name`. Note that the `class_name: data_pb2.SimpleSB3TrainingRunConfig` is
defined in `data.proto`.

You can start the run by:

```RUN_PARAMS=simple_sb3_lander cogment run start_run```
