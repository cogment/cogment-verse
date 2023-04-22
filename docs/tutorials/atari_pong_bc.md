# Atari Pong Behavior Cloning (BC)

The goal of this tutorial is to showcase various experiment setups possible with Cogment-Verse using behavior cloning.


- Offline data collection
  - Exploring the collected data in the trial datastore
- Training with data collected
  - Exploring the available models in the model registry
- Online behavior cloning


## Collecting Data
To play the game and collect data, use the following command:
```console
python -m main +experiment=adaptive_grid/data_collection
```
By default, only 20 trials will be played before the platform shuts down. To increase this number, go to `config/experiment/data_collection.yaml` and increase the `num_trials: 20` parameter.


The trial datastore process will be store the trial data to disk at the following location: `.cogment_verse/trial_datastore/trial_datastore.db`.

To check the number of trials collected and get general information about the trial datastore content (the content of the `trial_datastore.db` file), two options are available:

1. Go to the section Exploring the Trial Datastore and explore the data in a notebook.
2. If the game is not running (not in the process of playing trials), the open a terminal, activate the .venv, then run the following:
    ```console
    python -m launch_local_services
    ```
    It will launch the necessary processes to interact with the trial datastore and will keep the processes running until you interupt or close this terminal window (As opposed to `python -m main`, which launches all processes and close them automatically once trials are done).

    Open a new terminal window and run the following:
    ```console
    ./.cogment_verse/bin/cogment client trial_datastore --endpoint grpc://localhost:9001 list_trials --count=100
    ```
    If you have more than 100 trials, increase the `--count` attribute. No `--count` attribute will result in only the first 10 trials displayed.

### Clearing existing trial data
To clear all existing data in the trial datastore, simply delete the file `.cogment_verse/trial_datastore/trial_datastore.db`.

*WARNING* once you delete the trial_datastore.db, it is lost forever.

---

## Exploring the Trial Datastore

Go to the template notebook `trial_data/explore_trial_datastore.ipynb`. It contains instructions on the data available from the trial datastore.

---

## Exploring the Model Registry

A list of models available to disk will also be located at `.cogment_verse/model_registry`. The sub-folder names are the `model_ids`.

To list the currently available models and their available versions, go to the template notebook `trial_data/explore_model_registry.ipynb`. It contains instructions on the data available from the model registry.

- To save a model to disk, make sure that the config parameter `run.archive_model` is set to `True`. See `config/adaptive_grid/behavior_cloning.yaml` for an example.
- If no `run.model_id` is provided, it will randomly generate one.



---
