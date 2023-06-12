# Overcooked Environment

You can use the overcooked environment integration with cogment-verse by installing a fork of the repository [overcooked_ai](git+https://github.com/wduguay-air/overcooked_ai.git).

To install the forked repository, in addition to the original installation instructions, run the following command:
```console
pip install -r overcooked_requirements.txt
```

The integration uses the `play` run implementation as defined in
`runs/play.py`

Examples of experiment configurations are provided at `config/experiment/overcooked`


## Play
To play overcooked with the default layout: cramped_room, run the command:
```console
python -m main +experiment=overcooked/play
```

### Turn-based vs Real time
The game rendering can be either displayed in real time or in a turn-based fashion. The turn-based variant will wait for an input from the web actor before incrementing a step and rendering the game. A timer is also implemented to return a do nothing action in case the web actor doesn't act.

Turn-based:
```console
python -m main +experiment=overcooked/play
```

Real time:
```console
python -m main +experiment=overcooked/play services.environment.env_name=overcooked-real-time services.environment.turn_based=True
```

## Environment Layouts
Multiple envrionment layouts are available from the [overcooked_ai](git+https://github.com/wduguay-air/overcooked_ai.git) repository. To change the layout of the environment, you can simple change the layout parameter of the environment using the hydra command line options. For example:

```console
python -m main +experiment=overcooked/play services.environment.layout=counter_circuit
```

Alternatively, you can use or add a different environment configuration file under `config/services/environment/overcooked`. For example, to use a different environment layout configuration file, use the command:
```console
python -m main +experiment=overcooked/play services/environment=overcooked/counter_circuit
```

To see all available layouts, go to the [overcooked_ai](git+https://github.com/wduguay-air/overcooked_ai.git) repository under the directory `src/overcooked_ai_py/data/layouts`.

