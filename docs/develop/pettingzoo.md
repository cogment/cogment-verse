## PettingZoo

### Atari Games

#### Additional installation steps

Because of some issues with peting zoo versions and the pip version resolver the following packages need be installed separately after `pip install -r requirements.txt`:

```console
$ pip install SuperSuit==3.7.0
```

#### macOS

This step is only required for Apple silicon-based computers (e.g., M1&2 chips). Clone [Multi-Agent-ALE](https://github.com/Farama-Foundation/Multi-Agent-ALE) repository

```console
$ git clone https://github.com/Farama-Foundation/Multi-Agent-ALE.git
$ cd Multi-Agent-ALE
$ pip install .
```

#### License Activation

Activate [AutoROM](https://github.com/Farama-Foundation/AutoROM) license relating to Atari games.

```sh
AutoROM --accept-license
```
