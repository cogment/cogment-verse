## Isaac gym

Cogment Verse can use NVIDIA's [Isaac Gym](https://developer.nvidia.com/isaac-gym) environments.

> ⚠️ You'll need to use python3.8 (not python3.9)

1. download the zip file from [NVIDIA webpage](https://developer.nvidia.com/isaac-gym)
   , unzip and copy the `isaacgym` folder to this repo.
2. clone [IsaacGymEnvs](https://github.com/NVIDIA-Omniverse/IsaacGymEnvs) and copy the
   folder inside the `isaacgym` folder
3. comment out line-32 in `isaacgym/IsaacGymEnvs/isaacgymenvs/__init__.py`
4. (Assuming you already installed requirements.txt), run `pip install -r isaac_requirements.txt`.
5. run `nvidia-smi` to check that you have NVIDIA drivers and proper cuda installations.
6. (Assuming you already have mlflow running in a different terminal), Run `python -m main services/environment=ant`
