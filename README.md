# Damped Anderson Mixing for Deep Reinforcement Learning: Acceleration, Convergence, and Stabilization (NeurIPS 2021)

Please refer to [StableAA](https://arxiv.org/abs/2110.08896) (NeurIPS 2021) to look into the details of our paper.

## Dependencies
```
gym 0.12.1  
pytorch 1.7.0  
opencv-python  
atari-py
```

## Usage
Take Breakouk game as an example, the paper results can be reproduced by running:  
#### For DQN (DQN agent, AA=0, Soft=0:max operator):
```
python main.py --env_name="BreakoutNoFrameskip-v4" --agent_name="DuelingDQN" --seed=101 --gpu=0 --beta=0.05 --use_restart --reg_scale=0.1 --target_update_freq=2000 --max_steps=15000000 --AA=0 --soft=0
```
#### For DuelingDQN_RAA (DuelingDQN agent, AA=0: vanilla RAA, Soft=0:max operator):
```
python main.py --env_name="BreakoutNoFrameskip-v4" --agent_name="DuelingDQN_RAA" --seed=101 --gpu=0 --beta=0.05 --use_restart --reg_scale=0.1 --target_update_freq=2000 --max_steps=15000000 --AA=0 --soft=0 
```
#### For DuelingDQN_StableAA (DuelingDQN agent, AA=1: stable AA, Soft=1:MellowMax operator, Omega):
```
python main.py --env_name="BreakoutNoFrameskip-v4" --omega=5.0 --agent_name="DuelingDQN_RAA" --seed=101 --gpu=0 --beta=0.05 --use_restart --reg_scale=0.1 --target_update_freq=2000 --max_steps=15000000 --AA=1 --soft=1
```
Hyper-parameters can be modified with different arguments, e.g., Omega, AA, Soft, reg_scale. Please refer to the paper for more details.

## Results
Some experimental data and saved models are found under **logs/**, especially in **scalars.npy**. After training, we can leverage **plot_curve.py** based on the results to plot the learning curves, which is similar to **Figure 1** in our paper.

## Contact

Please refer to ksun6@ualberta in case you have any questions.

## Reference
Please cite our paper if you use this code in your own work:
```
@inproceedings{sun2021damped,
  title={Damped Anderson Mixing for Deep Reinforcement Learning: Acceleration, Convergence, and Stabilization},
  author={Sun, Ke and Wang, Yafei and Liu, Yi and Zhao, Yingnan and Pan, Bo and Jui, Shangling and Jiang, Bei and Kong, Linglong},
  booktitle={Advances in Neural Information Processing Systems},
  year={2021}
}
```

## Acknowledgement
We appreciate the following github repos a lot for their valuable code base:

https://github.com/shiwj16/raa-drl
