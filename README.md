# Fork of Generative Adversarial Imitation Learning

## Timing

(For Step 2, the major bottleneck)

- **Classic**: ~11 hours.
- **ModernStochastic**: ~60 hours. Gaaah.
- **ModernEntreg**: each regularization value takes ~10 hours, so ~30 hours total
  for the three values.
- **Humanoid**: N/A

Yeah, I wish there was a better way to do this sequentially. Probably Scala
would be the best option. Remember that the classic ones and Reacher
("ModernEntreg") used seven random initializations.


## Plots

![cartpole](figures/CartPole-v0.png)

![mountaincar](figures/MountainCar-v0.png)
