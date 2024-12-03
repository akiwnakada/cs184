# HW3 Writeup

## Choices for Hyperparameters:

## DAgger on the CartPole environment
lr: 1e-4
num_dataset_samples: 1,000
num_rollout_steps: 2,500
dagger_epochs: 20
batch_size: 64
## DAgger on the MountainCar environment
Identical to the Cartpole environment
## BC on the CartPole environment
lr: 1e-3 
num_dataset_samples: 10,000
num_rollout_steps: 2,500
bc_epochs: 100
batch_size: 64
## BC on the MountainCar environment
Identical to the Cartpole environment

## Discussion

Our choice of hyperparameters was mainly influenced by whether we were using the BC algorithm or the Dagger algorithm. Initially, we tested both algorithms using the same values for epochs and num_dataset_samples (20 epochs and 1000 num_dataset_samples), but while the Dagger algorithm achieved loss of below 0.2, the BC algorithm did not converge nearly as fast and did not achieve loss below 0.5. This showed us that we needed to provide more data and run more iterations for the BC algorithm compared to the Dagger algorithm. Intuitively this makes sense given that the BC algorithm is much more naive than the Dagger algorithm. Ultimately, we played around with the num_dataset_samples and epochs values for the BC algorithms in the two environments and settled on num_dataset_samples = 10000, bc_epochs = 100. These values allowed the agent to achieve loss under 0.2 while still running quickly. 

The next parameter we addressed was the learning rate. For all of the previous experiments we had been using a constant learning rate of 1e-3. While running our experiments, we noticed that, while the loss graphs for the BC algorithms were smooth, the loss graphs for the Dagger algorithms were quite jumpy. Because of this, we lowered the learning rate to 1e-4 for the Dagger algorithms to smoothen out the curve. Ultimately we are still not sure why this occured but it is an interesting topic for future discussion. 

We used the same hyperparameters for both environments because our algorithms (BC and Dagger) performed quite well in both environments, achieving sub 0.3 loss around the board. It is worth mentioning, however, that Dagger made larger improvements over BC in the CartPole environment than in the MountainCar environment. This is perhaps due to the fact that the MountainCar environment is more complex than the CartPole environment. 