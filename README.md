# nitrous-ema
Fast and simple post-hoc EMA (Karras et al., 2023) with minimal `.item()` calls.

A fork of https://github.com/lucidrains/ema-pytorch

Features added:
- No more `.item()` calls during update which would force a device synchronization and slows things down. `initted` and `step` are now stored as Python types on CPUs. They are still put into the state dict via `set_extra_state` and `get_extra_state`. 
- Added a `step_size_correction` parameter to scale the weighting term (with geometric mean) when `update_every` is larger than 1. Otherwise the effective update rate would be too slow

Starter script:
```python
import torch
import torch.nn as nn
import torch.optim as optim
from nitrous_ema import PostHocEMA

# simple EMA application
data = torch.randn(512, 128)
target = torch.randn(512, 1)
net = nn.Linear(128, 1)
optimizer = optim.SGD(net.parameters(), lr=0.01)
ema = PostHocEMA(net,
                    sigma_rels=[0.05, 0.1],
                    checkpoint_every_num_steps=100,
                    update_every=10,
                    step_size_correction=True)

for _ in range(1000):
    optimizer.zero_grad()
    sample_idx = torch.randint(0, 512, (32, ))
    loss = (net(data[sample_idx]) - target[sample_idx]).pow(2).mean()
    loss.backward()
    optimizer.step()
    ema.update()

# Evaluate the model on the test data
with torch.no_grad():
    loss = (net(data) - target).pow(2).mean()
    print("Loss: ", loss.item())

# Evaluate the EMA model on the test data
with torch.no_grad():
    ema_model = ema.synthesize_ema_model(sigma_rel=0.08, device='cpu')
    loss = (ema_model(data) - target).pow(2).mean()
    print("EMA Loss: ", loss.item())

```