import torch
import torch.nn as nn
import torch.optim as optim
from nitrous_ema import PostHocEMA


def test1():
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


def test2():
    # test loading and saving the EMA module
    net = nn.Linear(128, 1)
    ema = PostHocEMA(net, sigma_rels=[0.05, 0.1])
    for _ in range(100):
        ema.update()

    torch.save(ema.state_dict(), 'ema.pth')
    del ema
    new_ema = PostHocEMA(net, sigma_rels=[0.05, 0.1])
    new_ema.load_state_dict(torch.load('ema.pth', weights_only=True))
    print(new_ema.step, new_ema.ema_models[0].initted)
    for _ in range(100):
        new_ema.update()


if __name__ == "__main__":
    test1()
    test2()
