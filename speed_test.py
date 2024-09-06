import time
import torch
import torchvision
from nitrous_ema import PostHocEMA as NitrousPostHocEMA
from ema_pytorch import PostHocEMA as PytorchPostHocEMA


def test():

    network = torch.compile(torchvision.models.vit_h_14().cuda())
    pytorch_ema = PytorchPostHocEMA(network,
                                    sigma_rels=[0.05, 0.1],
                                    checkpoint_every_num_steps=1e10).cuda()
    nitrous_ema = NitrousPostHocEMA(network,
                                    sigma_rels=[0.05, 0.1],
                                    checkpoint_every_num_steps=1e10).cuda()

    inputs = torch.randn(8, 3, 224, 224).cuda()
    optim = torch.optim.SGD(network.parameters(), lr=0.01, fused=True)

    # warm-up
    for _ in range(10):
        optim.zero_grad()
        network(inputs).mean().backward()
        optim.step()
        pytorch_ema.update()
        nitrous_ema.update()

    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        optim.zero_grad()
        network(inputs).mean().backward()
        optim.step()
    torch.cuda.synchronize()
    print("Without EMA: ", time.time() - start)

    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        optim.zero_grad()
        network(inputs).mean().backward()
        optim.step()
        pytorch_ema.update()
    torch.cuda.synchronize()
    print("Pytorch EMA: ", time.time() - start)

    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        optim.zero_grad()
        network(inputs).mean().backward()
        optim.step()
        nitrous_ema.update()
    torch.cuda.synchronize()
    print("Nitrous EMA: ", time.time() - start)


if __name__ == "__main__":
    test()
