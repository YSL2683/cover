import argparse
import imageio
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--path")
args = parser.parse_args()

print(args.path)
payload = torch.load(args.path, weights_only=False)
demo = payload[0]

obs_list = [[], []]
for i in range(len(demo)):
    obs = demo[i]
    img1, img2 = obs[:3].transpose((1, 2, 0)), obs[3:6].transpose((1, 2, 0))
    obs_list[0].append(img1)
    obs_list[1].append(img2)

imageio.mimsave("buffer-front.mp4", obs_list[0], fps=30, quality=10, macro_block_size=1)
imageio.mimsave("buffer-hand.mp4", obs_list[1], fps=30, quality=10, macro_block_size=1)
