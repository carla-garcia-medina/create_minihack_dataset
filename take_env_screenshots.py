import gym
import minihack
import imageio
import os
import shutil
import numpy as np


def get_dataset(env_name, runs):
    env = gym.make(env_name, observation_keys=("glyphs", "pixel"))
    np.set_printoptions(threshold = np.inf)
    
    for run in range(runs):
        env.reset()
        counter = 0
        out_path = 'datasets/{0}/dataset_{1}/'.format(env, run)
        if os.path.exists(out_path):
            shutil.rmtree(out_path)
        os.makedirs(out_path)

        prompts_out_path = out_path + 'prompts.txt'
        prompts_file = open(prompts_out_path, "w")
        while True:
            action = env.action_space.sample()
            obs, rewards, done, info = env.step(action)
            if done:
                break
            img_out_path = out_path + '/{0}.jpg'.format(counter)
            imageio.imwrite(img_out_path, obs['pixel'])
            prompts_file.write(np.array2string(obs['glyphs'],  max_line_width = None)+'\n\n')

            counter += 1


def main():
    get_dataset('MiniHack-River-Monster-v0', 5)
    get_dataset('MiniHack-River-v0', 5)
    

if __name__ == '__main__':
    main()