import argparse
import sys
import pdb
import gymnasium as gym
import numpy as np
from gymnasium import wrappers, logger
import matplotlib.pyplot as plt


class Agent(object):

    """The world's simplest agent!"""
    action_number = 0
    elf_color1 = [181, 83, 40]
    elf_color2 = [45, 50, 184]
    elf_possible_colors = [elf_color1, elf_color2]
    centipede_color1 = [184, 70, 162]
    centipede_color2 = [184, 50, 50]
    centipede_possible_colors = [centipede_color1, centipede_color2]

    def __init__(self, action_space):
        self.action_space = action_space

    def get_location(self, img, colors):
        for color in colors:
            result = zip(*np.where(img == color)[:2])
            locations = [x for x in result]
            if locations:
                return locations[0]
            else:
                return None



    # You should modify this function
    # You may define additional functions in this
    # file or others to support it.
    def act(self, observation, reward, done):
        """Compute and action based on the current state.

        Args:
            observation: a 3d array of the game screen pixels.
                Format: rows x columns x rgb.
            reward: the reward associated with the current state.
            done: whether or not it is a terminal state.

        Returns:
            A numerical code giving the action to take. See
            See the Actions table at:
            https://gymnasium.farama.org/environments/atari/centipede/
        """
        new_array = observation.reshape((observation.shape[0] * observation.shape[1], 3))
        var = {tuple(x) for x in new_array}
        target_col = (184, 70, 162)
        longest_seq_cent = None
        result2 = zip(*np.where(observation == [181, 83, 40])[:2])
        locations2 = [x for x in result2]
        longest_seq_elf = self.longest_seq_color(locations2)
        if target_col in var:
            result = zip(*np.where(observation == [184, 70, 162])[:2])
            locations = [x for x in result]
            longest_seq_cent = self.longest_seq_color(locations)
        centipede_location = self.get_location(longest_seq_cent, self.centipede_possible_colors)
        elf_location = self.get_location(longest_seq_elf, self.elf_possible_colors,)
        offset = 0
        if centipede_location is not None and elf_location is not None:
            offset = centipede_location[1] - elf_location[1]
            print(offset)
        if centipede_location is None and elf_location is None:
            print("None")
            return 0
        if abs(offset) <= 10:
            return 10
        else:
            if offset < 0:
                return 4
            else:
                return 3


        # return self.action_space.sample()

    def longest_seq_color(self, locations):
        mask = np.zeros((210, 160), dtype=bool)
        for row, col in locations:
            mask[row, col] = True
        markers = np.zeros_like(mask, dtype=int)
        counter = 1
        for i in range(210):
            for j in range(160):
                if mask[i, j] == True and markers[i, j] == 0:
                    self.depth_first_search(i, j, counter, mask, markers)
                    counter += 1
        # Find the largest region region
        unique_markers, marker_counter = np.unique(markers, return_counts=True)
        # sliced off the zero and then added 1 to compensate for sliced index.
        largest_marker = unique_markers[np.argmax(marker_counter[1:]) + 1]

        # Creating a mask to match largest marker
        largest_connected_region = np.zeros_like(observation)
        largest_connected_region[markers == largest_marker] = observation[markers == largest_marker]
        return largest_connected_region

    def depth_first_search(self, row, col, counter, mask, markers):
        if mask[row, col] == True and markers[row, col] == 0 and (0 <= row < 210) and (0 <= col < 160):
            markers[row, col] = counter
            for i, j in ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)):
                self.depth_first_search(i, j, counter, mask, markers)


## YOU MAY NOT MODIFY ANYTHING BELOW THIS LINE OR USE
## ANOTHER MAIN PROGRAM
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--env_id', nargs='?', default='Centipede-v0', help='Select the environment to run')
    args = parser.parse_args()

    # You can set the level to logger.DEBUG or logger.WARN if you
    # want to change the amount of output.
    logger.set_level(logger.INFO)

    env = gym.make(args.env_id, render_mode="human")

    # You provide the directory to write to (can be an existing
    # directory, including one with existing data -- all monitor files
    # will be namespaced). You can also dump to a tempdir if you'd
    # like: tempfile.mkdtemp().
    outdir = 'random-agent-results'

    env.unwrapped.seed(0)
    agent = Agent(env.action_space)

    episode_count = 100
    reward = 0
    terminated = False
    score = 0
    special_data = {}
    special_data['ale.lives'] = 3
    observation = env.reset()[0]

    while not terminated:
        action = agent.act(observation, reward, terminated)
        observation, reward, terminated, truncated, info = env.step(action)
        # pdb.set_trace()
        score += reward
        env.render()

    # Close the env and write monitor result info to disk
    print("Your score: %d" % score)
    env.close()
