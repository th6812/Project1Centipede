import argparse
import sys
import pdb
import gymnasium as gym
import numpy as np
from gymnasium import wrappers, logger


class Agent(object):
    """The world's simplest agent!"""
    action_number = 0
    elf_possible_colors = {
        "level1": (181, 83, 40),
        "level2": (45, 50, 184)
    }
    centipede_possible_colors = {
        "level1": (184, 70, 162),
        "level2": (184, 50, 50)
    }

    def __init__(self, action_space):
        self.action_space = action_space

    def get_location(self, img, color):
        """
        Returns the location of the object to be considered
        :param img: the image to get the location out off
        :param color: the particular color of the object in search of
        :return: a singular point of th object.
        """
        result = zip(*np.where(img == color)[:2])
        locations = [x for x in result]
        if locations:
            return locations[3]
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
        var = {tuple(x) for x in new_array}  # the array of colors
        if self.elf_possible_colors.get("level1") in var:  # if condition to check if it's level 1 or 2
            elf_color = self.elf_possible_colors.get("level1")
            centipede_color = self.centipede_possible_colors.get("level1")
        else:
            elf_color = self.elf_possible_colors.get("level2")
            centipede_color = self.centipede_possible_colors.get("level2")
        result2 = zip(*np.where(observation == elf_color)[:2])
        elf_locations = [x for x in result2]  # gets the array of everywhere the elf color was found
        if not elf_locations:
            return 0  # returns a no-op to know when the screen is blinking
        longest_seq_elf = self.longest_seq_color(elf_locations)  # returns an image of just the elf on the screen
        result = zip(*np.where(observation == centipede_color)[:2])
        centipede_locations = list(result)
        if not centipede_locations:
            return 0  # returns a no-op if the centipede is not on the screen yet.
        longest_seq_cent = self.longest_seq_color(centipede_locations)  # returns the longest sequence of centipede
        # parts

        centipede_location = self.get_location(longest_seq_cent, centipede_color)  # the actual tuple of the position
        # of the centipede

        elf_location = self.get_location(longest_seq_elf, elf_color)  # the actual tuple of the position
        # of the elf

        self.action_number += 1

        if centipede_location is not None and elf_location is not None:
            offset = centipede_location[1] - elf_location[1]
        else:
            return 0  # return action number for no-op
        if self.action_number % 5 == 0:
            return 1  # return action number for shooting
        if offset < 0:
            return 4  # return action number for moving left
        else:
            return 3  # return action number for moving right

    def longest_seq_color(self, locations):
        """Finds the longest sequence of a particular color in an image given the locations
           that color is found. This code was edited from generative AI
        :param locations: The indexes a particular color is found at.
        :return: An observation that is just the longest sequence of a color against a black backdrop.

        """
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
        """
        Recursively search for all the connected pixels that are one color and mark them the same integer
        This code was lifted directly from generative AI.
        :param row: the row of the image currently on in the recursion
        :param col: the col of the image currently on in the recursion
        :param counter: the current int to mark the pixels
        :param mask: the mask of the colors we are checking for
        :param markers: a mirror of the mask that has the counters at the various pixel locations
        :return: updated markers with the counters at the indexes that fulfilled the checks
        """
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
