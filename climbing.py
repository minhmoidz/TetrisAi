import numpy as np
import random
from TetrisBattle.envs.tetris_env import TetrisSingleEnv
from Agent import Agent


class SimpleHillClimbing:
    def __init__(self, num_weights=34, max_attempts=20):
        self.num_weights = num_weights
        self.max_attempts = max_attempts
        self.env = TetrisSingleEnv(
            gridchoice="none", obs_type="grid", mode="rgb_array")
        # Bộ trọng số ban đầu
        self.current_weights = np.random.uniform(-1, 1, self.num_weights)
        self.best_weights = self.current_weights.copy()
        self.best_fitness = -9999  # Điểm tốt nhất hiện tại

    def fitness_function(self, weights):
        # Create a new agent and make sure it has a weights attribute
        agent = Agent(weights)
        agent.weights = weights  # Add this line to set weights as an attribute

        state = self.env.reset()
        done = False
        lines_cleared = 0
        blocks_dropped = 0
        while not done:
            actions = agent.choose_action(state, self.env)
            for action in actions:
                state, reward, done, _ = self.env.step(action)
                if done:
                    break
                if reward > 0:  # Nếu xóa dòng
                    lines_cleared += reward
            blocks_dropped += 1  # EDITED.
        # Fitness = số dòng hoàn thành
        fitness = lines_cleared
        # OR fitness = blocks_dropped
        return fitness

    def optimize_weights(self):
        nGame = 1000  # assume that playing 1000 games
        for game in range(nGame):
            for i in range(self.num_weights):  # Duyệt qua từng trọng số
                count_exit = 0
                best_fitness_for_weight = self.best_fitness
                best_weight_value = self.current_weights[i]

                while count_exit < self.max_attempts:
                    # Save original value
                    original_value = self.current_weights[i]

                    # Try a new random value
                    self.current_weights[i] = random.uniform(-1, 1)

                    # Test the new weights
                    candidate_fitness = self.fitness_function(
                        self.current_weights)

                    # If we found a better value, keep it
                    if candidate_fitness > best_fitness_for_weight:
                        best_fitness_for_weight = candidate_fitness
                        best_weight_value = self.current_weights[i]
                        # Also update best overall if needed
                        if candidate_fitness > self.best_fitness:
                            self.best_fitness = candidate_fitness
                            self.best_weights = self.current_weights.copy()
                            print(f"{game},{i}:{self.best_fitness}")
                    else:
                        # Revert to original value if no improvement
                        self.current_weights[i] = original_value

                    count_exit += 1

                # After all attempts, set to the best value found for this weight
                self.current_weights[i] = best_weight_value

            # Print progress every 10 games
            if game % 10 == 0:
                print(
                    f"Game {game}/{nGame}, Current best fitness: {self.best_fitness}")
        return self.best_weights


# Chạy thuật toán Hill Climbing
if __name__ == "__main__":
    hc = SimpleHillClimbing(num_weights=34, max_attempts=20)
    best_weights = hc.optimize_weights()
    print("Best Weights:", best_weights)
