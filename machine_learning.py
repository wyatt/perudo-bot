import math
import random
import time

from keras import Input
import numpy as np
from silence_tensorflow import silence_tensorflow

silence_tensorflow()

import tensorflow as tf
from keras.models import Sequential
from tensorflow.keras.layers import Dense
# from keras.src.saving import load_model
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import tqdm
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

tf.keras.utils.disable_interactive_logging()

physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)



DEBUG = False

def printd(*args, **kwargs):
    if (DEBUG):
        print(*args, **kwargs)

class PerudoModel:
    def __init__(self, input_shape, output_shape):
        self.model = self.create_model(input_shape, output_shape)
        self.fitness = 0

    def create_model(self, input_shape, output_shape):
        model = Sequential([
            Input(shape=(input_shape,)),
            Dense(30, activation='relu'),
            Dense(30, activation='relu'),
            Dense(output_shape, activation='linear')
        ])
        model.compile(optimizer='adam', loss='mse')
        return model

    def mutate(self, mutation_rate=0.1):
        weights = self.model.get_weights()
        for i in range(len(weights)):
            mask = np.random.random(weights[i].shape) < mutation_rate
            weights[i] += np.random.normal(0, 0.1, weights[i].shape) * mask
        self.model.set_weights(weights)

class Population:
    def __init__(self, size, input_shape, output_shape):
        self.models = [PerudoModel(input_shape, output_shape) for _ in range(size)]

    def select_parents(self):
        min_fitness = min(model.fitness for model in self.models)
        adjusted_fitness = [model.fitness - min_fitness + 1 for model in self.models]
        total_fitness = sum(adjusted_fitness)
        if total_fitness == 0:
            return random.sample(self.models, 2)
        probabilities = [fitness / total_fitness for fitness in adjusted_fitness]
        return np.random.choice(self.models, 2, p=probabilities, replace=False)

    def crossover(self, parent1, parent2):
        child = PerudoModel(parent1.model.input_shape[1], parent1.model.output_shape[1])
        child_weights = []
        for w1, w2 in zip(parent1.model.get_weights(), parent2.model.get_weights()):
            mask = np.random.random(w1.shape) < 0.5
            child_w = w1 * mask + w2 * (1 - mask)
            child_weights.append(child_w)
        
        child.model.set_weights(child_weights)
        return child

    def evolve(self):
        new_population = []
        for _ in range(len(self.models)):
            parent1, parent2 = self.select_parents()
            child = self.crossover(parent1, parent2)
            child.mutate()
            new_population.append(child)
        self.models = new_population

# Current bet count, 1x
# Current bet value, 1x
# Palifico round, 1x
# Total dice, 1x
# Num dice in hand, 1x
# n dice in hand (<=5), 5x (num of 1s, 2s, 3s, 4s, 5s, 6s)
# Is my turn 1x (Need to remember to add Palifico state)
# Previous 8 bets (count, value, type) 16x

# [count, value]

# Outputs:
# Dudo!
# Bet count
# Bet value
input_placeholders = [0] * 28

class Player():
    def __init__(self, name, game, model=None):
        self.name = name
        self.total_dice = 5
        self.is_palifico = False
        self.game = game
        self.model = model
        self.roll_dice()

    def roll_dice(self):
        self.dice = [random.randint(1, 6) for _ in range(self.total_dice)]

    def valid_outputs(self):
        valid_outputs = []

        min_bet = self.game.min_bet_count()
        min_bet_ones = self.game.min_bet_count(True)

        if self.game.current_bet[0] != 0:
            printd("Current bet ", self.game.current_bet)
            valid_outputs.append([1, 0, 0])

        printd("Min bet: ", min_bet)
        printd("Current bet: ", self.game.current_bet)

        # 1. Calculate all possible bets (within a reasonable range)
        # Calculating for 1s
        min_bet_temp_ones = min_bet_ones + 1 if self.game.current_bet[1] == 1 else min_bet_ones
        for x in range(max(min_bet_temp_ones,1), min(min_bet_temp_ones + 5, self.game.total_dice())):
            if self.game.current_bet[2] == 1 and self.game.current_bet[0] != 0:
                # Palifico round where you are not starting
                if 1 != self.game.current_bet[1]:
                    continue
            valid_outputs.append([0, x, 1])

        # Calculating for 2-6
        for val in range(max(self.game.current_bet[1],2), 7):
            if self.game.current_bet[2] == 1 and self.game.current_bet[0] != 0:
                # Palifico round where you are not starting
                if val != self.game.current_bet[1]:
                    continue
            min_bet_temp = min_bet + 1 if self.game.current_bet[1] == val else min_bet
            for x in range(max(min_bet_temp,1), min(min_bet_temp + 5, self.game.total_dice())):
                valid_outputs.append([0, x, val])

        return valid_outputs

    def select_action(self, model, epsilon, valid_actions: list[list[int]]):
        """Select an action based on epsilon-greedy strategy."""
        printd("Valid actions:", valid_actions)
        printd("State:", self.state())
        if random.random() < epsilon:
            return random.choice(valid_actions)  # Random action (exploration)
        else:
            state = np.array(self.state())[np.newaxis]
            q_values = model.predict(state)[0]

            action_mask = np.zeros(len(q_values), dtype=bool)
            for action in valid_actions:
                action_index = self.action_to_index(action)
                action_mask[action_index] = True


            printd("Q-values:", q_values)
            # Find the valid action with the highest Q-value
            masked_q_values = np.where(action_mask, q_values, -np.inf)
            best_action_index = np.argmax(masked_q_values)
            return self.index_to_action(best_action_index)
        

    def action_to_index(self, action):
        # Convert action [dudo, count, value] to a single index
        dudo, count, value = action
        
        if dudo:
            return 0  # Index for dudo action
        else:
            min_bet = self.game.min_bet_count()
            min_bet_ones = self.game.min_bet_count(True)
            min_bet_temp = min_bet + 1 if self.game.current_bet[1] == value else min_bet
            min_bet_temp_ones = min_bet_ones + 1 if self.game.current_bet[1] == 1 else min_bet_ones
            subtract_val = min_bet_temp_ones if value == 1 else min_bet_temp
            printd("action-to_index", action, subtract_val, ((value - 1) * 5) + (count - subtract_val))
            return 1 + ((value - 1) * 5) + (count - subtract_val) 

    def index_to_action(self, index):
        if index == 0:
            return [1, 0, 0]  # Dudo action
        else:
            index -= 1
            min_bet = self.game.min_bet_count()
            min_bet_ones = self.game.min_bet_count(True)
            add_val = min_bet_ones if index <= 5 else min_bet
            count = (index % 5) + 1 + add_val
            value = (index // 5) + 1
            printd("index-to-action", index + 1, add_val, [0, count, value])
            return [0, count, value]

    def state(self):
        [count, value, palifico] = self.game.current_bet
        total_dice = self.game.total_dice()
        num_dice_in_hand = len(self.dice)
        dice_counts = [self.dice.count(i) for i in range(1, 7)]

        previous_bets = self.game.previous_bets
        is_my_turn = 1 if self == self.game.current_player else 0

        return [count, value, 1 if palifico else 0, total_dice, num_dice_in_hand, *dice_counts, is_my_turn] + previous_bets

    def __str__(self):
        return f"{self.name} has {self.dice}"



class Game():
    def __init__(self, total_players):
        self.players = [Player(f"Player {i}", self) for i in range(total_players)]
        self.current_player = self.players[0]
        self.current_player_idx = 0
        self.previous_player = None
        # [count, value, palifico]
        self.current_bet = [0, 0, 0]
        self.previous_bets = [0] * 16

    def current_bet(self):
        return self.current_bet
    # Min bet count IF val is greater
    def min_bet_count(self, ones=False):
        if self.current_bet[0]:
            if self.current_bet[1] == 1:
                if ones:
                    return self.current_bet[0]
                else:
                    return (self.current_bet[0] * 2) + 1
            else:
                if ones:
                    return math.ceil(self.current_bet[0] / 2)
                else:
                    return self.current_bet[0]

        min_bet_init = int(max((self.total_dice() // 3.5), 1))

        if ones:
            return min_bet_init // 2
        return min_bet_init

    def take_action(self, action: list[int]):
        # There might be a reward immediately or after the next player
        # Returns [reward_for_current_player, reward_for_prev_player, game_over]
        # If reward is 0, then it should wait for the next players rount

        # [Dudo, Bet count, Bet value]
        if action[0] == 1:
            # There will be a reward immediately
            if self.was_dudo_successful():
                self.previous_player.total_dice -= 1
                is_game_over = self.is_game_over()
                if is_game_over:
                    return [100, -20, True]
                else:
                    is_player_dead = self.previous_player.total_dice == 0
                    if is_player_dead:
                        self.players.remove(self.previous_player)
                        self.next_round(self.current_player)
                        return [10, -20, False]
                    else:
                        self.next_round(self.current_player)
                        return [10, -5, False]
            else:
                self.current_player.total_dice -= 1
                is_game_over = self.is_game_over()
                if is_game_over:
                    return [-20, 10, True]
                else:
                    is_player_dead = self.current_player.total_dice == 0
                    if is_player_dead:
                        self.players.remove(self.current_player)
                        self.next_round(self.previous_player)
                        return [-20, 2, False]
                    else:
                        printd("Current player: ", self.current_player)
                        printd("Previous player: ", self.previous_player)
                        self.next_round(self.previous_player)
                        return [-5, 2, False]
        else:
            self.previous_bets = [action[1], action[2], *self.previous_bets[:14]]
            self.current_bet = [action[1], action[2], self.current_player.is_palifico]
            self.next_turn()
            return [0, 2, False]

    def was_dudo_successful(self):
        [count, value] = self.current_bet[:2]
        total_count = 0
        for player in self.players:
            total_count += player.dice.count(value)
            if value != 1:
                total_count += player.dice.count(1)
        return total_count >= count

    def is_game_over(self):
        # Check if only one player has dice left
        players_with_dice = 0
        for player in self.players:
            if player.total_dice > 0:
                players_with_dice += 1
        printd("Players with dice: ", players_with_dice)
        return players_with_dice == 1

    def next_turn(self):
        self.previous_player = self.current_player
        printd("Setting Previous player", self.previous_player)
        self.current_player = self.players[(self.current_player_idx + 1) % len(self.players)]
        self.current_player_idx = (self.current_player_idx + 1) % len(self.players)


    def next_round(self, winner: Player):
        for player in self.players:
            if len(player.dice) == 1:
                player.is_palifico = not player.is_palifico
            player.roll_dice()
        self.current_player = winner
        self.current_player_idx = self.players.index(winner)
        self.previous_player = None
        self.current_bet = [0,0,0]

    def total_dice(self):
        return sum(len(player.dice) for player in self.players)

# Training parameters
POPULATION_SIZE = 30  # Reduced from 50
NUM_GENERATIONS = 30  # Reduced from 100
NUM_GAMES_PER_GENERATION = 20  # Reduced from 100
EPSILON_START = 0.5  # Reduced from 0.5 for faster convergence
EPSILON_END = 0.01
EPSILON_DECAY = (EPSILON_END / EPSILON_START) ** (1 / NUM_GENERATIONS)

# Initialize population
input_shape = 28  # Adjust based on your state representation
output_shape = 31  # Adjust based on your action space (30 possible bets + 1 for Dudo)
population = Population(POPULATION_SIZE, input_shape, output_shape)

start_time = time.time()


def evaluate_model(model, generation, model_index, epsilon):
    model.fitness = 0
    for game_num in range(NUM_GAMES_PER_GENERATION):
        # Create a new game with random number of players (2-6)
        num_players = random.randint(3, 6)
        game = Game(num_players)
    
  
        # Assign different models to players
        for j, player in enumerate(game.players):
            # Use the current model for one player, and random models for others
            if j == 0:
                player.model = model
            else:
                player.model = random.choice(population.models)
        
        done = False
        prev_action = None
        prev_state = None
        prev_player_pointer = None
        game_turns = 0

        while not done and game_turns < 100:
            current_player = game.current_player
            action = current_player.select_action(current_player.model.model, epsilon, current_player.valid_outputs())
            printd(action, current_player, current_player.game.previous_player, current_player.valid_outputs())
            [curr_reward, prev_reward, done] = game.take_action(action)
            printd("Done: ", done)
            printd("Rewards: ", curr_reward, prev_reward)

            if current_player.model == model:  # Only update fitness for the model we're currently evaluating
                if curr_reward != 0:
                    model.fitness += curr_reward
            elif prev_action and prev_state and prev_player_pointer and prev_player_pointer.model == model:
                model.fitness += prev_reward

            prev_state = current_player.state()
            prev_action = action
            prev_player_pointer = game.current_player
            game_turns += 1


        if (game_num % 5 == 0):
            print(f"Generation {generation + 1}/{NUM_GENERATIONS}, Model {model_index + 1}/{POPULATION_SIZE}, {game_num / NUM_GAMES_PER_GENERATION * 100}% Complete")
    
    print(f"Generation {generation + 1}/{NUM_GENERATIONS}, Model {model_index + 1}/{POPULATION_SIZE} 100% Complete | Total Fitness: {model.fitness}")

    return model




def train_model():
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    plt.ion()  # Turn on interactive mode

    generation_avg_fitness = []
    all_fitness_history = []

    def update_plots(frame):
        ax1.clear()
        ax2.clear()

        # Plot individual model fitness as a bar chart
        if all_fitness_history:
            latest_fitness = [fitness_history[-1] for fitness_history in all_fitness_history]
            model_labels = [f'Model {i+1}' for i in range(len(latest_fitness))]
            ax1.bar(model_labels, latest_fitness)
        ax1.set_title('Individual Model Fitness')
        ax1.set_xlabel('Models')
        ax1.set_ylabel('Fitness')
        ax1.tick_params(axis='x', rotation=45)

        # Plot average fitness per generation
        ax2.plot(generation_avg_fitness, 'r-')
        ax2.set_title('Average Fitness per Generation')
        ax2.set_xlabel('Generation')
        ax2.set_ylabel('Average Fitness')

        plt.tight_layout()
        plt.draw()
        plt.pause(0.1)

    ani = FuncAnimation(fig, update_plots, interval=1000)

    for generation in range(NUM_GENERATIONS):
        epsilon = EPSILON_START * (EPSILON_DECAY ** generation)
        
        print(f"\nGeneration {generation + 1}/{NUM_GENERATIONS}")
        # Evaluate each model in the population using parallel processing
        use_multiprocessing = True  # Toggle this to switch between multi-threading and single-threading

        if use_multiprocessing:
            with ProcessPoolExecutor() as executor:
                futures = [executor.submit(evaluate_model, model, generation, i, epsilon) for i, model in enumerate(population.models)]
                population.models = [future.result() for future in as_completed(futures)]
        else:
            population.models = [evaluate_model(model, generation, i, epsilon) for i, model in enumerate(population.models)]

        # Sort models by fitness
        population.models.sort(key=lambda x: x.fitness, reverse=True)
        
        # Update fitness history
        if len(all_fitness_history) < len(population.models):
            all_fitness_history = [[model.fitness] for model in population.models]
        else:
            for i, model in enumerate(population.models):
                all_fitness_history[i].append(model.fitness)

        # Calculate and store average fitness
        avg_fitness = sum(model.fitness for model in population.models) / POPULATION_SIZE
        generation_avg_fitness.append(avg_fitness)

        # Print progress
        print(f"\nGeneration {generation + 1}/{NUM_GENERATIONS} Summary:")
        print(f"Best fitness: {population.models[0].fitness}")
        print(f"Average fitness: {avg_fitness}")
        print(f"Worst fitness: {population.models[-1].fitness}")
        print(f"Epsilon: {epsilon:.4f}")
        
        # Evolve population
        population.evolve()

        elapsed_time = time.time() - start_time
        print(f"\nCheckpoint at Generation {generation + 1}")
        print(f"Time elapsed: {elapsed_time:.2f} seconds")
        print(f"Average time per generation: {elapsed_time / (generation + 1):.2f} seconds")
        # Optionally save a checkpoint
        best_model = population.models[0]
        best_model.model.save(f'perudo_model_gen_{generation + 1}.h5')
        print(f"Checkpoint saved: perudo_model_gen_{generation + 1}.h5")

        print("\n" + "="*50 + "\n")  # Separator between generations

        # Update plots
        update_plots(generation + 1)

    # Save the best model
    best_model = population.models[0]
    best_model.model.save('best_perudo_model.h5')

    total_time = time.time() - start_time
    print(f"\nTraining Complete!")
    print(f"Total training time: {total_time:.2f} seconds")
    print(f"Average time per generation: {total_time / NUM_GENERATIONS:.2f} seconds")
    print(f"Final best fitness: {best_model.fitness}")
    print(f"Best model saved as: best_perudo_model.h5")

    plt.ioff()  # Turn off interactive mode
    plt.show()  # Keep the plot window open

if __name__ == "__main__":
    with tf.device('/GPU:0'):
        train_model()