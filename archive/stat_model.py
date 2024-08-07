import heapq
import json
import random
from numpy import random as np_random
from itertools import count
from typing import Dict, Union
from scipy.stats import binom

from test import binomial_probability_gte

# Simulation of a game of Perudo, calculating the optimal bet for each player

STARTING_DICE = 5
STARTING_PLAYERS = 4




def calc_all_bets(my_dice: list[int], total_dice_count: int, min_bet: int, palifico: bool):
    bets = {
        1: {},
        2: {},
        3: {},
        4: {},
        5: {},
        6: {}
    }

    for bet in range(min_bet // 2, (min_bet // 2) + 5):
        num_dice_to_find = max(bet - my_dice.count(1), 0)
        bets[1][bet] = binomial_probability_gte(total_dice_count - len(my_dice), 1/6, num_dice_to_find)


    for x in range(2, 7):
        for bet in range(min_bet, min_bet + 5):
            num_dice_to_find = max(bet - my_dice.count(x), 0)
            bets[x][bet] = binomial_probability_gte(total_dice_count - len(my_dice), (1/6 if palifico else 1/3), num_dice_to_find)

    return bets


def min_bet(total_dice: int, current_bet: Union[int, None]) -> int:
    if current_bet:
        return current_bet + 1

    return max((total_dice // 3), 1)

def main():
    total_dice = STARTING_DICE * STARTING_PLAYERS
    has_been_palifico = False

    print("Welcome to Perudo!")
    while True:
        print("Dice count: ",total_dice)
        dice = input("Enter dice separted by a comma: ")
        dice = [int(d) for d in dice.split(",")]

        starting = input("Are you starting? (y/n): ") == "y"
        if starting:
            palifico = False
            if len(dice) == 1 and not has_been_palifico:
                print("A palifico has been called!")
                palifico = True
                has_been_palifico = True
            bets = calc_all_bets(dice, total_dice, min_bet(total_dice, None), palifico)

            # List to store probabilities with their locations
            probabilities = []

            # Iterate through the dictionary to extract probabilities and their locations
            for outer_key, inner_dict in bets.items():
                for inner_key, probability in inner_dict.items():
                    probabilities.append((probability, (outer_key, inner_key)))

            # Get the top 3 probabilities
            top_3 = heapq.nlargest(3, probabilities, key=lambda x: x[0])

            for prob, (outer_key, inner_key) in top_3:
                print(f"Bet: (Number: {inner_key}, Type: {outer_key}), Probability: {prob}")

        else:
            current_bet = input("Enter current bet (count, type): ")
            [count, type] = [int(x) for x in current_bet.split(",")]
            is_palifico_round = input("Is this a palifico round? (y/n): ") == "y"

            bets = calc_all_bets(dice, total_dice, min_bet(total_dice, count * (2 if type == 1 else 1)), is_palifico_round)
            for k in [1,2,3,4,5,6]:
                if is_palifico_round and k != type:
                    del bets[k]
                elif k != 1 and int(k) < type:
                    del bets[k]

            prob_value = 1/6 if is_palifico_round else (1/3 if type != 1 else 1/6)
            prob_of_current_bet = binomial_probability_gte(total_dice - len(dice), prob_value, count - dice.count(type))
            print("Probability of prexisting bet is: ", prob_of_current_bet)

            # List to store probabilities with their locations
            probabilities = []

            # Iterate through the dictionary to extract probabilities and their locations
            for outer_key, inner_dict in bets.items():
                for inner_key, probability in inner_dict.items():
                    probabilities.append((probability, (outer_key, inner_key)))

            # Get the top 3 probabilities
            top_3 = heapq.nlargest(3, probabilities, key=lambda x: x[0])

            for prob, (outer_key, inner_key) in top_3:
                print(f"Bet: (Number: {inner_key}, Type: {outer_key}), Probability: {prob}")

            print("Best option", top_3[0][0])

            if (1 - prob_of_current_bet) > top_3[0][0]:
                print("Dudo!")

        # Track dice change
        dice_change = int(input("Enter dice change: "))
        total_dice += dice_change


if __name__ == "__main__":
    main()