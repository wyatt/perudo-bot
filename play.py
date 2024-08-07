import math

import numpy as np
from keras.src.saving import load_model

# Current bet count, 1x
# Current bet value, 1x
# Palifico round, 1x
# Total dice, 1x
# Num dice in hand, 1x
# n dice in hand (<=5), 5x
# Is my turn 1x (Need to remember to add Palifico state)
# Previous 8 bets (count, value, type) 16x

# Outputs:
# Dudo!
# Bet count
# Bet value

loaded_model = load_model('best_perudo_model.h5', custom_objects={'mse': 'mse'})

def state_from_input(total_dice: int, previous_bets: list[int], dice: list[int]):
    state = []
    dice_count = len(dice)
    dice = dice + [0] * (5 - len(dice))
    is_starting = input("Are you starting? (y/n): ") == "y"
    is_palifico = input("Is this a palifico round? (y/n): ") == "y"
    if is_starting:
        state += [0,0, 1 if is_palifico else 0]
    else:
        current_bet = input("Enter current bet (count, type): ")
        [count, type] = [int(x) for x in current_bet.split(",")]
        state += [count, type]
        state += [1 if is_palifico else 0]

    state += [total_dice]
    state += [dice_count]
    state += dice
    state += [1]
    state += previous_bets

    return state


def min_bet_count(current_bet, total_dice, ones=False):
    if current_bet[0]:
        if current_bet[1] == 1:
            if ones:
                return current_bet[0]
            else:
                return (current_bet[0] * 2) + 1
        else:
            if ones:
                return math.ceil(current_bet[0] / 2)
            else:
                return current_bet[0]

    min_bet_init = int(max((total_dice // 3.5), 1))

    if ones:
        return min_bet_init // 2
    return min_bet_init
        
def valid_outputs(state):
    [current_count, current_value, palifico, total_dice] = state[:4]
    current_bet = [current_count, current_value, palifico]
    valid_outputs = []
    min_bet = min_bet_count(current_bet, total_dice)
    min_bet_ones = min_bet_count(current_bet, total_dice, True)
    
    if current_bet[0] != 0:
        valid_outputs.append([1, 0, 0])
    
    # 1. Calculate all possible bets (within a reasonable range)
    # Calculating for 1s
    min_bet_temp_ones = min_bet_ones + 1 if current_bet[1] == 1 else min_bet_ones
    for x in range(max(min_bet_temp_ones,1), min(min_bet_temp_ones + 5, total_dice)):
        if current_bet[2] == 1 and current_bet[0] != 0:
            # Palifico round where you are not starting
            if 1 != current_bet[1]:
                continue
        valid_outputs.append([0, x, 1])
    
    # Calculating for 2-6
    for val in range(max(current_bet[1],2), 7):
        if current_bet[2] == 1 and current_bet[0] != 0:
            # Palifico round where you are not starting
            if val != current_bet[1]:
                continue
        min_bet_temp = min_bet + 1 if current_bet[1] == val else min_bet
        for x in range(max(min_bet_temp,1), min(min_bet_temp + 5, total_dice)):
            valid_outputs.append([0, x, val])
    
    return valid_outputs


total_dice = int(input("Enter total starting dice: "))
previous_bets: list[int] = [0] * 16
new_round = True
dice = []

while True:
    print("Dice in hand: ", dice)
    if new_round:
        dice = input("Enter dice separted by a comma: ")
        dice = [int(d) for d in dice.split(",")]
    state = state_from_input(total_dice, previous_bets, dice)
    print(len(state), state)
    q_values = loaded_model.predict(np.array(state)[np.newaxis])[0]
    valid_actions = valid_outputs(state)

    # Extract Q-values for each valid action
    prediction = max(valid_actions, key=lambda action: q_values[0] * action[0] + q_values[1] * action[1] + q_values[2] * action[2])


    print("Valid actions: ", valid_actions)
    print("Q-values: ", q_values)

    print("Output: ", prediction)

    if prediction[0] == 1:
        print("Dudo!")
        break
    else:
        print(f"Bet: (Count: {prediction[1]}, Type: {prediction[2]})")
        previous_bets = [prediction[1], prediction[2]] + previous_bets[:14]
        print("Total dice: ", total_dice)
        print("Previous bets: ", previous_bets)

    dice_change = int(input("Enter dice change: "))
    total_dice += dice_change
    if dice_change != 0:
        new_round = True
    else:
        new_round = False




