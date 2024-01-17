import numpy as np
import pandas as pd
import random
import csv
import copy
import tensorflow as tf
from keras.models import Sequential, load_model
from keras.layers import Dense, Conv2D, Flatten
from keras.optimizers import Adam

def convert(switch, grid):
    temp = []
    for row in grid:
        for num in row:
            if (switch):
                if (num == 2):
                    temp.append(1)
                    temp.append(0)
                elif (num == 1):
                    temp.append(0)
                    temp.append(1)
                else:
                    temp.append(0)
                    temp.append(0)
            else:
                if (num == 2):
                    temp.append(0)
                    temp.append(1)
                elif (num == 1):
                    temp.append(1)
                    temp.append(0)
                else:
                    temp.append(0)
                    temp.append(0)
    return temp
def record(list, col, switch, grid):
    temp = convert(switch, grid)
    for i in range(8):
        if (i == col):
            temp.append(1)
        else:
            temp.append(0)
    list.append(temp)
def record_heuristic(list, heuristic, grid):
    temp = convert(False, grid)
    temp.append(heuristic)
    list.append(temp)

def record_2D_heuristic(list, grid):
    result = []
    for i in range(8):
        row = []
        for j in range(8):
            if (grid[i][j] == 1):
                row.append([-1])
            elif (grid[i][j] == 2):
                row.append([1])
            else:
                row.append([0])
        result.append(row)
    list.append(result)
def record_2D_heuristic_flipped(list, grid):
    result = []
    for i in range(8):
        row = []
        for j in range(8):
            if (grid[i][7-j] == 1):
                row.append([-1])
            elif (grid[i][7-j] == 2):
                row.append([1])
            else:
                row.append([0])
        result.append(row)
    list.append(result)



def predict(model, switch, grid):
    data = convert(switch, grid)
    return model.predict([data])
def available(grid):
    result = []
    for i in range(8):
        for j in range(8):
            if (grid[j][i] == 0):
                result.append(i)
                break
    return result

def write_to_csv(name, final):
    file = open(name, "w")
    writer = csv.writer(file)
    header = []
    for i in range(64):
        header.append(i+1)
        header.append(i+1)
    for i in range(8):
        header.append("col"+str(i))
    writer.writerow(header)
    for row in final:
        writer.writerow(row)
    file.close()


def place_new_grid(col, num, grid):
    new_grid = copy.deepcopy(grid)
    for i in range(8):
        if (new_grid[7-i][col] == 0):
            new_grid[7-i][col] = num
            return new_grid
    return new_grid
def place(col, num, grid):
    for i in range(8):
        if (grid[7-i][col] == 0):
            grid[7-i][col] = num
            return True
    return False

def check_win(grid):
    for j in range(8):
        for i in range(5): # horizontal and vertical
            if (grid[j][i] != 0 and grid[j][i] == grid[j][i+1] and grid[j][i] == grid[j][i+2] and grid[j][i] == grid[j][i+3]):
                return grid[j][i]
            if (grid[i][j] != 0 and grid[i][j] == grid[i+1][j] and grid[i][j] == grid[i+2][j] and grid[i][j] == grid[i+3][j]):
                return grid[i][j]
    for i in range(3, 8): # diagonals
        for j in range(0, i-2):
            if (grid[i-j][j] != 0 and grid[i-j][j] == grid[i-j-1][j+1] and grid[i-j][j] == grid[i-j-2][j+2] and grid[i-j][j] == grid[i-j-3][j+3]):
                return grid[i-j][j]
            if (grid[7-j][7-i+j] != 0 and grid[7-j][7-i+j] == grid[7-j-1][7-i+j+1] and grid[7-j][7-i+j] == grid[7-j-2][7-i+j+2] and grid[7-j][7-i+j] == grid[7-j-3][7-i+j+3]):
                return grid[7-j][7-i+j]
            if (grid[i-j][7-j] != 0 and grid[i-j][7-j] == grid[i-j-1][6-j] and grid[i-j][7-j] == grid[i-j-2][5-j] and grid[i-j][7-j] == grid[i-j-3][4-j]):
                return grid[i-j][7-j]
            if (grid[7-j][i-j] != 0 and grid[7-j][i-j] == grid[7-j-1][i-j-1] and grid[7-j][i-j] == grid[7-j-2][i-j-2] and grid[7-j][i-j] == grid[7-j-3][i-j-3]):
                return grid[7-j][i-j]
    return 0
def disp(grid):
    for row in grid:
        for cell in row:
            print(cell, end=" ")
        print()
    print()
    for i in range(8):
        print(i, end=" ")
    print()

def interpret(array, available, randomness): # interprets binary output of model
    print(array)
    for i in range(len(array)):
        array[i] += randomness
    print(array)
    whole_sum = 0
    for i in available:
        whole_sum += array[i]
    rand = random.random()
    for i in range(len(available)):
        partial_sum = 0
        for j in available[:i+1]:
            partial_sum += array[j]
        if (rand < (partial_sum/whole_sum)):
            print(available[i])
            return available[i]

    return available[-1]

def single_heuristic(player, grid):
    sequences = [[0, player, player, player],
                [player, 0, player, player],
                [player, player, 0, player],
                [player, player, player, 0]]
    value = 0
    for j in range(8):
        for i in range(5): # horizontal and vertical
            block = [[grid[j][i], grid[j][i+1], grid[j][i+2], grid[j][i+3]],
                     [grid[i][j], grid[i+1][j], grid[i+2][j], grid[i+3][j]]]
            for k in block:
                for sequence in sequences:
                    if (k == sequence):
                        value += 1
    for i in range(3, 8): # diagonals
        for j in range(0, i-2):
            block = [[grid[i-j][j], grid[i-j-1][j+1], grid[i-j-2][j+2], grid[i-j-3][j+3]],
                     [grid[7-j][7-i+j], grid[7-j-1][7-i+j+1], grid[7-j-2][7-i+j+2], grid[7-j-3][7-i+j+3]],
                     [grid[i-j][7-j], grid[i-j-1][6-j], grid[i-j-2][5-j], grid[i-j-3][4-j]],
                     [grid[7-j][i-j], grid[7-j-1][i-j-1], grid[7-j-2][i-j-2], grid[7-j-3][i-j-3]]]
            for k in block:
                for sequence in sequences:
                    if (k == sequence):
                        value += 1
    return value
def heuristic(grid):
    return 10*(single_heuristic(2, grid) - single_heuristic(1, grid))


def minimax(grid, depth, currentPlayer, alpha, beta):
    available_cols = available(grid)
    random.shuffle(available_cols)
    winner = check_win(grid) # if win stops the branch and returns -100 or 100
    if (winner == 1):
        return [-100-depth, -1]
    elif (winner == 2):
        return [100+depth, -1]
    if (depth == 0 or len(available_cols) == 0): #if reached end of search or draw uses heuristic for value
        return [heuristic(grid), -1]
    if (currentPlayer == 2):
        value = -200
        index = -1
        for col in available_cols:
            benefit = minimax(place_new_grid(col, 2, grid), depth-1, 1, alpha, beta)
            if (value < benefit[0]):
                value = benefit[0]
                index = col
            if (value > beta):
                break
            if (value > alpha):
                alpha = value
        return [value, index]
    else:
        value = 200
        index = -1
        for col in available_cols:
            benefit = minimax(place_new_grid(col, 1, grid), depth-1, 2, alpha, beta)
            if (value > benefit[0]):
                value = benefit[0]
                index = col
            if (value < alpha):
                break
            if (value < beta):
                beta = value
        return [value, index]




 #play against model

# def play_optimal(depth):
#     while (True):
#         player = 1 # reset everything
#         grid = [[0, 0, 0, 0, 0, 0, 0, 0],
#                 [0, 0, 0, 0, 0, 0, 0, 0],
#                 [0, 0, 0, 0, 0, 0, 0, 0],
#                 [0, 0, 0, 0, 0, 0, 0, 0],
#                 [0, 0, 0, 0, 0, 0, 0, 0],
#                 [0, 0, 0, 0, 0, 0, 0, 0],
#                 [0, 0, 0, 0, 0, 0, 0, 0],
#                 [0, 0, 0, 0, 0, 0, 0, 0]]
#         disp(grid)
#         while (True):
#             if (player == 1):
#                 num = int(input("Column: "))
#                 place(num, 1, grid)
#             else:
#                 optimal = minimax(grid, depth, 2, -100, 100)
#                 place(optimal[1], 2, grid)
#             disp(grid)
#             if (check_win(grid)):
#                 print(f"Player {player} WIN") 
#                 break
#             #switch players
#             if (player == 1):
#                 player = 2
#             else:
#                 player = 1
#         keep_playing = input("Keep Playing? (Y/N) ")
#         if (keep_playing == "N"):
#             break
# play_optimal(3)


model = Sequential()
model.add(Conv2D(128, 4, input_shape=(8, 8, 1), activation='relu'))
model.add(Flatten())
model.add(Dense(units=64, activation='relu'))
model.add(Dense(units=64, activation='relu'))
model.add(Dense(units=1, activation='linear'))
model.compile(loss="mse", optimizer=Adam())
print(model.get_config())
model.summary()

from sklearn.metrics import mean_squared_error
mses = []
def optimal_data(batches, batch_size, depth):
    batch = 0
    while (batch < batches):
        print(f"Batch {batch + 1}")
        final=[]
        heuristics=[]
        successes = 0
        while (successes < batch_size):
            print(f"Game {successes + 1}")
            mod10 = 2*random.randint(0, 4)
            player = 1;
            grid = [[0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0]]
            move = 1
            while(True):
                options = available(grid)
                if (len(options) == 0):
                    break
                optimal = minimax(grid, 2, player, -200, 200)
                place(optimal[1], player, grid)
                if (check_win(grid)):
                    break
                if (player == 1): player = 2
                else: player = 1
                if (move % 10 == mod10):
                    h = minimax(grid, depth, player, -200, 200)[0]
                    record_2D_heuristic(final, grid)
                    record_2D_heuristic_flipped(final, grid)
                    heuristics.append([h])
                    heuristics.append([h])
                move += 1
            successes+=1
        mses.append(mean_squared_error(heuristics, model.predict(final)))
        model.fit(final, heuristics, epochs=20, batch_size=2)
        batch += 1
optimal_data(200, 30, 3)


import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
linear_reg = LinearRegression()
x_vals = np.array(range(len(mses))).reshape((-1, 1))
y_vals = np.array(mses)
linear_reg.fit(x_vals, y_vals)


file = open("mses.csv", "w")
writer = csv.writer(file)
writer.writerow(mses)
file.close()

model.save("player_2_optimized_16conv")

plt.plot(mses)
plt.plot(x_vals, linear_reg.predict(x_vals))
plt.show()
    
        
    

    






# model v model (continuous learning with added randomness)
# def random_learn(batch_size, num_batches, max_length, modelname):
#     model = load_model(modelname)
#     batches = 0
#     while (batches < num_batches):
#         successes = 0
#         final = []
#         while (successes < batch_size):
#             print(f"GAME {successes}")
#             player = 1 # reset everything
#             grid = [[0, 0, 0, 0, 0, 0, 0, 0],
#                     [0, 0, 0, 0, 0, 0, 0, 0],
#                     [0, 0, 0, 0, 0, 0, 0, 0],
#                     [0, 0, 0, 0, 0, 0, 0, 0],
#                     [0, 0, 0, 0, 0, 0, 0, 0],
#                     [0, 0, 0, 0, 0, 0, 0, 0],
#                     [0, 0, 0, 0, 0, 0, 0, 0],
#                     [0, 0, 0, 0, 0, 0, 0, 0]]
#             temp_1 = []
#             temp_2 = []
#             while (True):
#                 options = available(grid) # available columns
#                 if (len(options) == 0): # if board is compeltely filled call draw
#                     print("DRAW")
#                     break
#                 else:
#                     if (player == 1):
#                         choice = interpret(predict(model, True, grid)[0], options, 0) # model chooses
#                     else:
#                         choice = interpret(predict(model, False, grid)[0], options, 0)
#                     place(choice, player, grid)
#                     if (player == 1): # data should always be on player 2 side so switch if player 1
#                         record_2D(temp_1, choice, True, grid)
#                     else:
#                         record_2D(temp_2, choice, False, grid) # data is stored seperately for each player

#                 if (check_win(grid)):
#                     print(f"Player {player} WIN") 
#                     print(len(temp_1))
#                     print(len(temp_2))
#                     if (player == 1 and len(temp_1) <= max_length): # only winning data is collected
#                         successes += 1
#                         for boardstate in temp_1:
#                             final.append(boardstate)
#                     elif (player == 2 and len(temp_2) <= max_length):   
#                         successes += 1
#                         for boardstate in temp_2:
#                             final.append(boardstate)
#                     break
#                 #switch players
#                 if (player == 1):
#                     player = 2
#                 else:
#                     player = 1
#         nparray = np.asarray(final)
#         X = nparray[:, :1]
#         y = nparray[:, 1:]
#         model1.fit(X, y, epochs = 10, batch_size = max_length)
#         model2.fit(X, y, epochs = 10, batch_size = max_length)
#         batches += 1
#     model1.save(model1name)
#     model2.save(model2name)
# random_learn(1, 1000, 100, "optimized")

# def play():
#     while (True):
#         player = 1 # reset everything
#         grid = [[0, 0, 0, 0, 0, 0, 0, 0],
#                 [0, 0, 0, 0, 0, 0, 0, 0],
#                 [0, 0, 0, 0, 0, 0, 0, 0],
#                 [0, 0, 0, 0, 0, 0, 0, 0],
#                 [0, 0, 0, 0, 0, 0, 0, 0],
#                 [0, 0, 0, 0, 0, 0, 0, 0],
#                 [0, 0, 0, 0, 0, 0, 0, 0],
#                 [0, 0, 0, 0, 0, 0, 0, 0]]
#         disp(grid)
#         while (True):
#             num = int(input("Column: "))
#             place(num, player, grid)
#             disp(grid)
#             if (check_win(grid)):
#                 print(f"Player {player} WIN") 
#                 break
#             #switch players
#             if (player == 1):
#                 player = 2
#             else:
#                 player = 1
#         keep_playing = input("Keep Playing? (Y/N) ")
#         if (keep_playing == "N"):
#             break
# play()

# play against model
# def play_model():
#     model = load_model("optimized")
#     successes = 0
#     while (True):
#         player = 1 # reset everything
#         grid = [[0, 0, 0, 0, 0, 0, 0, 0],
#                 [0, 0, 0, 0, 0, 0, 0, 0],
#                 [0, 0, 0, 0, 0, 0, 0, 0],
#                 [0, 0, 0, 0, 0, 0, 0, 0],
#                 [0, 0, 0, 0, 0, 0, 0, 0],
#                 [0, 0, 0, 0, 0, 0, 0, 0],
#                 [0, 0, 0, 0, 0, 0, 0, 0],
#                 [0, 0, 0, 0, 0, 0, 0, 0]]
#         disp(grid)
#         while (True):
#             if (player == 1):
#                 num = int(input("Column: "))
#                 place(num, 1, grid)
#             else:
#                 options = available(grid) # available columns
#                 if (len(options) == 0): # if board is compeltely filled call draw
#                     print("DRAW")
#                     break
#                 else:
#                     choice = interpret(predict(model, False, grid)[0], options, 0)
#                     place(choice, 2, grid)
#             disp(grid)
#             if (check_win(grid)):
#                 print(f"Player {player} WIN") 
#                 break
#             #switch players
#             if (player == 1):
#                 player = 2
#             else:
#                 player = 1
#         keep_playing = input("Keep Playing? (Y/N) ")
#         if (keep_playing == "N"):
#             break
# play_model()

{
# collect random data
# def collect_data(trials, max_length):
#     successes = 0
#     while (successes < trials):
#         player = 1 # reset everything
#         grid = [[0, 0, 0, 0, 0, 0, 0, 0],
#                 [0, 0, 0, 0, 0, 0, 0, 0],
#                 [0, 0, 0, 0, 0, 0, 0, 0],
#                 [0, 0, 0, 0, 0, 0, 0, 0],
#                 [0, 0, 0, 0, 0, 0, 0, 0],
#                 [0, 0, 0, 0, 0, 0, 0, 0],
#                 [0, 0, 0, 0, 0, 0, 0, 0],
#                 [0, 0, 0, 0, 0, 0, 0, 0]]
#         temp_1 = []
#         temp_2 = []
#         while (True):
#             # num = int(input("Column: "))
#             options = available(grid) # available columns
#             if (len(options) == 0): # if board is compeltely filled call draw
#                 print("DRAW")
#                 break
#             else:
#                 choice = random.choice(options) # random choice between available
#                 place(choice, player, grid)
#                 if (player == 1): # data should always be on player 2 side so switch if player 1
#                     record(temp_1, choice, True, grid)
#                 else:
#                     record(temp_2, choice, False, grid) # data is stored seperately for each player

#             if (check_win(grid)):
#                 print(f"Player {player} WIN") 
#                 if (player == 1 and len(temp_1) <= max_length): # only winning data is collected
#                     successes += 1
#                     for boardstate in temp_1:
#                         final.append(boardstate)
#                 elif (player == 2 and len(temp_2) <= max_length):
#                     successes += 1
#                     for boardstate in temp_2:
#                         final.append(boardstate)
#                 break
#             #switch players
#             if (player == 1):
#                 player = 2
#             else:
#                 player = 1

# collect_data(50, 100)

# file = open("v1training.csv", "w")
# writer = csv.writer(file)
# header = []
# for i in range(64):
#     header.append(i+1)
#     header.append(i+1)
# for i in range(8):
#     header.append("col"+str(i))
# writer.writerow(header)
# for row in final:
#     writer.writerow(row)
# file.close()
    


# model v model 
# def collect_data(trials, max_length):
#     model = load_model("v1")
#     successes = 0
#     while (successes < trials):
#         print(f"GAME {successes}")
#         player = 1 # reset everything
#         grid = [[0, 0, 0, 0, 0, 0, 0, 0],
#                 [0, 0, 0, 0, 0, 0, 0, 0],
#                 [0, 0, 0, 0, 0, 0, 0, 0],
#                 [0, 0, 0, 0, 0, 0, 0, 0],
#                 [0, 0, 0, 0, 0, 0, 0, 0],
#                 [0, 0, 0, 0, 0, 0, 0, 0],
#                 [0, 0, 0, 0, 0, 0, 0, 0],
#                 [0, 0, 0, 0, 0, 0, 0, 0]]
#         temp_1 = []
#         temp_2 = []
#         while (True):
#             # num = int(input("Column: "))
#             options = available(grid) # available columns
#             if (len(options) == 0): # if board is compeltely filled call draw
#                 print("DRAW")
#                 break
#             else:
#                 if (player == 1):
#                     choice = interpret(predict(model, True, grid)[0], options, 0) # model chooses
#                 else:
#                     choice = interpret(predict(model, False, grid)[0], options, 0)
#                 place(choice, player, grid)
#                 if (player == 1): # data should always be on player 2 side so switch if player 1
#                     record(temp_1, choice, True, grid)
#                 else:
#                     record(temp_2, choice, False, grid) # data is stored seperately for each player

#             if (check_win(grid)):
#                 print(f"Player {player} WIN") 
#                 print(len(temp_1))
#                 print(len(temp_2))
#                 if (player == 1 and len(temp_1) <= max_length): # only winning data is collected
#                     successes += 1
#                     for boardstate in temp_1:
#                         final.append(boardstate)
#                 elif (player == 2 and len(temp_2) <= max_length):   
#                     successes += 1
#                     for boardstate in temp_2:
#                         final.append(boardstate)
#                 break
#             #switch players
#             if (player == 1):
#                 player = 2
#             else:
#                 player = 1

# collect_data(20, 8)

# file = open("v1training.csv", "w")
# writer = csv.writer(file)
# header = []
# for i in range(64):
#     header.append(i+1)
#     header.append(i+1)
# for i in range(8):
#     header.append("col"+str(i))
# writer.writerow(header)
# for row in final:
#     writer.writerow(row)
# file.close()



# random v model (continuous learning)
# def random_learn(batch_size, num_batches, max_length):
#     model = load_model("v2")
#     batches = 0
#     while (batches < num_batches):
#         successes = 0
#         final = []
#         while (successes < batch_size):
#             player = 1 # reset everything
#             grid = [[0, 0, 0, 0, 0, 0, 0, 0],
#                     [0, 0, 0, 0, 0, 0, 0, 0],
#                     [0, 0, 0, 0, 0, 0, 0, 0],
#                     [0, 0, 0, 0, 0, 0, 0, 0],
#                     [0, 0, 0, 0, 0, 0, 0, 0],
#                     [0, 0, 0, 0, 0, 0, 0, 0],
#                     [0, 0, 0, 0, 0, 0, 0, 0],
#                     [0, 0, 0, 0, 0, 0, 0, 0]]
#             temp_1 = []
#             while (True):
#                 options = available(grid) # available columns
#                 if (len(options) == 0): # if board is compeltely filled call draw
#                     print("DRAW")
#                     break
#                 else:
#                     if (player == 1):
#                         choice = random.choice(options) # random choice between available
#                     else:
#                         choice = interpret(predict(model, False, grid)[0], options, 0)
#                     place(choice, player, grid)
#                     if (player == 1): # only records random side
#                         record(temp_1, choice, True, grid)

#                 if (check_win(grid)):
#                     print(f"Player {player} WIN") 
#                     if (player == 1 and len(temp_1) <= max_length): # only winning data is collected
#                         successes += 1
#                         for boardstate in temp_1:
#                             final.append(boardstate)                
#                     break
#                 #switch players
#                 if (player == 1):
#                     player = 2
#                 else:
#                     player = 1
#         nparray = np.asarray(final)
#         X = nparray[:, :128]
#         y = nparray[:, 128:]
#         model.fit(X, y, epochs = 10, batch_size = max_length)
#         batches += 1 
#     model.save("v2")

# random_learn(1, 100, 6)



}
