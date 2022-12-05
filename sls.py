import csv
import glob
import random
import time

import generate_travelling_salesman_problem as tsp_input
import numpy as np


def stochasticLocalSearch(input_matrix, csvwriter):
    # input_matrix = tsp_input.write_distance_matrix(n, mean, sigma)
    # input_matrix = \
    #     [[0, 10, 15, 20],
    #      [10, 0, 35, 25],
    #      [15, 35, 0, 30],
    #      [20, 25, 30, 0]]

    # n = 5
    n = np.shape(input_matrix)[0]
    T1 = time.time()
    best_path = []
    best_result = float("inf")
    start_visited = np.zeros(n)

    # Greedy search first, find a local optimal path, start from 0
    random_times = n / 3
    if random_times > 100:
        random_times = 100
    global_improvement = True

    while random_times > 0 and global_improvement:
        start_node = random.randrange(n)
        if start_visited[start_node] == 1:
            continue
        global_improvement = False
        start_visited[start_node] = 1
        random_times -= 1

        visited = np.zeros(n)
        cur = start_node
        cur_result = 0.0
        visited[start_node] = 1
        greedy_best_path = [start_node]
        greedy_best_result = float("inf")
        while len(greedy_best_path) < n:
            minimum_edge = float("inf")
            next_node = 0
            for index in range(len(input_matrix[cur])):
                if visited[index] == 1:
                    continue
                num = input_matrix[cur][index]
                if num < minimum_edge:
                    minimum_edge = num
                    next_node = index
            greedy_best_path.append(next_node)
            visited[next_node] = 1
            cur_result += minimum_edge
            cur = next_node

        # initial path
        greedy_best_result = min(cur_result + input_matrix[greedy_best_path[-1]][start_node], greedy_best_result)
        greedy_best_path.append(start_node)

        # 2-opt
        improvement = True
        while improvement:
            improvement = False
            # print(id(greedy_best_path), id(improvement), id(greedy_best_result))
            greedy_best_result, improvement = try_2_opt(greedy_best_path, greedy_best_result, improvement)

        # compare local to global
        if greedy_best_result < best_result:
            best_result = greedy_best_result
            best_path = greedy_best_path[:]
            global_improvement = True

    T2 = time.time()
    result = [best_result, T2 - T1]
    csvwriter.writerow(result)
    print(" time: ", (T2 - T1) * 1000, "ms")
    # print(input_matrix)
    print("best result: %1.4f" % best_result)
    print(best_path)


def try_2_opt(greedy_best_path, greedy_best_result, improvement):
    for i in range(1, len(greedy_best_path) - 2):
        for j in range(i + 1, len(greedy_best_path)):
            if j - i <= 1:
                continue
            # old cost function, calculate all path's cost
            # new_path = best_path[:]
            # new_path[i:j] = best_path[j - 1:i - 1:-1]
            # new_cost = cost(input_matrix, new_path)
            # if new_cost < best_result:
            #     best_path = new_path
            #     best_result = new_cost
            #     improvement = True

            # new cost function, only calculate difference
            cost_change = improved_cost_change(input_matrix, greedy_best_path[i - 1], greedy_best_path[i],
                                               greedy_best_path[j - 1], greedy_best_path[j])
            if cost_change < 0:
                greedy_best_path[i:j] = greedy_best_path[j - 1:i - 1:-1]
                improvement = True
                greedy_best_result += cost_change
                # print(id(greedy_best_path), id(improvement), id(greedy_best_result))
                return greedy_best_result, improvement
            return greedy_best_result, improvement


# old cost, O(n)
def cost(matrix, route):
    res = 0.0
    for i in range(len(route) - 1):
        res += matrix[route[i]][route[i + 1]]
    return res


# new cost, O(1)
def improved_cost_change(matrix, n1, n2, n3, n4):
    return matrix[n1][n3] + matrix[n2][n4] - matrix[n1][n2] - matrix[n3][n4]


if __name__ == "__main__":
    # n = int(input("Enter the number of locations: "))
    # mean = float(input("Enter the mean: "))
    # sigma = float(input("Enter the standard deviation: "))

    file_path = str(input("Enter the file path to be read: "))
    output_file = "sls_out.csv"
    output = open(output_file, mode="w+")
    csvwriter = csv.writer(output)
    ids = [74445067, 0, 0]
    types = ["SLS"]
    csvwriter.writerow(ids)
    csvwriter.writerow(types)
    total_time_start = time.time()
    # competion/tsp-*.txt
    file_list = glob.glob(file_path)
    for filename in file_list:
        input_matrix = np.loadtxt(fname=filename, delimiter=" ", skiprows=1)
        stochasticLocalSearch(input_matrix, csvwriter)
    total_time_end = time.time()
    print(" time: ", (total_time_end - total_time_start) * 1000, "ms")
