import csv
import glob
import time
import numpy as np

MAX_INT = float('inf')

def find_min(adj, i):
    first, second = MAX_INT, MAX_INT
    for j in range(len(adj)):
        if i == j:
            continue
        if adj[i][j] <= first:
            second = first
            first = adj[i][j]

        elif adj[i][j] <= second and adj[i][j] != first:
            second = adj[i][j]

    return first, second

def get_lb(adj, neigh, pre_bound, visited):
    global_bound = 0
    pre_node = visited[-1]
    pre_first, pre_second = find_min(adj, pre_node)
    neigh_first, neigh_second = find_min(adj, neigh)
    if pre_node == 0:
        for i in range(len(adj)):
            first, second = find_min(adj, i)
            global_bound = global_bound + first + second
        lb = global_bound / 2.0 - (pre_first + neigh_first) / 2.0 + adj[pre_node][neigh]
    else:
        lb = pre_bound - (pre_second + neigh_first) / 2.0 + adj[pre_node][neigh]
    return lb

def get_neighs(adj, node, visited):
    neighs = []
    for i in range(len(adj)):
        if i != node and i not in visited:
            neighs.append(i)
    return neighs

def backtraking(adj, node, pre_bound, visited):
    n = len(adj)

    # end condition
    if len(visited) == n:
        return 0, []

    #recursion
    neighs = get_neighs(adj, node, visited)
    neigh_to_lb = [0] * n
    min_lb = MAX_INT
    for neigh in neighs:
        lb = get_lb(adj, neigh, pre_bound, visited)
        # print(lb)
        neigh_to_lb[neigh] = lb
        min_lb = min(min_lb, lb)

    min_neighs = []
    for i in range(len(neigh_to_lb)):
        if neigh_to_lb[i] == min_lb:
            min_neighs.append(i)

    best_cost = MAX_INT
    best_path = []
    for min_neigh in min_neighs:
        visited.append(min_neigh)
        n_cost, n_path = backtraking(adj, min_neigh, neigh_to_lb[min_neigh], visited)
        cost = n_cost + adj[node][min_neigh]
        if cost < best_cost:
            best_cost = cost
            best_path = n_path[:] + [min_neigh]
        visited.pop()

    return best_cost, best_path

def bnb(adj, csvwriter):
    T1 = time.time()
    best_cost, best_path = backtraking(adj, 0, 0, [0])
    cost = adj[best_path[0]][0] + best_cost
    path = [0] + best_path + [0]
    T2 = time.time()
    result = [cost, T2 - T1]
    csvwriter.writerow(result)
    print(" time: ", (T2 - T1) * 1000, "ms")
    print("best result: %1.4f" % cost)
    print(path)


if __name__ == "__main__":

    file_path = str(input("Enter the file path to be read: "))
    output_file = "bnb_out.csv"
    output = open(output_file, mode="w+")
    csvwriter = csv.writer(output)
    ids = [74445067, 0, 0]
    types = ["BnB"]
    csvwriter.writerow(ids)
    csvwriter.writerow(types)
    total_time_start = time.time()
    # competion/tsp-*.txt
    file_list = glob.glob(file_path)
    for filename in file_list:
        input_matrix = np.loadtxt(fname=filename, delimiter=" ", skiprows=1)
        print(input_matrix.tolist())
        bnb(input_matrix.tolist(), csvwriter)
    total_time_end = time.time()
    print(" time: ", (total_time_end - total_time_start) * 1000, "ms")

