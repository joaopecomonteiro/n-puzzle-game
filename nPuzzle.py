import numpy as np
import copy 
from timeit import default_timer as timer
import tracemalloc

n = 4

class node():
    def __init__(self, pieces_list = [], matrix = [], parent = None, f=0, g=0, h=0, mis=0):
        self.pieces_list = pieces_list
        self.matrix = matrix
        self.f = f
        self.g = g
        self.h = h
        self.mis = mis
        self.parent = parent

    def create_matrix(self, list):
        self.matrix = np.zeros((n,n))
        index = 0
        for i in range(4):
            for j in range(4):
                self.matrix[i][j] = list[index]
                index += 1
        # print(self.matrix)


    def isGoal(self):
        # print(self.matrix)
        return np.array_equal(self.matrix, goal.matrix)
    

    def findBlank(self):
        for i in range(n):
                for j in range(n):
                    if self.matrix[i][j] == 0:
                        return i, j


    def genChildren(self):
            x, y = self.findBlank()
            newMatrices = []
            if (x + 1) < n: #moving blank down / moving a tile up
                new = copy.deepcopy(self.matrix)
                new[x][y] = new[x+1][y]
                new[x+1][y] = 0
                newMatrices.append(new)
            if (y + 1) < n:  # moving blank right / moving a tile left
                new = copy.deepcopy(self.matrix)
                new[x][y]=new[x][y+1]
                new[x][y+1]= 0
                newMatrices.append(new)
            if (x - 1) > -1: #moving blank up / moving a tile down
                new = copy.deepcopy(self.matrix)
                new[x][y] = new[x-1][y]
                new[x-1][y] = 0
                newMatrices.append(new)
            if (y - 1) > -1: # moving blank left / moving a tile right
                new = copy.deepcopy(self.matrix)
                new[x][y] = new[x][y - 1]
                new[x][y-1] = 0
                newMatrices.append(new)
            ret = []
            for matrix in newMatrices: #create children nodes
                child = node()
                child.matrix = matrix
                child.parent = self
                # print(child.g)
                child.manhattan()
                child.misplaced()
                # print(child.g)
                ret.append(child)
            return ret


    def manhattan(self):
        # print("dakjdwadwkj")
        sum = 0
        # print(self.matrix)
        if np.array_equal(self.matrix, start.matrix):
            for i in range(n):
                for j in range(n):
                    if self.matrix[i][j] == 0:
                        continue
                    else:
                        x, y = findGoal(self.matrix[i][j], goal)
                        sum += abs(x - i) + abs(y - j)
            self.h = sum
            self.g = 0
            self.f = sum
        else:
            for i in range(n):
                for j in range(n):
                    if self.matrix[i][j] == 0:
                        continue
                    else:
                        x, y = findGoal(self.matrix[i][j], goal)
                        sum += abs(x - i) + abs(y - j)
            self.h = sum
            self.g = self.parent.g + 1
            self.f = self.g + sum


    def misplaced(self):
        total = 0
        for i in range(n):
            for j in range(n):
                if (self.matrix[i][j] != 0) and (self.matrix[i][j] != goal.matrix[i][j]):
                    total += 1
        self.mis = total


    def visited(self, visited_list):
        for node in visited_list:
            if np.array_equal(self.matrix, node.matrix):
                return True
        return False



    def print_matrix(self):
        for i in range(n):
            b = ""
            for j in range(n):
                b += str(int(self.matrix[i][j])) + '\t'
            print(b)
        print("---------------------------")





def solvability_to_std(position_list):
    inversions = 0
    for i in range(len(position_list)):
        inversions_temp = 0
        for j in range(i+1, len(position_list)):
            if position_list[i] > position_list[j] and position_list[j]!=0:
                inversions_temp += 1
        inversions += inversions_temp
    for i in range(len(position_list)):
        if position_list[i] == 0:
            if (i > -1 and i < 4) or (i > 8 and i < 13):
                if inversions % 2 != 0:
                    return True
            else:
                if inversions % 2 == 0: 
                    return True
    return False


def solvability(start_list, goal_list):
    start_std = solvability_to_std(start_list)
    goal_std = solvability_to_std(goal_list)
    return start_std == goal_std


def astar_manhattan():
    open_list = []
    closed_list = []
    open_list.append(start)
    start_time_astar_manhattan = timer()
    while(len(open_list) > 0):
        current_node = open_list[0]
        current_index = 0
        for index, item in enumerate(open_list):
            if item.f < current_node.f:
                current_node = item
                current_index = index
        open_list.pop(current_index)
        closed_list.append(current_node)
        if(current_node.isGoal()):
            path = []
            current = current_node
            while(current is not None):
                path.append(current)
                current = current.parent
            return path[::-1]
        children = current_node.genChildren()
        for child in children:
            for closed_child in closed_list:
                if(np.array_equal(child.matrix, closed_child.matrix)):
                    continue
            # print(child)
            child.manhattan()
            for open_child in open_list:
                if(np.array_equal(open_child.matrix, child.matrix) and child.g > open_child.g):
                    continue
            open_list.append(child)
        end_time_astar_manhattan = timer()
        if(end_time_astar_manhattan-start_time_astar_manhattan > 60):
            print("tempo limite excedido")
            return None



def astar_misplaced():
    open_list = []
    closed_list = []
    open_list.append(start)
    start_time_astar_misplaced = timer()
    while(len(open_list) > 0):
        current_node = open_list[0]
        current_index = 0
        for index, item in enumerate(open_list):
            if item.f < current_node.f:
                current_node = item
                current_index = index
        open_list.pop(current_index)
        closed_list.append(current_node)
        if(current_node.isGoal()):
            path = []
            current = current_node
            while(current is not None):
                path.append(current)
                current = current.parent
            return path[::-1]
        children = current_node.genChildren()
        for child in children:
            for closed_child in closed_list:
                if(np.array_equal(child.matrix, closed_child.matrix)):
                    continue
            for open_child in open_list:
                if(np.array_equal(open_child.matrix, child.matrix) and child.mis > open_child.mis):
                    continue
            open_list.append(child)
        end_time_astar_misplaced = timer()
        if(end_time_astar_misplaced-start_time_astar_misplaced > 60):
            print("tempo limite excedido")
            return None

        


def BFS():
    queue = []
    queue.append(start)
    queue.append(0)
    visited_nodes = []
    start_time_BFS = timer()
    while(len(queue) != 0):
        node = queue.pop(0)
        depth_node = queue.pop(0)
        visited_nodes.append(node)
        if(node.isGoal()):
            path = []
            while(node != None):
                path.insert(0, node)
                node = node.parent
            return path
        children = node.genChildren()
        for child in children:
            if(child not in visited_nodes):
                queue.append(child)
                queue.append(depth_node+1)
                visited_nodes.append(child)
        end_time_BFS = timer()
        if(end_time_BFS-start_time_BFS > 60):
            print("tempo limite excedido")
            return None


def IDFS():
    depth = 1
    bottom_reached = False 
    start_time_IDFS = timer()
    while not bottom_reached:
        path, bottom_reached = IDFSRec(start, 0, depth)
        if path is not None:
            return path
        depth += 1
        end_time_IDFS = timer()
        if(end_time_IDFS-start_time_IDFS > 60):
            print("tempo limite excedido")
            return None
    return None


def IDFSRec(node, current_depth, max_depth):
    if node.isGoal():
        print("Solucão encontrada")
        print("Profundidade da solução:", current_depth)
        path = []
        while(node != None):
            path.insert(0, node)
            node = node.parent
        return path, True
    children = node.genChildren()
    if current_depth == max_depth:
        return None, False
    bottom_reached = True
    for child in children:
        result, bottom_reached_rec = IDFSRec(child, current_depth + 1, max_depth)
        if result is not None:
            return result, True
        bottom_reached = bottom_reached and bottom_reached_rec
    return None, bottom_reached


def DFS():
    stack = []
    stack.insert(0, 0)
    stack.insert(0, start)
    visited_nodes = []
    start_time_DFS = timer()
    while(len(stack) != 0):
        node = stack.pop(0)
        depth_node = stack.pop(0)
        visited_nodes.append(node)
        if(node.isGoal()):
            path = []
            while(node != None):
                path.insert(0, node.matrix)
                node = node.parent
            return path
        children = node.genChildren()
        for child in children:
            if(child not in visited_nodes):
                stack.insert(0, depth_node+1)
                stack.insert(0, child)
                visited_nodes.append(child)
        end_time_DFS = timer()
        if(end_time_DFS-start_time_DFS > 60):
            print("tempo limite excedido")
            return None




def greedy_manhattan():
    path = []
    visited_nodes = []
    current = start
    current.manhattan()
    path.append(current)
    visited_nodes.append(current)
    start_time_greedy_manhattan = timer()
    while(1):
        if(current.isGoal()):
            return path
        children = current.genChildren()
        best_child = node()
        best_child.h = 99999
        for child in children:
            if(child.visited(visited_nodes) == False):
                if child.h < best_child.h:
                    best_child = child
        if(best_child.parent is None):
            current = current.parent
        else:
            current = best_child
            visited_nodes.append(current)
            path.append(current)
        end_time_greedy_manhattan = timer()
        if(end_time_greedy_manhattan-start_time_greedy_manhattan > 60):
            print("tempo limite excedido")
            return None
        
def greedy_misplaced():
    path = []
    visited_nodes = []
    current = start
    current.misplaced()
    path.append(current)
    visited_nodes.append(current)
    start_time_greedy_misplaced = timer()
    while(1):
        # print(current.matrix)
        if(current.isGoal()):
            return path
        children = current.genChildren()
        best_child = node()
        best_child.mis = 99999
        for child in children:
            if(child.visited(visited_nodes) == False):
                if child.mis < best_child.mis:
                    best_child = child
        if(best_child.parent is None):
            current = current.parent
        else:
            current = best_child
            visited_nodes.append(current)
            path.append(current) 
        end_time_greedy_misplaced = timer()
        if(end_time_greedy_misplaced-start_time_greedy_misplaced > 60):
            print("tempo limite excedido")
            return None       
        



start = node()
goal = node()
    
def get_input():
        global strategy, start_list, goal_list, start, goal, method
        start_list = list(map(int, input("Posição inicial: ").split()))
        goal_list = list(map(int, input("Posição final: ").split()))
        if (len(start_list) == n**2) and (len(goal_list) == n**2):
            start.create_matrix(start_list)
            goal.create_matrix(goal_list)
            method = input("Escolha uma destas estratégias de busca (DFS, BFS, IDFS, astar-misplaced, astar-manhattan, greedy-misplaced, greedy-manhattan): ")


def findGoal(num, goal):
    for i in range(n):
        for j in range(n):
            if goal.matrix[i][j] == num:
                return i, j

get_input()
valid = True

# visited_nodes = []
# visited_nodes.append(start)
# print(goal.visited(visited_nodes))








if (len(start_list) != n**2) or (len(goal_list) != n**2):
    print("As posições inseridas não são válidas")
else:
    if not solvability(start_list, goal_list):
        print("Configuração inicial não leva à configuração final proposta")
    else:
        path = []
        # tracemalloc.start()
        start_time = timer()
        if method == "DFS":
            path = DFS()
        elif method == "BFS":
            path = BFS()
        elif method == "IDFS":
            path = IDFS()
        elif method == "astar-misplaced":
            path = astar_misplaced()
        elif method == "astar-manhattan":
            path = astar_manhattan()
        elif method == "greedy-misplaced":
            path = greedy_misplaced()
        elif method == "greedy-manhattan":
            path = greedy_manhattan()
        else:
            valid = False
            print("Por favor escolha uma estratégia de pesquisa válida")
        if valid:
            if path is not None:
                for matrix in path:
                    matrix.print_matrix()
                print(f"Profundidade: {len(path)-1}")
            end_time = timer()
            print(f"Tempo demorado: {round(end_time-start_time, 3)} segundos")
    #         inst_mem, max_mem = tracemalloc.get_traced_memory()
    #         print(f"Máximo de memória usada: {max_mem} bytes")
    # tracemalloc.stop()

