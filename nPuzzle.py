import numpy as np
import math
import copy 

n = 4

class node():
    def __init__(self, pieces_list = [], parent = None, f=0, g=0, h=0):
        self.pieces_list = pieces_list
        self.f = f
        self.g = g
        self.h = h
        self.parent = parent


    def create_matrix(list):
        matrix = np.zeros((n,n))
        index = 0
        for i in range(4):
            for j in range(4):
                matrix[i][j] = list[index]
                index += 1
        return matrix


    def isGoal(current, goal):
        return np.array_equal(current, goal)


    def findBlank(current):
        for i in range(n):
                for j in range(n):
                    if current[i][j] == 0:
                        return i, j


    def genChildren(current):
            x, y = findBlank(current)
            newMatrices = []
            if (x + 1) < n: #moving blank down / moving a tile up
                new = copy.deepcopy(current)
                new[x][y] = new[x+1][y]
                new[x+1][y] = 0
                newMatrices.append(new)
            if (y + 1) < n:  # moving blank right / moving a tile left
                new = copy.deepcopy(current)
                new[x][y]=new[x][y+1]
                new[x][y+1]= 0
                newMatrices.append(new)
            if (x - 1) > -1: #moving blank up / moving a tile down
                new = copy.deepcopy(current)
                new[x][y] = new[x-1][y]
                new[x-1][y] = 0
                newMatrices.append(new)
            if (y - 1) > -1: # moving blank left / moving a tile right
                new = copy.deepcopy(current)
                new[x][y] = new[x][y - 1]
                new[x][y-1] = 0
                newMatrices.append(new)
            return newMatrices


    def findGoal(num, goal):
        for i in range(n):
            for j in range(n):
                if goal[i][j] == num:
                    return i, j


    def manhattan(current, goal):
        sum = 0
        for i in range(n):
            for j in range(n):
                if current[i][j] == 0:
                    continue
                else:
                    x, y = findGoal(current[i][j], goal)
                    sum += abs(x - i) + abs(y - j)
        return sum
    

    def aStar(current, goal):
        sequence = []
        sequence.append(current)
        while(1):
            print("ok")
            best_child = []
            best_manhattan = 9999999
            children = genChildren(current)
            # print(children)
            for child in children:
                # print(child)
                # print(manhattan(child, goal))
                # print("dwadwadwad")
                if(manhattan(child, goal) < best_manhattan):
                    best_manhattan = manhattan(child, goal)
                    best_child = child
            print(best_child)
            sequence.append(best_child)
            if(isGoal(best_child, goal)):
                return sequence
            current = best_child


    def DFS(current, goal):
        lst = [current]
        while(len(lst) != 0):
            print(len(lst))
            print(lst)
            for i in range(1):
                current = lst[0]
                lst.pop(0)
                if(isGoal(current, goal)):
                    return "solution found"
                children = genChildren(current)
                lst = children + lst
        return "solution not found"


    def BFS(current, goal):
        queue = []
        queue.append(current)
        queue.append(0)
        while(len(queue) != 0):
            node = queue.pop(0)
            depth_node = queue.pop(0)
            if(isGoal(node, goal)):
                return node, depth_node
            children = genChildren(node)
            for child in children:
                queue.append(child)
                queue.append(depth_node+1)
        return "solution not found"


    def iterativeDfs(start, goal):
        depth = 1
        bottom_reached = False  
        while not bottom_reached:
            result, bottom_reached = iterativeDfsRec(start, goal, 0, depth)
            if result is not None:
                return result
            depth += 1
        return None


    def iterativeDfsRec(node, goal, current_depth, max_depth):
        if isGoal(node, goal):
            print("solution found")
            print(current_depth)
            return node, True
        children = genChildren(node)
        if current_depth == max_depth:
            return None, False
        bottom_reached = True
        for child in children:
            result, bottom_reached_rec = iterativeDfsRec(child, goal, current_depth + 1, max_depth)
            if result is not None:
                return result, True
            bottom_reached = bottom_reached and bottom_reached_rec
        return None, bottom_reached


    def solvability(current):
        inversions = 0
        for i in range(len(current)):
            inversions_temp = 0
            for j in range(i+1, len(current)):
                if current[i] > current[j] and current[j]!=0:
                    inversions_temp += 1
            inversions += inversions_temp
        for i in range(len(current)):
            if current[i] == 0:
                if (i > -1 and i < 4) or (i > 8 and i < 13):
                    if inversions % 2 != 0:
                        return True
                else:
                    if inversions % 2 == 0: 
                        return True
        return False








start_list = list(map(int, input("Posição inicial: ").split()))
goal_list = list(map(int, input("Posição final: ").split()))
start_matrix = create_matrix(start_list)
goal_matrix = create_matrix(goal_list)
search_strategy = input("Escolha uma destas estratégias de busca (DFS, BFS, IDFS, A*-misplaced, A*-Manhattan, Greedy-misplaced, Greedy-Manhattan): ")
if solvability(start_list):
    if search_strategy == "DFS":
        print(DFS(start_matrix, goal_matrix))
    elif search_strategy == "BFS":
        print(BFS(start_matrix, goal_matrix))
    elif search_strategy == "IDSF":
        print(iterativeDfs(start_matrix, goal_matrix))
    elif search_strategy == "A*-misplaced":
        print(BFS(start_matrix, goal_matrix))
    elif search_strategy == "A*-Manhattan":
        print(BFS(start_matrix, goal_matrix))
    elif search_strategy == "A*-misplaced":
        print(BFS(start_matrix, goal_matrix))
    elif search_strategy == "Greedy-misplaced":
        print(BFS(start_matrix, goal_matrix))
    elif search_strategy == "Greedy-misplaced":
        print(BFS(start_matrix, goal_matrix))
    else:
        print("Por favor escolha uma estratégia de pesquisa válida")
else:
    print("nao exite solução")  


# print(aStar(start_matrix, goal_matrix))
# resultado, depth = BFS(start_matrix, goal_matrix)
# print(isGoal(start_matrix, goal_matrix))

# print(resultado)
# print(depth)

# print(iterativeDfs(start_matrix, goal_matrix))~

# print(solvability(start_list))