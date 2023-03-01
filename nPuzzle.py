import numpy as np
import math
import copy 

n = 4

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
                return "econtrada"
            children = genChildren(current)
            lst = children + lst
    return "solution not found"


def BFS(current, goal):
    queue = []
    queue.append(current)
    path = []
    while(len(queue) != 0):
        add = queue[0]
        print(len(queue))
        node = queue.pop(0)
        if(isGoal(node, goal)):
            return path
        children = genChildren(node)
        queue = queue + children
    return "solution not found"










start_list = list(map(int, input().split()))
goal_list = list(map(int, input().split()))
start_matrix = create_matrix(start_list)
goal_matrix = create_matrix(goal_list)

# print(aStar(start_matrix, goal_matrix))
# print(BFS(start_matrix, goal_matrix)
print(isGoal(start_matrix, goal_matrix))

