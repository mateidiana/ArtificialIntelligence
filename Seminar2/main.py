import heapq
import numpy as np
import matplotlib.pyplot as plt
import matplotlib


def distance(x, y):
    return (x[0] - y[0]) ** 2 + ((x[1] - y[1]) ** 2)


class Map:
    def __init__(self, m: np.ndarray) -> None:
        self.m = m

    def neighbors(self, cell):
        nrow, ncol = m.shape
        x, y = cell
        nb = []
        if x > 0:
            if m[x - 1, y] == 0:
                nb = nb + [(x - 1, y)]
        if x < (nrow - 1):
            if m[x + 1, y] == 0:
                nb = nb + [(x + 1, y)]
        if y > 0:
            if m[x, y - 1] == 0:
                nb = nb + [(x, y - 1)]
        if y < (ncol - 1):
            if m[x, y + 1] == 0:
                nb = nb + [(x, y + 1)]
        return nb


class PriorityQueue:
    def __init__(self):
        self.elements = []

    def empty(self):
        return len(self.elements) == 0

    def put(self, item, priority):
        heapq.heappush(self.elements, (priority, item))

    def get(self):
        return heapq.heappop(self.elements)[1]


class Stack:
    def __init__(self):
        self.elements = []

    def empty(self):
        return len(self.elements) == 0

    def put(self, x):
        self.elements.append(x)

    def get(self):
        return self.elements.pop()


def dfs(map: Map, start, goal):
    stack = Stack()
    stack.put(start)
    came_from = {start: None}
    while not stack.empty():
        current = stack.get()
        if current == goal:
            return came_from
        else:
            for next in map.neighbors(current):
                if next not in came_from:
                    stack.put(next)
                    came_from[next] = current
    return None


def make_path(came_from, start, goal):
    current = goal
    path = []
    while current != start:
        path.append(current)
        current = came_from[current]
    path.append(start)
    path.reverse()
    return path


def astar(map: Map, start, goal):
    queue = PriorityQueue()
    queue.put(start, 0)
    came_from = {start: None}
    cost_so_far = {start: 0}
    while not queue.empty():
        current = queue.get()

        if current == goal:
            return make_path(came_from, start, goal)

        for neighbour in map.neighbors(current):
            new_cost = cost_so_far[current] + distance(current, neighbour)

            if neighbour not in cost_so_far or new_cost < cost_so_far[neighbour]:
                cost_so_far[neighbour] = new_cost
                priority = distance(neighbour, goal) + new_cost
                queue.put(neighbour, priority)
                came_from[neighbour] = current
    return None



def show_path(path, map):
    cmap = matplotlib.colors.ListedColormap(['white', 'black'])

    plt.imshow(map.m, cmap=cmap)

    x, y = zip(*path)
    plt.plot(y, x, color='blue', linewidth=1)

    plt.show()


m = np.array(
    [[0, 1, 0, 0, 0, 0, 0],
     [0, 1, 0, 1, 0, 0, 1],
     [0, 1, 0, 1, 0, 1, 0],
     [0, 1, 0, 1, 0, 0, 0],
     [0, 0, 0, 1, 1, 0, 0],
     [0, 0, 0, 1, 1, 0, 0]])
mm = Map(m)
mm.neighbors((4, 1))


def main():
    start = (0, 0)
    goal = (5, 6)

    came_from = dfs(mm, start, goal)
    print(f'Came from dictionary:\n{came_from}\n')

    path = make_path(came_from, start, goal)
    print(f'Path with dfs:\n{path}\n')

    astar_path = astar(mm, start, goal)
    print(f'Path with A*:\n{astar_path}\n')

    show_path(astar_path, mm)


if __name__ == '__main__':
    main()


