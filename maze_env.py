# modified from repo gym-maze: https://github.com/tuzzer/gym-maze/tree/master/gym_maze/envs 

import numpy as np
import random
class Maze:

    COMPASS = {
        "N": (0, -1),
        "E": (1, 0),
        "S": (0, 1),
        "W": (-1, 0)
    }

    def __init__(self, maze_cells=None, maze_size=(10,10), has_loops=True, num_portals=0):

        # maze member variables
        self.maze_cells = maze_cells
        self.has_loops = has_loops
        self.__portals_dict = dict()
        self.__portals = []
        self.num_portals = num_portals

        # Use existing one if exists
        if self.maze_cells is not None:
            if isinstance(self.maze_cells, (np.ndarray, np.generic)) and len(self.maze_cells.shape) == 2:
                self.maze_size = tuple(maze_cells.shape)
            else:
                raise ValueError("maze_cells must be a 2D NumPy array.")
        # Otherwise, generate a random one
        else:
            # maze's configuration parameters
            if not (isinstance(maze_size, (list, tuple)) and len(maze_size) == 2):
                raise ValueError("maze_size must be a tuple: (width, height).")
            self.maze_size = maze_size

            self._generate_maze()

    def save_maze(self, file_path):

        if not isinstance(file_path, str):
            raise TypeError("Invalid file_path. It must be a str.")

        if not os.path.exists(os.path.dirname(file_path)):
            raise ValueError("Cannot find the directory for %s." % file_path)

        else:
            np.save(file_path, self.maze_cells, allow_pickle=False, fix_imports=True)

    @classmethod
    def load_maze(cls, file_path):

        if not isinstance(file_path, str):
            raise TypeError("Invalid file_path. It must be a str.")

        if not os.path.exists(file_path):
            raise ValueError("Cannot find %s." % file_path)

        else:
            return np.load(file_path, allow_pickle=False, fix_imports=True)

    def _generate_maze(self):

        # list of all cell locations
        self.maze_cells = np.zeros(self.maze_size, dtype=int)

        # Initializing constants and variables needed for maze generation
        current_cell = (random.randint(0, self.MAZE_W-1), random.randint(0, self.MAZE_H-1))
        num_cells_visited = 1
        cell_stack = [current_cell]

        # Continue until all cells are visited
        while cell_stack:

            # restart from a cell from the cell stack
            current_cell = cell_stack.pop()
            x0, y0 = current_cell

            # find neighbours of the current cells that actually exist
            neighbours = dict()
            for dir_key, dir_val in self.COMPASS.items():
                x1 = x0 + dir_val[0]
                y1 = y0 + dir_val[1]
                # if cell is within bounds
                if 0 <= x1 < self.MAZE_W and 0 <= y1 < self.MAZE_H:
                    # if all four walls still exist
                    if self.all_walls_intact(self.maze_cells[x1, y1]):
                    #if self.num_walls_broken(self.maze_cells[x1, y1]) <= 1:
                        neighbours[dir_key] = (x1, y1)

            # if there is a neighbour
            if neighbours:
                # select a random neighbour
                dir = random.choice(tuple(neighbours.keys()))
                x1, y1 = neighbours[dir]

                # knock down the wall between the current cell and the selected neighbour
                self.maze_cells[x1, y1] = self.__break_walls(self.maze_cells[x1, y1], self.__get_opposite_wall(dir))

                # push the current cell location to the stack
                cell_stack.append(current_cell)

                # make the this neighbour cell the current cell
                cell_stack.append((x1, y1))

                # increment the visited cell count
                num_cells_visited += 1

        if self.has_loops:
            self.__break_random_walls(0.2)

        if self.num_portals > 0:
            self.__set_random_portals(num_portal_sets=self.num_portals, set_size=2)

    def __break_random_walls(self, percent):
        # find some random cells to break
        num_cells = int(round(self.MAZE_H*self.MAZE_W*percent))
        cell_ids = random.sample(range(self.MAZE_W*self.MAZE_H), num_cells)

        # for each of those walls
        for cell_id in cell_ids:
            x = cell_id % self.MAZE_H
            y = int(cell_id/self.MAZE_H)

            # randomize the compass order
            dirs = random.sample(list(self.COMPASS.keys()), len(self.COMPASS))
            for dir in dirs:
                # break the wall if it's not already open
                if self.is_breakable((x, y), dir):
                    self.maze_cells[x, y] = self.__break_walls(self.maze_cells[x, y], dir)
                    break

    def __set_random_portals(self, num_portal_sets, set_size=2):
        # find some random cells to break
        num_portal_sets = int(num_portal_sets)
        set_size = int(set_size)

        # limit the maximum number of portal sets to the number of cells available.
        max_portal_sets = int(self.MAZE_W * self.MAZE_H / set_size)
        num_portal_sets = min(max_portal_sets, num_portal_sets)

        # the first and last cells are reserved
        cell_ids = random.sample(range(1, self.MAZE_W * self.MAZE_H - 1), num_portal_sets*set_size)

        for i in range(num_portal_sets):
            # sample the set_size number of sell
            portal_cell_ids = random.sample(cell_ids, set_size)
            portal_locations = []
            for portal_cell_id in portal_cell_ids:
                # remove the cell from the set of potential cell_ids
                cell_ids.pop(cell_ids.index(portal_cell_id))
                # convert portal ids to location
                x = portal_cell_id % self.MAZE_H
                y = int(portal_cell_id / self.MAZE_H)
                portal_locations.append((x,y))
            # append the new portal to the maze
            portal = Portal(*portal_locations)
            self.__portals.append(portal)

            # create a dictionary of portals
            for portal_location in portal_locations:
                self.__portals_dict[portal_location] = portal

    def is_open(self, cell_id, dir):
        # check if it would be out-of-bound
        x1 = cell_id[0] + self.COMPASS[dir][0]
        y1 = cell_id[1] + self.COMPASS[dir][1]

        # if cell is still within bounds after the move
        if self.is_within_bound(x1, y1):
            # check if the wall is opened
            this_wall = bool(self.get_walls_status(self.maze_cells[cell_id[0], cell_id[1]])[dir])
            other_wall = bool(self.get_walls_status(self.maze_cells[x1, y1])[self.__get_opposite_wall(dir)])
            return this_wall or other_wall
        return False

    def is_breakable(self, cell_id, dir):
        # check if it would be out-of-bound
        x1 = cell_id[0] + self.COMPASS[dir][0]
        y1 = cell_id[1] + self.COMPASS[dir][1]

        return not self.is_open(cell_id, dir) and self.is_within_bound(x1, y1)

    def is_within_bound(self, x, y):
        # true if cell is still within bounds after the move
        return 0 <= x < self.MAZE_W and 0 <= y < self.MAZE_H

    def is_portal(self, cell):
        return tuple(cell) in self.__portals_dict

    @property
    def portals(self):
        return tuple(self.__portals)

    def get_portal(self, cell):
        if cell in self.__portals_dict:
            return self.__portals_dict[cell]
        return None

    @property
    def MAZE_W(self):
        return int(self.maze_size[0])

    @property
    def MAZE_H(self):
        return int(self.maze_size[1])

    @classmethod
    def get_walls_status(cls, cell):
        walls = {
            "N" : (cell & 0x1) >> 0,
            "E" : (cell & 0x2) >> 1,
            "S" : (cell & 0x4) >> 2,
            "W" : (cell & 0x8) >> 3,
        }
        return walls

    @classmethod
    def all_walls_intact(cls, cell):
        return cell & 0xF == 0

    @classmethod
    def num_walls_broken(cls, cell):
        walls = cls.get_walls_status(cell)
        num_broken = 0
        for wall_broken in walls.values():
            num_broken += wall_broken
        return num_broken

    @classmethod
    def __break_walls(cls, cell, dirs):
        if "N" in dirs:
            cell |= 0x1
        if "E" in dirs:
            cell |= 0x2
        if "S" in dirs:
            cell |= 0x4
        if "W" in dirs:
            cell |= 0x8
        return cell

    @classmethod
    def __get_opposite_wall(cls, dirs):

        if not isinstance(dirs, str):
            raise TypeError("dirs must be a str.")

        opposite_dirs = ""

        for dir in dirs:
            if dir == "N":
                opposite_dir = "S"
            elif dir == "S":
                opposite_dir = "N"
            elif dir == "E":
                opposite_dir = "W"
            elif dir == "W":
                opposite_dir = "E"
            else:
                raise ValueError("The only valid directions are (N, S, E, W).")

            opposite_dirs += opposite_dir

        return opposite_dirs

class Portal:

    def __init__(self, *locations):

        self.__locations = []
        for location in locations:
            if isinstance(location, (tuple, list)):
                self.__locations.append(tuple(location))
            else:
                raise ValueError("location must be a list or a tuple.")

    def teleport(self, cell):
        if cell in self.locations:
            return self.locations[(self.locations.index(cell) + 1) % len(self.locations)]
        return cell

    def get_index(self, cell):
        return self.locations.index(cell)

    @property
    def locations(self):
        return self.__locations


class MazeEnv(object):
    ACTION = ["N", "S", "E", "W"]

    def __init__(self, maze_file=None, drift = 0, maze_size=None, mode=None):
        if mode == "plus":
            has_loops = True
            num_portals = 0 #int(round(min(maze_size)/3))
        else:
            has_loops = False
            num_portals = 0

        maze_cells = None
        if maze_file is not None: 
            maze_cells = self.load_from_path(maze_file)


        self.maze = Maze(maze_cells=maze_cells, maze_size=maze_size, has_loops=has_loops, num_portals=num_portals)
        self.maze_size = self.maze.maze_size
        # Simulation related variables.
        self.state = self.reset()
        self.goal = np.asarray([self.maze_size[0] - 1, self.maze_size[1] - 1])

        self.drift = drift

    def load_from_path(self, path):
        with open(path) as f:
            cells = f.readlines()
        cells = [ [int(wall) for wall in line.strip().split(" ")] for line in cells]
        return np.asarray(cells)

    def seed(self, seed=None):
        np.random.seed(seed)

    def step(self, action):
        p = [self.drift, self.drift, self.drift, self.drift]
        p[action] += (1 - 4 * self.drift)
        action = np.random.choice(4, 1, p=p)[0]

        if self.maze.is_open(self.state, self.ACTION[action]):
            delta = self.maze.COMPASS[self.ACTION[action]]
            self.state = (self.state[0] + delta[0], self.state[1] + delta[1])

        if np.array_equal(self.state, self.goal):
            reward = 1
            done = True
        else:
            reward = -0.1/(self.maze_size[0]*self.maze_size[1])
            done = False


        return self.state, reward, done

    def reset(self):
        self.state = (0, 0)
        return self.state

    def show_maze(self):
        maze_render = []
        x, y = self.maze_size
        for i in range(x):
            top_str = ""
            mid_str = "|"
            for _ in range(y):
                top_str += " __"
                mid_str += "  |"
            maze_render.append(top_str)
            maze_render.append(mid_str)
        maze_render.append(top_str)

        for i in range(x):
            for j in range(y):
                cell = self.maze.maze_cells[j, i]
                if self.maze.is_open((i, j), 'N'):
                    maze_render[2*j] = maze_render[2*j][:3*i+1] + '  ' +  maze_render[2*j][3*i+3:]
                if self.maze.is_open((i, j), 'E'):
                    maze_render[2*j+1] = maze_render[2*j+1][:3*i+3] + ' ' +  maze_render[2*j+1][3*i+4:]
                if self.maze.is_open((i, j), 'S'):
                    maze_render[2*j+2] = maze_render[2*j+2][:3*i+1] + '  ' +  maze_render[2*j+2][3*i+3:]
                if self.maze.is_open((i, j), 'W'):
                    maze_render[2*j+1] = maze_render[2*j+1][:3*i] + ' ' +  maze_render[2*j+1][3*i+1:]

        for line in maze_render:
            print(line)

# render
