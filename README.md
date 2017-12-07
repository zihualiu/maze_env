# Maze_env
Simple maze environment for reinforcement learning

Arguments: 
* mode: "plus" or None. If None is used, then the generated maze has no loops. If "plus" is used, there will be loops.
* maze_file: prebuilt maze file, see  [prebuilt mazes](prebuilt_maze/) for prebuilt T-maze and L-maze
* drift: probabiltiy of stochastic drift.
* maze_size: size of maze, tuple of width x height

Sample usage: 
```
from  maze_env import MazeEnv
>>> env = MazeEnv(maze_file=None, drift = 0, maze_size=(7, 7), mode=None)
>>> env.show_maze()
 __ __ __ __ __ __ __
|        |           |
 __       __    __ __
|     |     |        |
    __ __    __ __
|        |        |  |
    __    __ __
|     |  |     |  |  |
    __       __
|  |     |     |     |
 __       __    __
|     |     |        |
    __ __       __ __
|           |        |
 __ __ __ __ __ __ __
```

The starting point of maze is in top left corner. Every step that does not end in the finish (bottom right corner) will have a small negative penalty. Finishing the maze will result in a reward of +1

To step through the environment, it is very similar to openAI gym. first reset and then call ```.step(action)```. Actions are in range of 0 to 3, and step function returns tuple
```
next_state, reward, if_terminal, info = env.step(action)
```
