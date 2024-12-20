# SPACE simulator (fork)

This repo is fun for learning and tinkering, especially given the nice paper for explanation.

I appreciate the simplicity and clarity of the core logic, which makes it easy to see the relationship between Agents, Tasks, the BT, and the algorithms:
 - The core `Node` types in behavior_tree.py are beautifully simple.
 - Passing an agent and the blackboard to each `Node.run` is a nice clear pattern.
 - Providing `action` callbacks via SyncNode.run also seems nice.
 - Remarkably, a rich variety of collective phenomena can be achieved without even changing the tree. Simply changing the agents' `decide` method and other configurable parameters seems to go a long way.

It is also nice that the focus is on multi-agent tasking and execution without bogging down in sim fidelity and comms. Because of its simplicity, it could serve as great architectural starting point for other projects. For this reason, I decided to fork it and have a look at making some improvements.

## Ideas for improving this repo

### Code quality

 - [ ] Config handling
    * [ ] Develop schemas from existing config for auto-validation and stronger typing. Pydantic? CUE?? _Pydantic_.
    * [ ] Do away with globals and pass config objects in where needed.
 - [ ] Create cleaner model-view separation in Agent and Task classes 
    * [ ] Move pygame dependencies out to a vis module.
    * [ ] Vis abstraction to support other alternatives
    * [ ] Abstract kinematics/dynamics so fancier motion models could optionally be substituted in later
 - [ ] Dynamic plugin loading via config and importlib is a cool idea, but the implementation needs work
 - [ ] Rework (really remove) the SyncAction type hierarchy. Less inheritance = more better. Details below.
 - [ ] Behavior tree creation from XML
    * [ ] Should agent class really be responsible for this? Move to behavior_tree.py?? 
    * [ ] Refactor node instantiation to avoid `globals()` dict lookups that rely on `from behavior_tree import *`.
 - [ ] Add init.py to modules/ and consider renaming to src/ or something
 - [ ] Split utils.py into plotting, i/o, and config handling files.
 - [ ] Add type annotations throughout.
 - [ ] Unit tests
 - [ ] Make this an installable package:
    * [ ] Add a pyproject.toml and all that
    * [ ] Reorg the source tree to be more idomatic.
    * [ ] Add an examples/ dir and put main.py there?
 - [ ] main.py needs heavy refactoring and cleanup.
 - [ ] Apply consistent source formatting everywhere (black formatter)

### New capabilities and features

 - Introducing planning, a map, and obstacle / peer agent avoidance would be a cool improvement. 
    * See [SwarmLab](https://github.com/lis-epfl/swarmlab.git), which is a major inspiration for SPACE. [SwarmRobotics](https://github.com/xzlxiao/SwarmRobotics) appears to be a python port of SwarmLab. 
 - The sim models point particles in 2D, but could be generalized to model extended bodies in 3D using a library of platform models (warning: feature creep).

### Other thoughts

Consider using [`typing.Protocol`](https://peps.python.org/pep-0544/) to help constrain and solidify the decision-making plugin pattern, e.g. `DecisionLogic`.

Each agent holds a list of every other agent in the sim, when all it should know about is its neighbor set. See `Agent.all_agents`, renamed from `agents_info`. Can this be modeled more efficiently and realistically?

It seems strange that each agent owns a blackboard. Wouldn't it be simpler for `Node` to own it?
Also, the blackboard is just a dict. Pro: flexibility; Con: flexibility. What would it take to make this threadsafe?

I may rename `SyncAction` to `SyncActionNode` because it isn't an action. It is a `Node` that *has* an action.

The `SyncAction` node is nice and simple: aside from the inherited `name`, it has an `action` object whose only job is to be callable with signature `result: Status = action(agent: Agent, blackboard: dict)`. ~~This is great and there should be no need to change it.~~ Actually, why does the action need an `agent`, especially when each agent already has a tree as a member? `action` seems to be analogous to `tick` in BTCPP. `tick` takes no args. In BTCPP, access to the blackboard is provided through `NodeConfig`.
 
But then `LocalSensingNode`, `TaskExecutingNode`, `ExplorationNode`, and `DecisionMakingNode` subclass it, even though they are really just callback wrappers. To me, this doesn't seem to provide significant code reuse or interface standardization benefits. Actually, it's the opposite: the constructors are currently kind of messed up, with some taking an unused `agent` parameter; Agent dependency is needlessly coupled in; and most of these types don't even maintain state that persists outside their callback. So this introduces all the drawbacks of inheritance with none of the advantages.

Instead, for this minimal BT implementation, it seems super clean to build trees from only three concrete `Node` types: `Sequence`, `Fallback`, and `SyncActionNode`. Then `Leaf.action` is the canonical point of extensibility via dependency injection. A user can build an `action` object however they like, whether that's a plain function, a class method, a functor object, or even a lambda. Actions encapsulate whatever state they need, and the Nodes that run them don't need to care. Also, actions, being modular and standardized, can be easily reused across nodes. 

For some reason, the creator of BT.CPP recommends inheritance over this pattern, but I'd like to try it anyway. If I come to realize it is a bad turn, I will have learned something new.

Given this to-do list, starting with Pydantic config models and removing globals would probably provide the best bang for the buck.

---


# SPACE (Swarm Planning And Control Evaluation) Simulator

**SPACE** Simulator is a pygame-based application for simulating decentralized agent behavior using behavior trees. 
By integrating your custom decision-making algorithms as plugins, SPACE enables rigorous testing and comparative analysis against pre-existing algorithms. 

The official documentation of the SPACE simulator is available at [http://space-simulator.rtfd.io/](http://space-simulator.rtfd.io/). 


<div style="display: flex; flex-direction: row;">
    <img src="output/2024-07-13/RandomAssignment_100_agents_300_tasks_2024-07-13_00-41-18.gif" alt="GIF" width="400" height="300">
    <img src="output/2024-07-13/RandomAssignment_1000_agents_3000_tasks_2024-07-13_00-38-13.gif" alt="GIF" width="400" height="300">
</div>

- Example: (Left) `num_agents = 100`, `num_tasks = 300`; (Right) `num_agents = 1000`, `num_tasks = 3000`

<div style="display: flex; flex-direction: row;">
    <img src="output/2024-07-27/GRAPE_30_agents_200_tasks_2024-07-27_01-35-35.gif" alt="GIF" width="400" height="300">
    <img src="output/2024-07-27/CBBA_30_agents_200_tasks_2024-07-27_01-34-05.gif" alt="GIF" width="400" height="300">
</div>

- Example: (Left) `GRAPE`; (Right) `CBBA`; (Common) `num_agents = 30`, `num_tasks = 200 (static); 50 x 3 times (dynamic)`


## Features

- Simulates multiple agents performing tasks
- Agents use behavior trees for decision-making
- Real-time task assignment and execution
- Debug mode for visualizing agent behavior



## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/inmo-jang/space-simulator.git
    cd space-simulator
    ```

2. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

3. Run the simulator:
    ```sh
    python main.py
    ```


## Configuration

Modify the `config.yaml` file to adjust simulation parameters:
- Number of agents and tasks
- Screen dimensions
- Agent behavior parameters

Refer to the configuration guide [CONFIG_GUIDE.md](/docs/CONFIG_GUIDE.md)



## Usage

### Controls
- `ESC` or `Q`: Quit the simulation
- `P`: Pause/unpause the simulation
- `R`: Start/stop recording the simulation as a GIF file

### Monte Carlo Analysis

1. Set `mc_runner.yaml` for your purpose and run the following:
    ```sh
    python mc_runner.py
    ``` 

2. Set `mc_comparison.yaml` and run the following:
    ```sh
    python mc_analyzer.py
    ``` 



## Code Structure
- `main.py`: Entry point of the simulation, initializes pygame and manages the main game loop.
- `/modules/`
    - `agent.py`: Defines the Agent class and manages agent behavior.
    - `task.py`: Defines the Task class and manages task behavior.
    - `behavior_tree.py`: Implements behavior tree nodes and execution logic.
    - `utils.py`: Utility functions and configuration loading.
- `/plugins/`
    - `my_decision_making_plugin.py`: Template for decision-making algorithms for each agent.


## Contributing
Feel free to fork the repository and submit pull requests. 
Refer to [TODO.md](/docs/TODO.md) for a list of pending tasks and upcoming features.

## Citations
Please cite this work in your papers!
- [Inmo Jang, *"SPACE: A Python-based Simulator for Evaluating Decentralized Multi-Robot Task Allocation Algorithms"*, arXiv:2409.04230 [cs.RO], 2024](https://arxiv.org/abs/2409.04230)


## License
[GNU GPLv3](LICENSE)
