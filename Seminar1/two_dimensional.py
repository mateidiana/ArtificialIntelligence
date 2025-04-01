import random

rows = 6
columns = 6


class Thing:
    """This represents any physical object that can appear in an Environment.
    You subclass Thing to get the things you want. Each thing can have a
    .__name__  slot (used for output only)."""

    def __repr__(self):
        return '<{}>'.format(getattr(self, '__name__', self.__class__.__name__))

    def is_alive(self):
        """Things that are 'alive' should return true."""
        return hasattr(self, 'alive') and self.alive

    def show_state(self):
        """Display the agent's internal state. Subclasses should override."""
        print("I don't know how to show_state.")

    def display(self, canvas, x, y, width, height):
        """Display an image of this Thing on the canvas."""
        # Do we need this?
        pass


class Agent(Thing):
    """An Agent is a subclass of Thing with one required slot,
    .program, which should hold a function that takes one argument, the
    percept, and returns an action. (What counts as a percept or action
    will depend on the specific environment in which the agent exists.)
    Note that 'program' is a slot, not a method. If it were a method,
    then the program could 'cheat' and look at aspects of the agent.
    It's not supposed to do that: the program can only look at the
    percepts. An agent program that needs a model of the world (and of
    the agent itself) will have to build and maintain its own model.
    There is an optional slot, .performance, which is a number giving
    the performance measure of the agent in its environment."""

    def __init__(self, program):
        self.program = program


def TraceAgent(agent):
    """Wrap the agent's program to print its input and output. This will let
    you see what the agent is doing in the environment."""
    old_program = agent.program

    def new_program(percept):
        action = old_program(percept)
        print('{} perceives {} and does {}'.format(agent, percept, action))
        return action

    agent.program = new_program
    return agent


def ReflexVacuumAgent():
    """A reflex agent for the two-state vacuum environment."""
    def program(percept):
        """Makes a cycle for the traversal of the 2d array
        Example 4x4 2d array:
               0 1 2 3
            0: v < < <
            1: > > v ^
            2: v < < ^
            3: > > > ^ """
        location, status = percept
        row, col = location

        if status == 'Dirty':
            return 'Suck'

        if col == columns - 1:
            if row == 0:
                return 'Left'
            return 'Up'

        if row % 2 == 1:
            if col == columns - 2 and row != rows - 1:
                return 'Down'
            return 'Right'

        if row % 2 == 0:
            if col == 0:
                return 'Down'
            return 'Left'

    return Agent(program)


def ModelBasedVacuumAgent():
    """An agent that keeps track of what locations are clean or dirty."""
    model = {}

    def program(percept):
        """Same as ReflexVacuumAgent, except if everything is clean, do NoOp."""
        location, status = percept
        row, col = location
        model[location] = status

        if len(model) == rows * columns and all(value == 'Clean' for value in model.values()):
            return 'NoOp'

        if status == 'Dirty':
            return 'Suck'

        if col == columns - 1:
            if row == 0:
                return 'Left'
            return 'Up'

        if row % 2 == 1:
            if col == columns - 2 and row != rows - 1:
                return 'Down'
            return 'Right'

        if row % 2 == 0:
            if col == 0:
                return 'Down'
            return 'Left'

    return Agent(program)


class Environment:
    """Abstract class representing an Environment. 'Real' Environment classes
    inherit from this. Your Environment will typically need to implement:
        percept:           Define the percept that an agent sees.
        execute_action:    Define the effects of executing an action.
                           Also update the agent.performance slot.
    The environment keeps a list of .things and .agents (which is a subset
    of .things). Each agent has a .performance slot, initialized to 0.
    Each thing has a .location slot, even though some environments may not
    need this."""

    def __init__(self):
        self.things = []
        self.agents = []

    def thing_classes(self):
        return []  # List of classes that can go into environment

    def percept(self, agent):
        """Return the percept that the agent sees at this point. (Implement this.)"""
        raise NotImplementedError

    def execute_action(self, agent, action):
        """Change the world to reflect this action. (Implement this.)"""
        raise NotImplementedError

    def default_location(self, thing):
        """Default location to place a new thing with unspecified location."""
        return None

    def exogenous_change(self):
        """If there is spontaneous change in the world, override this."""
        pass

    def step(self):
        """Run the environment for one time step. If the
        actions and exogenous changes are independent, this method will
        do. If there are interactions between them, you'll need to
        override this method."""
        actions = []
        for agent in self.agents:
            actions.append(agent.program(self.percept(agent)))
        for (agent, action) in zip(self.agents, actions):
            self.execute_action(agent, action)
        self.exogenous_change()

    def run(self, steps=1000):
        """Run the Environment for given number of time steps."""
        for step in range(steps):
            self.step()

    def list_things_at(self, location, tclass=Thing):
        """Return all things exactly at a given location."""
        return [thing for thing in self.things
                if thing.location == location and isinstance(thing, tclass)]

    def some_things_at(self, location, tclass=Thing):
        """Return true if at least one of the things at location
        is an instance of class tclass (or a subclass)."""
        return self.list_things_at(location, tclass) != []

    def add_thing(self, thing, location=None):
        """Add a thing to the environment, setting its location. For
        convenience, if thing is an agent program we make a new agent
        for it. (Shouldn't need to override this.)"""
        if not isinstance(thing, Thing):
            thing = Agent(thing)
        if thing in self.things:
            print("Can't add the same thing twice")
        else:
            thing.location = location if location is not None else self.default_location(thing)
            self.things.append(thing)
            if isinstance(thing, Agent):
                thing.performance = 0
                self.agents.append(thing)

    def delete_thing(self, thing):
        """Remove a thing from the environment."""
        try:
            self.things.remove(thing)
        except ValueError as e:
            print(e)
            print("  in Environment delete_thing")
            print("  Thing to be removed: {} at {}".format(thing, thing.location))
            print("  from list: {}".format([(thing, thing.location) for thing in self.things]))
        if thing in self.agents:
            self.agents.remove(thing)


class TrivialVacuumEnvironment(Environment):
    """This environment has two locations, A and B. Each can be Dirty
    or Clean. The agent perceives its location and the location's
    status. This serves as an example of how to implement a simple
    Environment."""

    def __init__(self, rows, columns):
        super().__init__()
        self.status = self.init_status(rows, columns)

    # def thing_classes(self):
    #     return [Wall, Dirt, ReflexVacuumAgent, RandomVacuumAgent,
    #             TableDrivenVacuumAgent, ModelBasedVacuumAgent]

    def init_status(self, rows, columns):
        status_matrix = []
        for i in range(rows):
            row = []
            for j in range(columns):
                row.append(random.choice(['Clean', 'Dirty']))
            status_matrix.append(row)
        return status_matrix

    def percept(self, agent):
        """Returns the agent's location, and the location status (Dirty/Clean)."""
        row, col = agent.location
        return (agent.location, self.status[row][col])

    def execute_action(self, agent, action):
        """Change agent's location and/or location's status; track performance.
        Score 10 for each dirt cleaned; -1 for each move."""
        row, col = agent.location
        if action == 'Right':
            agent.location = (row, col + 1)
            agent.performance -= 1
        elif action == 'Left':
            agent.location = (row, col - 1)
            agent.performance -= 1
        elif action == 'Up':
            agent.location = (row - 1, col)
            agent.performance -= 1
        elif action == 'Down':
            agent.location = (row + 1, col)
            agent.performance -= 1
        elif action == 'Suck':
            if self.status[row][col] == 'Dirty':
                agent.performance += 10
            self.status[row][col] = 'Clean'

    def default_location(self, thing):
        """Agents start in either location at random."""
        random_row = random.randint(0, rows - 1)
        random_column = random.randint(0, columns - 1)
        return random_row, random_column


def main():
    # a = ReflexVacuumAgent()
    # a.program(((0, 0), 'Clean'))
    # a.program(((1, 1), 'Clean'))
    # a.program(((1, 2), 'Dirty'))
    # a.program(((2, 2), 'Dirty'))

    e = TrivialVacuumEnvironment(rows, columns)
    # e.add_thing(TraceAgent(ReflexVacuumAgent()))
    e.add_thing(TraceAgent(ModelBasedVacuumAgent()))
    e.run(72)


if __name__ == "__main__":
    main()