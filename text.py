class Agent():
    def __init__(self):
        super(Agent, self).__init__()
        # agents are movable by default
        self.movable = True
        # cannot send communication signals
        self.silent = False
        # cannot observe the world
        self.blind = False
        # physical motor noise amount
        self.u_noise = None
        # communication noise amount
        self.c_noise = None
        # control range
        self.u_range = 1.0
        # state
        # script behavior to execute
        self.action_callback = None


if __name__ == '__main__':
    num_agent = 3
    agents = [Agent() for i in range(num_agent)]  # 根据agent的数量创建list
    agents_Ss = [agent for agent in agents]

    ttb=5