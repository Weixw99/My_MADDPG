# 用作目标跟踪的V0版本
import numpy as np
from multienvs.core_v0 import World, Agent, Landmark
from multienvs.scenario import BaseScenario
# 用于目标跟踪的环境代码

class Scenario(BaseScenario):
    def make_world(self):
        # 创建居住在世界上的所有实体（地标、代理等），分配它们的能力（它们是否可以通信，或移动，或两者兼而有之）。在每个训练回合开始时调用一次
        world = World()
        world.collaborative = True
        # set any world properties first
        world.dim_c = 2  # 通信渠道维度
        num_jingtai_landmarks = 1
        num_dongtai_landmarks = 1
        num_agents = 3
        num_landmarks = num_jingtai_landmarks + num_dongtai_landmarks

        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
            agent.leader = True if i < 1 else False
            agent.size = 0.045

        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.adversary = True if i < num_dongtai_landmarks else False
            landmark.collide = True if landmark.adversary else True
            landmark.movable = True if landmark.adversary else False
            # landmark.size = 0.08
            landmark.size = 0.065 if landmark.adversary else 0.07
            landmark.boundary = False
        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world):
        agents = self.good_agents(world)
        adversaries = self.adversaries(world)
        # random properties for agents
        for i, agent in enumerate(world.agents):
            if i == 0:
                agent.color = np.array([1, 0.35, 0.85])
            else:
                agent.color = np.array([0.25, 0.35, 0.85])

            # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            if i == 0:
                landmark.color = np.array([0.25, 0.25, 0.25])
            else:
                landmark.color = np.array([0.85, 0.35, 0.35])
                
        # set random initial states
        for i, agent in enumerate(world.agents):
            agent.state.p_pos = np.array([np.random.uniform(0.00001, 0.00), np.random.uniform(0.00001, 0.00)])
            agent.state.p_vel = np.zeros(world.dim_p)  # 速度  [0 0]
            agent.state.c = np.zeros(world.dim_c)  # 状态 [0 0]

        for i, landmark in enumerate(world.landmarks):
            if not landmark.boundary:
                if i == 0:
                    landmark.state.p_pos = np.array([-0.97, 0.7])
                else:
                    landmark.state.p_pos = np.array([0, 1.6])
                landmark.state.p_vel = np.zeros(world.dim_p)

    def benchmark_data(self, agent, world):
        # returns data for benchmarking purposes
        if agent.adversary:
            collisions = 0
            for a in self.good_agents(world):
                if self.is_collision(a, agent):
                    collisions += 1
            return collisions
        else:
            return 0

    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size + 0.1
        return True if dist < dist_min else False

    # return all agents that are not adversaries
    def good_agents(self, world):
        return [agent for agent in world.agents]

    # return all adversarial agents
    def adversaries(self, world):
        return [landmark for landmark in world.landmarks if landmark.adversary]

    def leaders(self, world):
        return [agent for agent in world.agents if agent.leader]

    def reward(self, agent, world):  # 奖励
        # Agents are rewarded based on minimum agent distance to each landmark, penalized for collisions
        rew = 0
        agents = self.good_agents(world)
        adversaries = self.adversaries(world)

        shape1=False
        shape2=False
        shape3=False
        for i, a in enumerate(world.agents):
          if i==0:
              h1=a.state.p_pos
          elif i==1:
              h2=a.state.p_pos
          elif i==2:
              h3=a.state.p_pos
        H1=np.sqrt(np.sum(np.square(h1-h2)))
        H2 = np.sqrt(np.sum(np.square(h1 - h3)))
        H3 = np.sqrt(np.sum(np.square(h2 - h3)))
        if H1<=0.5  and h1[0]>h2[0] and h1[1]>h2[1]:
           rew+=1
           shape1=True
        if H2<=0.5  and h1[0]<h3[0] and h1[1]>h3[1]:
            rew+=1
            shape2=True
        if H3<=0.5  and h2[0]<h3[0] and h2[1]==h3[1]:
           rew+=0.3
           shape3=True
        # if shape1 ==True and shape2==True and shape3==True:
        #     rew +=1
        for i, a in enumerate(world.agents):
            # for l in world.landmarks:
                if i == 0:
                    dists = np.sqrt(np.sum(np.square(a.state.p_pos - world.landmarks[1].state.p_pos)))
                    # if dists<0.01:
                    #     rew + =0.8
        rew -= dists * 0.7


        if agent.collide:
            for a in world.agents:
                if agent is a :continue
                if self.is_collision(a, agent):
                    rew -= 10


        if agent.collide:
            # for a in adversaries:
                dist1 = np.sum(np.square(agent.state.p_pos - world.landmarks[0].state.p_pos))
                if self.is_collision(world.landmarks[0], agent):
                    rew -= 9  #这个可以适当增加一些
        for adv in adversaries:
            d = np.sqrt(np.sum(np.square(agent.state.p_pos - adv.state.p_pos)))
        # if d>0.2 :
        #    rew += 2 * np.sqrt(np.sum(np.square(agent.state.p_pos - adv.state.p_pos)))
        #     if d <= 0.28:
        #       rew -= 2.5
        return rew

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        entity_vel=[]
        for entity in world.landmarks:  # world.entities:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)
            entity_vel.append(entity.state.p_vel)
        # entity colors
        entity_color = []
        for entity in world.landmarks:  # world.entities:
            entity_color.append(entity.color)
        # communication of all other agents
        comm = []
        other_pos = []
        for other in world.agents:
            if other is agent: continue
            comm.append(other.state.c)
            other_pos.append(other.state.p_pos - agent.state.p_pos)
        return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos + comm)
