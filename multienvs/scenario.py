# 包含为所有场景扩展的基本场景对象。
# defines scenario upon which the world is built
class BaseScenario(object):
    # create elements of the world
    def make_world(self):
        raise NotImplementedError()

    # create initial conditions of the world
    def reset_world(self, world):
        raise NotImplementedError()
