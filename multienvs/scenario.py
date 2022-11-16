# 包含为所有场景扩展的基本场景对象。
# defines scenario upon which the world is built
class BaseScenario(object):
    # create elements of the world
    def make_world(self):
        raise NotImplementedError()  # 表示如果这个方法没有被子类重写，但是调用了，就会报错。

    # create initial conditions of the world
    def reset_world(self, world):
        raise NotImplementedError()
