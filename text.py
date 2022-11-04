import time

start_time = time.time()  # 获取初始时间,构建时间流
t = [0.0]  # 设置t_0为0


def step(m_i2):
    while True:
        curr_time = time.time() - start_time  # 获取当前的时间t
        e_iw_t = abs(u_i_ba(curr_time) - u_i_ba(t[-1]))  # 计算e_iwt
        if e_iw_t >= m_i2:  # 判断是否到达设定阈值
            t.append(curr_time)  # 将当前时间节点记为下一阶段的初始点，即t_k
            print(t)
            break


def u_i_ba(tt):  # u_i_ba 函数的表达式
    tt = tt * 2
    return tt


if __name__ == '__main__':
    while True:
        step(5)
