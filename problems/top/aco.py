import time
import math
import random
import warnings
import numpy as np
warnings.simplefilter("ignore", category=RuntimeWarning)


class ACO:
    def __init__(self, vehicle_num, target_num, vehicle_speed, target, time_lim, return2depot=False):
        self.num_type_ant = vehicle_num
        self.num_city = target_num + 1  # number of cities
        self.group = 200
        self.num_ant = self.group * self.num_type_ant  # number of ants
        self.ant_vel = vehicle_speed
        self.cut_time = time_lim
        self.return2depot = return2depot
        self.oneee = np.zeros((4, 1))
        self.target = target
        self.alpha = 1  # pheromone
        self.beta = 2
        self.k1 = 0.03
        self.iter_max = 30

    def name(self):
        return "ACO"

        # matrix of the distances between cities

    def distance_matrix(self):
        dis_mat = []
        for i in range(self.num_city):
            dis_mat_each = []
            for j in range(self.num_city):
                dis = math.sqrt(
                    pow(self.target[i][0] - self.target[j][0], 2) + pow(self.target[i][1] - self.target[j][1], 2))
                dis_mat_each.append(dis)
            dis_mat.append(dis_mat_each)
        return dis_mat

    def run(self):
        try:
            # print("ACO start, pid: %s" % os.getpid())
            start_time = time.time()
            # distances of nodes
            dis_list = self.distance_matrix()
            dis_mat = np.array(dis_list)
            value_init = self.target[:, 2].transpose()
            delay_init = self.target[:, 3].transpose()
            pheromone_mat = np.ones((self.num_type_ant, self.num_city, self.num_city))
            # velocity of ants
            path_new = [[0] for i in range(self.num_type_ant)]
            count_iter = 0
            while count_iter < self.iter_max:
                path_sum = np.zeros((self.num_ant, 1))
                time_sum = np.zeros((self.num_ant, 1))
                value_sum = np.zeros((self.num_ant, 1))
                path_mat = [[0] for i in range(self.num_ant)]
                value = np.zeros((self.group, 1))
                atten = np.ones((self.num_type_ant, 1)) * 0.2
                for ant in range(self.num_ant):
                    ant_type = ant % self.num_type_ant
                    visit = 0
                    if ant_type == 0:
                        unvisit_list = list(range(1, self.num_city))  # have not visit
                    for j in range(1, self.num_city):
                        if len(unvisit_list) == 0:
                            break
                        # choice of next city
                        trans_list = []
                        tran_sum = 0
                        trans = 0
                        # if len(unvisit_list)==0:
                        # print('len(unvisit_list)==0')
                        for k in range(len(unvisit_list)):  # to decide which node to visit
                            trans += np.power(pheromone_mat[ant_type][visit][unvisit_list[k]], self.alpha) * \
                                     np.power(
                                         value_init[unvisit_list[k]] * self.ant_vel[ant_type] / (
                                                 dis_mat[visit][unvisit_list[k]]  # * delay_init[unvisit_list[k]] + 1e-15
                                         ),
                                         self.beta
                                     )
                            # trans +=np.power(pheromone_mat[ant_type][unvisit_list[k]],self.alpha)*np.power(0.05*value_init[unvisit_list[k]],self.beta)
                            trans_list.append(trans)
                        tran_sum = trans
                        rand = random.uniform(0, tran_sum)  # seed 3546
                        if rand > max(trans_list):
                            rand = random.uniform(min(trans_list), tran_sum) if tran_sum > 0 else random.uniform(max(trans_list), tran_sum)
                        # rand = random.uniform(min(trans_list), tran_sum)
                        for t in range(len(trans_list)):
                            if rand <= trans_list[t]:
                                visit_next = unvisit_list[t]
                                break
                            else:
                                # visit_next = random.choice(unvisit_list)
                                continue
                        path_mat[ant].append(visit_next)
                        path_sum[ant] += dis_mat[path_mat[ant][j - 1]][path_mat[ant][j]]
                        time_sum[ant] = path_sum[ant] / self.ant_vel[ant_type]  # + delay_init[visit_next]
                        t_flyback = dis_mat[path_mat[ant][0]][visit_next] / self.ant_vel[ant_type] \
                            if self.return2depot else 0
                        if time_sum[ant] + t_flyback > self.cut_time:
                            time_sum[ant] -= path_sum[ant] / self.ant_vel[ant_type]  # + delay_init[visit_next]
                            path_mat[ant].pop()
                            break
                        value_sum[ant] += value_init[visit_next]
                        unvisit_list.remove(visit_next)  # update
                        visit = visit_next
                    if ant_type == self.num_type_ant - 1:
                        small_group = int(ant / self.num_type_ant)
                        for k in range(self.num_type_ant):
                            value[small_group] += value_sum[ant - k]
                # iteration
                if count_iter == 0:
                    value_new = max(value)
                    value = value.tolist()
                    for k in range(0, self.num_type_ant):
                        path_new[k] = path_mat[value.index(value_new) * self.num_type_ant + k]
                        path_new[k].remove(0)
                else:
                    if max(value) > value_new:
                        value_new = max(value)
                        value = value.tolist()
                        for k in range(0, self.num_type_ant):
                            path_new[k] = path_mat[value.index(value_new) * self.num_type_ant + k]
                            path_new[k].remove(0)

                # update pheromone
                pheromone_change = np.zeros((self.num_type_ant, self.num_city, self.num_city))
                for i in range(self.num_ant):
                    length = len(path_mat[i])
                    m = i % self.num_type_ant
                    n = int(i / self.num_type_ant)
                    for j in range(length - 1):
                        pheromone_change[m][path_mat[i][j]][path_mat[i][j + 1]] += value_init[path_mat[i][j + 1]] * \
                               self.ant_vel[m] / (
                                       dis_mat[path_mat[i][j]][path_mat[i][j + 1]]
                                       # * delay_init[path_mat[i][j + 1]] + 1e-15
                               )
                    atten[m] += (value_sum[i] / (np.power((value_new - value[n]), 4) + 1)) / self.group

                for k in range(self.num_type_ant):
                    pheromone_mat[k] = (1 - atten[k]) * pheromone_mat[k] + pheromone_change[k]
                count_iter += 1
        except Exception as ex:
            print(str(ex))
        # print("ACO result:", path_new)
        end_time = time.time()
        # print("ACO time:", end_time - start_time)
        return path_new, end_time - start_time
