#################################
# Your name: Saar Barak
#################################

import numpy as np
import matplotlib.pyplot as plt
import intervals


class Assignment2(object):
    """Assignment 2 skeleton.

    Please use these function signatures for this assignment and submit this file, together with the intervals.py.
    """

    def sample_from_D(self, m):
        """Sample m data samples from D.
        Input: m - an integer, the size of the data sample.

        Returns: np.ndarray of shape (m,2) :
                A two dimensional array of size m that contains the pairs where drawn from the distribution P.
        """
        x = np.sort(np.random.uniform(size=m))
        I_1 = (x >0.2)&(x < 0.4)
        I_2 = (x >0.6)&(x < 0.8)
        x_intersection = ((~I_1) & (~I_2))
        y = np.array([self.sample_for_y(x_intersection[i]) for i in range(m)]).reshape(m, )
        return np.column_stack((x, y))


    def experiment_m_range_erm(self, m_first, m_last, step, k, T):
        """Runs the ERM algorithm.
        Calculates the empirical error and the true error.
        Plots the average empirical and true errors.
        Input: m_first - an integer, the smallest size of the data sample in the range.
               m_last - an integer, the largest size of the data sample in the range.
               step - an integer, the difference between the size of m in each loop.
               k - an integer, the maximum number of intervals.
               T - an integer, the number of times the experiment is performed.

        Returns: np.ndarray of shape (n_steps,2).
            A two dimensional array that contains the average empirical error
            and the average true error for each m in the range accordingly.
        """
        experiment = np.arange(m_first, m_last + 1, step)
        empirical_err = np.zeros(experiment.shape)
        true_err = np.zeros(experiment.shape)
        for e in range(experiment.shape[0]):
            empirical_err_j = 0
            true_for_e = 0
            for t in range(T):
                sample = self.sample_from_D(experiment[e])
                ERM_intervals, ERM_empirical_err = intervals.find_best_interval(sample[:, 0], sample[:, 1], k)
                empirical_err_j += ERM_empirical_err
                true_for_e += self.calc_true_error(ERM_intervals)
            empirical_err[e] = empirical_err_j / (T * experiment[e])
            true_err[e] = true_for_e / T
        plt.plot(experiment, empirical_err, label='Empirical Error')
        plt.plot(experiment, true_err, label='True Error',color='r')
        plt.legend()
        plt.xlabel("sampales ")
        plt.ylabel("Empirical Error")
        plt.show()
        return np.stack((empirical_err, true_err))

    def experiment_k_range_erm(self, m, k_first, k_last, step):
        """Finds the best hypothesis for k= 1,2,...,10.
        Plots the empirical and true errors as a function of k.
        Input: m - an integer, the size of the data sample.
               k_first - an integer, the maximum number of intervals in the first experiment.
               m_last - an integer, the maximum number of intervals in the last experiment.
               step - an integer, the difference between the size of k in each experiment.

        Returns: The best k value (an integer) according to the ERM algorithm.
        """
        k_choices = np.arange(k_first, k_last + 1, step)
        empirical_err = np.zeros(k_choices.shape[0])
        true_err = np.zeros(k_choices.shape[0])
        sample = self.sample_from_D(m)
        xs = sample[:, 0]
        ys = sample[:, 1]

        for k in range(k_choices.shape[0]):
            ERM_intervals, ERM_empirical_err = intervals.find_best_interval(xs, ys, k_choices[k])
            empirical_err[k] = ERM_empirical_err / m
            true_err[k] = self.calc_true_error(ERM_intervals)

        plt.scatter(k_choices,true_err)
        plt.scatter(k_choices, empirical_err)
        plt.plot(k_choices, empirical_err, label='Empirical Error',color='r')
        plt.plot(k_choices, true_err, label='True Error',color='purple')
        plt.legend()
        plt.title(' empirical and true errors as a function of k')
        plt.xlabel("k")
        plt.show()

        return np.argmin(empirical_err)
        pass

    def experiment_k_range_srm(self, m, k_first, k_last, step):
        """Run the experiment in (c).
        Plots additionally the penalty for the best ERM hypothesis.
        and the sum of penalty and empirical error.
        Input: m - an integer, the size of the data sample.
               k_first - an integer, the maximum number of intervals in the first experiment.
               m_last - an integer, the maximum number of intervals in the last experiment.
               step - an integer, the difference between the size of k in each experiment.

        Returns: The best k value (an integer) according to the SRM algorithm.
        """
        k_choices = np.arange(k_first, k_last + 1, step)
        K_penalty = self.penalty(k_choices, m)
        empirical_err = np.zeros(k_choices.shape[0])
        true_err = np.zeros(k_choices.shape[0])
        sample = self.sample_from_D(m)
        for i in range(k_choices.shape[0]):
            ERM_intervals, ERM_empirical_err = intervals.find_best_interval(sample[:, 0], sample[:, 1], k_choices[i])
            empirical_err[i] = ERM_empirical_err / m
            true_err[i] = self.calc_true_error(ERM_intervals)
        pen_and_emp_err = K_penalty + empirical_err


        plt.scatter(k_choices, empirical_err,color='purple')
        plt.scatter(k_choices, true_err, color='r')
        plt.plot(k_choices, empirical_err, label='Empirical Error')
        plt.plot(k_choices, true_err, label='True Error')
        plt.plot(k_choices, K_penalty, label='Penalty')
        plt.plot(k_choices, pen_and_emp_err, label='Sum of Penalty and Empirical Error')
        plt.legend()
        plt.xlabel("k")
        plt.show()

    def cross_validation(self, m):
        """Finds a k that gives a good test error.
        Input: m - an integer, the size of the data sample.

        Returns: The best k value (an integer) found by the cross validation algorithm.
        """
        sample = self.sample_from_D(m)
        np.random.shuffle(sample)
        validation = sample[:int(m / 5)]
        train = np.array(sorted(sample[int(m / 5):], key=lambda x: x[0]))
        xt = train[:, 0]
        yt = train[:, 1]
        xv = validation[:, 0]
        yv = validation[:, 1]
        e_S2_ERM = []
        for k in range(1, 11):
            ERM_k_intervals, empirical_error_k = intervals.find_best_interval(xt, yt, k)
            e_S2_ERM.append(self.find_validation_error(ERM_k_intervals, xv, yv))
        print(e_S2_ERM)
        return


    #################################
    def sample_for_y(self, bol):
        if bol:
            return np.random.choice([0.0, 1.0], size=1, p=[0.2, 0.8])
        return np.random.choice([0.0, 1.0], size=1, p=[0.9, 0.1])

    def find_validation_error(self, ERM_k_intervals, xv, yv):
        error = 0.0
        for i in range(len(xv)):
            predication_x = 0
            for interval in ERM_k_intervals:
                if xv[i] >= interval[0] and xv[i] >= interval[1]:
                    predication_x += 1
                    break
                if xv[i] < interval[0]:
                    break
            error += (predication_x != yv[i])
        return error / len(xv)

    def calc_true_error(self, intervals1):
        probably1 = [(0, 0.2), (0.4, 0.6), (0.8, 1)]
        probably0 = self.rest_of_intervals(probably1)
        p1 = 0.8
        p0 = 0.1
        ret = 0.0
        intervals0 = self.rest_of_intervals(intervals1)
        for interval in intervals1:
            for i1 in probably1:
                ret += (self.calc_intersect(interval, i1)) * (1 - p1)
            for i0 in probably0:
                ret += (self.calc_intersect(interval, i0)) * (1 - p0)
        for interval in intervals0:
            for i1 in probably1:
                ret += (self.calc_intersect(interval, i1)) * p1
            for i0 in probably0:
                ret += (self.calc_intersect(interval, i0)) * p0
        return ret

    def penalty(self, k, n):
        ret = (2*k + np.log(20))/n
        return 2*np.sqrt(ret)

    def rest_of_intervals(self, intervals1):
        intervals0 = [(0.0, intervals1[0][0])]
        for i in range(len(intervals1) - 1):
            intervals0.append((intervals1[i][1], intervals1[i + 1][0]))
        intervals0.append((intervals1[len(intervals1) - 1][1], 1.0))
        return intervals0

    def calc_intersect(self, interval1, interval2):
        if (interval1[1] <= interval2[0]) or (interval1[0] >= interval2[1]):
            return 0.0
        if (interval1[0] <= interval2[0]) and (interval1[1] >= interval2[1]):
            return interval2[1] - interval2[0]
        if (interval1[0] >= interval2[0]) and (interval1[1] <= interval2[1]):
            return interval1[1] - interval1[0]
        if (interval1[0] <= interval2[0]) and (interval1[1] <= interval2[1]):
            return interval1[1] - interval2[0]
        if (interval1[0] >= interval2[0]) and (interval1[1] >= interval2[1]):
            return interval2[1] - interval1[0]
    #################################


if __name__ == '__main__':
    ass = Assignment2()
    ass.experiment_m_range_erm(10, 100, 5, 3, 100)
    ass.experiment_k_range_erm(1500, 1, 10, 1)
    ass.experiment_k_range_srm(1500, 1, 10, 1)
