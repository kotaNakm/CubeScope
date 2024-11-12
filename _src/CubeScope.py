""" Python implementation of CubeScope """

import numpy as np
from scipy.special import digamma, gammaln
from tqdm import trange, tqdm
import numba
import copy
from itertools import groupby
import time
import warnings

warnings.simplefilter("ignore")

FB = 8
# FB = 2, 4, 8, 16, 32

ZERO = 1.0e-8
REGIME_R = 5.0e-3
LAMBDA = 0.1
MAX_INI_r = 2
SEED = 0
TOL_R = 2 ** (-FB)

class Regime(object):
    def __init__(self):
        self.costM = np.inf
        self.costC = np.inf
        self.costT = np.inf

    def compute_costM(self, cnt, verbose: bool):
        if cnt == 0:
            return np.inf
        cost = 0

        each_non_zero_counts = np.zeros(self.n_modes)

        non_zeros = np.sum((self.factors[0] > TOL_R).astype(int), axis=0)
        non_zeros[np.argmax(non_zeros)] = 0
        each_non_zero_counts[0] = non_zeros.sum()

        for mode_ in range(1, self.n_modes):
            non_zeros = np.sum((self.factors[mode_] > TOL_R).astype(int), axis=1)
            non_zeros[np.argmax(non_zeros)] = 0
            each_non_zero_counts[mode_] = non_zeros.sum()

        if verbose:
            print(f"each_non_zero_counts: {each_non_zero_counts}")
            print(f"n_dims: {self.n_dims}")
            print(f"FB: {FB}, TOL_R: {TOL_R}")
        self.n_dims_refined = copy.deepcopy(self.n_dims)
        self.n_dims_refined[1:][self.n_dims_refined[1:] < 2] = 2  # avoid log(0)
        cost += each_non_zero_counts[0] * (
            np.log2((self.k - 1) * self.n_dims_refined[0]) + FB
        )
        cost += log_s(each_non_zero_counts[0])
        for mode_ in range(1, self.n_modes):
            cost += each_non_zero_counts[mode_] * (
                np.log2(self.k * (self.n_dims_refined[mode_] - 1)) + FB
            )
            cost += log_s(each_non_zero_counts[mode_])
        self.costM = cost
        return cost

    def compute_costC(self, X):
        if len(X) == 0:
            return 0

        self.factors = _normalize_factors(self.factors)
        return _compute_costC(X.to_numpy(), self.factors)

    def log_likelihood(
        self,
    ):
        def log_multi_beta(param, K=None):
            """
            Logarithm of the multivariate beta function.
            """
            if K is None:
                # param is assumed to be a vector
                return np.sum(gammaln(param)) - gammaln(np.sum(param))
            else:
                # param is assumed to be a scalar
                return K * gammaln(param) - gammaln(K * param)

        llh = 0
        for i in range(self.n_dims[0]):
            llh += log_multi_beta(self.counterM[0][i, :] + self.alpha[i])
            llh -= log_multi_beta(self.alpha[i], self.k)  # ??

        for mode_ in range(self.n_modes - 1):
            for i in range(self.k):
                llh += log_multi_beta(
                    self.counterM[mode_ + 1][:, i] + self.betas[mode_][i]
                )
                llh -= log_multi_beta(self.betas[mode_][i], self.n_dims[0])

        return llh

    def compute_factors(self):
        self.factors[0] = (
            (self.counterM[0][:] + self.prev_terms[0][:]).T
            / (self.counterA + self.l * self.alpha)
        ).T

        for mode_ in range(1, self.n_modes):
            self.factors[mode_] = (self.counterM[mode_] + self.prev_terms[mode_]) / (
                self.counterK + self.l * self.betas[mode_ - 1]
            )

        self.factors = _normalize_factors(self.factors)
        return self.factors


class CubeScope(object):
    def __init__(
        self,
        tensor,
        k: int,
        width: int,
        init_len: int,
        outputdir: str,
        args: object,
        verbose: bool,
        keep_best_factors: bool = True,
        early_stoppping: bool = False,
        tensor_shape=[],
    ):
        # initialze
        self.k = k  # # of topics/components
        self.width = width
        self.init_len = init_len
        self.n_dims = tensor.max().values + 1
        self.n_modes = len(self.n_dims)

        self.categorical_idxs = (
            args.categorical_idxs.split("/") if len(args.categorical_idxs) > 0 else []
        )

        self.n_dims[0] = width

        self.n_dims = self.n_dims.astype(int)

        print("##tensor dimensions##")
        print(self.n_dims)

        self.cur_n = 0

        self.init_params(args.alpha, args.beta)  # init alpha, beta and factors
        self.regimes = []

        self.keep_best_factors = keep_best_factors
        self.early_stoppping = early_stoppping

        # for anormaly detection
        self.anomaly = args.anomaly
        if self.anomaly:
            self.anomaly_scores = []

        # for visualization
        self.outputdir = outputdir
        self.verbose: bool = verbose
        self.sampling_log_likelihoods = []

        # for update parameters
        self.max_alpha = 100
        self.max_beta = 100

    def init_params(self, alpha, beta):
        """Initialize alpha, beta and factors"""
        self.alpha = np.full(self.n_dims[0], alpha)
        self.betas = [np.full(self.k, beta) for _ in range(self.n_modes - 1)]
        self.factors = [np.full((d, self.k), 1, dtype=float) for d in self.n_dims]

    def init_status(self, tensor):
        """Initialize Counters for current tensor"""
        self.counterM = [np.zeros((d, self.k), dtype=int) for d in self.n_dims]
        self.counterK = np.zeros(self.k, dtype=int)
        self.counterA = np.zeros(self.n_dims[0], dtype=int)

        self.n_events = len(tensor)
        self.assignment = np.full(self.n_events, -1, dtype=int)

        Asum = tensor.groupby(self.keys[0]).size()
        self.counterA[Asum.index] = Asum.values

    def init_infer(
        self,
        tensor_train,
        n_iter=10,
        init_l: bool = True,
        return_inference_time=False,
        calc_computational_cost=False,
    ):
        """Initialize model parameters i.e, training process
        1. batch estimation for each subtensor
        2. Initialize model parameters employing subtensors given by 1.
        """
        self.time_idx = list(tensor_train.columns)[0]
        self.keys = list(tensor_train.keys())

        self.l = int(tensor_train[self.time_idx].max() // self.width)
        # the oldest regime index is assigned 0, l is the newest index
        self.prev_distributions = [
            np.full((self.l, d, self.k), 1, dtype=float) for d in self.n_dims
        ]

        # self.l = len(tensor_train) // self.width
        print(f"l: {self.l}")

        if self.l < 1:
            print("Data[:init_len] does not have enough data for initialization")
            print("Inlier records in data[:init_len] must be longer than width")
            exit()

        # 1. process for subtensors that given by train tensor devided by l
        best_llh_in_L = -np.inf
        best_l = 0

        for ini in range(self.l):
            tensor = tensor_train[
                (tensor_train[self.time_idx] >= ini * self.width)
                & (tensor_train[self.time_idx] < (ini + 1) * self.width)
            ]
            tensor.loc[:, self.time_idx] -= ini * self.width
            print(tensor)
            cnt = len(tensor)
            self.init_status(tensor)

            if self.verbose:
                print("Gibbs Sampling")
            self.each_samp_llh = []
            best_llh = -np.inf

            start_inference_time = time.process_time()
            for iter_ in range(n_iter):
                self.assignment = self.sample_topic(tensor.to_numpy())

                # compute log likelihood
                llh = self.log_likelihood()
                if self.verbose or iter_ == 0 or iter_ == (n_iter - 1):
                    print(f"llh_{iter_+1}: {llh}")
                self.each_samp_llh.append(llh)

                if (self.early_stoppping) and (
                    self.each_samp_llh[-2] > self.each_samp_llh[-1]
                ):
                    break

                if (self.keep_best_factors) and (llh > best_llh):
                    best_counterM = copy.deepcopy(self.counterM)
                    best_counterK = copy.deepcopy(self.counterK)
                    best_llh = llh
            inference_time = time.process_time() - start_inference_time
            if calc_computational_cost:
                return 0, inference_time
            # print(self.assignment)
            if self.keep_best_factors:
                self.counterM = best_counterM
                self.counterK = best_counterK
                print(f"best llh: {self.log_likelihood()}")

            self.sampling_log_likelihoods.append(self.each_samp_llh)
            self.compute_factors_batch()
            self.update_prev_dist_init(cnt, ini)

            if best_llh_in_L < best_llh:  # higher is better
                best_l = ini

        # # choice only best prev distribution
        initial_prev_dist = [
            self.prev_distributions[m][best_l] for m in range(self.n_modes)
        ]
        self.prev_distributions = [
            np.full((self.l, d, self.k), 0, dtype=float) for d in self.n_dims
        ]
        for m in range(self.n_modes):
            self.prev_distributions[m][0] = initial_prev_dist[m]
        self.vscost_log = []

        # 2. initialize model parameters
        if return_inference_time:
            return self.model_initialization(tensor_train, n_iter), inference_time
        else:
            return self.model_initialization(
                tensor_train,
                n_iter,
            )

    def model_initialization(
        self,
        tensor_all,
        n_iter,
    ):
        """
        * infer tracking factors using factors derived by batch estimation (init_infer)
        * determine optimal l
        """

        def segment_len(rgm_nums, alloc):
            """
            return a half of max segment length
            return a max segment length
            """
            max_L = 0
            for num in rgm_nums:
                # print(num)
                tmp = np.where(alloc == num, 1, -1)
                dst = [
                    sum(1 for e in it) for _, it in groupby(tmp, key=lambda x: x > 0)
                ]
                # print(dst)
                if tmp[0] > 0:
                    max_ = np.max(dst[::2])
                else:
                    max_ = np.max(dst[1::2])

                if max_ > max_L:
                    max_L = max_
            return int(max_L)

        ini_regimes = []
        regimes_cost = {}
        for ini in range(self.l):
            # get partial tensor as current tensor
            tensor = tensor_all[
                (tensor_all[self.time_idx] >= ini * self.width)
                & (tensor_all[self.time_idx] < (ini + 1) * self.width)
            ]
            cnt = len(tensor)
            tensor.loc[:, self.time_idx] -= ini * self.width

            self.init_status(tensor)

            if self.verbose:
                print("Online Gibbs Sampling")
            print(f"# of events: {cnt}")
            self.each_samp_llh = []
            best_llh = -np.inf
            for iter_ in range(n_iter):
                self.assignment, self.prev_terms = self.sample_topic_online(
                    tensor.to_numpy()
                )

                # compute log likelihood
                llh = self.log_likelihood()
                if self.verbose or iter_ == 0 or iter_ == (n_iter - 1):
                    print(f"llh_{iter_+1}: {llh}")
                self.each_samp_llh.append(llh)

                if (self.early_stoppping) and (
                    self.each_samp_llh[-2] > self.each_samp_llh[-1]
                ):
                    break

                if (self.keep_best_factors) and (llh > best_llh):
                    best_counterM = copy.deepcopy(self.counterM)
                    best_counterK = copy.deepcopy(self.counterK)
                    best_llh = llh

            if self.keep_best_factors:
                self.counterM = best_counterM
                self.counterK = best_counterK
                print(f"best llh: {best_llh}")
            self.sampling_log_likelihoods.append(self.each_samp_llh)
            self.compute_factors()
            self.update_prev_dist_init(cnt, ini)

            # find opt regime
            # choose the regime which has most lowest coding cost
            tmp_rgm = Regime()
            tmp_rgm = self.regime_initialize(tmp_rgm)

            tmp_costC = tmp_rgm.compute_costC(tensor)
            tmp_costM = tmp_rgm.compute_costM(len(tensor), verbose=self.verbose)

            ini_regimes.append(tmp_rgm)
            regimes_cost[ini] = tmp_costC + tmp_costM
            print(f"initialized costC for train segment #{ini}: {tmp_costC}")
            print(f"initialized costM for train segment #{ini}: {tmp_costM}")

        if self.anomaly:
            self.aggregate_initials(ini_regimes)
            self.l = 1
            regime_assignments = [[0, 0]]
            self.regimes.append(self.all_comp_regime)
            # update prev dist by all_comp regime as a regime
            for mode_ in range(self.n_modes):
                self.prev_distributions[mode_][0, :, :] = copy.deepcopy(
                    self.all_comp_regime.factors[mode_]
                )
            self.prev_distributions = [
                copy.deepcopy(prev_dist_mode[:1])
                for prev_dist_mode in self.prev_distributions
            ]
            self.cur_n = self.init_len
            self.prev_rgm_id = 0
            self.n_segment = 0

            return regime_assignments

        # determine initial regime set and regime assignments by computing total cost for train tensor
        opt_rgm_nums, opt_alloc = self.determine_initial_regimeset_assignments(
            regimes_cost, tensor_all, ini_regimes
        )

        # add optimal regimes to regime set for stream process
        for i, r_id in enumerate(opt_rgm_nums):
            self.regimes.append(ini_regimes[r_id])
            opt_alloc[opt_alloc == r_id] = i

        self.prev_rgm_id = opt_alloc[-1]

        # determine optimal l following segment length and update prev_dist
        L = segment_len(range(len(opt_rgm_nums)), opt_alloc)
        self.l = L
        print(f"dependencies length L:{L}")
        self.prev_distributions = [
            copy.deepcopy(prev_dist_mode[:L])
            for prev_dist_mode in self.prev_distributions
        ]

        regime_assignments = [[0, 0]]

        return regime_assignments

    def determine_initial_regimeset_assignments(
        self, regimes_cost, tensor_all, ini_regimes
    ):
        cost_sorted = sorted(
            regimes_cost.items(), key=lambda x: x[1]
        )  # order of regime numbers which has smaller cost
        min_ = np.inf
        opt_alloc = np.full(len(ini_regimes), 0)
        opt_rgm_nums = [
            0,
        ]
        for max_r_num in range(1, MAX_INI_r + 1):
            candidate_rgm_nums = [num[0] for num in cost_sorted[:max_r_num]]
            print(f"candidate regimes for initial regimes:{candidate_rgm_nums}")
            cost, alloc = self.compute_total_cost_batch(
                tensor_all, ini_regimes, candidate_rgm_nums
            )

            if cost < min_:
                opt_alloc = alloc
                opt_rgm_nums = np.unique(alloc)
                min_ = cost
            else:
                print("break")
                print(f"top samllest {max_r_num-1} regimes")
                print(f"MAX_INI_r: {MAX_INI_r}")
                break
        # count segment
        self.n_segment = np.count_nonzero(
            np.array(opt_alloc[1:]) - np.array(opt_alloc[:-1])
        )

        print(f"initial regimes: {opt_rgm_nums}")
        print(f"cost and assignment: {min_} & {opt_alloc}")

        return opt_rgm_nums, opt_alloc

    def compute_total_cost_batch(self, tensor_all, ini_regimes, cadidate_rgm_nums):
        """
        compute total cost employed ini_regimes[candidate_rgm_nums] and assign them
        Return:
        cost: total cost computed by ini_regimes[candidate_rgm_nums]
        alloc: candidate regime assignments
        """

        alloc = np.full(self.l, -1)
        cost = 0
        # calc total coding cost and assign
        for ini in range(self.l):
            min_ = np.inf
            target_tensor = tensor_all[
                (tensor_all[self.time_idx] >= ini * self.width)
                & (tensor_all[self.time_idx] < (ini + 1) * self.width)
            ]
            target_tensor.loc[:, self.time_idx] -= ini * self.width
            for r in cadidate_rgm_nums:
                r_cost = ini_regimes[r].compute_costC(target_tensor)
                if min_ > r_cost:
                    alloc[ini] = r
                    min_ = r_cost
            cost += min_
        # add model cost to cost
        for r_id in np.unique(alloc):
            cost += ini_regimes[r_id].costM
        return cost, alloc

    def infer_online(self, tensor, alpha, beta, n_iter=10):
        # whether apply tuned parameter in initialization
        if True:
            self.init_params(alpha, beta)

        self.init_status(tensor)

        self.cur_n += self.width
        if self.verbose:
            print("Online Gibbs Sampling")
        print(f"# of events: {len(tensor)}")

        self.each_samp_llh = []
        best_llh = -np.inf

        start_time = time.process_time()
        for iter_ in range(n_iter):
            self.assignment, self.prev_terms = self.sample_topic_online(
                tensor.to_numpy()
            )
            # compute log likelihood
            llh = self.log_likelihood()
            if self.verbose or iter_ == 0 or iter_ == (n_iter - 1):
                print(f"llh_{iter_+1}: {llh}")
            self.each_samp_llh.append(llh)

            if (self.early_stoppping) and (
                self.each_samp_llh[-2] > self.each_samp_llh[-1]
            ):
                break
            if (self.keep_best_factors) and (llh > best_llh):
                best_counterM = copy.deepcopy(self.counterM)
                best_counterK = copy.deepcopy(self.counterK)
                best_llh = llh
        decomp_time = time.process_time() - start_time
        print(f"decomp_time:{decomp_time}")

        start_time = time.process_time()
        if self.keep_best_factors:
            self.counterM = best_counterM
            self.counterK = best_counterK
            print(f"best llh: {self.log_likelihood()}")

        self.sampling_log_likelihoods.append(self.each_samp_llh)
        self.cnt = len(tensor)
        self.compute_factors()
        self.update_prev_dist(self.cnt, self.l)

        time_mat = self.factors[0]

        shift_id = self.model_compressinon(self.cur_n, tensor)

        compress_time = time.process_time() - start_time
        print(f"compress_time:{compress_time}")

        return shift_id

    def sample_topic(self, X):
        return _gibbs_sampling(
            X,
            self.assignment,
            self.counterM,
            self.counterK,
            self.counterA,
            self.alpha,
            self.betas,
            self.k,
            self.n_dims,
        )

    def sample_topic_online(self, X):
        return _gibbs_sampling_online(
            X,
            self.assignment,
            self.counterM,
            self.counterK,
            self.counterA,
            self.alpha,
            self.betas,
            self.k,
            self.prev_distributions,
            self.l,
            self.n_dims,
        )

    def log_likelihood(
        self,
    ):
        def log_multi_beta(param, K=None):
            """
            Logarithm of the multivariate beta function.
            """
            if K is None:
                # param is assumed to be a vector
                return np.sum(gammaln(param)) - gammaln(np.sum(param))
            else:
                # param is assumed to be a scalar
                return K * gammaln(param) - gammaln(K * param)

        llh = 0
        for i in range(self.n_dims[0]):
            llh += log_multi_beta(self.counterM[0][i, :] + self.alpha[i])
            llh -= log_multi_beta(self.alpha[i], self.k)  # ??

        for mode_ in range(self.n_modes - 1):
            for i in range(self.k):
                llh += log_multi_beta(
                    self.counterM[mode_ + 1][:, i] + self.betas[mode_][i]
                )
                llh -= log_multi_beta(self.betas[mode_][i], self.n_dims[0])

        return llh

    def update_prev_dist_init(self, cnt, l):
        """add estimated factors to prev_dist[l]"""
        if cnt:
            for mode_ in range(self.n_modes):
                self.prev_distributions[mode_][l, :, :] = copy.deepcopy(
                    self.factors[mode_]
                )
        # for visualize
        if False:
            print(f"prev distributions for {l}")
            for mode_ in range(self.n_modes):
                print(self.prev_distributions[mode_])

    def update_prev_dist(self, cnt, l):
        """
        push and pop prev_dist queue
        """

        if cnt:
            for mode_ in range(self.n_modes):
                self.prev_distributions[mode_][:-1] = copy.deepcopy(
                    self.prev_distributions[mode_][1:]
                )
                self.prev_distributions[mode_][-1, :, :] = copy.deepcopy(
                    self.factors[mode_]
                )

    def compute_factors_batch(self):
        for i in range(self.n_dims[0]):
            self.factors[0][i, :] = (self.counterM[0][i, :] + self.alpha[i]) / (
                self.counterA[i] + self.alpha[i] * self.k
            )

        for mode_ in range(1, self.n_modes):
            for i in range(self.n_dims[mode_]):
                self.factors[mode_][i, :] = (
                    self.counterM[mode_][i, :] + self.betas[mode_ - 1]
                ) / (self.counterK + self.betas[mode_ - 1] * self.n_dims[mode_])
        self.factors = _normalize_factors(self.factors)
        return self.factors

    def compute_factors(self):
        self.factors[0] = (
            (self.counterM[0][:] + self.prev_terms[0][:]).T
            / (self.counterA + self.l * self.alpha)
        ).T

        for mode_ in range(1, self.n_modes):
            self.factors[mode_] = (self.counterM[mode_] + self.prev_terms[mode_]) / (
                self.counterK + self.l * self.betas[mode_ - 1]
            )

        self.factors = _normalize_factors(self.factors)
        return self.factors

    def aggregate_initials(self, init_regimes):
        tmp_rgm = Regime()
        tmp_rgm = self.regime_initialize(tmp_rgm)

        all_count = 0
        for ini_rgm in init_regimes:
            tmp_rgm.counterM += ini_rgm.counterM
            tmp_rgm.counterK += ini_rgm.counterK
            tmp_rgm.counterA += ini_rgm.counterA
            all_count += ini_rgm.counterK.sum()
        tmp_rgm.compute_factors()

        tmp_rgm.counterM = [m.astype(float) for m in ini_rgm.counterM]
        tmp_rgm.counterK = ini_rgm.counterK.astype(float)
        tmp_rgm.counterA = ini_rgm.counterA.astype(float)
        self.all_comp_regime = tmp_rgm
        self.all_count = all_count

    def model_compressinon(self, cur_n, X):
        shift_id = False
        self.n_regime = len(self.regimes)
        prev_rgm = self.regimes[self.prev_rgm_id]
        candidate_rgm = Regime()
        candidate_rgm = self.regime_initialize(candidate_rgm)

        ## compute_costM
        costM = candidate_rgm.compute_costM(self.cnt, verbose=self.verbose)
        costM += log_s(self.n_regime + 1) - log_s(self.n_regime)
        costM += (
            (log_s(self.n_segment + 1) - log_s(self.n_segment))
            + log_s(self.cnt)
            + np.log2(self.n_regime)
        )
        costC = candidate_rgm.compute_costC(X)
        cost_1 = costC + costM

        print(f"cand costC: {costC}")
        print(f"cand costM: {costM}")
        print(f"cand costT: {cost_1}")

        cost_0 = prev_rgm.compute_costC(X)
        print(f"prev costT: {cost_0}")

        self.vscost_log.append([cost_0, cost_1, cost_1 - cost_0, costC, costM])

        print("=========================================")
        print(f"Previous vs Candidate")
        print(f"{cost_0} vs {cost_1}")
        print("=========================================")
        print(f"diff::{cost_1 - cost_0}")

        if cost_0 < cost_1:  # stay on previous regime
            print("STAY")
            prev_rgm = self.regimes[self.prev_rgm_id]
            self.regime_update(self.prev_rgm_id)


        else:  # shift to any regime
            self.n_segment += 1
            shift_id = len(self.regimes)  # index + 1
            min_ = cost_1 + REGIME_R * cost_1
            add_flag = True

            # regime comparison
            for rgm_id, rgm in enumerate(self.regimes):
                if rgm_id == self.prev_rgm_id:
                    continue
                else:
                    rgm_costC = rgm.compute_costC(X)
                    rgm_costM = (
                        (log_s(self.n_segment + 1) - log_s(self.n_segment))
                        + log_s(self.cnt)
                        + np.log2(self.n_regime)
                    )
                    cost_0 = rgm_costC + rgm_costM
                    if cost_0 < min_:
                        shift_id = rgm_id
                        add_flag = False
                        min_ = cost_0

            print(f"SHIFT at {cur_n}")
            print(f"{self.prev_rgm_id}===>>>{shift_id}")

            if add_flag:  # add candidate regime to regime set
                self.regimes.append(candidate_rgm)
                self.prev_rgm_id = shift_id

            else:  # use existing regime
                self.regime_update(shift_id)
                shift_rgm = self.regimes[shift_id]
                self.alpha = copy.deepcopy(shift_rgm.alpha)
                self.betas = copy.deepcopy(shift_rgm.betas)
                self.prev_rgm_id = shift_id

        # For anormaly detection
        if self.anomaly:
            current_count = self.n_events
            self.all_count += current_count
            # only if current regime is initial regime
            if self.prev_rgm_id == 0:
                self.all_comp_regime.counterM = [
                    m_comp + m
                    for m_comp, m in zip(self.all_comp_regime.counterM, self.counterM)
                ]
                self.all_comp_regime.counterK = (
                    self.all_comp_regime.counterK
                ) + self.counterK
                self.all_comp_regime.counterA = (
                    self.all_comp_regime.counterA
                ) + self.counterA
                self.all_comp_regime.compute_factors()
            observed_cost = costC
            expected_cost = self.all_comp_regime.compute_costC(X)

            a_score = self.calc_anomaly_score(observed_cost, expected_cost, len(X))
            self.anomaly_scores.append(a_score)
            print("===Anomaly score===")
            print(a_score)

        return shift_id

    def calc_anomaly_score(self, observed_cost, expected_cost, len_X):
        """calculate anomaly score by chi-squared static"""
        if expected_cost == 0:
            return 0
        else:
            a_scroe = expected_cost / len_X

        return a_scroe

    def regime_initialize(self, regime_instance):
        regime_instance.k = self.k
        regime_instance.n_dims = self.n_dims
        regime_instance.n_modes = self.n_modes
        regime_instance.l = self.l
        regime_instance.factors = copy.deepcopy(self.factors)
        regime_instance.counterM = copy.deepcopy(self.counterM)
        regime_instance.counterK = copy.deepcopy(self.counterK)
        regime_instance.counterA = copy.deepcopy(self.counterA)
        regime_instance.alpha = self.alpha
        regime_instance.betas = self.betas
        regime_instance.prev_terms = self.prev_terms

        return regime_instance

    def regime_update(self, rgm_id):
        regime = self.regimes[rgm_id]

        regime.counterK = regime.counterK.astype(float)
        regime.counterK += np.round(self.counterK * LAMBDA)
        regime.counterA = regime.counterA.astype(float)
        regime.counterA += np.round(self.counterA * LAMBDA)

        for mode_ in range(self.n_modes):
            regime.counterM[mode_] = regime.counterM[mode_].astype(float)
            regime.counterM[mode_] += np.round(self.counterM[mode_] * LAMBDA)
        regime.factors = regime.compute_factors()

    def rgm_update_fin(self):
        if self.prev_rgm_id == (len(self.regimes) - 1):
            rgm = self.regimes[self.prev_rgm_id]
            rgm.alpha = copy.deepcopy(self.alpha)
            rgm.betas = copy.deepcopy(self.betas)
            rgm.factors = copy.deepcopy(self.factors)

    def save(self, outdir):
        """
        Save all of parameters for CubeScope
        """
        if len(self.vscost_log) > 0:
            np.savetxt(outdir + "vs_cost.txt", self.vscost_log[-1])
        np.savetxt(outdir + "llh.txt", self.each_samp_llh)
        np.savetxt(outdir + "alpha.txt", self.alpha)
        np.savetxt(outdir + "betas.txt", self.betas)
        for i, M in enumerate(self.factors):
            np.savetxt(outdir + "factor_{}.txt".format(i), M)
        for i, M in enumerate(self.regimes[self.prev_rgm_id].factors):
            np.savetxt(outdir + "c_regime_factor_{}.txt".format(i), M)


@numba.jit(nopython=True)
def _gibbs_sampling(X, Z, counterM, counterK, counterA, alpha, betas, k, n_dims):
    np.random.seed(SEED)
    """
    X: event tensor
    Z: topic/component assignments of the previous iteration
    """
    n_modes = X.shape[1]
    for e, x in enumerate(X):
        # for each non-zero event entry,
        # assign latent component, z
        pre_topic = Z[e]
        if not pre_topic == -1:
            counterK[pre_topic] -= 1
            for mode_, d in enumerate(x):
                counterM[mode_][d, pre_topic] -= 1

        """ compute posterior distribution """
        posts = np.full(k, 1.0, dtype=np.float64)
        posts *= counterM[0][x[0]] + alpha[x[0]]  # return (k,) vector
        posts /= counterA[x[0]] + alpha[x[0]] * k
        for j in range(1, n_modes):
            posts *= counterM[j][x[j]] + betas[j - 1]
            posts /= counterK + betas[j - 1] * n_dims[j]
        posts = posts / posts.sum()

        try:
            new_topic = draw_one(posts)
        except:
            print("cannot calc assignment posterior:")
            return

        Z[e] = new_topic
        counterK[new_topic] += 1
        for mode_, d in enumerate(x):
            counterM[mode_][d, new_topic] += 1

    return Z


@numba.jit(nopython=True)
def _gibbs_sampling_online(
    X, Z, counterM, counterK, counterA, alpha, betas, k, prev_distributions, l, n_dims
):
    np.random.seed(SEED)

    # ready for (hypa * prev)
    prev_terms = []
    for mode_, d in enumerate(n_dims):
        prev_terms.append(np.zeros((d, k)))
    n_modes = X.shape[1]

    for p in range(l):
        for i in range(n_dims[0]):
            prev_terms[0][i, :] += alpha[i] * prev_distributions[0][p, i, :]
        for mode_ in range(1, n_modes):
            for i in range(k):
                prev_terms[mode_][:, i] += (
                    betas[mode_ - 1][i] * prev_distributions[mode_][p][:, i]
                )

    for e, x in enumerate(X):
        # for each non-zero event entry,
        # assign latent topic/component, z
        pre_topic = Z[e]
        if not pre_topic == -1:
            counterK[pre_topic] -= 1
            for mode_, d in enumerate(x):
                counterM[mode_][d, pre_topic] -= 1

        """ compute posterior distribution """
        posts = np.full(k, 1.0, dtype=np.float64)
        posts *= counterM[0][x[0]] + prev_terms[0][x[0]]  # return (k,) vector
        posts /= counterA[x[0]] + l * alpha[x[0]]
        for j in range(1, n_modes):
            posts *= counterM[j][x[j]] + prev_terms[j][x[j]]
            posts /= counterK + l * betas[j - 1]

        posts = posts / posts.sum()
        try:
            new_topic = draw_one(posts)
        except:
            print("cannot calc assignment posterior:")
            print(posts)
            return
        Z[e] = new_topic
        counterK[new_topic] += 1
        for mode_, d in enumerate(x):
            counterM[mode_][d, new_topic] += 1
    return Z, prev_terms


def log_s(x):
    if x == 0:
        return 0
    return 2.0 * np.log2(x) + 1


@numba.jit(nopython=True)
def _compute_costC(X, factors):
    k = factors[0].shape[1]
    all_L = 0
    for x in X:
        val_ = ZERO
        for r in range(k):
            rval_ = 0
            for att_idx, factor in zip(x, factors):
                rval_ += np.log(factor[att_idx, r] + ZERO)
            val_ = rval_ if r == 0 else np.logaddexp(val_, rval_)
        all_L -= val_
    # transform base
    return all_L / np.log(2.0)


# calc log(sum(exp(x))) in numerically stable way
@numba.jit(nopython=True)
def logsumexp(x, y, k):
    if k == 0:
        return y
    if x == y:
        return x + 0.69314718055  # ln(2)
    vmax = x if x > y else y
    vmin = y if x > y else x
    if vmax > (vmin + 50):
        return vmax
    else:
        return vmax + np.log1p(np.exp(vmin - vmax))


# @numba.jit(nopython=True)
def _normalize_factors(factors):
    """
    refine each components
    factors[0]: item-wise topic/component distribution
    factors[1:]: topic/component-wise item distribution
    """

    for d in range(factors[0].shape[0]):
        sum_ = np.sum(factors[0], axis=1) + ZERO
        factors[0] = (factors[0].T / sum_).T

    for ind, factor in enumerate(factors[1:]):
        sum_ = np.sum(factor, axis=0) + ZERO
        factors[ind + 1] /= sum_
    return factors


@numba.jit(nopython=True)
def draw_one(posts):
    residual = np.random.uniform(0, np.sum(posts))
    return_sample = 0
    for sample, prob in enumerate(posts):
        residual -= prob
        if residual < 0.0:
            return_sample = sample
            break
    return return_sample
