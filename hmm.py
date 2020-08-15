# -*- coding: utf-8 -*
# python detect_loss.py

import numpy as np
import pandas as pd



class GenHMM():
	''' generation observation series of hmm '''

	def __init__(self, InitStateProb, StateTrans, StateToObser, T, burnIn_len):
		''' initialization '''
		self.InitStateProb = InitStateProb
		self.StateTrans = StateTrans
		self.StateToObser = StateToObser
		self.T = T 
		self.burnIn_len = burnIn_len
		return


	def run(self):
		''' run '''
		state_series, observation_series = self.gen_main(self.InitStateProb, self.StateTrans, self.StateToObser, self.T, self.burnIn_len)
		return state_series, observation_series


	def gen_initState(self, InitStateProb):
		''' fall in the initial state '''
		InitStateProb_cumsum = np.cumsum(InitStateProb)
		u = np.random.uniform(0, 1, 1)[0]
		return np.argmax(u < InitStateProb_cumsum)

	
	def gen_stateTransform(self, StateTrans, i):
		''' state transform '''
		cumsum = np.cumsum(StateTrans[i])
		u = np.random.uniform(0, 1, 1)[0]
		return np.argmax(u < cumsum)


	def gen_stateToObservation(self, StateToObser, i):
		''' gen observation from state ''' 
		cumsum = np.cumsum(StateToObser[i])
		u = np.random.uniform(0, 1, 1)[0]
		return np.argmax(u < cumsum)


	def gen_main(self, InitStateProb, StateTrans, StateToObser, T, burnIn_len):
		''' main generation process '''
		T += burnIn_len
		state_series = []
		observation_series = []
		state = self.gen_initState(InitStateProb)
		observation = self.gen_stateToObservation(StateToObser, state)
		state_series.append(state)
		observation_series.append(observation)
		for _ in range(T - 1):
			state = self.gen_stateTransform(StateTrans, state)
			observation = self.gen_stateToObservation(StateToObser, state)
			state_series.append(state)
			observation_series.append(observation)
		state_series = np.array(state_series[burnIn_len:], dtype='int32')
		observation_series = np.array(observation_series[burnIn_len:], dtype='int32')
		return state_series, observation_series



class DetectHMM():
	''' hmm predict state '''

	def __init__(self, observation_series, param_type, StateTrans, StateToObser, InitStateProb=[1, 0], fix_InitStateProb=True, stateToObser_0_vector=[0.9, 0.1], state_num=2, observation_num=2, em_iterNum=20):
		''' 
		initialization
		observation_series: observation series
		param_type: different parameter estimating methods  'em_semi', 'em_all', 'input'
		StateTrans: state transform problilities matrix
		StateToObser: probability matrix of state to observaiton
		InitStateProb: probablity of falling into inital state
		fix_InitStateProb: set InitStateProb fixed
		stateToObser_0_vector: if param_type == 'em_semi', then this would be the first line of StateToObser
		state_num: number of state types 
		observation_num: number of observation types
		em_iterNum: iterating times of em algorithm
		'''
		self.observation_series = observation_series
		self.T = len(observation_series)
		self.param_type = param_type
		self.StateTrans = StateTrans
		self.StateToObser = StateToObser
		self.InitStateProb = InitStateProb
		self.fix_InitStateProb = fix_InitStateProb
		self.stateToObser_0_vector = stateToObser_0_vector
		self.state_num = state_num
		self.observation_num = observation_num
		self.em_iterNum = em_iterNum
		return 


	def predict_state(self):
		''' predict state, different parameter estimating methods  '''
		# em algorithm estimate all parameters
		if self.param_type == 'em_all':
			InitStateProb, StateTrans, StateToObser = self.em_main(self.state_num, self.observation_num, self.InitStateProb, self.fix_InitStateProb, self.T, self.observation_series, self.em_iterNum, semi_par=False, stateToObser_0_vector=None)
		# em algorithm estimate part of parameters (fix first line of StateToObser)
		elif self.param_type == 'em_semi':
			InitStateProb, StateTrans, StateToObser = self.em_main(self.state_num, self.observation_num, self.InitStateProb, self.fix_InitStateProb, self.T, self.observation_series, self.em_iterNum, semi_par=True, stateToObser_0_vector=self.stateToObser_0_vector)
		# input parameters, instead of em algorithm
		elif self.param_type == 'input':
			InitStateProb = self.InitStateProb
			StateTrans = self.StateTrans
			StateToObser = self.StateToObser
		# viterbi algorithm for predicting
		p_max, state_series_est = self.viterbi_main(InitStateProb, StateTrans, StateToObser, self.state_num, self.T, self.observation_num, self.observation_series)
		return p_max, state_series_est


	def get_pi_i(self, InitStateProb, i):
		''' get initial state probablity ''' 
		return InitStateProb[i]


	def get_a_ij(self, StateTrans, i, j):
		''' get state transform probabliity '''
		return StateTrans[i][j]


	def get_b_jk(self, StateToObser, j, k):
		''' get state to observation probablity '''
		return StateToObser[j][k]

	 
	def get_stateIndexToLabel(self, state_omega, state):
		''' state index number to state label '''
		return state_omega[state]


	def get_obserIndexToLabel(self, observation_omega, observation):
		''' observation index number to observation label '''
		return observation_omega[observation]


	def get_alpha_ti(self, Alpha, t, i):
		''' get element in forward probalibity matrix '''
		return Alpha[t][i]


	def get_beta_ti(self, Beta, t, i):
		''' get element in backword probablity matrix '''
		return Beta[t][i]


	def prob_alpha(self, InitStateProb, StateTrans, StateToObser, state_num, T, observation_series):
		''' calculate forward probabliity matrix '''
		Alpha = np.zeros(T * state_num).reshape(T, state_num)
		# initial values
		for i in range(state_num):
		    alpha_0_i = self.get_pi_i(InitStateProb, i) * self.get_b_jk(StateToObser, i, observation_series[0])
		    Alpha[0][i] = alpha_0_i
		# iteration
		for t in range(T - 1):
		    for i in range(state_num):
		        temp_sum = 0
		        for j in range(state_num):
		            temp_sum += self.get_alpha_ti(Alpha, t, j) * self.get_a_ij(StateTrans, j, i)
		        alpha_tAdd1_i = temp_sum * self.get_b_jk(StateToObser, i, observation_series[t + 1])
		        Alpha[t + 1][i] = alpha_tAdd1_i        
		return Alpha	


	def prob_beta(self, InitStateProb, StateTrans, StateToObser, state_num, T, observation_series):
		''' calculate backward probalibity matrix '''
		Beta = np.zeros(T * state_num).reshape(T, state_num)
		# initail values
		for i in range(state_num):
			beta_Tsub1_i = 1
			Beta[T - 1][i] = beta_Tsub1_i
		# iteration
		for t in range(T - 2, -1, -1):
			for i in range(state_num):
				beta_t_i = 0
				for j in range(state_num):
					beta_t_i += self.get_a_ij(StateTrans, i, j) * self.get_b_jk(StateToObser, j, observation_series[t + 1]) * self.get_beta_ti(Beta, t + 1, j)
				Beta[t][i] = beta_t_i
		return Beta
 

	def prob_gamma_ti(self, Alpha, Beta, state_num, t, i):
		'''  probablity of state at certain time ''' 
		numerator = self.get_alpha_ti(Alpha, t, i) * self.get_beta_ti(Beta, t, i)
		denominator = 0
		for j in range(state_num):
			denominator += self.get_alpha_ti(Alpha, t, j) * self.get_beta_ti(Beta, t, j)
		if denominator == 0:
			result = 0
		else:
			result = numerator / denominator
		return result


	def prob_xi_tij(self, Alpha, Beta, StateTrans, StateToObser, state_num, observation_series, t, i, j):
		''' probality of state at certain time and state at next time '''
		numerator = self.get_alpha_ti(Alpha, t, i) * self.get_a_ij(StateTrans, i, j) * self.get_b_jk(StateToObser, j, observation_series[t + 1]) * self.get_beta_ti(Beta, t + 1, j)
		denominator = 0
		for k in range(state_num):
			for h in range(state_num):
				denominator += self.get_alpha_ti(Alpha, t, k) * self.get_a_ij(StateTrans, k, h) * self.get_b_jk(StateToObser, h, observation_series[t + 1]) * self.get_beta_ti(Beta, t + 1, h)
		if denominator == 0:
			result = 0
		else:
			result = numerator / denominator
		return result


	def em_setInitPara(self, state_num, observation_num, InitStateProb, fix_InitStateProb):
		''' set em algorithm initial parameters randomly '''
		# state transformation matrix
		StateTrans = []
		for i in range(state_num):
			splits = list(np.random.uniform(0, 1, state_num - 1))
			splits.append(0)
			splits.sort()
			splits = list(np.diff(splits))
			p_last = 1 - sum(splits)
			splits.append(p_last)		
			StateTrans.append(splits)
		# state to observation matrix
		StateToObser = []
		for i in range(state_num):
			splits = list(np.random.uniform(0, 1, observation_num - 1))
			splits.append(0)
			splits.sort()
			splits = list(np.diff(splits))
			p_last = 1 - sum(splits)
			splits.append(p_last)		
			StateToObser.append(splits)
		# initial state probability
		if not fix_InitStateProb:
			InitStateProb = list(np.random.uniform(0, 1, state_num - 1))
			InitStateProb.append(0)
			InitStateProb.sort()
			InitStateProb = list(np.diff(InitStateProb))
			p_last = 1 - sum(InitStateProb)
			InitStateProb.append(p_last)		
		return InitStateProb, StateTrans, StateToObser


	def em_main(self, state_num, observation_num, InitStateProb, fix_InitStateProb, T, observation_series, em_iterNum, semi_par, stateToObser_0_vector):
		''' em algorithm estimate parameters '''
		InitStateProb, StateTrans, StateToObser = self.em_setInitPara(state_num, observation_num, InitStateProb, fix_InitStateProb)
		if semi_par:
			StateToObser[0] = stateToObser_0_vector[:]
		for em_iter in range(em_iterNum):
			#print(em_iter)
			Alpha_temp = self.prob_alpha(InitStateProb, StateTrans, StateToObser, state_num, T, observation_series)
			Beta_temp = self.prob_beta(InitStateProb, StateTrans, StateToObser, state_num, T, observation_series)
			InitStateProb_temp, StateTrans_temp, StateToObser_temp = self.em_setInitPara(state_num, observation_num, InitStateProb, fix_InitStateProb)
			# update StateTrans 
			for i in range(state_num):
				denominator = 0
				for t in range(T - 1):
					denominator += self.prob_gamma_ti(Alpha_temp, Beta_temp, state_num, t, i)
				for j in range(state_num):
					numerator = 0
					for t in range(T - 1):
						numerator += self.prob_xi_tij(Alpha_temp, Beta_temp, StateTrans, StateToObser, state_num, observation_series, t, i, j)
					if denominator == 0:
						frac = 0
					else:
						frac = numerator / denominator
					StateTrans_temp[i][j] = frac
			# update StateTtoObser
			if semi_par:
				StateToObser_temp[0] = stateToObser_0_vector[:]		
			for j in range(state_num):
				if j == 0 and semi_par:
					continue
				denominator = 0 
				for t in range(T):
					denominator += self.prob_gamma_ti(Alpha_temp, Beta_temp, state_num, t, j)
				for k in range(observation_num):
					numerator = 0
					for t in range(T):
						numerator += self.prob_gamma_ti(Alpha_temp, Beta_temp, state_num, t, j) * (observation_series[t] == k)
					if denominator == 0:
						frac = 0
					else:
						frac = numerator / denominator
					StateToObser_temp[j][k] = frac
			# update InitStateProb
			if not fix_InitStateProb:
				for i in range(state_num):
					InitStateProb_temp[i] = InitStateProb_temp[i] = self.prob_gamma_ti(Alpha_temp, Beta_temp, state_num, 0, i)
			StateTrans = StateTrans_temp
			StateToObser = StateToObser_temp
			if not fix_InitStateProb:
				InitStateProb = InitStateProb_temp
		return InitStateProb, StateTrans, StateToObser


	def viterbi_main(self, InitStateProb, StateTrans, StateToObser, state_num, T, observation_num, observation_series):
		''' viterbi algorithm estimate state '''
		delta = np.zeros(2 * state_num).reshape(2, state_num)
		phi = np.zeros(T * state_num).reshape(T, state_num)
		phi = phi.astype(np.int8)
		# initialization
		for i in range(state_num):
		    delta[0][i] = self.get_pi_i(InitStateProb, i) * self.get_b_jk(StateToObser, i, observation_series[0])
		    phi[0][i] = 0
		# iteration
		for t in range(1, T):
		    for i in range(state_num):
		        temp = []
		        for j in range(state_num):
		            temp.append(delta[0][j] * self.get_a_ij(StateTrans, j, i))
		        delta[1][i] = max(temp) * self.get_b_jk(StateToObser, i, observation_series[t])
		        phi[t][i] = np.argmax(temp)
		    delta[0] = delta[1].copy()
		# stop
		p_max = max(delta[0])
		state_T = np.argmax(delta[0])
		# path tracking
		state_series_est = []
		state_series_est.append(state_T)
		for t in range(T - 2, -1, -1):
		    state_T = phi[t+1][state_T]
		    state_series_est.append(state_T)
		state_series_est.reverse()
		state_series_est = np.array(state_series_est, dtype='int32')
		return p_max, state_series_est



def detect_scores(state_series, state_series_est, target_state=1, fbeta=1):
	''' criterion for two categories classification '''

	state_series = np.array(state_series, dtype='int32')
	state_series_est = np.array(state_series_est, dtype='int32')
	tp = ((state_series == target_state) & (state_series_est == target_state)).sum()	
	fn = ((state_series == target_state) & (state_series_est != target_state)).sum()	
	fp = ((state_series != target_state) & (state_series_est == target_state)).sum()	
	tn = ((state_series != target_state) & (state_series_est != target_state)).sum()	
	deno = tp + fp
	# precision rate
	if deno == 0:
		precision = 0
	else:
		precision = tp / deno 
	# recall rate
	deno = tp + fn
	if deno == 0:
		recall = 0
	else:
		recall = tp / deno
	# f score
	deno = (fbeta ** 2) * precision + recall
	if deno == 0:
		fscore = 0
	else:
		fscore = ((1 + fbeta ** 2) * precision * recall) / deno
	# tpr and fpr
	tpr = recall 
	deno = fp + tn 
	if deno == 0:
		fpr = 0
	else:
		fpr = fp / deno
	return {'precision':precision, 'recall':recall, 'fscore':fscore, 'tpr':tpr, 'fpr':fpr}







if __name__ == '__main__':

	# hmm gen
	InitStateProb = [1, 0]			# initial state probablity
	p_00 = 0.8						# state transform prolibities matrix
	p_10 = 0.5
	StateTrans = [																
	    [p_00, 1 - p_00],
	    [p_10, 1 - p_10]
	]
	p_00 = 0.8						# generate observation on certain state
	p_10 = 0.2
	StateToObser = [
	    [p_00, 1 - p_00],
	    [p_10, 1 - p_10]
	]
	T = 1000						# series length
	burnIn_len = 50					# burn in length
	gen_hmm = GenHMM(InitStateProb, StateTrans, StateToObser, T, burnIn_len)
	state_series, observation_series = gen_hmm.run()


	# hmm predict state：em algorithm for part of the parameters
	param_type = 'em_semi'				# em 算法估计全部参数
	StateTrans_e = None					
	StateToObser_e = None	
	detect_hmm = DetectHMM(observation_series, param_type, StateTrans_e, StateToObser_e)
	_, state_series_est = detect_hmm.predict_state()

	scores = detect_scores(state_series, state_series_est)
	print(format('precision: ', '20s'), scores['precision'])
	print(format('recall: ', '20s'), scores['recall'])


