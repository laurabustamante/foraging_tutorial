//aria:compile=0
data {
	int NS;
	int K;
	int choose_stay[K];
	int<lower=1,upper=NS> which_S[K];
	vector[K] total_reward;
	vector[K] total_harvest_periods;
	vector[K] number_trees;
	vector[K] expected_reward;
	vector[K] is_Matching;
	vector[K] is_Mismatching;
	vector[K] is_small;
	vector[K] is_large;
}

transformed data{
	vector[5] tau_prior_OM = [.5,40,30,40,30]';
	vector[5] beta_prior_OM = [0,0,0,0,0]';

}

parameters {
	vector<offset=beta_prior_OM,multiplier=tau_prior_OM>[5] beta_ms; // group betas
	vector<lower=0>[5] tau_raw ;	// betas covariance scale
	matrix[NS,5] beta_s; //per subject betas
	cholesky_factor_corr[5] omega;   // betas covariance correlation
	// beta_ms[1] = inv_temp
	// beta_ms[2] = cost_Matching
	// beta_ms[3] = cost_Mismatching
	// beta_ms[4] = cost_small
	// beta_ms[5] = cost_large 
}

transformed parameters{
	vector[5] tau = tau_raw .* tau_prior_OM ;
}

model {
	omega ~ lkj_corr_cholesky(1); // prior on correlation
	tau_raw ~ normal(1,1); // prior on covariance scale
	beta_ms ~ normal(beta_prior_OM, tau_prior_OM); // prior on group betas
	matrix[5,5] L_omega = diag_pre_multiply(tau, omega) ;
	for (s in 1:NS) {
		beta_s[s] ~ multi_normal_cholesky(
			beta_ms
			, L_omega
		);
	}
	target += bernoulli_logit_lpmf(choose_stay |
		(beta_s[which_S,1]).*(
			expected_reward
			- is_Matching.*(
				(total_reward - beta_s[which_S,2].*number_trees)./total_harvest_periods
			)
			- is_Mismatching.*(
				(total_reward - (beta_s[which_S,2]+(beta_s[which_S,3])).*number_trees)./total_harvest_periods
			)
			- is_small.*(
				(total_reward - beta_s[which_S,4].*number_trees)./total_harvest_periods
			)
			- is_large.*(
				(total_reward - (beta_s[which_S,4]+(beta_s[which_S,5])).*number_trees)./total_harvest_periods
			)
		)
	) ;
}
generated quantities{
	// r: correlations
	matrix[5,5] r = multiply_lower_tri_self_transpose(omega) ;
}
