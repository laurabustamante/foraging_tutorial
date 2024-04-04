//aria:compile=0
data {
  int NS; // number of particiants
  int K; // length of all trials for all participants 
  array[K] int choose_stay; // whether participant exits the patch (0) or not (1)
  array[K] int<lower=1, upper=NS> which_S;
  vector[K] total_reward; // total rewards per block type
  vector[K] total_harvest_periods; // time spent per block type
  vector[K] number_trees; // how many trees visited per block type
  vector[K] expected_reward; // reward expected on a trial 
  vector[K] is_Matching; // 1 = congruent trial block, 0 = not 
  vector[K] is_Mismatching; // 1 = interference trial block, 0 = not 
  vector[K] is_small; // 1 = physical low efffort block, 0 = not 
  vector[K] is_large; // 1 = physical high efffort block, 0 = not 
} 

transformed data {
  // fixed effects prior ~ N(X_mu_prior, X_s_prior)
  // [1] = log inv_temp
  // [2] = cost_congruent
  // [3] = cost_interference
  // [4] = cost_small
  // [5] = cost_large
  vector[5] gam_mu_prior = [0, 0, 0, 0, 0]';     // mean of fixed effect prior
  vector[5] gam_s_prior  = [0.5, 40, 30, 40, 30]'; // sd of fixed effect prior
  vector[5] tau_mu_prior = [0.5, 20, 15, 20, 15]'; // mean of random variance prior
  vector[5] tau_s_prior  = [0.5, 20, 15, 20, 15]'; // sd of random variance prior
} 

parameters { 
  // [1] = log inv_temp
  // [2] = cost_congruent
  // [3] = cost_interference
  // [4] = cost_small
  // [5] = cost_large
  // session mean parameters
  vector[5] mu;           //  fixed effects 
  vector<lower=0>[5] tau; // spread of random effects
  // reparameterization, all random effects in a single matrix
  matrix[5, NS] U_pr;            // uncorreleated random effects
  cholesky_factor_corr[5] L_u;  // decomposition of corr matrix to lower triangular matrix
}

transformed parameters {
  matrix[NS, 5] U;      // random effects
  matrix[NS, 5] beta_s; // fixed + random effects
  // transform random effects
  // these have dimensions NS x Number of parameters
  // for example,
  // U[1,1]   = random effect inv. temp for subject 1
  // U[10, 3] = random effect of cost incongruent for subject 10
  U = (diag_pre_multiply(tau, L_u) * U_pr)';
  
  // compute fixed + random effect for each subject      
  for(i in 1:NS) {
  	for(p in 1:5) {
    beta_s[i,p] = mu[p] + U[i,p]; 
    }
  }
}

model {
  // priors (adjusted for bounds)
  target += normal_lpdf(mu  | gam_mu_prior, gam_s_prior);
  target += normal_lpdf(tau | tau_mu_prior, tau_s_prior) - 5*normal_lccdf(0 | tau_mu_prior, tau_s_prior);
  target += lkj_corr_cholesky_lpdf(L_u | 1);
  target += std_normal_lpdf(to_vector(U_pr));
  // likelihood
    choose_stay ~ bernoulli_logit((exp(beta_s[which_S, 1]))
                                .* (expected_reward
                                    - is_Matching
                                      .* ((total_reward
                                           - beta_s[which_S, 2]
                                             .* number_trees)
                                          ./ total_harvest_periods)
                                    - is_Mismatching
                                      .* ((total_reward
                                           - (beta_s[which_S, 2]
                                              + (beta_s[which_S, 3]))
                                             .* number_trees)
                                          ./ total_harvest_periods)
                                    - is_small
                                      .* ((total_reward
                                           - beta_s[which_S, 4]
                                             .* number_trees)
                                          ./ total_harvest_periods)
                                    - is_large
                                      .* ((total_reward
                                           - (beta_s[which_S, 4]
                                              + (beta_s[which_S, 5]))
                                             .* number_trees)
                                          ./ total_harvest_periods)));
}
  
generated quantities {
  // "recompose" Choleksy to corr matrix
  corr_matrix[5] rho = L_u * L_u';
}
