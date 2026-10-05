#fundamentals based model - time constant
#POSTERIOR PREDICTIVE CHECK

data {
  int<lower=1> N_e; //nr of historical elections
  int<lower=1> N_p; //nr of parties (top 5)
  int<lower=1> N_obs; //nr of observations (N_e * N_p)
  int<lower=1> K; //nr of predictors (covariates)
  
  matrix[N_obs, K] X; //matrix of predictors (covariates)
  vector[N_obs] v_obs; //observed vote shares
  
  int<lower=1, upper=N_p> p_id[N_obs]; //party id
  int<lower=1, upper=N_e> e_id[N_obs]; //election id
  
  int<lower=1> N_train; //how many elections to train on?
  int<lower=1> N_pred; //how many elections to predict?
}

parameters {
  #intercepts
  vector[N_p] beta_0; //beta_0[p] = base support for party p
  #regression coeffs (time constant)
  matrix[N_p, K] beta; //beta[p, k] = effect of predictor k on party p
}

transformed parameters {
  #eq.2 appendix: calculate shape parameters
  vector[N_obs] a;
  
  for (i in 1:(N_obs)) {
    a[i] = exp(beta_0[p_id[i]] + dot_product(beta[p_id[i], ], X[i, ]'));
  }
}

model {
  #priors for coeffs beta
  beta_0 ~ normal(5.5, 0.25); //intercepts
  to_vector(beta) ~ normal(0, 0.1); //slopes
  
  #eq.1 (historical likelihood)
  #train the model only with previous elections
  for (e in 1:N_train) {
    int start_idx = (e-1) * N_p + 1;
    int end_idx = e * N_p;
    
    #alpha for all parties in training elections
    vector[N_p] a_slice = a[start_idx:end_idx];
    vector[N_p] v_slice = v_obs[start_idx:end_idx];
    
    v_slice ~ dirichlet(a_slice);
  }
}

generated quantities {
  vector[N_pred * N_p] v_pred; //simulated vote shares for elections we want to predict
  
  #simulate vote shares for elections we want to predict based on what the model learned from previous elections
  for (t in 1:N_pred) {
    int e = N_train + t;
    
    int start_idx = (e-1) * N_p + 1;
    int end_idx = e * N_p;
    
    #alpha for all parties in prediction elections (rows after the training years)
    vector[N_p] a_slice = a[start_idx:end_idx];
    
    #simulate outcome based on learned parameters
    #predict only the party vote shares for the elections we want to predict
    v_pred[((t-1)*N_p + 1):(t*N_p)] = dirichlet_rng(a_slice);
  }
}
