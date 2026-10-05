// DYNAMIC MODEL
// NEW COMBINED MODEL
// since fundamentals code is static simply plug in fundamentals prediction into this model as anchor

data{
  int<lower=1> N_p; //nr of parties
  int<lower=1> N_comp; //nr of polling companies
  int<lower=1> N_polls; //nr of polls conducted
  int<lower=1> N_periods; //nr of days from start until election (t=1,...,T)
  
  int<lower=1, upper=N_comp> comp_id[N_polls]; //polling company id
  int<lower=1, upper=N_periods> date[N_polls]; //date when the poll was published
  int<lower=0> y[N_polls, N_p]; //poll counts
  
  #fundamentals anchor
  simplex[N_p] vE; //mean prediction extracted from fundamentals model
  vector[N_p] vE_sd; //sd of prediction extracted from fundamentals model
}


parameters {
  matrix[N_comp-1, N_p-1] house_effect_raw; //house effects (polling company biases) (relative to last one)
  matrix[N_periods, N_p-1] alphastar; //latent support (log ratio space)
  vector<lower=0>[N_p-1] sigma_evo; //evolution standard deviation (weekly volatility)
  cholesky_factor_corr[N_p-1] S_cor; //cholesky factor of the correlation matrix
}


transformed parameters {
  simplex[N_p] alpha[N_periods]; //convert log ratio to simplex (vote shares) for output and likelihood

  for (t in 1:N_periods) {
    row_vector[N_p] ea;
    for (p in 1:(N_p-1)) {
      ea[p] = exp(alphastar[t, p]);
    }
    ea[N_p] = 1.0; //baseline
    for (p in 1:N_p){
      alpha[t, p] = ea[p] / sum(ea);
    }
  }
}


model {
  #priors
  to_vector(house_effect_raw) ~ normal(0, 0.05); //house effects
  sigma_evo ~ normal(0, 0.025); //evolution standard deviation: weekly changes should be small (assuming swiss vote share is stable)
  S_cor ~ lkj_corr_cholesky(50); //cholesky factor of correlation matrix
  
  #fundamentals prior
  //fundamentals mean to log ratio space
  //last party as baseline
  vector[N_p-1] log_ratio_mean;
  for (p in 1:(N_p-1)) {
    log_ratio_mean[p] = log(vE[p] / vE[N_p]);
  }
  //fundamentals sd to log ratio space (delta method approximation: sd_log = sd_linear / mean_linear)
  vector[N_p-1] sigma_anchor_log;
  for (p in 1:(N_p-1)) {
    sigma_anchor_log[p] = vE_sd[p] / vE[p];
  }
  //apply prior (end anchor)
  alphastar[N_periods] ~ multi_normal_cholesky(
    log_ratio_mean,
    diag_matrix(sigma_anchor_log)
  );
  
  #random walk prior
  //alpha[t] ~ normal(alpha[t+1], sigma)
  for (t in 1:(N_periods-1)) {
    alphastar[t] ~ multi_normal_cholesky(
      alphastar[t+1],
      diag_pre_multiply(sigma_evo, S_cor)
    );
  }
  
  #likelihood
  for (k in 1:N_polls) {
    int t = date[k];
    int c = comp_id[k];
    //expected probability for poll, log ratio: latent + house effect
    vector[N_p-1] latent_lr = alphastar[t]';
    //assuming house effects raw is defined relative to baseline
    vector[N_p-1] he_shift = rep_vector(0, N_p-1);
    if (c < N_comp) {
      he_shift = house_effect_raw[c]';
    }
    #combine latent support + he
    vector[N_p-1] poll_log_ratio = latent_lr + he_shift;
    #back to simplex
    vector[N_p] poll_prob;
    for (p in 1:(N_p-1)) {
      poll_prob[p] = exp(poll_log_ratio[p]);
    }
    poll_prob[N_p] = 1.0; //baseline
    poll_prob = poll_prob / sum(poll_prob);
    
    #likelihood
    y[k] ~ multinomial(poll_prob);
  }
}


generated quantities {
  #final forecast as vote shares
  simplex[N_p] forecast_final;
  forecast_final = alpha[N_periods];
  
  #trajectory for plots
  matrix[N_periods, N_p] trajectory;
  for (t in 1:N_periods) {
    for (p in 1:N_p) {
      trajectory[t, p] = alpha[t, p];
    }
  }
}















