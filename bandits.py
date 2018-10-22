#!/usr/bin/env python3
# Linear reward penalty multi armed bandit?
# https://michaelpacheco.net/blog/RL-multi-armed-bandit-2

import numpy as np

NUM_TRIALS = 5000
E = 0.33
ALPHA = 0.2
BETA = 0.8

AD_CONCURRENCY = 10

ground_truth_adperf = np.random.rand(AD_CONCURRENCY)
ad_counts = np.zeros(AD_CONCURRENCY)
ads_prob = np.ones(AD_CONCURRENCY) * (1.0/AD_CONCURRENCY)
reward_count = np.zeros(AD_CONCURRENCY)


for _ in range(NUM_TRIALS):

    # Either choose a greedy action or explore
    if E > np.random.uniform():
        which_ad_to_try = np.random.randint(0, AD_CONCURRENCY)
    else:
        which_ad_to_try = np.random.choice(AD_CONCURRENCY, p=ads_prob)

    # now run the ad
    # binary here but we would use some kind of ad performance based reward
    # something like (daily rev - daily cost) if # impressions > k ?
    # or converted vs not?
    if ground_truth_adperf[which_ad_to_try] > np.random.uniform():
        reward = 1
    else:
        reward = 0

    # now update the ad num and expected value
    ad_counts[which_ad_to_try] += 1
    reward_count[which_ad_to_try] = reward_count[which_ad_to_try] + reward
    mask1 = np.zeros(AD_CONCURRENCY)
    mask2 = np.ones(AD_CONCURRENCY)
    mask1[which_ad_to_try] = 1
    mask2[which_ad_to_try] = 0
    if reward == 1:
        ad_won_vector = (ads_prob + ALPHA * (1 - ads_prob)) * mask1
        rest_of_ads = ((1 - ALPHA) * ads_prob) * mask2
        ads_prob = ad_won_vector + rest_of_ads
    else:
        rest_of_ads = (BETA / (AD_CONCURRENCY - 1) + (1 - BETA) * ads_prob) * mask2
        ad_lose_vector = ((1 - BETA) * ads_prob) * mask1
        ads_prob = ad_lose_vector + rest_of_ads

print("num times each ad was displayed: ", ad_counts)
# print("check sum ad display == trials : ", ad_counts.sum() == NUM_TRIALS)

print("reward over time tracking : ", reward_count)
print("agent's weight : ", ads_prob)
print("agent's probability guess : ", reward_count / ad_counts)
print("agents guess best ad : ", np.argmax(reward_count / ad_counts))

print("ground truth full list : ", ground_truth_adperf)
print("ground truth best ad : ", np.argmax(ground_truth_adperf))
