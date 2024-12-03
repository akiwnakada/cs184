Plot Comparison/Analysis: 

List of observed means, 95th and 5th percentile values (in that order): 

Explore: 15, 15, 15
Greedy: 6, 17.5, 1
ETC: 10, 15, 8
Epgreedy: 12.5, 15.5, 9.5
UCB: 11, 13, 9
Thompson_sampling: 7, 12.5, 4

* Algorithms
    * Explore
        - The average regret observed across 30 trials was linear in terms of T (this is given by the slope). Additionally, the average regret increased at a constant rate relative to T. This makes sense since the explore algorithm chooses an arm uniformly randomely at each time step and thus has the same expected regret for each time step. Out of all of the algorithms, based on the data observed, the explore algorithm performs the worst. This also makes sense since we are not learning anything from the data, we simply choose a random arm each time. An interesting characteristic of the explore algorithm is that there was not a lot of variance in the observed regret across the 30 trials (as shown by the 5th and 9th quantiles depicted by the light blue). The low variance is expected as each arm is chosen randomely, reducing the chance that any one run is significantly different than another run. At the final time step the explore algorithm produced an average total regret of around 14. 
    * Greedy
        - The greedy algorithm also shows that the average regret is linear relative to T. We found that this is theoretically true in lecture. One notable observation is that the average regret jumps initially then stabilizes, increasing with a lower slope compared to the first few time steps. Intuitively this makes sense as with more time steps, more samples are taken from each arm so the empirical mean estimates should get more accurate. Perhaps the most distinguishable feature of the results using this algorithm is the huge difference between the 5th (total regret of around 1) and 95th (total regreat of around 17) percentile regret observed in the 30 trials. The large variance of total regret observed using the Greedy algorithm is due to the potential for huge deviations in the arms chosen based on the first few arm pulls. If the first initial arm pulls lead to the optimal arm being chosen early, the optimal arm will continuously get pulled and the regret will grow very slowly. On the other hand, if the sample from the optimal arm happens to be low and an arm with a lower mean (which happens to have a higher sample value) is chosen, this arm may repeatedly get pulled thus increasing the regret dramatically. Overall, the Greedy algorithm performed poorly. Even though the average total regret was around 6, huge variance in the results made it so that the 95th percentile trial had a total regret which was higher than any of the regret values observed using the Explore algorithm at the final time step. 
    * ETC
        - The explore then commit algorithm followed the same average total regret observed using the explore algorithm until a certain point (the point at which we commit). Once this critical step was reached, the algorithm chose the empirically optimal arm and commited. Once the algorithm commited to this arm, the slope of the regret decreased but did not go to 0, indicating that, on average the optimal arm was not the one chosen. The variance in this algorithm was predictable since the variance in the explore part was the same as observed in the explore algorithm. During the commit phase, results varied from having a slope of 0 (optimal arm being chosen) to having a high slope (suboptimal arm chosen). Overall the ETC algorithm performed poorly on average with an average total regret of around 10.
    * Epsilon Greedy
        - The epsilon greedy algorithm exhibited linear regret with respect to T. There was relatively low variance in the regret observed throughout the 30 trials. When compared to Greedy in terms of average total regret, epsilon greedy actually performed worse than Greedy. This is a surprising result that we will analyze later. 
    * UCB
        - It was not clear from the graph that the UCB algorithm exhibits sublinear average regret (even though we know this is true from lecture). This is due to the fact that we ran this experiment with only T = 100 time steps. As T increased, the sublinearity would have been more obsvious. One nice characteristic about this algorithm was that there was relatively little variance in the regret observed in each trial. 
    * Thompson Sampling
        - The Thompson Sampling algorithm performed the best out of all of the algorithms. We can see that the cumulative regret is sublinear with respect to T. The average regret observed is actually a bit higher than the greedy algorithm (7 vs 6), but the 95 percentile trial had significantly less regret using the Thompson sampling algorithm vs the Greedy algorithm. 

Why was Greedy so good? 

The greedy algorithm produced the lowest average cumulative regret of any of the algorithms. This result was due to several choices made in the experiment design. For one, there were only 100 time steps. In short time periods, algorithms which exploit more perform better because exploration based algorithms do not have enough time to learn about the true means of the arms. Secondly, the true arm means chosen for the experiment were quite close together (mu_list = [0.3,0.5,0.6,0.65,0.7]). This makes it harder for learning based algorithms to make the optimal choice in a short time frame since they take uncertainty into account with their choices. 

Despite the short term success of Greedy, the Thompson Sampling algorithm still showed it produces sublinear cumulative regret while Greedy exhibited linear regret with respect to T, implying that asymptotically Thompson Sampling is more optimal. 