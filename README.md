MDP & POMDP Learning Experiments

This repository contains implementations and experimental results for Markov Decision Processes (MDP) and Partially Observable Markov Decision Processes (POMDP). The experiments focus on state abstraction, error rates relative to sample size/depth, and the sample complexity involved in history-based extraction in POMDPs.

1. MDP Experiments (Richens et al.)

These experiments replicate and extend the work found in Richens et al. (arXiv:2506.01622). We evaluate the error rates of the learned model against training sample size and goal depth.

Results

Figure 1: MDP Performance Analysis

A. Error vs. Training Samples

Fixed Depth = 50

As the number of training samples ($N$) increases, the error rate decreases significantly, plateauing around $N=5000$.

Samples ($N$)

Error ($\pm$ std)

500

0.1118 $\pm$ 0.0091

1000

0.0853 $\pm$ 0.0141

2000

0.0552 $\pm$ 0.0075

3000

0.0532 $\pm$ 0.0066

5000

0.0388 $\pm$ 0.0044

7000

0.0383 $\pm$ 0.0084

10000

0.0307 $\pm$ 0.0046

B. Error vs. Goal Depth

Fixed Samples ($N$) = 5000

The model demonstrates robustness to increasing goal depth, maintaining low error rates even as depth extends to 400.

Depth

Error ($\pm$ std)

10

0.0580 $\pm$ 0.0019

20

0.0385 $\pm$ 0.0094

50

0.0440 $\pm$ 0.0030

75

0.0469 $\pm$ 0.0037

100

0.0319 $\pm$ 0.0062

200

0.0380 $\pm$ 0.0100

400

0.0421 $\pm$ 0.0009

2. POMDP Experiments

These experiments investigate the difficulty of extracting transition probabilities in a POMDP setting using histories of varying lengths ($h$). We measure the "Sample Complexity"—the number of total episodes required to achieve a valid trial for a specific history.

Results

Figure 2: POMDP Extraction & Sample Complexity

Sample Complexity Analysis

The defining challenge in POMDP extraction is the rarity of specific histories. We define $\rho(h)$ as the probability of encountering history $h$.

History Length

Est. $\rho(h)$

Sample Complexity Multiplier

0 (Empty)

~1.000

1.0x episodes needed

1

~0.406

2.5x episodes needed

2

~0.109

11.2x episodes needed

Key Insights

Inverse Relationship: Each 'valid trial' requires $1/\rho(h)$ episodes on average.

History Decay: As history length increases, $\rho(h)$ drops, causing the required number of episodes to spike.

Episodic Necessity: Without episodic resets, reliable extraction of these statistics is impossible.

Detailed Trial Data

The following logs detail the extraction testing across different history lengths and trial counts.

<details>
<summary>Click to view full experiment logs</summary>

Testing extraction with different history lengths and trial counts...

History (length=0): []
  True K(h,0) = [0.37 0.37 0.26]
  True P(o=0|h,a=0) = 0.370
    n= 10: p̂=0.400, error=0.030, attempts=  10, ρ(h)≈1.000
    n= 20: p̂=0.350, error=0.020, attempts=  20, ρ(h)≈1.000
    n= 50: p̂=0.400, error=0.030, attempts=  50, ρ(h)≈1.000
    n=100: p̂=0.430, error=0.060, attempts= 100, ρ(h)≈1.000

History (length=1): [(1, 0)]
  True K(h,0) = [0.37        0.30459459 0.32540541]
  True P(o=0|h,a=0) = 0.370
    n= 10: p̂=0.200, error=0.170, attempts=  28, ρ(h)≈0.357
    n= 20: p̂=0.450, error=0.080, attempts=  46, ρ(h)≈0.435
    n= 50: p̂=0.400, error=0.030, attempts= 114, ρ(h)≈0.439
    n=100: p̂=0.300, error=0.070, attempts= 255, ρ(h)≈0.392

History (length=2): [(1, 0), (2, 1)]
  True K(h,0) = [0.385125    0.37504167 0.23983333]
  True P(o=0|h,a=0) = 0.385
    n= 10: p̂=0.500, error=0.115, attempts= 217, ρ(h)≈0.046
    n= 20: p̂=0.300, error=0.085, attempts= 215, ρ(h)≈0.093
    n= 50: p̂=0.480, error=0.095, attempts= 784, ρ(h)≈0.064
    n=100: p̂=0.350, error=0.035, attempts=1377, ρ(h)≈0.073

History (length=2): [(0, 0), (1, 1)]
  True K(h,0) = [0.47171953 0.29417272 0.23410776]
  True P(o=0|h,a=0) = 0.472
    n= 10: p̂=0.300, error=0.172, attempts=  73, ρ(h)≈0.137
    n= 20: p̂=0.550, error=0.078, attempts= 142, ρ(h)≈0.141
    n= 50: p̂=0.480, error=0.008, attempts= 268, ρ(h)≈0.187
    n=100: p̂=0.580, error=0.108, attempts= 772, ρ(h)≈0.130


</details>

References

Richens et al. (2025). Title of the paper. arXiv preprint arXiv:2506.01622
