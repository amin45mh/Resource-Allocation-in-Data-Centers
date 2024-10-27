This project addresses the challenge of optimizing resource allocation in distributed data centers
that serve users with diverse computing and memory requirements. These data centers operate in
dynamic environments with unpredictable task arrivals, and varying resource demands, making
traditional resource allocation approaches inefficient and prone to imbalances. We propose a
solution based on Deep Reinforcement Learning to overcome these limitations. Unlike traditional
Q-Learning, which relies on a Q-Table and struggles with scalability in large state spaces, Deep
RL uses neural networks to approximate Q-values, enabling effective management of continuous
state spaces. This report details the formulation of the resource allocation problem, the
implementation of classical Q-Learning, and the transition to a Deep RL-based approach. By
leveraging Deep RL, our solution can dynamically learn and adapt to the environment's variability,
resulting in improved decision-making processes for resource allocation. The proposed method
aims to achieve balanced utilization and availability of memory and CPU resources across data
centers.