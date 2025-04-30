# Cersyve

**Cersyve** is a benchmark for neural safety <u>cer</u>tificate <u>sy</u>nthesis and <u>ve</u>rification in control systems.

Neural safety certificate verification is a special type of verification problems that require neural networks to satisfy certain properties everywhere in the state space, i.e., the input set of the verification problem is the entire state space.
This is different from most exisiting benchmarks, which only consider verification in either a small disturbance set around data samples or part of the state space, such as MNIST, CIFAR, and ACAS Xu.
Another distinct feature of safety certificate verification is that it not only involves a single safety certificate network, but also needs system dynamics and control policies for verification, and certain conversions are required before these components can be formulated as a standard verification problem.

Cersyve contains nine commonly used control tasks with state dimensions ranging from two to six.
These tasks include both linear and nonlinear dynamics and safety constraints.
We provide two ONNX models for each tasks.
One is a pretrained safety certificate and the other is finetuned.
The models are already integrated with all necessary elements for verification, such as system dynamics, constraints, and control policies.
Users can view each of them as a single neural network for standard verification.

While this version of Cersyve only contains necessary tools for verification, our full version also includes a set of neural safety certificate synthesis tools, including pre-training, adversarial training, and verification-guided training modules, as well as evaluation tools for synthesized certificates.
These tools facilitate secondary development and performance comparison of different synthesis algorithms.
Moreover, we also include an MILP-based neural safety certificate verification algorithm, as well as neural value functions synthesized and verified by our framework on all nine tasks for comparing different verification algorithms.
For a detailed description of Cersyve, please refer to our paper: [Scalable Synthesis of Formally Verified Neural Value Function for Hamilton-Jacobi Reachability Analysis](https://arxiv.org/abs/2407.20532).