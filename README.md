# Dissipative Hamiltonian Neural Networks (D-HNN)

**Python recreation and extension** of the Julia-based HNN and D-HNN experiments from the Advanced Machine Learning for Physics course.

## Project Structure

```
D_HNN/
├── src/dhnn/            # Core library
│   ├── models.py        # BaselineNN, HNN, DHNN architectures
│   ├── integrators.py   # RK4, Störmer–Verlet, learned rollout
│   ├── data.py          # ODE definitions, samplers, dataset generation
│   ├── training.py      # Loss functions, generic training loop
│   └── plotting.py      # Publication-quality plotting helpers
├── notebooks/
│   └── experiments.ipynb # Main experiment notebook
├── figures/             # Saved plots
├── Refrence_Notebooks/  # Original Julia notebooks (read-only reference)
├── Thesis_style.mplstyle
├── requirements.txt
└── pyproject.toml
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Background

A **Hamiltonian Neural Network** parameterises the Hamiltonian $\mathcal{H}_\theta(q,p)$ and
derives dynamics via Hamilton's equations using automatic differentiation — guaranteeing
energy conservation by construction.

A **Dissipative HNN** extends this by splitting the vector field into a symplectic
(conservative) part from $\mathcal{H}_\theta$ and a gradient (dissipative) part from
$\mathcal{D}_\theta$, enabling modelling of systems with friction.

### Systems studied

| System | Type | Notebook Part |
|--------|------|---------------|
| Simple harmonic oscillator | Conservative | I |
| Ideal pendulum | Conservative | I |
| Damped harmonic oscillator | Dissipative | II |
| Real pendulum (experimental data) | Dissipative | I & II |
| Lennard-Jones particle interaction | Conservative (N-body) | I |

## References

- Greydanus, Dzamba & Sosanya. *Hamiltonian Neural Networks*. NeurIPS 2019.
- Sosanya & Greydanus. *Dissipative Hamiltonian Neural Networks*. arXiv:2201.10085, 2022.
