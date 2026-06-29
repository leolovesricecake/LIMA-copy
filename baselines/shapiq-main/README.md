# shapiq: Shapley Interactions for Machine Learning

> An interaction may speak more than a thousand main effects.

Shapley Interaction Quantification (`shapiq`) is a Python package for (1) approximating any-order Shapley interactions, (2) benchmarking game-theoretical algorithms for machine learning, (3) explaining feature interactions of model predictions. `shapiq` extends the well-known [shap](https://github.com/shap/shap) package for both researchers working on game theory in machine learning, as well as the end-users explaining models. SHAP-IQ extends individual Shapley values by quantifying the **synergy** effect between entities (aka **players** in the jargon of game theory) like explanatory features, data points, or weak learners in ensemble models. Synergies between players give a more comprehensive view of machine learning models.

## 🛠️ Install
`shapiq` is intended to work with **Python 3.12 and above**.
Installation can be done via `uv` :
```sh
uv add shapiq
```

or via `pip`:

```sh
pip install shapiq
```

## 👀 Upcoming
See what’s on the horizon for the library in our [GitHub Project Board](https://github.com/users/mmschlk/projects/4). We plan and track upcoming features, improvements, and maintenance tasks there including new explainers, performance optimizations, and expanded model support.

## ⭐ Quickstart

You can explain your model with `shapiq.explainer` and visualize Shapley interactions with `shapiq.plot`.
If you are interested in the underlying game theoretic algorithms, then check out the `shapiq.approximator` and `shapiq.games` modules.

### Use ProxySPEX (Proxy SParse EXplainer) <img src="https://raw.githubusercontent.com/mmschlk/shapiq/main/docs/source/_static/images/spex_logo.png" alt="spex_logo" align="right" height="75px"/>
For large-scale use-cases you can also check out the [👓``ProxySPEX``](https://shapiq.readthedocs.io/en/latest/api/shapiq.approximator.sparse.html#shapiq.approximator.sparse.SPEX) approximator.

```python
# load your data and model with large number of features
data, model, n_features = ...

# use the ProxySPEX approximator directly
approximator = shapiq.ProxySPEX(n=n_features, index="FBII", max_order=2)
fbii_scores = approximator.approximate(budget=2000, game=model.predict)

# or use ProxySPEX with an explainer
explainer = shapiq.Explainer(
    model=model,
    data=data,
    index="FBII",
    max_order=2,
    approximator="proxyspex"  # specify ProxySPEX as approximator
)
explanation = explainer.explain(data[0])
```

### Visualize feature interactions

A handy way of visualizing interaction scores up to order 2 are network plots.
You can see an example of such a plot below.
The nodes represent feature **attributions** and the edges represent the **interactions** between features.
The strength and size of the nodes and edges are proportional to the absolute value of attributions and interactions, respectively.

```python
shapiq.network_plot(
    first_order_values=interaction_values.get_n_order_values(1),
    second_order_values=interaction_values.get_n_order_values(2)
)
# or use
interaction_values.plot_network()
```

The pseudo-code above can produce the following plot (here also an image is added):

<p align="center">
  <img width="500px" src="https://raw.githubusercontent.com/mmschlk/shapiq/main/docs/source/_static/network_example2.png" alt="network_plot_example">
</p>

### Explain TabPFN

With ``shapiq`` you can also explain [``TabPFN``](https://github.com/PriorLabs/TabPFN) by making use of the _remove-and-recontextualize_ explanation paradigm implemented in ``shapiq.TabPFNExplainer``.

```python
import tabpfn, shapiq
data, labels = ...                    # load your data
model = tabpfn.TabPFNClassifier()     # get TabPFN
model.fit(data, labels)               # "fit" TabPFN (optional)
explainer = shapiq.TabPFNExplainer(   # setup the explainer
    model=model,
    data=data,
    labels=labels,
    index="FSII"
)
fsii_values = explainer.explain(data[0])  # explain with Faithful Shapley values
fsii_values.plot_force()               # plot the force plot
```

<p align="center">
  <img width="800px" src="https://raw.githubusercontent.com/mmschlk/shapiq/main/docs/source/_static/images/fsii_tabpfn_force_plot_example.png" alt="Force Plot of FSII values">
</p>


## 📖 Documentation with tutorials
The documentation of ``shapiq`` can be found at https://shapiq.readthedocs.io.
If you are new to Shapley values or Shapley interactions, we recommend starting with the [introduction](https://shapiq.readthedocs.io/en/latest/introduction/) and the [examples & tutorials](https://shapiq.readthedocs.io/en/latest/auto_examples/index.html).
There is a lot of great resources available to get you started with Shapley values and interactions.

## 💬 Citation

If you use ``shapiq`` and enjoy it, please consider citing our [NeurIPS paper](https://arxiv.org/abs/2410.01649) or consider starring this repository.

```bibtex
@inproceedings{Muschalik.2024b,
  title     = {shapiq: Shapley Interactions for Machine Learning},
  author    = {Maximilian Muschalik and Hubert Baniecki and Fabian Fumagalli and
               Patrick Kolpaczki and Barbara Hammer and Eyke H\"{u}llermeier},
  booktitle = {The Thirty-eight Conference on Neural Information Processing Systems Datasets and Benchmarks Track},
  year      = {2024},
  url       = {https://openreview.net/forum?id=knxGmi6SJi}
}
```
