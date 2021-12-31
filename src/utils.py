import matplotlib.pyplot as plt
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
import numpy as np


def plot_function(f: callable, cs: CS.ConfigurationSpace, samples_per_axis=100) -> plt.Figure:
    fig = plt.figure()
    assert isinstance(fig, plt.Figure)

    ax = fig.gca()
    assert isinstance(ax, plt.Subplot)

    parameters = cs.get_hyperparameters()
    n_parameter = len(parameters)
    if n_parameter > 2:
        raise ValueError("Plotting currently only supports max 2 dimensions")
    if n_parameter == 0:
        raise ValueError("Requires at least 1 parameter for plotting")

    ranges = []
    for parameter in parameters:
        assert isinstance(parameter, CSH.NumericalHyperparameter)
        if parameter.log:
            space_function = np.logspace
        else:
            space_function = np.linspace

        ranges.append(space_function(parameter.lower, parameter.upper, num=samples_per_axis))

    x = ranges[0]
    ax.set_xlabel(parameters[0].name)
    if n_parameter == 1:
        y = [f(p) for p in x]
        ax.plot(x, y, label=f.__name__)
        ax.set_ylabel(f"f({parameters[0].name})")
        plt.legend()

    elif n_parameter == 2:
        y = ranges[1]
        ax.set_ylabel(parameters[1].name)
        xx, yy = np.meshgrid(x, y)
        z = []
        for x1, x2 in zip(xx.reshape(-1), yy.reshape(-1)):
            X = {parameters[0].name: x1, parameters[1].name: x2}
            z.append(f(**X))

        z = np.reshape(z, xx.shape)
        plt.pcolormesh(x, y, z)

        plt.axis('scaled')

    return fig
