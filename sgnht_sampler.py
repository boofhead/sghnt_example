# -*- coding: utf-8 -*-

"""
 Stochastic Gradient Nosé Hoover Thermostat samplers for bayesian parameter fitting
 a described in 
"""

from mxnet.optimizer import Optimizer
from mxnet.ndarray import NDArray, clip, array, sum, power, full, ones, exp, random_normal, sqrt
import math
import random

# convenience wrapper for Optimizer.Register
register = Optimizer.register


@register
class SGNHT(Optimizer):
    """Stochastic Gradient Nosé Hoover Thermostat
    
    This class implements the sampler described in the paper *Bayesian Sampling Using Stochastic
    Gradient Thermostats*, available at
    http://people.ee.duke.edu/~lcarin/sgnht-4.pdf
    
    Parameters
    ----------

    a : float
        level of added random noise
    
    """

    def __init__(self, a=0.5, **kwargs):
        super(SGNHT, self).__init__(**kwargs)
        self.a = a
        self.gamma = a
        self.delta_gamma = 0

    def create_state(self, index, weight):
        """Create additional optimizer state such as momentum.

        Parameters
        ----------
        weight : NDArray
            The weight data


        use this to initialize wd_mult

        """
        lr = self._get_lr(index)
        return random_normal(0, math.sqrt(lr), weight.shape, weight.context, dtype=weight.dtype)  # momentum

    def _update_count(self, index):
        """
        update num_update

        Parameters:
        index : int
            The index will be updated
        """
        if index not in self._index_update_count:
            self._index_update_count[index] = self.begin_num_update
        else:
            self._index_update_count[index] += 1
        if self._index_update_count[index] > self.num_update:
            self.num_update = self._index_update_count[index]
            self.gamma += self.rescale_grad * self.delta_gamma - self._get_lr(index)
            self.delta_gamma = 0

    def update(self, index, weight, grad, state):
        """Update the parameters.

        Parameters
        ----------
        index : int
            An unique integer key used to index the parameters

        weight : NDArray
            weight ndarray

        grad : NDArray
            grad ndarray

        state : NDArray or other objects returned by init_state
            The auxiliary state used in optimization.
        """
        assert (isinstance(weight, NDArray))
        assert (isinstance(grad, NDArray))
        lr = self._get_lr(index)
        wd = self._get_wd(index)
        self._update_count(index)

        mom = state

        grad[:] = (grad + wd * weight) * self.rescale_grad
        if self.clip_gradient is not None:
            grad = clip(grad, -self.clip_gradient, self.clip_gradient)
        mom[:] *= (1 - self.gamma)
        mom[:] -= lr * grad + random_normal(0, math.sqrt(2 * self.a * lr), weight.shape, weight.context)
        weight[:] += mom
        self.delta_gamma += sum(power(mom, 2)).asscalar()


@register
class SGNHTP(SGNHT):
    """Stochastic Gradient Nosé Hoover Thermostat sampler
    with gamma priors on weight decay (precision prior on weights)

    Parameters
    ----------

    alpha, beta : float
        initial parameters for gamma prior on wd

    wd_update_step : integer
        batches per wd update
    """

    def __init__(self, alpha=1., beta=1., wd_update_step=500, **kwargs):
        super(SGNHTP, self).__init__(**kwargs)
        self.alpha = alpha
        self.beta = beta
        self.wd_update_step = wd_update_step
        self.wd = 1.0

    def create_state(self, index, weight):
        """Create additional optimizer state such as momentum.

        Parameters
        ----------
        
        weight : NDArray
            The weight data

            
        use this to initialize wd_mult

        """
        lr = self._get_lr(index)
        self._set_wd(index, random.gammavariate(self.alpha, self.beta))
        return (array([self.alpha, self.beta], weight.context, dtype=weight.dtype),  # wd prior parameters
                random_normal(0, math.sqrt(lr), weight.shape, weight.context, dtype=weight.dtype))  # momentum

    def update(self, index, weight, grad, state):
        """Update the parameters.

        Parameters
        ----------
        index : int
            An unique integer key used to index the parameters

        weight : NDArray
            weight ndarray

        grad : NDArray
            grad ndarray

        state : NDArray or other objects returned by init_state
            The auxiliary state used in optimization.
        """
        assert (isinstance(weight, NDArray))
        assert (isinstance(grad, NDArray))
        lr = self._get_lr(index)
        wd = self._get_wd(index)
        self._update_count(index)
        num_step = self.num_update - self.begin_num_update

        prior, mom = state
        prior += array([weight.size / 2.0, sum(power(weight, 2)) / 2.0])

        if num_step % self.wd_update_step == 0:
            wd = random.gammavariate(prior[0].asscalar(), prior[1].asscalar())
            self._set_wd(index, wd)

        grad = (grad + wd * weight) * self.rescale_grad
        if self.clip_gradient is not None:
            grad = clip(grad, -self.clip_gradient, self.clip_gradient)
        mom[:] *= (1 - self.gamma)
        mom[:] -= lr * grad + random_normal(0, math.sqrt(2 * self.a * lr), weight.shape, weight.context)
        weight[:] += mom
        self.delta_gamma += sum(power(mom, 2)).asscalar()


@register
class mSGNHT(Optimizer):
    """modified Stochastic Gradient Nosé Hoover Thermostat

    This class implements the modified  Stochastic Gradient Nosé Hoover Thermostat
    sampler described in the paper *Scalable Deep Poisson Factor Analysis For Topic Modelling*,
    available at http://people.ee.duke.edu/~lcarin/DeepPFA_ICML2015.pdf

    Parameters
    ----------

    a : float
        level of added random noise

    """

    def __init__(self, a=0.5, **kwargs):
        super(mSGNHT, self).__init__(**kwargs)
        self.a = a

    def create_state(self, index, weight):
        """Create additional optimizer state such as momentum.

        Parameters
        ----------
        weight : NDArray
            The weight data


        use this to initialize wd_mult

        """
        lr = self._get_lr(index)
        return (full(weight.shape, self.a, weight.context, dtype=weight.dtype),  # thermostat
                random_normal(0, math.sqrt(lr), weight.shape, weight.context, dtype=weight.dtype))  # momentum

    def update(self, index, weight, grad, state):
        """Update the parameters.

        Parameters
        ----------
        index : int
            An unique integer key used to index the parameters

        weight : NDArray
            weight ndarray

        grad : NDArray
            grad ndarray

        state : NDArray or other objects returned by init_state
            The auxiliary state used in optimization.
        """
        assert (isinstance(weight, NDArray))
        assert (isinstance(grad, NDArray))
        lr = self._get_lr(index)
        wd = self._get_wd(index)
        self._update_count(index)

        gamma, mom = state

        grad = (grad + wd * weight) * self.rescale_grad
        if self.clip_gradient is not None:
            grad = clip(grad, -self.clip_gradient, self.clip_gradient)
        mom[:] -= lr * (grad + gamma * mom) + \
            random_normal(0, math.sqrt(2 * self.a * lr), weight.shape, weight.context)
        weight[:] += mom
        gamma[:] += lr * (power(mom, 2) - ones(weight.shape, weight.context))


@register
class mSGNHT_S_ABOBA(mSGNHT):
    """modified Stochastic Gradient Nosé Hoover Thermostat

    Uses the Symmetric splitting integration method described in the paper
    *High-Order Stochastic Gradient Thermostats for Bayesian Learning of Deep Models*
    https://arxiv.org/pdf/1512.07662.pdf

    """

    def update(self, index, weight, grad, state):
        """Update the parameters.

        Parameters
        ----------
        index : int
            An unique integer key used to index the parameters

        weight : NDArray
            weight ndarray

        grad : NDArray
            grad ndarray

        state : NDArray or other objects returned by init_state
            The auxiliary state used in optimization.
        """
        assert (isinstance(weight, NDArray))
        assert (isinstance(grad, NDArray))
        lr = self._get_lr(index)
        wd = self._get_wd(index)
        self._update_count(index)

        gamma, mom = state

        grad = (grad + wd * weight) * self.rescale_grad
        if self.clip_gradient is not None:
            grad = clip(grad, -self.clip_gradient, self.clip_gradient)

        mom[:] *= exp(-lr / 2. * gamma)  # B
        mom[:] -= lr * grad + random_normal(0, math.sqrt(2. * self.a * lr), weight.shape, weight.context)  # O
        mom[:] *= exp(-lr / 2. * gamma)  # B
        weight[:] += lr * mom  # A
        gamma[:] += lr * (power(mom, 2.) - ones(weight.shape, weight.context))  # A


@register
class mSGNHT_S_BADODAB(mSGNHT):
    """modified Stochastic Gradient Nosé Hoover Thermostat

    Uses the Symmetric splitting integration method described in the paper
    *Adaptive Thermostats for Noisy Gradient Systems*
    https://arxiv.org/pdf/1505.06889.pdf
    
    different to the paper uses the modfied (anisotropic friction) SGNHT algorithm

    Parameters
    ----------

    mu : float
        Thermal Mass

    """

    def __init__(self, mu=1.0, **kwargs):
        super(mSGNHT_S_BADODAB, self).__init__(**kwargs)
        self.mu = mu

    def update(self, index, weight, grad, state):
        """Update the parameters.

        Parameters
        ----------
        index : int
            An unique integer key used to index the parameters

        weight : NDArray
            weight ndarray

        grad : NDArray
            grad ndarray

        state : NDArray or other objects returned by init_state
            The auxiliary state used in optimization.
        """
        assert (isinstance(weight, NDArray))
        assert (isinstance(grad, NDArray))
        lr = self._get_lr(index)
        wd = self._get_wd(index)
        self._update_count(index)

        gamma, mom = state

        grad = (grad + wd * weight) * self.rescale_grad
        if self.clip_gradient is not None:
            grad = clip(grad, -self.clip_gradient, self.clip_gradient)

        mom[:] -= lr * grad
        weight[:] += lr / 2.0 * mom
        gamma[:] += lr / 2.0 / self.mu * (power(mom, 2.) - ones(weight.shape, weight.context))
        mom[:] *= exp(-lr / 2. * gamma)
        mom[:] += self.a * sqrt((ones(weight.shape, weight.context) - exp(-lr * 2.0 * gamma)) / (2.0 * gamma)) * \
            random_normal(shape=weight.shape, ctx=weight.context)
        gamma += lr / 2.0 / self.mu * (power(mom, 2.) - ones(weight.shape, weight.context))
        weight[:] += lr / 2.0 * mom


@register
class mSGNHTP_S_BADODAB(mSGNHT):
    """modified Stochastic Gradient Nosé Hoover Thermostat

    Uses the Symmetric splitting integration method described in the paper
    *Adaptive Thermostats for Noisy Gradient Systems*
    https://arxiv.org/pdf/1505.06889.pdf
    
    different to the paper uses the modfied (anisotropic friction) SGNHT algorithm
    
    Includes a gamma prior on the weight penalty

    Parameters
    ----------

    mu : float
        Thermal Mass

    """

    def __init__(self, alpha=1., beta=1., wd_update_step=500, mu=1.0, **kwargs):
        super(mSGNHTP_S_BADODAB, self).__init__(**kwargs)
        self.alpha = alpha
        self.beta = beta
        self.wd_update_step = wd_update_step
        self.wd = 1.0
        self.mu = mu

    def create_state(self, index, weight):
        """Create additional optimizer state such as momentum.

        Parameters
        ----------
        
        weight : NDArray
            The weight data

            
        use this to initialize wd_mult

        """
        lr = self._get_lr(index)
        self._set_wd(index, random.gammavariate(self.alpha, self.beta))
        return (array([self.alpha, self.beta], weight.context, dtype=weight.dtype),  # wd prior parameters
                full(weight.shape, self.a, weight.context, dtype=weight.dtype),  # thermostat
                random_normal(0, math.sqrt(lr), weight.shape, weight.context, dtype=weight.dtype))  # momentum

    def update(self, index, weight, grad, state):
        """Update the parameters.

        Parameters
        ----------
        index : int
            An unique integer key used to index the parameters

        weight : NDArray
            weight ndarray

        grad : NDArray
            grad ndarray

        state : NDArray or other objects returned by init_state
            The auxiliary state used in optimization.
        """
        assert (isinstance(weight, NDArray))
        assert (isinstance(grad, NDArray))
        lr = self._get_lr(index)
        wd = self._get_wd(index)
        self._update_count(index)
        num_step = self.num_update - self.begin_num_update

        prior, gamma, mom = state
        prior += array([weight.size / 2.0, sum(power(weight, 2)) / 2.0])

        if num_step % self.wd_update_step == 0:
            wd = random.gammavariate(prior[0].asscalar(), prior[1].asscalar())
            self._set_wd(index, wd)

        grad = (grad + wd * weight) * self.rescale_grad
        if self.clip_gradient is not None:
            grad = clip(grad, -self.clip_gradient, self.clip_gradient)

        mom[:] -= lr * grad
        weight[:] += lr / 2.0 * mom
        gamma[:] += lr / 2.0 / self.mu * (power(mom, 2.) - ones(weight.shape, weight.context))
        mom[:] *= exp(-lr / 2. * gamma)
        mom[:] += self.a * sqrt((ones(weight.shape, weight.context) - exp(-lr * 2.0 * gamma)) / (2.0 * gamma)) * \
            random_normal(shape=weight.shape, ctx=weight.context)
        gamma += lr / 2.0 / self.mu * (power(mom, 2.) - ones(weight.shape, weight.context))
        weight[:] += lr / 2.0 * mom
