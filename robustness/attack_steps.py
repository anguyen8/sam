import torch as ch

class AttackerStep:
    '''
    Generic class for attacker steps, under perturbation constraints
    specified by an "origin input" and a perturbation magnitude. 
    Must implement project, make_step, and random_perturb
    '''
    def __init__(self, orig_input, eps, step_size):
        '''
        Initialize the attacker step with a given perturbation magnitude
        - eps (float): the perturbation magnitude
        - orig_input (ch.tensor): the original input
        '''
        self.orig_input = orig_input 
        self.eps = eps
        self.step_size = step_size

    def project(self, x):
        '''
        Given an input x, project it back into the feasible set
        - x (ch.tensor): the input to project back
        Returns (ch.tensor): the projected input
        '''
        raise NotImplementedError

    def make_step(self, g):
        '''
        Given a gradient, make the appropriate step according to the
        perturbation constraint (e.g. dual norm maximization for lp norms).
        - g (ch.tensor): the raw gradient
        Returns (ch.tensor): the new input
        '''
        raise NotImplementedError

    def random_perturb(self, x):
        '''
        Given a starting input, take a random step within the feasible set
        '''
        raise NotImplementedError

### Instantiations of the AttackerStep class

# L-infinity threat model
class LinfStep(AttackerStep):
    def project(self, x):
        diff = x - self.orig_input
        diff = ch.clamp(diff, -self.eps, self.eps)
        return diff + self.orig_input

    def make_step(self, g):
        step = ch.sign(g) * self.step_size
        return step

    def random_perturb(self, x):
         return 2 * (ch.rand_like(x) - 0.5) * self.eps

# L2 threat model
class L2Step(AttackerStep):
    def project(self, x):
        diff = x - self.orig_input
        diff = diff.renorm(p=2, dim=0, maxnorm=self.eps)
        return self.orig_input + diff

    def make_step(self, g):
        # Scale g so that each element of the batch is at least norm 1
        g_norm = ch.norm(g.view(g.shape[0], -1), dim=1).view(-1, 1, 1, 1)
        scaled_g = g / (g_norm + 1e-10)
        return scaled_g * self.step_size

    def random_perturb(self, x):
        return (ch.rand_like(x) - 0.5).renorm(p=2, dim=1, maxnorm=self.eps)

# Unconstrained threat model
class UnconstrainedStep(AttackerStep):
    def project(self, x):
        return x

    def make_step(self, g):
        return g * self.step_size

    def random_perturb(self, x):
        return (ch.rand_like(x) - 0.5).renorm(p=2, dim=1, maxnorm=step_size)
