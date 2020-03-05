import torch as ch
import dill
import os
if int(os.environ.get("NOTEBOOK_MODE", 0)) == 1:
    from tqdm import tqdm_notebook as tqdm
else:
    from tqdm import tqdm

from . import helpers
from . import attack_steps

STEPS = {
    'inf': attack_steps.LinfStep,
    '2': attack_steps.L2Step,
    'unconstrained': attack_steps.UnconstrainedStep
}

class Attacker(ch.nn.Module):
    """
    Attacker (INTERNAL CLASS)

    Attacker class, used to make adversarial examples. 

    This is primarily an internal class, you probably want to be looking at
    AttackerModel, which is how models are actually served (AttackerModel
    uses Attacker).
    """
    def __init__(self, model, dataset):
        """
        Initialize the Attacker 
        - model (PyTorch model [nn.Module]) -- the model to attack
        - dataset (a Dataset class [see datasets.py]) -- only used to get mean and std for normalization
        """
        super(Attacker, self).__init__()
        self.normalize = helpers.InputNormalize(dataset.mean, dataset.std)
        self.model = model

    def forward(self, x, target, *_, constraint, eps, step_size, iterations, criterion,
                random_start=False, random_restarts=False, do_tqdm=False,
                targeted=False, custom_loss=None, should_normalize=True, 
                orig_input=None, use_best=True):
        """
        Implementation of forward (finds adversarial examples)
        - x (ch.tensor): original input
        - See below (AttackerModel forward) for description of named arguments
        Returns: adversarial example for x
        """
        
        # Can provide a different input to make the feasible set around
        # instead of the initial point
        if orig_input is None: orig_input = x.detach()
        orig_input = orig_input.cuda()

        # Multiplier for gradient ascent [untargeted] or descent [targeted]
        m = -1 if targeted else 1

        # Initialize step class
        step = STEPS[constraint](eps=eps, orig_input=orig_input, step_size=step_size)

        def calc_loss(inp, target):
            '''
            Calculates the loss of an input with respect to target labels
            Uses custom loss (if provided) otherwise the criterion
            '''
            if should_normalize:
                inp = self.normalize(inp)
            output = self.model(inp)
            if custom_loss:
                return custom_loss(self.model, inp, target)

            return criterion(output, target), output

        # Main function for making adversarial examples
        def get_adv_examples(x):
            # Random start (to escape certain types of gradient masking)
            if random_start:
                x = ch.clamp(x + step.random_perturb(x), 0, 1)

            iterator = range(iterations)
            if do_tqdm: iterator = tqdm(iterator)

            # Keep track of the "best" (worst-case) loss and its
            # corresponding input
            best_loss = None
            best_x = None

            # A function that updates the best loss and best input
            def replace_best(loss, bloss, x, bx):
                if bloss is None:
                    bx = x.clone().detach()
                    bloss = losses.clone().detach()
                else:
                    replace = m * bloss < m * loss
                    bx[replace] = x[replace].clone().detach()
                    bloss[replace] = loss[replace]

                return bloss, bx

            # PGD iterates    
            for _ in iterator:
                x = x.clone().detach().requires_grad_(True)
                losses, out = calc_loss(x, target)
                assert losses.shape[0] == x.shape[0], 'Shape of losses must match input!'

                loss = ch.mean(losses)
                grad, = ch.autograd.grad(loss, [x])

                with ch.no_grad():
                    args = [losses, best_loss, x, best_x]
                    best_loss, best_x = replace_best(*args) if use_best else (losses, x)

                    x = step.make_step(grad) * m + x
                    x = ch.clamp(x, 0, 1)
                    x = step.project(x)
                    if do_tqdm: iterator.set_description("Current loss: {l}".format(l=loss))

            # Save computation (don't compute last loss) if not use_best
            if not use_best: return x.clone().detach()

            losses, _ = calc_loss(x, target)
            args = [losses, best_loss, x, best_x]
            best_loss, best_x = replace_best(*args)
            return best_x

        # Random restarts: repeat the attack and find the worst-case
        # example for each input in the batch
        if random_restarts:
            to_ret = None

            orig_cpy = x.clone().detach()
            for _ in range(random_restarts):
                adv = get_adv_examples(orig_cpy)

                if to_ret is None:
                    to_ret = adv.detach()

                _, output = calc_loss(adv, target)
                corr, = helpers.accuracy(output, target, topk=(1,), exact=True)
                corr = corr.byte()
                misclass = ~corr
                to_ret[misclass] = adv[misclass]

            adv_ret = to_ret
        else:
            adv_ret = get_adv_examples(x)

        return adv_ret

class AttackerModel(ch.nn.Module):
    def __init__(self, model, dataset):
        super(AttackerModel, self).__init__()
        self.normalizer = helpers.InputNormalize(dataset.mean, dataset.std)
        self.model = model
        self.attacker = Attacker(model, dataset)

    def forward(self, inp, target=None, make_adv=False, with_latent=False,
                    fake_relu=False, with_image=True, **attacker_kwargs):
        if make_adv:
            assert target is not None
            prev_training = bool(self.training)
            self.eval()
            adv = self.attacker(inp, target, **attacker_kwargs)
            if prev_training:
                self.train()

            inp = adv

        if with_image:
            normalized_inp = self.normalizer(inp)
            output = self.model(normalized_inp, with_latent=with_latent,
                                                    fake_relu=fake_relu)
        else:
            output = None

        return output #(output, inp)


## This takes in the standard PyTorch input (No need to normalize like Madry)
## This would allow you to adversarially perturb the input
class MyAttackerModel(ch.nn.Module):
    def __init__(self, model, dataset):
        super(MyAttackerModel, self).__init__()
        import ipdb; ipdb.set_trace
        self.normalizer = helpers.InputNormalize(dataset.mean, dataset.std)
        self.model = model
        self.attacker = Attacker(model, dataset)

    def forward(self, inp, target=None, make_adv=False, with_latent=False,
                    fake_relu=False, with_image=True, **attacker_kwargs):
        if make_adv:
            assert target is not None
            prev_training = bool(self.training)
            self.eval()
            adv = self.attacker(inp, target, **attacker_kwargs)
            if prev_training:
                self.train()

            inp = adv

        if with_image:
            normalized_inp = inp
            output = self.model(normalized_inp, with_latent=with_latent,
                                                    fake_relu=fake_relu)
        else:
            output = None

        return output #(output, inp)
