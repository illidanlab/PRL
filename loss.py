import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

sys.path.append("./torchdp/")

from torchdp.autograd_grad_sample import add_hooks, remove_hooks
from torch.autograd import Variable
import numpy as np

# Loss functions
def loss_coteaching(y_1, y_2, t, forget_rate, ind, noise_or_not):
    noise_or_not = torch.Tensor(noise_or_not).cuda()
    ind = torch.Tensor(ind).cuda().long()
    loss_1 = F.cross_entropy(y_1, t, reduction="none")
    ind_1_sorted = torch.argsort(loss_1).cuda()
    loss_1_sorted = loss_1[ind_1_sorted]

    loss_2 = F.cross_entropy(y_2, t, reduction="none")
    ind_2_sorted = torch.argsort(loss_2).cuda()
    loss_2_sorted = loss_2[ind_2_sorted]

    remember_rate = 1 - forget_rate
    num_remember = int(remember_rate * len(loss_1_sorted))

    pure_ratio_1 = torch.sum(noise_or_not[ind[ind_1_sorted[:num_remember]]]) / float(
        num_remember
    )
    pure_ratio_2 = torch.sum(noise_or_not[ind[ind_2_sorted[:num_remember]]]) / float(
        num_remember
    )

    ind_1_update = ind_1_sorted[:num_remember]
    ind_2_update = ind_2_sorted[:num_remember]
    # exchange
    loss_1_update = F.cross_entropy(y_1[ind_2_update], t[ind_2_update])
    loss_2_update = F.cross_entropy(y_2[ind_1_update], t[ind_1_update])

    return (
        torch.sum(loss_1_update) / num_remember,
        torch.sum(loss_2_update) / num_remember,
        pure_ratio_1,
        pure_ratio_2,
    )


def loss_spl(y_1, y_2, t, forget_rate, ind, noise_or_not):
    noise_or_not = torch.Tensor(noise_or_not).cuda()
    ind = torch.Tensor(ind).cuda().long()
    loss_1 = F.cross_entropy(y_1, t, reduction="none")
    ind_1_sorted = torch.argsort(loss_1).cuda()
    loss_1_sorted = loss_1[ind_1_sorted]

    loss_2 = F.cross_entropy(y_2, t, reduction="none")
    ind_2_sorted = torch.argsort(loss_2).cuda()
    loss_2_sorted = loss_2[ind_2_sorted]

    remember_rate = 1 - forget_rate
    num_remember = int(remember_rate * len(loss_1_sorted))

    pure_ratio_1 = torch.sum(noise_or_not[ind[ind_1_sorted[:num_remember]]]) / float(
        num_remember
    )
    pure_ratio_2 = torch.sum(noise_or_not[ind[ind_2_sorted[:num_remember]]]) / float(
        num_remember
    )

    ind_1_update = ind_1_sorted[:num_remember]
    ind_2_update = ind_2_sorted[:num_remember]
    # exchange
    loss_1_update = F.cross_entropy(y_1[ind_1_update], t[ind_1_update])
    loss_2_update = F.cross_entropy(y_2[ind_2_update], t[ind_2_update])

    return (
        torch.sum(loss_1_update) / num_remember,
        torch.sum(loss_2_update) / num_remember,
        pure_ratio_1,
        pure_ratio_2,
    )


def loss_CO_IGFilter(x, model1, model2, t, forget_rate, ind, noise_or_not):
    # you should assert the model1 and model2 are added hooks
    noise_or_not = torch.Tensor(noise_or_not).cuda()
    ind = torch.Tensor(ind).cuda().long()
    add_hooks(model1)
    add_hooks(model2)
    y_1 = model1(x)
    y_2 = model2(x)

    loss_1 = F.cross_entropy(y_1, t, reduction="none")
    loss_1_bp = torch.sum(loss_1)
    loss_1_bp.backward()

    loss_2 = F.cross_entropy(y_2, t, reduction="none")
    loss_2_bp = torch.sum(loss_2)
    loss_2_bp.backward()
    # get individual gradient
    with torch.no_grad():
        n = y_1.shape[0]
        # computed_sample_grads_1 = []
        computed_grad_norm_1 = torch.zeros(n).cuda()

        for p in model1.parameters():
            grad_layer = torch.flatten(p.grad_sample.detach(), start_dim=1).squeeze()
            del p.grad_sample
            computed_grad_norm_1 = computed_grad_norm_1 + torch.sum(
                grad_layer ** 2, dim=1
            )
        computed_grad_norm_2 = torch.zeros(n).cuda()
        for p in model2.parameters():
            grad_layer = torch.flatten(p.grad_sample.detach(), start_dim=1).squeeze()
            computed_grad_norm_2 = computed_grad_norm_2 + torch.sum(
                grad_layer ** 2, dim=1
            )
            del p.grad_sample

        _, ind_1_sorted = torch.sort(computed_grad_norm_1)
        _, ind_2_sorted = torch.sort(computed_grad_norm_2)

    # ind_1_sorted = np.argsort(loss_1.data).cuda()

    loss_1_sorted = loss_1[ind_1_sorted.squeeze()]

    # loss_2 = F.cross_entropy(y_2, t, reduce = False)
    # ind_2_sorted = np.argsort(loss_2.data).cuda()
    loss_2_sorted = loss_2[ind_2_sorted.squeeze()]

    remember_rate = 1 - forget_rate
    num_remember = int(remember_rate * len(loss_1_sorted))
    num_start = int((1 - remember_rate) * len(loss_1_sorted))

    pure_ratio_1 = torch.sum(noise_or_not[ind[ind_1_sorted[:num_remember]]]) / float(
        num_remember
    )
    pure_ratio_2 = torch.sum(noise_or_not[ind[ind_2_sorted[:num_remember]]]) / float(
        num_remember
    )

    ind_1_update = ind_1_sorted[:num_remember]
    ind_2_update = ind_2_sorted[:num_remember]

    remove_hooks(model1)
    remove_hooks(model2)
    # exchange
    model1.zero_grad()
    model2.zero_grad()
    y_1 = model1(x[ind_2_update, :, :, :])
    y_2 = model2(x[ind_1_update, :, :, :])
    loss_1_update = F.cross_entropy(y_1, t[ind_2_update])
    loss_2_update = F.cross_entropy(y_2, t[ind_1_update])
    return (
        torch.sum(loss_1_update) / num_remember,
        torch.sum(loss_2_update) / num_remember,
        pure_ratio_1,
        pure_ratio_2,
    )


def loss_co_prl_l(
    x,
    model1,
    model2,
    t,
    forget_rate,
    ind,
    noise_or_not,
    which_model_grad=1,
    num_classes=10,
):
    noise_or_not = torch.Tensor(noise_or_not).cuda()
    ind = torch.Tensor(ind).cuda().long()

    y_1 = model1(x)
    y_2 = model2(x)

    loss_1 = F.cross_entropy(y_1, t, reduction="none")
    loss_1_bp = torch.sum(loss_1)
    # loss_1_bp.backward()
    loss_1 = F.mse_loss(
        F.softmax(y_1, dim=1),
        torch.nn.functional.one_hot(t, num_classes=num_classes),
        reduction="none",
    )
    loss_1 = loss_1.sum(dim=1)
    loss_2 = F.cross_entropy(y_2, t, reduction="none")
    loss_2_bp = torch.sum(loss_2)
    # loss_2_bp.backward()
    loss_2 = F.mse_loss(
        F.softmax(y_2, dim=1),
        torch.nn.functional.one_hot(t, num_classes=num_classes),
        reduction="none",
    )
    loss_2 = loss_2.sum(dim=1)
    # get individual gradient
    with torch.no_grad():
        n = y_1.shape[0]
        remember_rate = 1 - forget_rate
        num_remember = int(remember_rate * n)
        num_start = int((1 - remember_rate) * n)

        _, ind_1_sorted = torch.sort(loss_1)
        ind_1_update = ind_1_sorted[:num_remember]

        _, ind_2_sorted = torch.sort(loss_2)
        ind_2_update = ind_2_sorted[:num_remember]

    pure_ratio_1 = torch.sum(noise_or_not[ind[ind_1_sorted[:num_remember]]]) / float(
        num_remember
    )
    pure_ratio_2 = torch.sum(noise_or_not[ind[ind_2_sorted[:num_remember]]]) / float(
        num_remember
    )

    # exchange

    model1.zero_grad()
    model2.zero_grad()
    y_1 = model1(x[ind_2_update, :, :, :])
    y_2 = model2(x[ind_1_update, :, :, :])
    loss_1_update = F.cross_entropy(y_1, t[ind_2_update])
    loss_2_update = F.cross_entropy(y_2, t[ind_1_update])
    return (
        torch.sum(loss_1_update) / num_remember,
        torch.sum(loss_2_update) / num_remember,
        pure_ratio_1,
        pure_ratio_2,
    )


def loss_prl_l(
    x,
    model1,
    model2,
    t,
    forget_rate,
    ind,
    noise_or_not,
    which_model_grad=1,
    num_classes=10,
):
    noise_or_not = torch.Tensor(noise_or_not).cuda()
    ind = torch.Tensor(ind).cuda().long()

    y_1 = model1(x)
    y_2 = model2(x)

    loss_1 = F.cross_entropy(y_1, t, reduction="none")
    loss_1_bp = torch.sum(loss_1)
    # loss_1_bp.backward()
    loss_1 = F.mse_loss(
        F.softmax(y_1, dim=1),
        torch.nn.functional.one_hot(t, num_classes=num_classes),
        reduction="none",
    )
    loss_1 = loss_1.sum(dim=1)
    loss_2 = F.cross_entropy(y_2, t, reduction="none")
    loss_2_bp = torch.sum(loss_2)
    # loss_2_bp.backward()
    loss_2 = F.mse_loss(
        F.softmax(y_2, dim=1),
        torch.nn.functional.one_hot(t, num_classes=num_classes),
        reduction="none",
    )
    loss_2 = loss_2.sum(dim=1)
    # get individual gradient
    with torch.no_grad():
        n = y_1.shape[0]
        remember_rate = 1 - forget_rate
        num_remember = int(remember_rate * n)
        num_start = int((1 - remember_rate) * n)

        _, ind_1_sorted = torch.sort(loss_1)
        ind_1_update = ind_1_sorted[:num_remember]

        _, ind_2_sorted = torch.sort(loss_2)
        ind_2_update = ind_2_sorted[:num_remember]

    pure_ratio_1 = torch.sum(noise_or_not[ind[ind_1_sorted[:num_remember]]]) / float(
        num_remember
    )
    pure_ratio_2 = torch.sum(noise_or_not[ind[ind_2_sorted[:num_remember]]]) / float(
        num_remember
    )

    # do not exchange: spl-mse

    model1.zero_grad()
    model2.zero_grad()
    y_1 = model1(x[ind_1_update, :, :, :])
    y_2 = model2(x[ind_2_update, :, :, :])
    loss_1_update = F.cross_entropy(y_1, t[ind_1_update])
    loss_2_update = F.cross_entropy(y_2, t[ind_2_update])
    return (
        torch.sum(loss_1_update) / num_remember,
        torch.sum(loss_2_update) / num_remember,
        pure_ratio_1,
        pure_ratio_2,
    )


def loss_bootstrapping(
    x,
    model1,
    model2,
    t,
    forget_rate,
    ind,
    noise_or_not,
    which_model_grad=1,
    num_classes=10,
):
    noise_or_not = torch.Tensor(noise_or_not).cuda()
    ind = torch.Tensor(ind).cuda().long()
    n = t.shape[0]
    remember_rate = 1 - forget_rate
    num_remember = int(remember_rate * n)

    y_1 = model1(x)
    outputs1 = F.softmax(y_1, dim=1)
    _, pred1 = torch.max(outputs1.data, 1)

    loss_1_update = 0.8 * F.cross_entropy(y_1, t) + 0.2 * F.cross_entropy(y_1, pred1)

    y_2 = model2(x)
    outputs2 = F.softmax(y_2, dim=1)
    _, pred2 = torch.max(outputs2.data, 1)

    # loss 2 does not affect bootstrapping method, here is only for main.py to be consistent
    loss_2_update = 0.8 * F.cross_entropy(y_2, t) + 0.2 * F.cross_entropy(y_2, pred2)

    # pure ratio does not exist in booststrapping method, here pure ratio is meaningless, only for main.py to be consistent.
    pure_ratio_1 = torch.sum(noise_or_not[ind[:num_remember]]) / float(num_remember)
    pure_ratio_2 = torch.sum(noise_or_not[ind[:num_remember]]) / float(num_remember)

    return loss_1_update, loss_2_update, pure_ratio_1, pure_ratio_2


def loss_CO_DoubleRobust(
    x, model1, model2, t, forget_rate, ind, noise_or_not, which_model_grad=1
):
    # you should assert the model1 and model2 are added hooks
    noise_or_not = torch.Tensor(noise_or_not).cuda()
    ind = torch.Tensor(ind).cuda().long()
    add_hooks(model1)
    add_hooks(model2)

    y_1 = model1(x)
    y_2 = model2(x)

    loss_1 = F.cross_entropy(y_1, t, reduction="none")
    loss_1_bp = torch.sum(loss_1)
    loss_1_bp.backward()
    loss_1 = F.mse_loss(y_1, t)

    loss_2 = F.cross_entropy(y_2, t, reduction="none")
    loss_2_bp = torch.sum(loss_2)
    loss_2_bp.backward()
    loss_2 = F.mse_loss(y_2, t)
    # get individual gradient
    with torch.no_grad():
        n = y_1.shape[0]
        remember_rate = 1 - forget_rate
        num_remember = int(remember_rate * n)
        num_start = int((1 - remember_rate) * n)

        computed_grad_norm_1 = torch.zeros(n).cuda()
        for p in model1.parameters():
            grad_layer = torch.flatten(p.grad_sample.detach(), start_dim=1).squeeze()
            grad_layer = grad_layer - grad_layer.mean(dim=0)
            del p.grad_sample
            computed_grad_norm_1 = computed_grad_norm_1 + torch.sum(
                grad_layer ** 2, dim=1
            )

        _, ind_1_sorted_gd = torch.sort(computed_grad_norm_1)
        _, ind_1_sorted_loss = torch.sort(loss_1)
        ind_1_sorted_gd = ind_1_sorted_gd[:num_remember]
        ind_1_sorted_loss = ind_1_sorted_loss[:num_remember]
        # print(ind_1_sorted_loss.shape)
        # print(ind_1_sorted_gd.shape)
        # print(num_remember)
        ind_1_update = torch.from_numpy(
            np.intersect1d(
                ind_1_sorted_gd.cpu().numpy(), ind_1_sorted_loss.cpu().numpy()
            )
        )
        # print(ind_1_update.shape)
        computed_grad_norm_2 = torch.zeros(n).cuda()
        for p in model2.parameters():
            grad_layer = torch.flatten(p.grad_sample.detach(), start_dim=1).squeeze()
            grad_layer = grad_layer - grad_layer.mean(dim=0)
            del p.grad_sample
            computed_grad_norm_2 = computed_grad_norm_2 + torch.sum(
                grad_layer ** 2, dim=1
            )

        _, ind_2_sorted_gd = torch.sort(computed_grad_norm_2)
        _, ind_2_sorted_loss = torch.sort(loss_2)
        ind_2_sorted_gd = ind_2_sorted_gd[:num_remember]
        ind_2_sorted_loss = ind_2_sorted_loss[:num_remember]
        ind_2_update = torch.from_numpy(
            np.intersect1d(
                ind_2_sorted_gd.cpu().numpy(), ind_2_sorted_loss.cpu().numpy()
            )
        )

    pure_ratio_1 = torch.sum(noise_or_not[ind_1_update]) / float(ind_1_update.shape[0])
    pure_ratio_2 = torch.sum(noise_or_not[ind_2_update]) / float(ind_1_update.shape[0])

    remove_hooks(model1)
    remove_hooks(model2)
    # exchange

    model1.zero_grad()
    model2.zero_grad()
    y_1 = model1(x[ind_2_update, :, :, :])
    y_2 = model2(x[ind_1_update, :, :, :])
    loss_1_update = F.cross_entropy(y_1, t[ind_2_update])
    loss_2_update = F.cross_entropy(y_2, t[ind_1_update])
    return (
        torch.sum(loss_1_update) / num_remember,
        torch.sum(loss_2_update) / num_remember,
        pure_ratio_1,
        pure_ratio_2,
    )
