import torch
import torch.nn.functional as F
from torch import Tensor

from .logreg import LogReg


def masked_accuracy(logits: Tensor, labels: Tensor):
    if len(logits) == 0:
        return 0
    pred = torch.argmax(logits, dim=1)
    acc = pred.eq(labels).sum() / len(logits) * 100
    return acc.item()


def accuracy(logits: Tensor, labels: Tensor, masks: list[Tensor]):
    accs = []
    for mask in masks:
        acc = masked_accuracy(logits[mask], labels[mask])
        accs.append(acc)
    return accs


def linear_evaluation(z, labels, masks, lr=0.01, max_epoch=100):
    z = z.detach()
    hid_dim, num_classes = z.shape[1], int(labels.max()) + 1

    classifier = LogReg(hid_dim, num_classes).to(z.device)
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=lr, weight_decay=0.0)

    for epoch in range(1, max_epoch + 1):
        classifier.train()
        optimizer.zero_grad(set_to_none=True)

        logits = classifier(z[masks[0]])
        loss = F.cross_entropy(logits, labels[masks[0]])
        loss.backward()
        optimizer.step()
    
    with torch.no_grad():
        classifier.eval()
        logits = classifier(z)
        accs = accuracy(logits, labels, masks)

    return accs
