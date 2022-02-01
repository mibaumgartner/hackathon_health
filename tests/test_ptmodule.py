from medhack.ptmodule.module import BaselineClassification


def test_dummy_input():
    import torch

    model = BaselineClassification()

    batch = (
        torch.rand(4, 3, 64, 64),
        torch.randint(0, 2, (4,))
    )

    model.training_step(batch)
    model.validation_step(batch)
