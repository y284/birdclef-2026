import torch

from birdclef import config as C
from birdclef.model import LogMel, Net, cosine_loss, masked_ce


def test_one_frame_per_second():
    assert LogMel()(torch.randn(2, C.WINDOW * C.SR)).shape == (2, 1, C.N_MELS, C.WINDOW * C.FRAMES_PER_SECOND)
    model = Net(7, "tf_efficientnet_b0", pretrained=False).eval()
    with torch.no_grad():
        for seconds in (5, 10):
            clip, frame = model(torch.randn(2, seconds * C.SR))
            assert clip.shape == (2, 7) and frame.shape == (2, seconds, 7)
            # attention pooling: each clip logit is a weighted mean of that class's frame logits
            assert (clip >= frame.amin(1) - 1e-5).all() and (clip <= frame.amax(1) + 1e-5).all()
        assert model.forward_with_embedding(torch.randn(2, 5 * C.SR))[2].shape == (2, 1536)


def test_masked_ce_ignores_masked_classes():
    logits = torch.tensor([[2.0, 5.0, 0.0]])
    target, mask = torch.tensor([[1.0, 0, 0]]), torch.tensor([[1.0, 0, 1]])
    expected = -torch.log_softmax(torch.tensor([2.0, 0.0]), 0)[0]  # class 1 leaves the softmax
    assert torch.isclose(masked_ce(logits, target, mask), expected)
    assert torch.isclose(masked_ce(logits, target, torch.ones(1, 3)), torch.nn.functional.cross_entropy(logits, target))


def test_cosine_loss_skips_samples_without_embedding():
    a = torch.tensor([[1.0, 0], [1.0, 0], [0, 1.0]])
    b = torch.tensor([[2.0, 0], [-1.0, 0], [5.0, 5.0]])
    assert torch.isclose(cosine_loss(a, b, torch.tensor([1.0, 1, 0])), torch.tensor(1.0))  # (0 + 2) / 2
