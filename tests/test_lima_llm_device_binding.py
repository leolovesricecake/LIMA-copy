from lima_llm.backbone.hf_backbone import HFBackbone


class _FakeDevice:
    def __init__(self, spec: str) -> None:
        self.spec = spec
        if ":" in spec:
            self.type, raw_index = spec.split(":", 1)
            self.index = int(raw_index)
        else:
            self.type = spec
            self.index = None

    def __str__(self) -> str:
        return self.spec


class _FakeCuda:
    def __init__(self) -> None:
        self.set_device_calls = []

    def set_device(self, device) -> None:
        self.set_device_calls.append(device)


class _FakeTorch:
    def __init__(self) -> None:
        self.cuda = _FakeCuda()

    def device(self, spec: str) -> _FakeDevice:
        return _FakeDevice(spec)


def test_prepare_device_binds_exact_cuda_ordinal() -> None:
    torch = _FakeTorch()

    device = HFBackbone._prepare_device(torch, "cuda:2")

    assert str(device) == "cuda:2"
    assert device.index == 2
    assert torch.cuda.set_device_calls == [2]


def test_prepare_device_leaves_cpu_unbound() -> None:
    torch = _FakeTorch()

    device = HFBackbone._prepare_device(torch, "cpu")

    assert str(device) == "cpu"
    assert torch.cuda.set_device_calls == []
