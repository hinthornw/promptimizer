from promptim.optimizers import (
    load_optimizer,
    FeedbackGuidedOptimizer,
    MetaPromptOptimizer,
    FewShotOptimizer,
)


def test_config_kind():
    optimizers = [FewShotOptimizer, MetaPromptOptimizer, FeedbackGuidedOptimizer]
    _MAP = {OptimizerCls.config_cls.kind: OptimizerCls for OptimizerCls in optimizers}
    assert len(_MAP) == len(optimizers)

    for kind in _MAP:
        loaded = load_optimizer({"kind": kind})
        assert isinstance(loaded, _MAP[kind])
