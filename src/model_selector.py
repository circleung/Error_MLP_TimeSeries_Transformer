from models.rnn import RNNBaseModel
from models.model_lightning import LitRNNBaseModel
from models.trnasformer_decoder import SimpleDecoderOnlyTransformer
from models.crossattention_transformer import CrossAttentionTransformer
from models.lstm import LSTMBaseModel
from models.transformer_encoder import SimpleEncoderOnlyTransformer


class ModelSelector:
    available_models = {
        "rnn": RNNBaseModel,
        "transformer_decoder": SimpleDecoderOnlyTransformer,
        "crossattention_transformer": CrossAttentionTransformer,
        "lstm": LSTMBaseModel,
        "transformer_encoder": SimpleEncoderOnlyTransformer,
    }

    def __new__(cls, model_name: str, backbone_kwargs=None, lightning_kwargs=None):
        if model_name not in cls.available_models:
            raise ValueError(
                f"Unknown model: {model_name}. Choose from {list(cls.available_models)}"
            )
        base_model_cls = cls.available_models[model_name]
        if not backbone_kwargs:
            raise TypeError("You must provide backbone_kwargs for this model!")
        base_model = base_model_cls(**backbone_kwargs)
        lit_model = LitRNNBaseModel(backbone=base_model, **(lightning_kwargs or {}))
        return base_model, lit_model


def test_model_selector_success():

    backbone_kwargs = {
        "input_size": 17,
        "hidden_size": 32,
        "num_layers": 1,
        "device": "cpu",
        "num_continuous": 7,
    }
    lightning_kwargs = {"tolerance": 0.05}

    base, lit = ModelSelector(
        "RNN", backbone_kwargs=backbone_kwargs, lightning_kwargs=lightning_kwargs
    )
    assert isinstance(base, RNNBaseModel), f"Expected RNNBaseModel, got {type(base)}"
    assert isinstance(
        lit, LitRNNBaseModel
    ), f"Expected LitRNNBaseModel, got {type(lit)}"
    assert hasattr(lit, "backbone")
    assert lit.backbone is base
    # Test that model is functional
    import torch

    dummy_input = torch.randn(2, 24, 17)
    output = base(dummy_input)
    assert output.shape[-1] == 7, "Output should have num_continuous=7 features"

    print("✅ test_model_selector_success passed!")


def test_model_selector_failure():
    try:
        ModelSelector("NONEXISTENT", backbone_kwargs={}, lightning_kwargs={})
    except ValueError as e:
        assert "Unknown model" in str(e)
        print("✅ test_model_selector_failure passed!")
    else:
        raise AssertionError("ModelSelector did not raise ValueError for unknown model")


def test_model_selector_missing_kwargs():
    try:
        ModelSelector("RNN")
    except TypeError:
        print("✅ test_model_selector_missing_kwargs passed!")
    else:
        raise AssertionError(
            "ModelSelector did not raise TypeError for missing required kwargs."
        )


if __name__ == "__main__":
    test_model_selector_success()
    test_model_selector_failure()
    test_model_selector_missing_kwargs()
