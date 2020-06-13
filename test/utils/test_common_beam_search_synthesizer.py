import unittest
import torch.nn as nn
import numpy as np
from mlprogram.nn import TrainModel
from mlprogram.synthesizers import CommonBeamSearchSynthesizer


class TestCommonBeamSearchSynthesizer(unittest.TestCase):
    def test_create(self):
        dummy = None
        synthesizer = CommonBeamSearchSynthesizer.create(
            1, dummy, dummy, dummy,
            TrainModel(nn.Linear(1, 1), nn.Linear(2, 2), nn.Linear(3, 3),
                       nn.Linear(5, 5)),
            dummy, dummy)
        self.assertEqual((1, 1), synthesizer.input_reader.weight.shape)

    def test_state_dict(self):
        dummy = None
        synthesizer = CommonBeamSearchSynthesizer(
            1, dummy, dummy, dummy,
            nn.Linear(1, 1), nn.Linear(2, 2), nn.Linear(3, 3), nn.Linear(5, 5),
            dummy, dummy)
        state_dict = synthesizer.state_dict()
        self.assertEqual(8, len(state_dict))
        self.assertEqual((1, 1), state_dict["input_reader.weight"].shape)
        self.assertEqual((1,), state_dict["input_reader.bias"].shape)
        self.assertEqual(
            (2, 2), state_dict["action_sequence_reader.weight"].shape)
        self.assertEqual(
            (2,), state_dict["action_sequence_reader.bias"].shape)
        self.assertEqual((3, 3), state_dict["decoder.weight"].shape)
        self.assertEqual((3,), state_dict["decoder.bias"].shape)
        self.assertEqual((5, 5), state_dict["predictor.weight"].shape)
        self.assertEqual((5,), state_dict["predictor.bias"].shape)

    def test_load_state_dict(self):
        dummy = None
        synthesizer = CommonBeamSearchSynthesizer(
            1, dummy, dummy, dummy,
            nn.Linear(1, 1), nn.Linear(2, 2), nn.Linear(3, 3), nn.Linear(5, 5),
            dummy, dummy)
        state_dict = synthesizer.state_dict()
        synthesizer.load_state_dict(state_dict)

    def test_load_state_dict_from_train_model(self):
        dummy = None
        train_model = TrainModel(
            nn.Linear(1, 1), nn.Linear(2, 2), nn.Linear(3, 3), nn.Linear(5, 5))
        synthesizer = CommonBeamSearchSynthesizer(
            1, dummy, dummy, dummy,
            nn.Linear(1, 1), nn.Linear(2, 2), nn.Linear(3, 3), nn.Linear(5, 5),
            dummy, dummy)
        synthesizer.load_state_dict(train_model.state_dict())
        self.assertTrue(np.array_equal(
            train_model.input_reader.weight.detach().numpy(),
            synthesizer.input_reader.weight.detach().numpy())
        )
        self.assertTrue(np.array_equal(
            train_model.action_sequence_reader.weight.detach().numpy(),
            synthesizer.action_sequence_reader.weight.detach().numpy())
        )
        self.assertTrue(np.array_equal(
            train_model.decoder.weight.detach().numpy(),
            synthesizer.decoder.weight.detach().numpy())
        )
        self.assertTrue(np.array_equal(
            train_model.predictor.weight.detach().numpy(),
            synthesizer.predictor.weight.detach().numpy())
        )


if __name__ == "__main__":
    unittest.main()
