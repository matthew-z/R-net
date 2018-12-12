from allennlp.common.testing import ModelTestCase
from qa.squad.rnet import RNet

class RNetDynamicTest(ModelTestCase):
    def setUp(self):
        super().setUp()
        self.set_up_model('tests/fixtures/rnet/experiment_dynamic.jsonnet',
                          'tests/fixtures/data/squad.json')

    def test_model_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.param_file)
