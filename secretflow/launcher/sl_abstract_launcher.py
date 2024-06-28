import secretflow as sf
from secretflow.ml.nn import SLModel


class SLAbstractLauncher:
    def __init__(self, self_party='alice', parties=['alice', 'bob'], label_party='bob', address="local",
                 cluster_config=None, **kwargs):
        sf.shutdown()
        sf.init(parties, address=address, cluster_config=cluster_config, log_to_driver=True)

        self.parties = dict([(p, sf.PYU(p)) for p in (parties if parties else list(cluster_config['parties'].keys()))])
        self.party = self.parties[self_party]
        self.device_y = self.parties[label_party]
        self.hold_label = True if self_party == label_party else False
        self.batch_size = kwargs.pop('batch_size', 128)
        self.repeat_count = kwargs.pop('repeat_count', 5)
        self.map_data_set = kwargs.pop('map_data_set', None)
        self.label_col = kwargs.pop('label_col', 'y')
        self.epochs = kwargs.pop('epochs', 5)
        self.random_seed = kwargs.pop('random_seed', 123)
        self.base_dnn_units_size = kwargs.pop('base_dnn_units_size', [256, 32])
        self.base_dnn_activation = kwargs.pop('base_dnn_activation', "relu")
        self.fuse_dnn_units_size = kwargs.pop('fuse_dnn_units_size', [256, 256, 32])
        self.fuse_dnn_activation = kwargs.pop('fuse_dnn_activation', "relu")
        self.preprocess_layer = kwargs.pop('preprocess_layer', None)
        self.kwargs = kwargs

    def _create_dataset_builder(self, party, batch_size=128, repeat_count=5):
        def dataset_builder(x):
            import pandas as pd
            import tensorflow as tf

            x = [dict(t) if isinstance(t, pd.DataFrame) else t for t in x]
            x = x[0] if len(x) == 1 else tuple(x)
            data_set = (
                tf.data.Dataset.from_tensor_slices(x).batch(batch_size).repeat(repeat_count)
            )

            map_data_set = self.map_data_set[party] if isinstance(self.map_data_set, dict) else map_data_set
            if map_data_set is not None:
                data_set = data_set.map(map_data_set)
            return data_set

        return dataset_builder

    def create_base_model(self, party):
        pass

    def create_fuse_model(self):
        pass

    def get_party(self, party=None):
        return self.parties[party if party else str(self.party)]

    def run(self, data):
        # data_builder_dict = dict([(self.parties[p_name], self._create_dataset_builder(party=p_name,
        #                                                                               batch_size=self.batch_size,
        #                                                                               repeat_count=self.repeat_count))
        #                           for p_name in self.parties])
        data_builder_dict = {self.party: self._create_dataset_builder(str(self.party),
                                                                      self.batch_size, self.repeat_count)}
        label = data[self.label_col] if self.hold_label else None
        if self.hold_label:
            data = data.drop(columns=self.label_col)
        base_model_dict = dict([(self.parties[p_name], self.create_base_model(p_name)) for p_name in self.parties])
        # base_model_dict = {self.party: self.create_base_model(str(self.party))}
        model_fuse = self.create_fuse_model

        sl_model = SLModel(
            base_model_dict=base_model_dict,
            device_y=self.device_y,
            model_fuse=model_fuse,
        )
        history = sl_model.fit(
            data,
            label,
            epochs=self.epochs,
            batch_size=self.batch_size,
            random_seed=self.random_seed,
            dataset_builder=data_builder_dict,
        )
        global_metric = sl_model.evaluate(
            data,
            label,
            batch_size=self.batch_size,
            random_seed=self.random_seed,
            dataset_builder=data_builder_dict,
        )
