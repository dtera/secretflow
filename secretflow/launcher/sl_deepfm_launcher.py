from secretflow.launcher.sl_abstract_launcher import SLAbstractLauncher
from examples.app.v_recommendation.deep_fm.model.deepfm_model import DeepFMbase, DeepFMfuse


class SLDeepFMLauncher(SLAbstractLauncher):
    def __init__(self, self_party='alice', parties=['alice', 'bob'], label_party='bob', address="local", **kwargs):
        super().__init__(self_party, parties, label_party, address, **kwargs)
        self.fm_embedding_dim = kwargs.pop('fm_embedding_dim', 16)

    def create_base_model(self, party):
        def create_base_model():
            import tensorflow as tf

            dnn_units_size = self.base_dnn_units_size[party] if isinstance(self.base_dnn_units_size,
                                                                           dict) else self.base_dnn_units_size
            dnn_activation = self.base_dnn_activation[party] if isinstance(self.base_dnn_activation,
                                                                           dict) else self.base_dnn_activation
            pl = self.preprocess_layer[party] if isinstance(self.preprocess_layer, dict) else self.preprocess_layer
            fm_dim = self.fm_embedding_dim[party] if isinstance(self.fm_embedding_dim, dict) else self.fm_embedding_dim
            model = DeepFMbase(
                dnn_units_size=dnn_units_size,
                dnn_activation=dnn_activation,
                preprocess_layer=pl,
                fm_embedding_dim=fm_dim,
                **self.kwargs
            )
            model.compile(
                loss=tf.keras.losses.binary_crossentropy,
                optimizer=tf.keras.optimizers.Adam(),
                metrics=[
                    tf.keras.metrics.AUC(),
                    tf.keras.metrics.Precision(),
                    tf.keras.metrics.Recall(),
                ],
            )
            return model

        return create_base_model

    def create_fuse_model(self):
        import tensorflow as tf

        model = DeepFMfuse(dnn_units_size=self.fuse_dnn_units_size,
                           dnn_activation=self.fuse_dnn_activation,
                           **self.kwargs)
        model.compile(
            loss=tf.keras.losses.binary_crossentropy,
            optimizer=tf.keras.optimizers.Adam(),
            metrics=[
                tf.keras.metrics.AUC(),
                tf.keras.metrics.Precision(),
                tf.keras.metrics.Recall(),
            ],
        )
        return model
