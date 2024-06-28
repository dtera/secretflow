# coding: utf-8
# Copyright (c) Antfin, Inc. All rights reserved.
import sys
import tensorflow as tf
from itertools import chain

import secretflow as sf
from secretflow.launcher.sl_deepfm_launcher import SLDeepFMLauncher
from examples.app.v_recommendation.deep_fm.data.dataset import load_ml_1m

NUM_USERS = 6040
NUM_MOVIES = 3952
GENDER_VOCAB = ["F", "M"]
AGE_VOCAB = [1, 18, 25, 35, 45, 50, 56]
OCCUPATION_VOCAB = [i for i in range(21)]
GENRES_VOCAB = [
    "Action",
    "Adventure",
    "Animation",
    "Children's",
    "Comedy",
    "Crime",
    "Documentary",
    "Drama",
    "Fantasy",
    "Film-Noir",
    "Horror",
    "Musical",
    "Mystery",
    "Romance",
    "Sci-Fi",
    "Thriller",
    "War",
    "Western",
]

feat_cols = {
    'alice': [
        "UserID",
        "Gender",
        "Age",
        "Occupation",
        "Zip-code",
    ],
    'bob': [
        "MovieID",
        "Rating",
        "Title",
        "Genres",
        "Timestamp",
    ],
}

id_col = {
    'alice': "UserID",
    'bob': "MovieID"
}

drop_cols = {
    'alice': [
        "Zip-code",
    ],
    'bob': [
        "Title",
        "Timestamp",
    ],
}


def load_data(party, feat_dict):
    vdf = load_ml_1m(part=feat_dict, num_sample=50000)
    # preprocess
    data = vdf.drop(columns=list(chain.from_iterable(drop_cols.values())))
    # data = vdf.drop(columns=drop_cols[party])
    data[id_col[party]] = data[id_col[party]].astype("string")
    return data


def preprocess_alice():
    inputs = {
        "UserID": tf.keras.Input(shape=(1,), dtype=tf.string),
        "Gender": tf.keras.Input(shape=(1,), dtype=tf.string),
        "Age": tf.keras.Input(shape=(1,), dtype=tf.int64),
        "Occupation": tf.keras.Input(shape=(1,), dtype=tf.int64),
    }
    user_id_output = tf.keras.layers.Hashing(
        num_bins=NUM_USERS, output_mode="one_hot"
    )
    user_gender_output = tf.keras.layers.StringLookup(
        vocabulary=GENDER_VOCAB, output_mode="one_hot"
    )

    user_age_out = tf.keras.layers.IntegerLookup(
        vocabulary=AGE_VOCAB, output_mode="one_hot"
    )
    user_occupation_out = tf.keras.layers.IntegerLookup(
        vocabulary=OCCUPATION_VOCAB, output_mode="one_hot"
    )

    outputs = {
        "UserID": user_id_output(inputs["UserID"]),
        "Gender": user_gender_output(inputs["Gender"]),
        "Age": user_age_out(inputs["Age"]),
        "Occupation": user_occupation_out(inputs["Occupation"]),
    }
    return tf.keras.Model(inputs=inputs, outputs=outputs)


def preprocess_bob():
    inputs = {
        "MovieID": tf.keras.Input(shape=(1,), dtype=tf.string),
        "Genres": tf.keras.Input(shape=(1,), dtype=tf.string),
    }

    movie_id_out = tf.keras.layers.Hashing(
        num_bins=NUM_MOVIES, output_mode="one_hot"
    )
    movie_genres_out = tf.keras.layers.TextVectorization(
        output_mode='multi_hot', split="whitespace", vocabulary=GENRES_VOCAB
    )
    outputs = {
        "MovieID": movie_id_out(inputs["MovieID"]),
        "Genres": movie_genres_out(inputs["Genres"]),
    }
    return tf.keras.Model(inputs=inputs, outputs=outputs)


def parse_bob(row_sample, label):
    y_t = label["Rating"]
    y = tf.expand_dims(
        tf.where(
            y_t > 3,
            tf.ones_like(y_t, dtype=tf.float32),
            tf.zeros_like(y_t, dtype=tf.float32),
        ),
        axis=1,
    )
    return row_sample, y


def run(party='alice'):
    addresses = {
        'alice': '127.0.0.1:8001',
        'bob': '127.0.0.1:8002',
    }
    cluster_config = {
        'parties': {
            'alice': {
                'address': '127.0.0.1:20001',
                'listen_addr': '0.0.0.0:20001'
            },
            'bob': {
                'address': '127.0.0.1:20002',
                'listen_addr': '0.0.0.0:20002'
            },
        },
        'self_party': party
    }
    preprocess_layer = {'alice': preprocess_alice(), 'bob': preprocess_bob()}
    launcher = SLDeepFMLauncher(self_party=party, parties=None, label_party='bob',
                                address=None, cluster_config=cluster_config,
                                preprocess_layer=preprocess_layer,
                                map_data_set={'alice': None, 'bob': parse_bob}, label_col='Rating')
    # feat_dict = dict([(launcher.get_party(p), feat_cols[p]) for p in feat_cols])
    feat_dict = {launcher.get_party(party): feat_cols[party]}
    data = load_data(party, feat_dict)
    launcher.run(data)


if __name__ == "__main__":
    run(party=sys.argv[1] if len(sys.argv) > 1 else 'alice')
