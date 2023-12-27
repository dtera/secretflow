import sys
import time
import logging

import secretflow as sf
from secretflow.ml.boost.ss_xgb_v import Xgb
from secretflow.device.driver import wait, reveal
from secretflow.data import FedNdarray, PartitionWay
from secretflow.data.split import train_test_split
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score, classification_report
from sklearn.datasets import load_svmlight_file

# init log
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

# init all nodes in local Standalone Mode.
sf.init(['alice', 'bob'], address='local')

# init PYU, the Python Processing Unit, process plaintext in each node.
alice = sf.PYU('alice')
bob = sf.PYU('bob')
# carol = sf.PYU('carol')

# init SPU, the Secure Processing Unit,
#           process ciphertext under the protection of a multi-party secure computing protocol
spu = sf.SPU(sf.utils.testing.cluster_def(['alice', 'bob']))


# read data in each party
def read_x(start, end):
    from sklearn.datasets import load_breast_cancer
    x = load_breast_cancer()['data']
    return x[:, start:end]


def read_y():
    from sklearn.datasets import load_breast_cancer
    return load_breast_cancer()['target']


def train(dataset='a9a.train', label_holder_feat_num=50, sample_cnt=-1):
    x, y = load_svmlight_file(f"../../data/{dataset}")
    x = x.toarray()
    if sample_cnt != -1:
        x = x[:sample_cnt]
        y = y[:sample_cnt]
    # alice / bob / carol each hold one third of the features of the data
    v_data = FedNdarray(
        partitions={
            alice: (alice(lambda: x[:, :label_holder_feat_num])()),
            bob: (bob(lambda: x[:, label_holder_feat_num:])()),
            # alice: alice(read_x)(0, 10),
            # bob: bob(read_x)(10, 20),
            # carol: carol(read_x)(20, 30),
        },
        partition_way=PartitionWay.VERTICAL,
    )
    # Y label belongs to alice
    label_data = FedNdarray(
        {alice: (alice(lambda: y)())},
        # partitions={alice: alice(read_y)()},
        partition_way=PartitionWay.VERTICAL,
    )
    # wait IO finished
    wait([p.data for p in v_data.partitions.values()])
    wait([p.data for p in label_data.partitions.values()])
    # split train data and test date
    random_state = 1234
    split_factor = 0.8
    v_train_data, v_test_data = train_test_split(v_data, train_size=split_factor, random_state=random_state)
    v_train_label, v_test_label = train_test_split(label_data, train_size=split_factor, random_state=random_state)
    # run SS-XGB
    xgb = Xgb(spu)
    start = time.time()
    params = {
        # for more detail, see Xgb API doc
        'num_boost_round': 3,
        'max_depth': 5,
        'learning_rate': 0.1,
        'sketch_eps': 0.08,
        'objective': 'logistic',
        'reg_lambda': 0.1,
        'subsample': 1,
        'colsample_by_tree': 1,
        'base_score': 0.5,
    }
    model = xgb.train(params, v_train_data, v_train_label)
    logging.info(f"train time: {time.time() - start}")

    # Do predict
    start = time.time()
    # Now the result is saved in the spu by ciphertext
    spu_yhat = model.predict(v_test_data)
    # reveal for auc, acc and classification report test.
    yhat = reveal(spu_yhat)
    logging.info(f"predict time: {time.time() - start}")
    y = reveal(v_test_label.partitions[alice])
    # get the area under curve(auc) score of classification
    logging.info(f"auc: {roc_auc_score(y, yhat)}")
    binary_class_results = np.where(yhat > 0.5, 1, 0)
    # get the accuracy score of classification
    logging.info(f"acc: {accuracy_score(y, binary_class_results)}")
    # get the report of classification
    print("classification report:")
    print(classification_report(y, binary_class_results))


if __name__ == "__main__":
    # train()
    train('wepay.train', 2, 100000)
