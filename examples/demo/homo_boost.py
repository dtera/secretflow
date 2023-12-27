from sklearn.datasets import load_svmlight_file

import secretflow as sf
from secretflow.data import FedNdarray, PartitionWay
from secretflow.data.base import Partition
from secretflow.data.horizontal.dataframe import HDataFrame
from secretflow.ml.boost.homo_boost import SFXgboost
from secretflow.security.aggregation import SecureAggregator
from secretflow.security.compare import SPUComparator
from secretflow.utils.simulation.datasets import load_dermatology

# In case you have a running secretflow runtime already.
sf.shutdown()

sf.init(['alice', 'bob', 'charlie'], address='local')
alice, bob, charlie = sf.PYU('alice'), sf.PYU('bob'), sf.PYU('charlie')

aggr = SecureAggregator(charlie, [alice, bob])
spu = sf.SPU(sf.utils.testing.cluster_def(['alice', 'bob']))
comp = SPUComparator(spu)


def horizental_fed_train(dataset="a9a.train", data_size=10000):
    params = {
        'max_depth': 5,  # max depth
        'eta': 0.3,  # learning rate
        'objective': 'binary:logistic',
        # objection function，support "binary:logistic","reg:logistic","multi:softmax","multi:softprob","reg:squarederror"
        'min_child_weight': 1,  # The minimum value of weight
        'lambda': 0.1,  # L2 regularization term on weights (xgb's lambda)
        'alpha': 0,  # L1 regularization term on weights (xgb's alpha)
        'max_bin': 10,  # Max num of binning
        'num_class': 6,  # Only required in multi-class classification
        'gamma': 0,  # Same to min_impurity_split,The minimux gain for a split
        'subsample': 1.0,  # Subsample rate by rows
        'colsample_bytree': 1.0,  # Feature selection rate by tree
        'colsample_bylevel': 1.0,  # Feature selection rate by level
        'eval_metric': 'auc',  # supported eval metric：
        'hess_key': 'hess',
        'grad_key': 'grad',
        'label_key': 'label',
    }

    x, y = load_svmlight_file(f"../../data/{dataset}")
    x = x.toarray()
    ndarr = FedNdarray(
        {
            alice: (alice(lambda: x[:data_size])()),
            bob: (bob(lambda: x[data_size:])()),
        },
        partition_way=PartitionWay.HORIZONTAL,
    )
    data = HDataFrame(
        aggregator=aggr,
        comparator=comp,
        partitions={pyu: Partition(pyuobj) for pyu, pyuobj in ndarr.partitions.items()},
    )

    bst = SFXgboost(server=charlie, clients=[alice, bob])
    bst.train(data, data, params=params, num_boost_round=6)


def train():
    data = load_dermatology(parts=[alice, bob], aggregator=aggr,
                            comparator=comp)
    data.fillna(value=0, inplace=True)
    params = {
        # XGBoost parameter tutorial
        # https://xgboost.readthedocs.io/en/latest/parameter.html
        'max_depth': 4,  # max depth
        'eta': 0.3,  # learning rate
        'objective': 'multi:softmax',
        # objection function，support "binary:logistic","reg:logistic","multi:softmax","multi:softprob","reg:squarederror"
        'min_child_weight': 1,  # The minimum value of weight
        'lambda': 0.1,  # L2 regularization term on weights (xgb's lambda)
        'alpha': 0,  # L1 regularization term on weights (xgb's alpha)
        'max_bin': 10,  # Max num of binning
        'num_class': 6,  # Only required in multi-class classification
        'gamma': 0,  # Same to min_impurity_split,The minimux gain for a split
        'subsample': 1.0,  # Subsample rate by rows
        'colsample_bytree': 1.0,  # Feature selection rate by tree
        'colsample_bylevel': 1.0,  # Feature selection rate by level
        'eval_metric': 'auc',  # supported eval metric：
        # 1. rmse
        # 2. rmsle
        # 3. mape
        # 4. logloss
        # 5. error
        # 6. error@t
        # 7. merror
        # 8. mlogloss
        # 9. auc
        # 10. aucpr
        # Special params in SFXgboost
        # Required
        'hess_key': 'hess',
        # Required, Mark hess columns, optionally choosing a column name that is not in the data set
        'grad_key': 'grad',  # Required，Mark grad columns, optionally choosing a column name that is not in the data set
        'label_key': 'class',
        # Required，ark label columns, optionally choosing a column name that is not in the data set
    }

    bst = SFXgboost(server=charlie, clients=[alice, bob])
    bst.train(data, data, params=params, num_boost_round=6)


if __name__ == "__main__":
    train()
    # horizental_fed_train()
