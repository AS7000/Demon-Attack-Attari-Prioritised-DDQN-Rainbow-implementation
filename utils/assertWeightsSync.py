import numpy as np
def assert_weight_sync(model1,model2):
    q_weights = model1.get_weights()
    target_weights = model2.get_weights()
    for qw, tw in zip(q_weights, target_weights):
        assert np.array_equal(qw, tw), "Q-Network and Target-Network weights differ!"