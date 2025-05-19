Function RANDOM_FOREST(Dataset D, Integer N_trees, Integer m_features):
    Forest ← []

    For i from 1 to N_trees do:
        Sample Di ← BOOTSTRAP_SAMPLE(D)
        Tree Ti ← BUILD_DECISION_TREE(Di, m_features)
        Add Ti to Forest

    Return Forest

Function BUILD_DECISION_TREE(Data Di, Integer m):
    If stopping_condition_met(Di):
        Return LEAF_NODE(Most_common_label(Di))

    Features ← RANDOM_SUBSET(All_features, m)
    Best_feature, threshold ← FIND_BEST_SPLIT(Di, Features)

    Left_data, Right_data ← SPLIT_DATA(Di, Best_feature, threshold)

    Left_node ← BUILD_DECISION_TREE(Left_data, m)
    Right_node ← BUILD_DECISION_TREE(Right_data, m)

    Return NODE(Best_feature, threshold, Left_node, Right_node)

Function PREDICT_RANDOM_FOREST(Forest, Instance x):
    Predictions ← [PREDICT_TREE(Ti, x) for each Ti in Forest]
    Return MAJORITY_VOTE(Predictions)  // or AVERAGE for regression
