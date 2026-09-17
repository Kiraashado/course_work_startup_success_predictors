import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from modeling_report import mean_pr_curve


def test_pr_curves_never_mix_scales_of_distinct_models():
    # Each fold ranks perfectly; pooling scores would incorrectly reduce AP.
    pred = pd.DataFrame({'repeat': [0]*4, 'fold': [0, 0, 1, 1],
                         'y': [1, 0, 1, 0], 'probability': [.4, .3, .9, .8]})
    assert average_precision_score(pred.y, pred.probability) < 1
    np.testing.assert_allclose(mean_pr_curve(pred, np.array([.25, .75])), 1)


def test_pr_step_area_agrees_with_weighted_fold_ap():
    pred = pd.DataFrame({'repeat': [0]*8, 'fold': [0]*3+[1]*5,
                         'y': [1, 0, 1, 0, 1, 0, 0, 1],
                         'probability': [.9, .8, .7, .95, .85, .75, .65, .55]})
    # Midpoints avoid discontinuities; 10000 cells resolve all recall steps here.
    grid = (np.arange(10000)+.5)/10000
    expected = sum(len(g)*average_precision_score(g.y,g.probability)
                   for _,g in pred.groupby('fold'))/len(pred)
    assert abs(mean_pr_curve(pred, grid).mean()-expected) < 1e-10
