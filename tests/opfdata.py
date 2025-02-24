import sys

# add ai4climate package to path
sys.path.append("../ai4climate")

from load import load_task

# Set up base paths
root_path = "../../donti_group_shared/AI4Climate/tests"

load_task("OPFData", "train_small_test_large", root_path)