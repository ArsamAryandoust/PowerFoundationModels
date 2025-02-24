import sys
sys.path.append("../ai4climate")
from load import load_task

root_path = "../../donti_group_shared/AI4Climate/tests"
(
    train_data, 
    val_data, 
    test_data
) = load_task(
    "OPFData", 
    "train_small_test_medium", 
    root_path
)

print(train_data)
print(val_data)
print(test_data)