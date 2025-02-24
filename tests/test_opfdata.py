import sys
sys.path.append("../ai4climate")
import load

root_path = "../../donti_group_shared/AI4Climate/tests"
(
    train_data, 
    val_data, 
    test_data
) = load.load_task(
    "OPFData", 
    "train_small_test_medium", 
    root_path,
    data_frac = 0.01,
    train_frac = 0.1
)

print(train_data)
print(val_data)
print(test_data)