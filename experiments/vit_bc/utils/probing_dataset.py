from torch.utils.data import Dataset


class ProbingDataset(Dataset):
    def __init__(self, data: list):
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> dict:
        return self.data[index]

    def get_feature_range(self, feature: str) -> tuple[int, int]:
        assert (
            feature in self.data[0].keys()
        ), f"Please enter a feature in dataset: {self.data[0].keys()}"
        min_feature_value, max_feature_value = (
            self.data[0][feature],
            self.data[0][feature],
        )
        for entry in self.data:
            if entry[feature] > max_feature_value:
                max_feature_value = entry[feature]
            elif entry[feature] < min_feature_value:
                min_feature_value = entry[feature]
        return (min_feature_value, max_feature_value)


class ProbingDatasetCleaned(Dataset):
    def __init__(self, data: list):
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> tuple:
        return self.data[index]
