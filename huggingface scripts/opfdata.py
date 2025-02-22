import datasets

class MyDataset(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("0.0.1")

    def _info(self):
        return datasets.DatasetInfo(
            description="OPFData from AI4Climate collection."
        )

    def _split_generators(self, dl_manager):
        # Download or locate your dataset files.
        data_files = {
            "train": "path/to/train.csv",
            "test": "path/to/test.csv"
        }
        downloaded_files = dl_manager.download_and_extract(data_files)
        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": downloaded_files["train"]}),
            datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={"filepath": downloaded_files["test"]}),
        ]

    def _generate_examples(self, filepath):
        # Read your file and yield examples.
        with open(filepath, encoding="utf-8") as f:
            for id, line in enumerate(f):
                # Example: process each line into an example
                # Here you would parse the line as needed.
                yield id, {"text": line.strip(), "label": "positive"}  # adjust logic as required