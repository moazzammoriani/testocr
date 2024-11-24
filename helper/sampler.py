import pandas
import random


class WordsSampler:
    """Helper class to sample IAM word image paths and their transcriptions"""

    def __init__(self, csv_path) -> None:
        self.words: pandas.DataFrame = pandas.read_csv(csv_path, dtype=str)

    def get_random_form_ids(self, nforms: int) -> list[str]:
        """Returns `nform` number of random form_ids from the words csv.
        For example,
        s = Sampler('words.csv')
        s.get_random_form_ids(3) -> ['c03-007f', 'c01-014', 'l07-085']
        """
        words = self.words
        unique_form_ids = words.form_id.unique().tolist()
        return random.choices(population=unique_form_ids, k=nforms)

    def get_paths_and_transcriptions(
        self, nforms: int = 7
    ) -> tuple[list[str], list[str]]:
        """Returns a tuple containing randomly sample of word image file paths and their
        corresponding transcriptions, respectively."""

        words = self.words
        form_ids = self.get_random_form_ids(nforms)
        filtered_words = words.loc[words["form_id"].isin(form_ids)]
        image_file_paths = filtered_words.loc[:, "file_path"].tolist()
        transcriptions = filtered_words.loc[:, "transcription"].tolist()
        return image_file_paths, transcriptions


class LinesSampler:
    """Helper class to sample IAM word image paths and their transcriptions"""

    def __init__(self, csv_path) -> None:
        self.lines: pandas.DataFrame = pandas.read_csv(csv_path, dtype=str)

    def get_random_form_ids(self, nforms: int) -> list[str]:
        """Returns `nform` number of random form_ids from the words csv.
        For example,
        s = Sampler('words.csv')
        s.get_random_form_ids(3) -> ['c03-007f', 'c01-014', 'l07-085']
        """
        words = self.lines
        unique_form_ids = words.form_id.unique().tolist()
        return random.choices(population=unique_form_ids, k=nforms)

    def get_paths_and_transcriptions(
        self, nforms: int = 7
    ) -> tuple[list[str], list[str]]:
        """Returns a tuple containing randomly sample of word image file paths and their
        corresponding transcriptions, respectively."""

        lines = self.lines
        form_ids = self.get_random_form_ids(nforms)
        filtered_words = lines.loc[lines["form_id"].isin(form_ids)]
        image_file_paths = filtered_words.loc[:, "file_path"].tolist()
        transcriptions = filtered_words.loc[:, "transcription"].tolist()
        return image_file_paths, transcriptions
