from abc import ABC


class OcrInterface(ABC):
    @classmethod
    def init(cls):
        pass

    @classmethod
    def from_word_images(cls, image_paths: list[str]) -> list[str]:
        pass

    @classmethod
    def from_line_images(cls, image_paths: list[str]) -> list[str]:
        pass
