from keras_ocr.pipeline import Pipeline
import keras_ocr
from .ocr_interface import OcrInterface
import tqdm


class KerasOcr(OcrInterface):
    pipeline: Pipeline | None = None

    def __init__(self):
        pass

    @classmethod
    def init(cls):
        # Downloads weights
        cls.pipeline = keras_ocr.pipeline.Pipeline()

    @classmethod
    def from_word_images(cls, image_paths: list[str]) -> list[str]:
        images = [keras_ocr.tools.read(url) for url in image_paths]
        if not cls.pipeline:
            cls.init()
        assert cls.pipeline is not None
        prediction_groups = cls.pipeline.recognize(images)
        predictions = list(
            map(lambda x: x[0][0] if len(x) > 0 else "", prediction_groups)
        )
        return predictions

    @classmethod
    def string_from_line_image(cls, pipeline, image_path) -> str:
        image = [keras_ocr.tools.read(url) for url in [image_path]]
        prediction_groups = pipeline.recognize(image)
        sorted_line = sorted(prediction_groups[0], key=lambda word: word[1][0][0])
        return " ".join(list(map(lambda x: x[0], sorted_line)))

    @classmethod
    def from_line_images(cls, image_paths: list[str]) -> list[str]:
        if cls.pipeline is None:
            cls.init()
        assert cls.pipeline is not None
        return list(
            tqdm.tqdm(
                map(lambda p: cls.string_from_line_image(cls.pipeline, p), image_paths),
                total=len(image_paths),
                desc="Getting OCR results",
            )
        )
