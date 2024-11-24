from easyocr.easyocr import Reader
from .ocr_interface import OcrInterface
import easyocr
import tqdm


class EasyOcr(OcrInterface):
    reader: Reader | None = None

    def __init__(self):
        pass

    @classmethod
    def init(cls):
        cls.reader = easyocr.Reader(["en"])

    @classmethod
    def string_from_word_image(cls, reader, image_path: str):
        ocr_result = reader.readtext(image_path)
        prediction = ocr_result[0][1] if len(ocr_result) > 0 else ""
        return prediction

    @classmethod
    def from_word_images(cls, image_paths: list[str]) -> list[str]:
        if not cls.reader:
            cls.init()
        assert cls.reader is not None
        return list(
            tqdm.tqdm(
                map(lambda p: cls.string_from_word_image(cls.reader, p), image_paths),
                total=len(image_paths),
                desc="Getting OCR results",
            )
        )

    @classmethod
    def string_from_line_image(cls, reader, image_path: str):
        ocr_result = reader.readtext(image_path)
        return " ".join(list(map(lambda x: x[1], ocr_result)))

    @classmethod
    def from_line_images(cls, image_paths: list[str]) -> list[str]:
        if not cls.reader:
            cls.init()
        assert cls.reader is not None
        return list(
            tqdm.tqdm(
                map(lambda p: cls.string_from_line_image(cls.reader, p), image_paths),
                total=len(image_paths),
                desc="Getting OCR results",
            )
        )
