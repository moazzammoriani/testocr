from .ocr_interface import OcrInterface
from paddleocr import PaddleOCR
import tqdm


class PaddleOcr(OcrInterface):
    ocr: PaddleOCR | None = None

    def __init__(self):
        pass

    @classmethod
    def init(cls):
        # Downloads weights
        cls.ocr = PaddleOCR(use_angle_cls=True, lang="en")

    @classmethod
    def string_from_word_image(cls, ocr, image_path):
        # Passing `ocr` as an argument ensures it has already passed the None check
        ocr_result = ocr.ocr(image_path, cls=True)
        return ocr_result[0][0][1][0] if ocr_result[0] else ""

    @classmethod
    def from_word_images(cls, image_paths: list[str]) -> list[str]:
        if not cls.ocr:
            cls.init()
        assert cls.ocr is not None
        return list(
            tqdm.tqdm(
                map(lambda p: cls.string_from_word_image(cls.ocr, p), image_paths),
                total=len(image_paths),
                desc="Getting OCR results",
            )
        )

    @classmethod
    def string_from_line_image(cls, ocr, image_path):
        ocr_result = ocr.ocr(image_path, cls=True)
        return (
            " ".join(list(map(lambda x: x[1][0], ocr_result[0])))
            if ocr_result[0]
            else ""
        )

    @classmethod
    def from_line_images(cls, image_paths: list[str]) -> list[str]:
        if not cls.ocr:
            cls.init()
        assert cls.ocr is not None
        return list(
            tqdm.tqdm(
                map(lambda p: cls.string_from_line_image(cls.ocr, p), image_paths),
                total=len(image_paths),
                desc="Getting OCR results",
            )
        )
