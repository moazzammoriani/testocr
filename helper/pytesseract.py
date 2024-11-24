from .ocr_interface import OcrInterface
import pytesseract
from PIL import Image
import tqdm


class PytesseractOcr(OcrInterface):

    def __init__(self):
        pass

    @classmethod
    def init(cls):
        pass

    @classmethod
    def string_from_image(cls, image_path: str):
        result = ""
        with Image.open(image_path) as pillow_image:
            converted_image = pytesseract.image_to_string(pillow_image)
            result = converted_image.replace("\n", "").replace("\x0c", "")
        return result

    @classmethod
    def from_word_images(cls, image_paths: list[str]) -> list[str]:
        return list(
            tqdm.tqdm(
                map(cls.string_from_image, image_paths),
                total=len(image_paths),
                desc="Getting OCR results",
            )
        )

    @classmethod
    def from_line_images(cls, image_paths: list[str]) -> list[str]:
        return list(
            tqdm.tqdm(
                map(cls.string_from_image, image_paths),
                total=len(image_paths),
                desc="Getting OCR results",
            )
        )
