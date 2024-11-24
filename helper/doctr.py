from .ocr_interface import OcrInterface
from doctr.models.predictor import OCRPredictor
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
import tqdm


class DoctrOcr(OcrInterface):
    model: OCRPredictor | None = None

    def __init__(self):
        pass

    @classmethod
    def init(cls):
        cls.model = ocr_predictor("db_resnet50", "crnn_vgg16_bn", pretrained=True)

    @staticmethod
    def string_from_image(model, image_path) -> str:
        document = model(DocumentFile.from_images(image_path))
        result = ""
        for page in document.pages:
            for block in page.blocks:
                for line in block.lines:
                    for word in line.words:
                        result += word.value
        return result

    @classmethod
    def from_word_images(cls, image_paths: list[str]) -> list[str]:
        if not cls.model:
            cls.init()
        assert cls.model is not None
        return list(
            tqdm.tqdm(
                map(
                    lambda p: cls.string_from_image(cls.model, p),
                    image_paths,
                ),
                total=len(image_paths),
                desc="Getting OCR results",
            )
        )

    @classmethod
    def from_line_images(cls, image_paths: list[str]) -> list[str]:
        if not cls.model:
            cls.init()
        assert cls.model is not None
        return list(
            tqdm.tqdm(
                map(
                    lambda p: cls.string_from_image(cls.model, p),
                    image_paths,
                ),
                total=len(image_paths),
                desc="Getting OCR results",
            )
        )
