from helper.ocr_interface import OcrInterface

from helper.sampler import WordsSampler
from helper.cer import Cer

from helper.sampler import LinesSampler


class Tester:
    def __init__(self):
        pass

    @staticmethod
    def get_words_cer(ocr: OcrInterface, nforms: int):
        paths, transcriptions = WordsSampler(
            "./words.csv"
        ).get_paths_and_transcriptions(nforms)

        print("Initializing OCR")
        ocr.init()

        print("Converting image words to text")
        predictions = ocr.from_word_images(paths)

        cer = Cer.cer(predictions=predictions, transcriptions=transcriptions)
        print(f"The CER score is {cer}")
        return cer

    @staticmethod
    def get_lines_cer(ocr: OcrInterface, nforms: int):
        paths, transcriptions = LinesSampler(
            "./lines.csv"
        ).get_paths_and_transcriptions(nforms)

        print("Initializing OCR")
        ocr.init()

        print("Converting image lines to text")
        predictions = ocr.from_line_images(paths)

        cer = Cer.cer(predictions=predictions, transcriptions=transcriptions)
        print(f"The CER score is {cer}")
        return cer
