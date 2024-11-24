from evaluate import load


class Cer:
    def __init__(self):
        pass

    @staticmethod
    def cer(predictions, transcriptions):
        return load("cer").compute(predictions=predictions, references=transcriptions)
