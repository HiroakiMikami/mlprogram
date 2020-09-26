import fairseq.optim

types = {
    "fairseq.optim.Adafactor": lambda: fairseq.optim.adafactor.Adafactor
}
