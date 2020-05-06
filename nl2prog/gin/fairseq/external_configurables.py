import gin
import fairseq.optim

gin.external_configurable(fairseq.optim.adafactor.Adafactor,
                          module="fairseq.optim")
