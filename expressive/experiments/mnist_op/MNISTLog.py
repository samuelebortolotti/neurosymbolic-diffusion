from expressive.methods.logger import Log


class MNISTLog(Log):
    loss = 0
    semantic_loss = 0
    KL_loss = 0

    def create_dict(self, iterations: int) -> dict:
        return {
            "loss": self.loss / iterations,
            "semantic_loss": self.semantic_loss / iterations,
            "KL_loss": self.KL_loss / iterations,
        }
