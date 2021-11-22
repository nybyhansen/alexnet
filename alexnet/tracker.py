class Tracker:
    def __init__(self):
        self.losses = []

    def submit(self, batch_loss):
        self.losses.append(batch_loss)
