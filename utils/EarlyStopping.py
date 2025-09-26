class EarlyStopping:
    def __init__(self, patience=50, min_delta=1e-3):
        self.patience = patience
        self.min_delta = min_delta
        self.best_score = -float("inf")
        self.counter = 0
        self.should_stop = False

    def step(self, current_score):
        if current_score > self.best_score + self.min_delta:
            self.best_score = current_score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
