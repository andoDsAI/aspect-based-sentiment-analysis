from tensorflow.keras import callbacks


class EarlyStop(callbacks.EarlyStopping):
    def __init__(self, args, **kwargs):
        super(EarlyStop, self).__init__(**kwargs)
        self.num_aspects = num_aspects

    def get_monitor_value(self, logs):
        logs = logs if logs is not None else {}
        monitor_value = sum([logs.get(f"val_aspect_{i}_f1_score") for i in range(self.num_aspects)]) // self.num_aspects
        return monitor_value
