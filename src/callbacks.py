from tensorflow.keras import callbacks


class EarlyStop(callbacks.EarlyStopping):
    def __init__(self, args, **kwargs):
        super(EarlyStop, self).__init__(**kwargs)
        self.args = args

    def get_monitor_value(self, logs):
        logs = logs if logs is not None else {}
        monitor_value = logs.get("val_aspect_f1_score") * self.args.aspect_coef + logs.get(
            "val_polarity_f1_score"
        ) * (1 - self.args.aspect_coef)
        return monitor_value
