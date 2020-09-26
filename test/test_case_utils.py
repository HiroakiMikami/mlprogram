import os


def integration_test(f):
    def wrapped(*args, **kwargs):
        self = args[0]
        if "MLPROGRAM_INTEGRATION_TEST" not in os.environ:
            self.skipTest("MLPROGRAM_INTEGRATION_TEST is not set")
        return f(*args, **kwargs)
    return wrapped
