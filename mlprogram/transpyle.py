import logging

import transpyle  # noqa

# Disable logging to file
del logging.root.handlers[1]
