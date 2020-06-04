class ActionEnum(object):
    """Enumeration of actions that a client of the Trainer may request by
    implementing the Trainer's action request function, which a clinet may
    optionally provide in order to request early termination or saving is
    used to allow the checkpoint to be saved when stopping execution with
    a SIGINT (Ctrl-c)."""
    NONE = 0  # Take no special action
    STOP = 1  # Stop training. checkpoint_after_train controls whether a checkpoint is cteated
    CHECKPOINT = 2  # Take a checkpoint, and keep training
