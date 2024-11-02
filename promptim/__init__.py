import tqdm


# Deactivate tqdm by replacing it with a no-op function
def tqdm_noop(*args, **kwargs):
    if args and hasattr(args[0], "__iter__"):
        return args[0]
    return args


tqdm.auto = tqdm_noop
