__all__ = ['logs']


def load_file(file):
    '''
    Load imported log file (imported via one of the import scripts)
    '''
    import pickle
    return pickle.load(file)
