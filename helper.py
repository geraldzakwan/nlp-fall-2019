start_token = 'START'
stop_token = 'STOP'
unk_token = 'UNK'

class Helper:
    @staticmethod
    def is_empty(list_of_dict):
        if list_of_dict is None:
            return True

        if len(list_of_dict) == 0:
            return True

        return False

    @staticmethod
    def is_valid_n_grams(n_grams, n):
        if n_grams is None:
            return False

        if len(n_grams) == 0:
            return False

        if not isinstance(n_grams, tuple):
            return False

        if len(n_grams) != n:
            return False

        return True
