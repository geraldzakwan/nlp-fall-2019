class Helper:
    @staticmethod
    def is_empty(list_of_dict):
        if list_of_dict is None:
            return True

        if len(list_of_dict) == 0:
            return True

        return False
