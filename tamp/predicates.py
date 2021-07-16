class On:
    def __init__(self, bottom_num, top_num):
        self.name = 'On'
        self.bottom_num = bottom_num
        self.top_num = top_num

    def in_state(self, logical_state):
        for other_fluent in logical_state:
            if (other_fluent.bottom_num == self.bottom_num or self.bottom_num == '*') \
                and (other_fluent.top_num == self.top_num or self.top_num == '*'):
                return True
        return False
