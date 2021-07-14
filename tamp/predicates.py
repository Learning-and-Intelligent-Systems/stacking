class On:
    def __init__(self, bottom, top):
        self.name = 'On'
        self.bottom = bottom
        self.top = top

    def in_state(self, logical_state):
        for other_fluent in logical_state:
            if (other_fluent.bottom == self.bottom or self.bottom == '*') \
                and (other_fluent.top == self.top or self.top == '*'):
                return True
        return False
