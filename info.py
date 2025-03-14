class Timeout(object):
    def __init__(self):
        pass

    def __str__(self):
        return 'Timeout'
    
class DigitCrazy(object):
    def __init__(self):
        pass

    def __str__(self):
        return 'Digit is crazy'
    
class OutArea(object):
    def __init__(self):
        pass

    def __str__(self):
        return 'Out area'
    
class OverOrientation(object):
    def __init__(self):
        pass

    def __str__(self):
        return 'over orientation'
    
class LIPOutRange(object):
    def __init__(self):
        pass

    def __str__(self):
        return 'LIP out range'


class ReachGoal(object):
    def __init__(self):
        pass

    def __str__(self):
        return 'Reaching goal'


class Danger(object):
    def __init__(self, min_dist):
        self.min_dist = min_dist

    def __str__(self):
        return 'Too close'


class Collision(object):
    def __init__(self):
        pass

    def __str__(self):
        return 'Collision'


class Nothing(object):
    def __init__(self):
        pass

    def __str__(self):
        return ''

