from random import randint


class Rnd(int):

    max_val = 4294967295

    def __new__(cls, *args, **kwargs):
        val = randint(0, Rnd.max_val)
        print "Seed: ", val
        return super(Rnd, cls).__new__(cls, val)
