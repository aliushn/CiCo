class instrument(object):
    def __init__(self, name):
        self.name = name

    def play(self):
        print('Please play {}!'.format(self.name))


def testplay():
    p = instrument('Piano')
    p.play()
    p = instrument('Violin')
    p.play()

testplay()


