from surface import Parabola
from concentrator import Concentrator
def main():
    parabola = Parabola(1)
    config = {
        'nsources': 5,
        'radius': .1,
        'xlim': [-1, 1],
        'ylim': [0, 1],
        }
    concentrator = Concentrator(parabola, config)
    concentrator.savefig()

if __name__ == '__main__':
    main()

