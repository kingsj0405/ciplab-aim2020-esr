import fire

from MSRResNet import test


def hello():
    print("Hello, World!")


if __name__ == '__main__':
    fire.Fire({
        'test': test,
        'hello': hello
    })