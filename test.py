import threading
from data import *




if __name__ == "__main__":


    def a(bar, baz):

        return 1
    def b(bar, baz):

        return 1
    def c(bar, baz):

        return 1

    def d(bar, baz):

        return 1





    from multiprocessing.pool import ThreadPool

    pool = ThreadPool(processes=2)

    a = pool.apply_async(a, ('world', 'foo'))  # tuple of args for foo
    b= pool.apply_async(b, ('world', 'foo'))
    c=pool.apply_async(c, ('world', 'foo'))
    # do some other stuff in the main process

    a = a.get()
    b=b.get()
    print(a,b)