from numba import njit, prange
import numpy as np
def simple_vectorize(text, scopes_map):
    _text = text
    def simple_vectorize_inner(f):
        f_ = njit(f)
        f_.text = _text

        def simple_vectorized(self,scopes):
            l = np.shape(scopes)[0]
            #try:
            for i in prange(l):
                f_(scopes[i,:],i)
            #except Exception:
            #    raise Exception()

                #for j,v in enumerate(scopes[i,:]):
                #    var_name = scopes_map[j]
                #    print(f"scope.{var_name}=scope[{j}]={v}")
                #print(i)
                #raise e
        return simple_vectorized
    return simple_vectorize_inner