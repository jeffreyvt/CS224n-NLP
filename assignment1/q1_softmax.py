import numpy as np


def softmax(x):
    """Compute the softmax function for each row of the input x.

    It is crucial that this function is optimized for speed because
    it will be used frequently in later code. You might find numpy
    functions np.exp, np.sum, np.reshape, np.max, and numpy
    broadcasting useful for this task.

    Numpy broadcasting documentation:
    http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html

    You should also make sure that your code works for a single
    N-dimensional vector (treat the vector as a single row) and
    for M x N matrices. This may be useful for testing later. Also,
    make sure that the dimensions of the output match the input.

    You must implement the optimization in problem 1(a) of the
    written assignment!

    Arguments:
    x -- A N dimensional vector or M x N dimensional numpy matrix.

    Return:
    x -- You are allowed to modify x in-place
    """
    orig_shape = x.shape
    if len(x.shape) > 1:
        # Matrix
        ### YOUR CODE HERE
        min = np.amin(x, axis=1)
        x = np.subtract(x.transpose(), min).transpose()
        y = np.exp(x)
        z = np.array([np.sum(y,axis=1),]*y.shape[1]).transpose()
        x = np.true_divide(y,z)
        ### END YOUR CODE
    else:
        # Vector
        ### YOUR CODE HERE
        min = np.amin(x)
        x = np.subtract(x.transpose(), min)
        x = np.divide(np.exp(x),np.sum(np.exp(x)))
        ### END YOUR CODE

    assert x.shape == orig_shape
    return x


def test_softmax_basic():
    """
    Some simple tests to get you started.
    Warning: these are not exhaustive.
    """
    print "Running basic tests..."
    test1 = softmax(np.array([1,2]))
    print test1
    ans1 = np.array([0.26894142,  0.73105858])
    assert np.allclose(test1, ans1, rtol=1e-05, atol=1e-06)

    test2 = softmax(np.array([[1001,1002],[3,4]]))
    print test2
    ans2 = np.array([
        [0.26894142, 0.73105858],
        [0.26894142, 0.73105858]])
    assert np.allclose(test2, ans2, rtol=1e-05, atol=1e-06)

    test3 = softmax(np.array([[-1001,-1002]]))
    print test3
    ans3 = np.array([0.73105858, 0.26894142])
    assert np.allclose(test3, ans3, rtol=1e-05, atol=1e-06)

    print "You should be able to verify these results by hand!\n"


def test_softmax():
    """
    Use this space to test your softmax implementation by running:
        python q1_softmax.py
    This function will not be called by the autograder, nor will
    your tests be graded.
    """
    print "Running your tests..."
    ### YOUR CODE HERE
    x = np.array([[-2.64686423e+00, -2.75463235e+00, -3.20158442e+00,
                   2.39812882e-01, -1.03778789e+00, -1.68634181e+00,
                   -3.32472303e-01, -2.34244404e+00, -2.32417480e+00,
                   -1.92003506e-01],
                  [-1.20339084e+00, -2.30409426e+00, -8.67176976e-01,
                   -1.69039539e+00, -9.53913687e-01, 9.56845978e-01,
                   -1.60882767e+00, 1.10609417e+00, -1.67409366e+00,
                   -3.42659747e-01],
                  [-1.65222502e+00, -2.85469965e+00, -9.34289636e-01,
                   -6.94683644e-01, -1.39938302e+00, -1.48309836e+00,
                   -7.63891145e-01, -1.21522142e+00, -1.35181968e+00,
                   3.91321003e-05],
                  [-1.10873153e+00, -2.84338905e+00, -2.20019325e+00,
                   -8.88771028e-01, -1.14047466e+00, 1.59170262e-01,
                   -5.83463274e-01, -1.01228826e+00, -2.24018230e+00,
                   -8.34434540e-01],
                  [-1.28474074e+00, -1.99492313e+00, -4.06241068e-01,
                   -1.54540107e+00, -1.59290073e+00, 8.13817812e-01,
                   -1.34216750e+00, 3.85370917e-01, -2.88340379e+00,
                   -1.16256165e-01],
                  [-3.08172064e+00, -2.24335490e+00, -1.91988200e+00,
                   -5.49837935e-01, -1.25391394e+00, -1.22819688e+00,
                   -1.44629086e+00, -5.21814938e-01, -2.03519192e+00,
                   5.18414578e-01],
                  [-2.72346057e+00, -2.79374446e+00, -1.61681432e+00,
                   -1.91632866e-01, -2.25044689e+00, -2.49447265e+00,
                   -6.20917521e-01, -2.58757723e+00, -2.54607641e+00,
                   4.80060659e-01],
                  [-3.01370399e+00, -2.87840742e+00, -3.35078654e+00,
                   8.27177453e-01, -5.94070863e-01, -2.77714780e+00,
                   -7.88050977e-02, -2.85289012e+00, -1.55610656e+00,
                   -3.64610349e-03],
                  [-1.78140259e+00, -1.57968199e+00, -6.11542652e-01,
                   -6.43771871e-01, -3.67227233e-01, 1.49545388e-01,
                   -1.02400702e+00, 3.83082176e-01, -2.22162561e+00,
                   7.54131285e-02],
                  [-6.96103378e-01, -2.89572964e+00, -8.29860492e-01,
                   -2.03342435e+00, -2.42634515e+00, 7.74497874e-01,
                   -1.26417347e+00, -2.84721472e-01, -2.72250551e+00,
                   -5.70354427e-01],
                  [-1.47364051e+00, -2.50692369e+00, -1.84993068e+00,
                   -4.44188732e-01, -3.16991871e-01, -3.75446455e-01,
                   -5.95299585e-01, -6.12120167e-01, -1.46479321e+00,
                   -5.12553916e-01],
                  [-1.45559837e+00, -1.74559219e+00, -3.51566614e-01,
                   -1.41803959e+00, -1.55972930e+00, 8.58785519e-01,
                   -1.31485574e+00, 3.72507606e-01, -3.23774023e+00,
                   -1.65531527e-03],
                  [-1.32995780e+00, -2.39554709e+00, -4.16776642e-01,
                   -1.17018291e+00, -1.72130129e+00, -3.05386818e-01,
                   -9.70798780e-01, -5.82057432e-01, -2.39850709e+00,
                   -4.96296706e-02],
                  [-1.38654578e+00, -2.55876480e+00, -1.38357522e+00,
                   -4.07534199e-01, -7.92805926e-01, -7.90857801e-01,
                   -4.07367026e-01, -1.18277208e+00, -1.78162618e+00,
                   -3.75526328e-01],
                  [-2.34514539e+00, -1.63132765e+00, -9.15517903e-01,
                   -8.21587746e-01, -1.81919147e+00, -7.45760229e-02,
                   -1.13347030e+00, -6.01932247e-01, -3.82211024e+00,
                   3.81135433e-01],
                  [-1.89630621e+00, -1.78301121e+00, -1.64583546e+00,
                   2.67048926e-01, 3.08633916e-01, -5.98732462e-01,
                   -2.67875597e-01, -7.54964245e-01, -1.95006637e+00,
                   -2.22618932e-01],
                  [-3.11785951e+00, -2.51970652e+00, -2.24235332e+00,
                   3.41720473e-01, -8.93408684e-01, -2.52734796e+00,
                   -6.34464681e-01, -1.92918805e+00, -1.60563492e+00,
                   4.48139323e-01],
                  [-3.21197638e+00, -2.72312258e+00, -3.39266096e+00,
                   8.90695833e-01, -1.96643873e-01, -2.72877611e+00,
                   -2.39343969e-01, -2.39942698e+00, -1.22340823e+00,
                   8.56484450e-02],
                  [-1.77628149e+00, -1.32594568e+00, -2.22548810e-01,
                   -1.22361645e+00, -7.15223994e-01, 9.57297846e-01,
                   -1.54089679e+00, 1.24045812e+00, -2.67049501e+00,
                   1.97947331e-01],
                  [-1.37794913e+00, -1.97813438e+00, -2.47243010e-02,
                   -9.79333061e-01, -1.28211304e+00, -1.87591698e-01,
                   -9.30901272e-01, -2.34389197e-01, -2.42193758e+00,
                   8.44479499e-02]])
    print x.shape
    print softmax(x)
    ### END YOUR CODE


if __name__ == "__main__":
    test_softmax_basic()
    test_softmax()
