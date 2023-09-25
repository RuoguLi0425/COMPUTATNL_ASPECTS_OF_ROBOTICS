import unittest
import numpy as np
from transforms import *

class TestTransforms(unittest.TestCase):
    """Unit test transforms.py.
    """

    def test_transform_is_valid(self):
        """Test transforms.transform_is_valid.
        """
        # fix random seed
        np.random.seed(8)

        a = np.eye(4)
        self.assertTrue(transform_is_valid(a))
        a[:3, :3] = self._rand_rotation_matrix()
        self.assertTrue(transform_is_valid(a))
        a[:3, :3] = self._rand_rotation_matrix()
        self.assertTrue(transform_is_valid(a))
        a[:3, :3] = self._rand_rotation_matrix()
        self.assertTrue(transform_is_valid(a))

        # rotation deteminant of -1
        a = np.array([[0.00314318, 0.98352366, -0.18075212, -0.10442753],
                    [-0.21180265, 0.17730691 , 0.96109412 , -0.14890335],
                    [-0.9773074,  -0.03526289, -0.20887023, 0.11372389],
                    [ 0.,         0.         , 0.         , 1.        ]])
        self.assertFalse(transform_is_valid(a))

        # fail as last row is not properly formatted
        a = np.eye(4)
        a[3, 3] = 2.
        self.assertFalse(transform_is_valid(a))

        # some general cases that should fail
        self.assertFalse(transform_is_valid(np.random.rand(4,4)))
        self.assertFalse(transform_is_valid(np.random.rand(3,3)))

    def test_transform_concat(self):
        """Test transforms.transform_concat.
        """
        # transform on left side of concat operator
        t1 = np.eye(4)
        t1[:3, :3] = np.array([[-0.19140606, -0.59014793, 0.78427619],
                            [-0.3289964, -0.71424818, -0.61774664],
                            [ 0.92472974, -0.37626449, -0.05744512]])

        # transform on right side of concat operator
        t2 = np.eye(4)
        t2[:3, :3] = np.array([[0.54037545, -0.39685982, 0.74195461],
                            [-0.44571062, -0.88291785, -0.14764184],
                            [ 0.71367809, -0.25091502, -0.65399177]])

        # answer post concat
        ans = np.array([[0.7193248, 0.40022683, -0.56779426, 0.],
                        [-0.30030582, 0.91618983, 0.26535374, 0.],
                        [0.62640901, -0.0203636, 0.77922851, 0.],
                        [0., 0., 0., 1.]])

        # user function evaluation
        t_concat = transform_concat(t1, t2)

        self.assertTrue(transform_is_valid(t_concat))
        self.assertTrue(np.isclose(t_concat, ans).all())

    def test_transform_point3s(self):
        """Test transforms.transform_point3s.
        """
        # 2 point3s
        p = np.array([[1., 2., 3.], [2., 7., -1.]])

        # example transform
        t = np.array([[1., 0., 0., 1.],
                    [0., 0., 1., 1.],
                    [0., -1., 0., 1.],
                    [0., 0., 0., 1.]])

        self.assertTrue(np.isclose(transform_point3s(t, p), [[2., 4., -1.], [ 3., 0., -6.]]).all())

    def test_transform_inverse(self):
        """Test transforms.transform_inverse.
        """
        # random SE(3)
        t = np.eye(4)
        t[:3, :3] = self._rand_rotation_matrix()

        t_inv = transform_inverse(t)

        # make sure inverse is both right and left inverse
        self.assertTrue(np.isclose(np.matmul(t, t_inv), np.eye(4)).all())
        self.assertTrue(np.isclose(np.matmul(t_inv, t), np.eye(4)).all())

    def _rand_rotation_matrix(self):
        """Creates a random rotation matrix.
            Based on code from here:
            http://www.realtimerendering.com/resources/GraphicsGems/gemsiii/rand_rotation.c

        Returns:
            numpy.array [3, 3]: Rotation matrix.
        """
        theta, phi, z = np.random.uniform(size=(3,))
        theta = theta * 2.0 * np.pi  # Rotation about the pole (Z).
        phi = phi * 2.0 * np.pi  # For direction of pole.
        z = z * 2.0

        # Compute a vector V used for distributing points over the sphere
        # via the reflection I - V Transpose(V).  This formulation of V
        # will guarantee that if x[1] and x[2] are uniformly distributed,
        # the reflected points will be uniform on the sphere.  Note that V
        # has length sqrt(2) to eliminate the 2 in the Householder matrix.
        r = np.sqrt(z)
        Vx, Vy, Vz = V = (
            np.sin(phi) * r,
            np.cos(phi) * r,
            np.sqrt(2.0 - z))

        st = np.sin(theta)
        ct = np.cos(theta)

        R = np.array(((ct, st, 0), (-st, ct, 0), (0, 0, 1)))

        # Construct the rotation matrix  ( V Transpose(V) - I ) R.
        M = (np.outer(V, V) - np.eye(3)).dot(R)

        return M

if __name__ == '__main__':
    unittest.main()
