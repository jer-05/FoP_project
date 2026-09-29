import numpy as np

from constants import EPS

class Parabola:
    def __init__(self, a):
        self.a = a
        self.focus = np.array([0, a/4])
    def get_inward_normal(self, x, d):
        """
        Compute the normal vector to the parabola's surface to the same side as d is incident from
        """
        gradient =  2 * self.a * x
        normal = np.array([gradient, -1])
        normal = normal / np.linalg.norm(normal)
        if np.dot(normal, d) > 0:
            return normal
        else:
            return -normal
    def get_y(self, x):
        return self.a * x**2
    def get_sources_on_circle(self, radius, n):
        """
        Generate a circle of sources surrounding the focal point of the parabolic mirror
        """
        angles = np.linspace(0, 2*np.pi *(1 - 1/n), n)
        sources = radius * np.stack((np.sin(angles), np.cos(angles))).T + self.focus
        return sources
    def get_intersect(self, O, d_vec):
        """
        Return the intersection point (x, y) and parameter value t
        """
        print(f"Parabola.get_intersect: Finding reflection from {O} with direction {d_vec}")
        a = self.a * d_vec[0]**2
        b = 2 * self.a * d_vec[0] * O[0] - d_vec[1]
        c = self.a * O[0]**2 - O[1]
        intersect = False
        if np.abs(a) < EPS: # if the leading coefficient of the quadratic equation is small
            t = -c / b
            if t > 0:
                intersect = True
        else:
            D = b**2 - 4*a*c
            print(f"get_refl: a={a}, b={b}, c={c} => D={D}")
            if D > 0:
                t1 = (-b + D**.5) / (2*a)
                t2 = (-b - D**.5) / (2*a)
                print(f"t1: {t1}, t2: {t2}")
                if t1 > 0 or t2 > 0:
                    intersect = True
            if intersect:
                if t1 > 0 and t2 > 0:
                    t = min(t1, t2)
                else:
                    t = max(t1, t2)
        if not intersect:
            print(f"Parabola.get_intersect: No intersection found")
            return None, None
        r_intersect = O + t * d_vec
        print(f"Parabola.get_intersect: returning intersect {r_intersect} at parameter {t}")
        return (r_intersect, t)


