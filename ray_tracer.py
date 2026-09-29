import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.lines import Line2D

EPS = 1e-5
IMG_DIR = "figures"
RAY_COLOR = "green"

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

class Concentrator:
    """
    Uses a parabolic mirror and collection of light sources to simulate
    ray scattering using ray tracing; plotting is done only within the given x and y limits
    """
    def __init__(self, parabola, config):
        self.xlim = config['xlim']
        self.ylim = config['ylim']
        self.parabola = parabola
        self.source_plot_object = None
        self.lines = []
        self.fig, self.ax = plt.subplots()
        plt.subplots_adjust(bottom=.40)

        # define sliders
        self.ax_slider_angle = self.fig.add_axes((.2, 0, .6, .08))
        self.slider_angle = Slider(
                ax = self.ax_slider_angle,
                label = "Angle",
                valmin = 0,
                valmax = 360,
                valfmt = "%d Deg",
                valinit = 0
                )
        self.slider_angle.on_changed(self.propagate_ray)

        self.ax_slider_radius = self.fig.add_axes((.2, .1, .6, .08))
        self.slider_radius = Slider(
                ax = self.ax_slider_radius,
                label = "Radius",
                valmin = 0,
                valmax = self.parabola.a / 5,
                valinit = self.parabola.a / 10,
                )
        self.slider_radius.on_changed(self.set_sources)

        self.ax_slider_n = self.fig.add_axes((.2, .2, .6, .08))
        self.slider_n = Slider(
                ax = self.ax_slider_n,
                label = "Number of Sources",
                valmin = 1,
                valmax = 30,
                valinit = 5,
                valstep=1,
                )
        self.slider_n.on_changed(self.set_sources)

        # plot parabola and sources
        self.set_sources(None)
        x = np.linspace(self.xlim[0], self.xlim[1], 100)
        y = self.parabola.get_y(x)
        self.ax.plot(x, y, color="blue", label="mirror")
        self.ax.scatter(*self.parabola.focus, marker="x", color="black", label="focus")
        self.ax.set_xlim(self.xlim)
        self.ax.set_ylim(self.ylim)
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")
        self.ax.axis('equal')
        self.ax.set_title("Parabolic Mirror Simulation")
        handles, labels = self.ax.get_legend_handles_labels()
        handles += [Line2D([], [], color=RAY_COLOR)]
        labels += ["ray"]
        self.ax.legend(handles, labels)

        plt.show()

    def in_limits(self, r):
        """
        Check if a point is still in the visible plotting range
        """
        if r[0] < self.xlim[0] or r[0] > self.xlim[1] or r[1] < self.ylim[0] or r[1] > self.ylim[1]:
            return False
        return True

    def get_tlim(self, r, d):
        """
        Compute smallest line parameter t so that the line is still in the visible plotting range
        """
        d = np.where(np.abs(d) < EPS, np.copysign(EPS, d), d) # to prevent zero-division
        t_min = (np.array([self.xlim[0], self.ylim[0]]) - r) / d
        t_max = (np.array([self.xlim[1], self.ylim[1]]) - r) / d
        t_min[t_min < 0] = np.inf
        t_max[t_max < 0] = np.inf
        t_lim = min(np.min(t_min), np.min(t_max))
        return t_lim

    def propagate_ray(self, angle):
        """
        Propagate a ray from the source with reflections and plot the result
        """
        print(f"\n-")
        print(f"Propagating ray with angle {angle}")
        for line in self.lines:
            line.remove()
        self.lines = []
        rad = angle * np.pi / 180 
        d_start = np.array([np.cos(rad), np.sin(rad)])
        for pos in self.sources:
            d = d_start
            while True:
                r, t = self.parabola.get_intersect(pos, d)
                if r is None or not self.in_limits(r):
                    t = self.get_tlim(pos, d)
                t_range = np.linspace(0, t, 100)
                r_range = pos[None, :] + t_range[:, None] * d[None, :]
                self.lines += self.ax.plot(r_range[:, 0], r_range[:, 1], color=RAY_COLOR)
                if r is None or not self.in_limits(r):
                    break
                normal = self.parabola.get_inward_normal(r[0], d)
                ip = np.dot(normal, d)
                d = d - 2 * ip * normal
                pos = r + EPS * d
        self.fig.canvas.draw_idle()

    def set_sources(self, _):
        """
        Get a circle of points around the focus of the parabola and set them as light sources.
        The function uses slider values as input for the radius of the circle and number
        of sources. Afterwards, light rays are plotted again.
        """
        if self.source_plot_object is not None:
            self.source_plot_object.remove()
        self.sources = self.parabola.get_sources_on_circle(self.slider_radius.val, int(self.slider_n.val))
        self.source_plot_object = self.ax.scatter(*self.sources.T, marker=".", color="red", label="source")
        self.propagate_ray(self.slider_angle.val)

    def savefig(self, fname="image.png"):
        full_name = IMG_DIR + "/" + fname
        print(f"Concentrator.savefig: Saving figure as {full_name}")
        self.fig.savefig(full_name)

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
