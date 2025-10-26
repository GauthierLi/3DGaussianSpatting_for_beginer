"""
Implementation of Spherical Harmonics Basis Functions
Includes calculation of spherical harmonics and 3D visualization features.
"""

import numpy as np
from scipy.special import sph_harm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as colors
from matplotlib.widgets import Slider, Button
import warnings

warnings.filterwarnings('ignore')


class SphericalHarmonics:
    """
    Spherical Harmonics basis function class.
    Supports calculation of spherical harmonics of any degree and order.
    """
    
    def __init__(self, l=0, m=0):
        """
        Initializes the spherical harmonic.
        
        Args:
            l (int): The degree of the spherical harmonic, l >= 0.
            m (int): The order of the spherical harmonic, -l <= m <= l.
        """
        if l < 0:
            raise ValueError("Degree l must be non-negative.")
        if abs(m) > l:
            raise ValueError(f"Order m must satisfy -l <= m <= l, but l={l}, m={m}")
        
        self.l = l
        self.m = m
        
    def __call__(self, theta, phi):
        """
        Calculates the value of the spherical harmonic at given angles.
        
        Args:
            theta (float or array): Polar angle, in range [0, π].
            phi (float or array): Azimuthal angle, in range [0, 2π].
            
        Returns:
            complex or array: The complex value of the spherical harmonic.
        """
        # The parameter order for scipy's sph_harm is (m, l, phi, theta)
        return sph_harm(self.m, self.l, phi, theta)
    
    def real_part(self, theta, phi):
        """Returns the real part of the spherical harmonic."""
        return np.real(self(theta, phi))
    
    def imag_part(self, theta, phi):
        """Returns the imaginary part of the spherical harmonic."""
        return np.imag(self(theta, phi))
    
    def magnitude(self, theta, phi):
        """Returns the magnitude of the spherical harmonic."""
        return np.abs(self(theta, phi))
    
    def phase(self, theta, phi):
        """Returns the phase of the spherical harmonic."""
        return np.angle(self(theta, phi))
    
    def __str__(self):
        return f"Y_{self.l}^{self.m}(θ,φ)"
    
    def __repr__(self):
        return f"SphericalHarmonics(l={self.l}, m={self.m})"


class SphericalHarmonicsVisualizer:
    """
    Interactive 3D visualizer for Spherical Harmonics.
    Supports real-time parameter adjustment and different visualization modes.
    """
    
    def __init__(self, l=2, m=1, resolution=50):
        """
        Initializes the visualizer.
        
        Args:
            l (int): Initial degree.
            m (int): Initial order.
            resolution (int): Resolution of the spherical mesh.
        """
        self.l = l
        self.m = m
        self.resolution = resolution
        self.sh = SphericalHarmonics(l, m)
        
        self.theta = np.linspace(0, np.pi, resolution)
        self.phi = np.linspace(0, 2*np.pi, resolution)
        self.THETA, self.PHI = np.meshgrid(self.theta, self.phi)
        
        self.axis_limit = 2.0
        self.ax_rect = [0.1, 0.25, 0.7, 0.7]
        self.cbar_rect = [0.85, 0.3, 0.03, 0.55]

        self.fig = None
        self.ax = None
        self.cax = None
        self.surf = None
        self.colorbar = None
        self.mode = 'magnitude'
        self._initial_position_locked = False
        self._initial_dist = None
        
    def _compute_harmonics(self):
        """Computes the spherical harmonic values for the current parameters."""
        self.sh = SphericalHarmonics(self.l, self.m)
        return self.sh(self.THETA, self.PHI)
    
    def _get_visualization_data(self, harmonics_values):
        """Gets the corresponding data and coordinates based on the visualization mode."""
        if self.mode == 'magnitude':
            values = np.abs(harmonics_values)
            title = f"|Y_{self.l}^{self.m}(θ,φ)|"
        elif self.mode == 'real':
            values = np.real(harmonics_values)
            title = f"Re[Y_{self.l}^{self.m}(θ,φ)]"
        elif self.mode == 'imag':
            values = np.imag(harmonics_values)
            title = f"Im[Y_{self.l}^{self.m}(θ,φ)]"
        elif self.mode == 'phase':
            values = np.angle(harmonics_values)
            title = f"Phase[Y_{self.l}^{self.m}(θ,φ)]"
        else:
            values = np.abs(harmonics_values)
            title = f"|Y_{self.l}^{self.m}(θ,φ)|"
        
        x_base = np.sin(self.THETA) * np.cos(self.PHI)
        y_base = np.sin(self.THETA) * np.sin(self.PHI)
        z_base = np.cos(self.THETA)
        
        base_radius = 0.8
        
        if self.mode == 'phase':
            normalized_phase = (values + np.pi) / (2 * np.pi)
            radius_factor = base_radius + 0.6 * np.sin(2 * normalized_phase * np.pi)
        else:
            max_abs = np.max(np.abs(values))
            if max_abs > 1e-12:
                normalized_values = np.abs(values) / max_abs
                radius_factor = base_radius + 0.8 * normalized_values
            else:
                radius_factor = np.ones_like(values) * base_radius
        
        x = radius_factor * x_base
        y = radius_factor * y_base  
        z = radius_factor * z_base
        
        return x, y, z, values, title
    
    def _lock_axis_limits(self):
        """Locks the axis limits to prevent auto-scaling."""
        if self.ax is None:
            return
        
        self.ax.set_xlim3d([-self.axis_limit, self.axis_limit])
        self.ax.set_ylim3d([-self.axis_limit, self.axis_limit])
        self.ax.set_zlim3d([-self.axis_limit, self.axis_limit])
        
        self.ax.set_box_aspect([1, 1, 1])
        
        self.ax.set_autoscale_on(False)
        
        self.ax.grid(True, alpha=0.3)
        
        ticks = np.linspace(-self.axis_limit, self.axis_limit, 5)
        self.ax.set_xticks(ticks)
        self.ax.set_yticks(ticks) 
        self.ax.set_zticks(ticks)
    
    def _update_plot(self):
        """Updates the 3D plot, keeping the view stable (no more recursive shrinking)."""
        if self.ax is None:
            return

        try:
            self.ax.set_position(self.ax_rect)
            if not self._initial_position_locked:
                self._initial_position_locked = True

            elev, azim = self.ax.elev, self.ax.azim

            harmonics_values = self._compute_harmonics()
            x, y, z, values, title = self._get_visualization_data(harmonics_values)

            if self.surf is not None:
                self.surf.remove()
                self.surf = None

            if self.mode == 'phase':
                mapped_colors = plt.cm.hsv((values + np.pi) / (2 * np.pi))
                show_colorbar = False
                vmin, vmax = -np.pi, np.pi
            else:
                vmin, vmax = float(np.min(values)), float(np.max(values))
                if abs(vmax - vmin) < 1e-12:
                    vmin, vmax = 0.0, 1.0
                    norm_values = np.ones_like(values) * 0.5
                else:
                    norm_values = (values - vmin) / (vmax - vmin)
                mapped_colors = plt.cm.RdYlBu_r(norm_values)
                show_colorbar = True

            self.surf = self.ax.plot_surface(
                x, y, z,
                facecolors=mapped_colors,
                linewidth=0,
                antialiased=True,
                shade=False,
                alpha=0.95
            )

            self._lock_axis_limits()
            self.ax.view_init(elev=elev, azim=azim)

            self.ax.set_xlabel('X', fontsize=10)
            self.ax.set_ylabel('Y', fontsize=10)
            self.ax.set_zlabel('Z', fontsize=10)
            self.ax.set_title(title, fontsize=12, pad=12)

            if self.cax is None:
                self.cax = self.fig.add_axes(self.cbar_rect)
            if self.colorbar is None:
                from matplotlib.cm import ScalarMappable
                norm = plt.Normalize(vmin=vmin, vmax=vmax)
                sm = ScalarMappable(norm=norm, cmap=plt.cm.RdYlBu_r)
                sm.set_array([])
                if show_colorbar:
                    self.colorbar = self.fig.colorbar(sm, cax=self.cax)
                else:
                    self.cax.set_visible(False)
            else:
                if show_colorbar:
                    self.cax.set_visible(True)
                    self.colorbar.mappable.set_clim(vmin, vmax)
                    self.colorbar.mappable.set_cmap(plt.cm.RdYlBu_r)
                    self.colorbar.update_normal(self.colorbar.mappable)
                else:
                    self.cax.set_visible(False)

            self.fig.canvas.draw_idle()
        except Exception as e:
            print(f"Plot update error: {e}")
            self._lock_axis_limits()
    
    def _on_l_change(self, val):
        """Callback for l-slider change."""
        new_l = int(val)
        if abs(self.m) > new_l:
            self.m = 0
            if hasattr(self, 'slider_m'):
                self.slider_m.set_val(0)
        self.l = new_l
        self._update_plot()
    
    def _on_m_change(self, val):
        """Callback for m-slider change."""
        new_m = int(val)
        if abs(new_m) <= self.l:
            self.m = new_m
            self._update_plot()
        else:
            if hasattr(self, 'slider_m'):
                self.slider_m.set_val(self.m)
    
    def _on_mode_change(self, label):
        """Callback for mode change."""
        mode_map = {
            'Magnitude': 'magnitude',
            'Real Part': 'real',
            'Imaginary Part': 'imag',
            'Phase': 'phase'
        }
        self.mode = mode_map.get(label, 'magnitude')
        self._update_plot()
    
    def visualize(self, interactive=True):
        """
        Creates the 3D visualization.
        
        Args:
            interactive (bool): Whether to show interactive controls.
        """
        self.fig = plt.figure(figsize=(12, 8), constrained_layout=False)
        
        if interactive:
            self.ax = self.fig.add_axes(self.ax_rect, projection='3d')
            
            ax_l = plt.axes([0.1, 0.15, 0.3, 0.03])
            self.slider_l = Slider(ax_l, 'l (degree)', 0, 8, valinit=self.l, valfmt='%d')
            self.slider_l.on_changed(self._on_l_change)
            
            ax_m = plt.axes([0.5, 0.15, 0.3, 0.03])
            self.slider_m = Slider(ax_m, 'm (order)', -8, 8, valinit=self.m, valfmt='%d')
            self.slider_m.on_changed(self._on_m_change)
            
            ax_mode = plt.axes([0.1, 0.05, 0.7, 0.08])
            from matplotlib.widgets import RadioButtons
            self.radio = RadioButtons(ax_mode, ('Magnitude', 'Real Part', 'Imaginary Part', 'Phase'))
            self.radio.on_clicked(self._on_mode_change)
            
        else:
            self.ax = self.fig.add_axes(self.ax_rect, projection='3d')
        
        self._lock_axis_limits()
        
        self.ax.view_init(elev=20, azim=45)
        
        self.ax.set_xlabel('X', fontsize=10)
        self.ax.set_ylabel('Y', fontsize=10)
        self.ax.set_zlabel('Z', fontsize=10)
        
        self._update_plot()
        
        plt.show()
    
    def save_plot(self, filename, dpi=300):
        """
        Saves the current plot.
        
        Args:
            filename (str): The filename to save to.
            dpi (int): The image resolution.
        """
        if self.fig is not None:
            self.fig.savefig(filename, dpi=dpi, bbox_inches='tight')
            print(f"Plot saved to: {filename}")
        else:
            print("No plot to save. Please run visualize() first.")


def demo():
    """Demonstration function."""
    print("Spherical Harmonics Basis Functions Demo")
    print("=" * 50)
    
    harmonics = [
        SphericalHarmonics(0, 0),
        SphericalHarmonics(1, 0),
        SphericalHarmonics(1, 1),
        SphericalHarmonics(2, 0),
        SphericalHarmonics(2, 2),
    ]
    
    theta_test = np.array([0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi])
    phi_test = np.array([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
    
    for sh in harmonics:
        print(f"\n{sh}:")
        for i, (theta, phi) in enumerate(zip(theta_test[:3], phi_test[:3])):
            value = sh(theta, phi)
            print(f"  θ={theta:.2f}, φ={phi:.2f}: {value:.4f}")
    
    print("\n" + "=" * 50)
    print("Launching interactive visualization...")
    print("Use the sliders to adjust l and m parameters.")
    print("Use the radio buttons to switch visualization modes.")
    
    visualizer = SphericalHarmonicsVisualizer(l=2, m=1, resolution=80)
    visualizer.visualize(interactive=True)


if __name__ == "__main__":
    demo()
