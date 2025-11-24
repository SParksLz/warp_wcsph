import numpy as np
import warp as wp
import warp.render
import json
from wcsph_kernel import *

class sph_material:
    def __init__(self, 
                rho = 1000.0, # rest density
                stiffness = 45000.0,
                exponent = 7.0, 
                mu=0.005,
                tension=0.01) -> None:
        self.rho = rho
        self.stiffness = stiffness
        self.exponent = exponent
        self.mu = mu
        self.tension = tension # tension
 
class sph_model :
    def __init__(self, bound_size, particle_distance) -> None:

        # self.lower_bound: tuple = (-1.0, -1.0, 0.0)
        # self.upper_bound: tuple = (1.0, 1.0, 1.0)
        self.particle_distance = particle_distance

        self.bound_width = bound_size
        self.bound_height = bound_size
        self.bound_length = bound_size

        self.liquid_material = sph_material()

    def build_hash_grid(self):
        grid_cell_size = int(self.bound_height / self.particle_distance)

        self.grid = wp.HashGrid(grid_cell_size, grid_cell_size, grid_cell_size)
        # self.grid.build(particle_q, self.smoothing_length)

class wcsph:
    def __init__(self) -> None:
        self.verbose = False
        self.sim_time = 0.0
        self.sim_dt = 0.0002
        self.particle_radius = 0.01
        self.particle_distance = self.particle_radius * 2.0
        self.smoothing_length = self.particle_distance * 1.6
        # self.particle_distance = self.smoothing_length 
        self.bound_size = 1.2
        self.sph_model = sph_model(self.bound_size, self.smoothing_length)
        self.p_volume = 0.8 * (self.particle_distance ** 3)
        self.sub_step_num = 5
        self.gravity = -9.8


        self.n = int(
            self.bound_size * self.bound_size * self.bound_size / (self.smoothing_length**3)
        )  # number particles (small box in corner)
        print(f"particle count : {self.n}")

        self.x = wp.empty(self.n, dtype=wp.vec3)
        self.mass = self.p_volume * self.sph_model.liquid_material.rho

        self.v = wp.zeros(self.n, dtype=wp.vec3)
        self.rho = wp.zeros(self.n, dtype=float)
        self.a = wp.zeros(self.n, dtype=wp.vec3)
        self.nei_count = wp.zeros(self.n, dtype=wp.int32)
        self.pressure = wp.zeros(self.n, dtype=float)

        # set random positions
        wp.launch(
            kernel=initialize_particles,
            dim=self.n,
            inputs=[self.x, self.smoothing_length, self.bound_size, self.bound_size, self.bound_size],
        )  # initialize in small area

        # self.save_particle_to_json()

        self.sph_model.build_hash_grid()

        self.renderer = wp.render.OpenGLRenderer(up_axis="Z")
        # self.renderer = wp.render.UsdRenderer(up_axis="Z", stage="wcsph_test.usd")

    def sub_step(self) :
        with wp.ScopedTimer("grid build", active=True):
                    # build grid
            self.sph_model.grid.build(self.x, self.smoothing_length)
            wp.launch(
                kernel=get_neighbor,
                dim=self.n,
                inputs=[
                    self.sph_model.grid.id,
                    self.smoothing_length,
                    self.x,
                    self.nei_count,
                ]
            )
        with wp.ScopedTimer("calculate density", active = self.verbose) :

            wp.launch(
                kernel=rho,
                dim=self.n,
                inputs=[
                    self.sph_model.grid.id,
                    self.mass,
                    self.smoothing_length,
                    self.rho,
                    self.x,
                ]
            )
        with wp.ScopedTimer("pressure", active = self.verbose):
            wp.launch(
                kernel=pressure,
                dim=self.n,
                inputs=[
                    self.sph_model.liquid_material.stiffness,
                    self.sph_model.liquid_material.exponent,
                    self.sph_model.liquid_material.rho,
                    self.rho,
                    self.pressure,
                ]
            )
        with wp.ScopedTimer("calculate acceleration", active = self.verbose) :
            wp.launch(
                kernel=acceleration,
                dim=self.n,
                inputs=[
                    self.sph_model.grid.id,
                    self.sph_model.liquid_material.rho,
                    self.sph_model.liquid_material.tension,
                    self.sph_model.liquid_material.mu,
                    self.mass,
                    self.particle_radius,
                    self.p_volume,
                    self.smoothing_length,
                    self.gravity,
                    self.x,
                    self.v,
                    self.rho,
                    self.a,
                    self.pressure,
                ]
            )
        # # kick
        wp.launch(kernel=kick, dim=self.n, inputs=[self.v, self.a, self.sim_dt])
        # # drift
        wp.launch(kernel=drift, dim=self.n, inputs=[self.x, self.v, self.sim_dt])
        # # ground collision
        wp.launch(kernel=apply_bounds, dim=self.n, inputs=[self.x, self.v ,1.2 ,-0.1])


    def step(self) :
        with wp.ScopedTimer("step"):
            for _ in range(self.sub_step_num):
                with wp.ScopedTimer("sub_step"):
                    self.sub_step()
                    self.sim_time += self.sim_dt

    def render(self):
        pass
        # if self.renderer is None:
        #     return

        with wp.ScopedTimer("render"):
            self.renderer.begin_frame(self.sim_time)
            self.renderer.render_points(
                points=self.x.numpy(), radius=self.smoothing_length *0.5, name="points", colors=(0.2, 0.3, 0.7)
            )
            self.renderer.end_frame()

    def save_particle_to_json(self, filename="particles.json"):
        """
            Save particle position data to JSON file
        """
        # Convert Warp array to numpy array, then to list
        positions = self.x.numpy()
        
        # Convert to list format: [[x1, y1, z1], [x2, y2, z2], ...]
        particles_data = positions.tolist()
        
        # Save to JSON file
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(particles_data, f, indent=2)
        
        print(f"Particle data saved to {filename}, total {self.n} particles")
if __name__ == "__main__" :
    test = wcsph()
    pt_array = test.x
    pt_nei_count = test.nei_count
    print(pt_nei_count)
    print(wp.__version__)


    for i in range(1200) :
        test.render()
        test.step()
    # test.renderer.save()
    # print(pt_array)
