import numpy as np
import os

from src.parameter_paser import parse_args_area_2


class FFZ:

    def __init__(self, node_coord, gateway_coord, λ, h, ρ):
        self.node_coord = np.array(node_coord)
        self.gateway_coord = np.array(gateway_coord)
        self.λ = λ  # Wavelength of LoRa
        self.h = h  # thickness of the slice
        self.ρ = ρ  # density of points per unit volume
        
        # distance between node and gateway
        self.d = np.linalg.norm(self.node_coord - self.gateway_coord)
        self.a = self.b = np.sqrt((self.λ * self.d) / 2)

        # the main axis, which aline with node gateway line
        self.c = self.d / 2
        
        self.center = (self.node_coord + self.gateway_coord) / 2
        self.direction = self.gateway_coord - self.node_coord
        self.direction /= np.linalg.norm(self.direction)
        self.num_cylinders = int(np.ceil(self.d / self.h))


    def __rotation_matrix(self, axis, theta):
        """
        Return the rotation matrix associated with counterclockwise rotation about
        the given axis by theta radians.
        """

        axis = np.asarray(axis)
        axis = axis / np.sqrt(np.dot(axis, axis))
        a = np.cos(theta / 2.0)
        b, c, d = -axis * np.sin(theta / 2.0)
        aa, bb, cc, dd = a * a, b * b, c * c, d * d
        bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
  
        return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                        [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                        [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])


    def __point_in_ellipsoid(self, point):
        x, y, z = point
        return (x / self.a)**2 + (y / self.b)**2 + (z / self.c)**2 <= 1
    

    def generate_points(self):
        all_points = []

        for i in range(self.num_cylinders):
            z1 = -self.c + i * self.h
            z2 = z1 + self.h

            if self.c**2 - z1**2 < 0 or self.c**2 - z2**2 < 0:
                continue
            
            # r = sqrt(c^2 - z^2) * b / c
            r1 = np.sqrt(self.c**2 - z1**2) * self.b / self.c 
            r2 = np.sqrt(self.c**2 - z2**2) * self.b / self.c
            V = np.pi * self.h / 3 * (r1**2 + r2**2 + r1*r2)

            num_points = int(np.round(V * self.ρ))

            # number of layers along z-axis
            num_layers = int(np.round(np.cbrt(num_points)))

            for j in range(num_layers):
                z = z1 + self.h * (j / num_layers)     # z-coordinate
                r = r1 + (r2 - r1) * (j / num_layers)  # radius at z

                # number of points on the layer
                layer_points = int(num_points / num_layers)

                for k in range(layer_points):
                    theta = 2 * np.pi * (k / layer_points)  # angle
                    x = r * np.cos(theta)
                    y = r * np.sin(theta)
                    point = np.array([x, y, z])
                    if self.__point_in_ellipsoid(point):
                        all_points.append(point)

        all_points = np.array(all_points)

        # Transform points to the correct coordinate system
        all_points = self.transform_points(all_points)

        return all_points


    def transform_points(self, points):
        # Calculate the rotation axis and angle

        # The cross product gives the rotation axis
        rotation_axis = np.cross([0, 0, 1], self.direction)
        
        # The dot product gives the cos of the angle
        rotation_angle = np.arccos(np.dot([0, 0, 1], self.direction))

        # Apply rotation and translation to points
        rotation = self.__rotation_matrix(rotation_axis, rotation_angle)
        points_rotated = np.dot(points, rotation.T)
        transformed_points = points_rotated + self.center

        return transformed_points


class Tree:

    def __init__(self, r_cylinder, h_cylinder, r_ellipsoid, h_ellipsoid, position):
        self.r_cylinder = r_cylinder    # trunk_radius
        self.h_cylinder = h_cylinder    # trunk_height
        self.r_ellipsoid = r_ellipsoid  # canopy_radius
        self.h_ellipsoid = h_ellipsoid  # canopy_height
        self.position = np.array(position)


    def is_inside(self, point):
        # Check if the point is inside the trunk
        local_point = point - self.position
        x, y, z = local_point

        # Check if the point is inside the trunk
        if z <= self.h_cylinder:
            if (x**2 + y**2) <= self.r_cylinder**2:
                return True
        # Check if the point is inside the canopy
        else:
            if ((x**2 + y**2) / self.r_ellipsoid**2 + ((z - self.h_cylinder)**2) / self.h_ellipsoid**2) <= 1:
                return True

        # The point is outside the tree
        return False


class Orchard:
    
    def __init__(self, num_rows, num_cols, row_dist, col_dist, tree_parameters, orchard_origin):
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.row_dist = row_dist
        self.col_dist = col_dist
        self.orchard_origin = np.array(orchard_origin)

        self.trees = []
        for i in range(self.num_rows):
            for j in range(self.num_cols):
                x = self.orchard_origin[0] + i * self.row_dist
                # Offset y coordinate for alternate rows to create a rhombic pattern.
                y = self.orchard_origin[1] + j * self.col_dist + (i % 2) * self.col_dist / 2
                position = (x, y, 0)
                tree = Tree(*tree_parameters, position)
                self.trees.append(tree)


    def is_inside(self, point):
        for tree in self.trees:
            if tree.is_inside(point):
                return True
        return False


def calculate_ratios_one_ffz(node_coord, 
                             gateway_coord, 
                             density, 
                             orchard, 
                             maximum_comm_dis, 
                             lora_frequency):
    
    lora_wavelength = 299792458.0 / lora_frequency  # wavelength in meters
    thickness = lora_wavelength                     # thickness of the FFZ slice

    ffz = FFZ(node_coord, gateway_coord, lora_wavelength, thickness, density)

    # Generate points in FFZ
    ffz_points = ffz.generate_points()

    # Count the points
    total_points = 0
    points_within_trees = 0
    points_below_ground = 0
    points_air = 0

    # Iterate over the points and count the number that are within trees
    for point in ffz_points:
        total_points += 1
        if orchard.is_inside(point):
            points_within_trees += 1
        elif point[2] < 0:  # z-coordinate is below ground level
            points_below_ground += 1
        else:
            points_air += 1

    tree_ratio = points_within_trees / total_points
    air_ratio = points_air / total_points
    ground_ratio = points_below_ground / total_points

    location_vector = (tree_ratio, air_ratio, ground_ratio, ffz.direction[0], ffz.direction[1], ffz.direction[2], ffz.d / maximum_comm_dis)

    return location_vector


if __name__ == '__main__':

    base_dir = os.path.dirname(os.path.realpath(__file__))

    args = parse_args_area_2(base_dir)
    print(f"\nUsing configuration file: {args.config}\n")

    output_dir = os.path.join(base_dir, "output")
    os.makedirs(output_dir, exist_ok=True)

    location_vector_path = os.path.join(output_dir, args.location_vector_name)

    num_rows_m = args.num_rows
    num_cols_m = args.num_cols

    row_dist_m = args.row_dist 
    col_dist_m = args.col_dist

    orchard_origin_m = (0, col_dist_m, 0)
    node_origin_m = (1.5, col_dist_m, 0)
    gateway_coord_m = (0, 0, 10)

    r_cylinder_m = args.r_cylinder      
    h_cylinder_m = args.h_cylinder
    r_ellipsoid_m = args.r_ellipsoid
    h_ellipsoid_m = args.h_ellipsoid

    tree_parameters_m = (r_cylinder_m, h_cylinder_m, r_ellipsoid_m, h_ellipsoid_m)

    orchard_m = Orchard(num_rows_m, num_cols_m, row_dist_m, col_dist_m, tree_parameters_m, orchard_origin_m)
    
    density_m = args.density
    maximum_comm_dis_m = args.maximum_comm_dis
    lora_frequency = args.lora_frequency

    in_file = open(location_vector_path, "w")

    for i in range(num_rows_m):
        for j in range(num_cols_m):

            print(f"Node: row {i+1}, col {j+1}")

            x_m = node_origin_m[0] + i * row_dist_m
            y_m = node_origin_m[1] + j * col_dist_m + (i % 2) * col_dist_m / 2.0

            node_coord_m = (x_m, y_m, 0.5)

            location_vector_node_ij = calculate_ratios_one_ffz(node_coord_m, gateway_coord_m, density_m,
                                                               orchard_m, 
                                                               maximum_comm_dis_m, 
                                                               lora_frequency)

            in_str = str(i + 1) + "_" + str(j + 1) + ", " + ", ".join(str(item) for item in location_vector_node_ij) + "\n"
            in_file.write(in_str)

    in_file.close()
    
    print()


