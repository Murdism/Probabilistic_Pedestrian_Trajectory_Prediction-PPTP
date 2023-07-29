
from shapely.geometry import Point, MultiPoint, MultiPolygon, LineString
from shapely.ops import unary_union
from shapely.geometry import Point, MultiPoint, MultiPolygon, LineString
from shapely.ops import unary_union
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from shapely.geometry import Point
from config import ROS_PARAMS
from shapely.ops import unary_union
from shapely.plotting import plot_polygon

vehicle_radius = 1 # the distance between the center of the vehicle and its front
collisions = []
class Intersection:
    def __init__(self, trajectory_vehicle, trajectory_ped,wgt):
        self.trajectory_vehicle = trajectory_vehicle
        self.trajectory_ped = trajectory_ped
        self.wgt = wgt
        self.vehicle_radius = 1 # the distance between the center of the vehicle and its front
        #self.point_dict = {((float("{:.3f}".format(x))), float("{:.3f}".format(y))): index + 1 for index, (x, y) in enumerate(trajectory_ped)}

    def create_circles(self):
        circles = [Point(center).buffer(self.vehicle_radius) for center in self.trajectory_vehicle]
        return unary_union(circles)
    def find_intersection(self):
            # Create circles for each point in the vehicle trajectory and merge them
            merged_circles = self.create_circles()

            # Find the intersection points that are within the vehicle_radius of any circle center
            intersection_points_list = []
            intersection_circle_centers = []
            #print("trajectory_ped",self.trajectory_ped)
            #print("------------------------------------\n")
            for i,point in enumerate(self.trajectory_ped):
                ped_point = Point(point[0], point[1])  # Convert to Shapely Point
                for center in self.trajectory_vehicle:
                    circle_center_point = Point(center)
                    if ped_point.distance(circle_center_point) <= self.vehicle_radius:
                        # check index
                        #print("pedestrian point : ",ped_point)
                        #calculate the time of collision by multiplying the index with the time step of the pedestrian
                        collision_time = (1/ROS_PARAMS.ped_frequency)* (i+1)
                        collision =[center,self.wgt[0],collision_time]
                        print("collision at index : ",(i+1),collision)
                        collisions.append(collision)
                        #intersection_circle_centers.append(center)
                        # break  # Break after finding the first intersection

            return collisions

def check_intersection(trajectory_vehicle,pedestrian_trajs,traj_weights):
    collisions = []
    
    for trajs,weights in zip(pedestrian_trajs,traj_weights):
        
        for j,(traj,wgt) in enumerate(zip(trajs,weights)):
            #print("traj: ",traj)
            # traj_tuples = [tuple(x) for x in traj]
            intersection_class = Intersection(trajectory_vehicle,traj,wgt)
            # This will update the collisions list
            intersection_points = intersection_class.find_intersection()
            #print("collision: ",collisions)
            #print("---------------------------------------------------------------------------------------------")

    return collisions

if __name__ == '__main__':
    # Define the trajectories as lists of (x, y) points
    trajectory_pedestrian = [(12, 1), (11, 1.5), (10, 1.8), (9, 2.5), (8, 3), (7, 3.5), (6, 4), (5, 4.25), (4, 4.6), (3, 4.7)]
    trajectory_vehicle = [(0, 2), (0.05, 2.15875), (0.1, 2.314), (0.15, 2.46575), (0.2, 2.614), (0.25, 2.75875), (0.3, 2.9), (0.35, 3.03775), (0.4, 3.172), (0.45, 3.30275), (0.5, 3.43), (0.55, 3.55375), (0.6, 3.674), (0.65, 3.79075), (0.7, 3.904), (0.75, 4.01375), (0.8, 4.12), (0.85, 4.22275), (0.9, 4.322), (0.95, 4.41775), (1, 4.51), (1.05, 4.59875), (1.1, 4.684), (1.15, 4.76575), (1.2, 4.844), (1.25, 4.91875), (1.3, 4.99), (1.35, 5.05775), (1.4, 5.122), (1.45, 5.18275), (1.5, 5.24), (1.55, 5.29375), (1.6, 5.344), (1.65, 5.39075), (1.7, 5.434), (1.75, 5.47375), (1.8, 5.51), (1.85, 5.54275), (1.9, 5.572), (1.95, 5.59775), (2, 5.62), (2.05, 5.63875), (2.1, 5.654), (2.15, 5.66575), (2.2, 5.674), (2.25, 5.67875), (2.3, 5.68), (2.35, 5.67775), (2.4, 5.672), (2.45, 5.66275), (2.5, 5.65), (2.55, 5.63375), (2.6, 5.614), (2.65, 5.59075), (2.7, 5.564), (2.75, 5.53375), (2.8, 5.5), (2.85, 5.46275), (2.9, 5.422)]

    # Create an instance of the Intersection class
    intersection = Intersection(trajectory_vehicle, trajectory_pedestrian)



    # Create an instance of the Intersection class
    intersection = Intersection(trajectory_vehicle, trajectory_pedestrian)

    # Find the intersections
    intersection_points = intersection.find_intersection()

    # Print the intersection points
    #print("Intersection Points:", intersection_points)

    # Plot the unary_union(circles) and pedestrian trajectory
    circles = [Point(center).buffer(4.4) for center in trajectory_vehicle]
    circles = intersection.create_circles()
    pedestrian_line = LineString(trajectory_pedestrian)
    vehicle_line = LineString(trajectory_vehicle)

    plt.figure(figsize=(8, 6))
    u = unary_union(circles)
    BLUE = '#6699cc'
    plot_polygon(u, add_points=False, color=BLUE)
    x_ped, y_ped = pedestrian_line.xy
    x_veh, y_veh = vehicle_line.xy
    plt.plot(x_ped, y_ped, 'r-', label='Pedestrian trajectory')
    plt.plot(x_veh, y_veh, 'r-', label='vehicle trajectory')

    # Plot the intersection points
    x_inter, y_inter = zip(*intersection_points)
    plt.scatter(x_inter, y_inter, color='g', label='Intersection Points')

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.title('Intersection Points')
    plt.axis('equal')
    plt.show()