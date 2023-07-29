#!/usr/bin/env python3
import rospy
from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import Point, PoseArray, Pose
import sys
from visualization_msgs.msg import Marker, MarkerArray
import tf
import math
from copy import deepcopy
from nav_msgs.msg import Path
import numpy as np
import torch
from utils import generate_square_mask
from trajectory_candidates import run_cluster
from config import CFG ,Args,cfg_zed,ROS_PARAMS
from utils import person
from train import train_attn_mdn 
from intersection import check_intersection



class MarkerBase(object):
    def __init__(self, pedestrians, frequency, points,predict=True):
        self.data = None
        self.sub = rospy.Subscriber('gazebo/model_states', ModelStates, self.ModelStates_callback)
        # Subscribe to the global_plan topic
        self.sub_vehilce_path = rospy.Subscriber("/move_base/TebLocalPlannerROS/global_plan", Path, self.local_plan_callback)

        #Publishers for the markers
        self.marker_publisher = rospy.Publisher('pedestrian_marker', MarkerArray, queue_size=10)
        self.predicted_publisher = rospy.Publisher('pedestrian_predicted_position', MarkerArray, queue_size=10)
        self.publish_collision = rospy.Publisher('pedestrian_collision', MarkerArray, queue_size=10)

        self.rate = rospy.Rate(int(frequency))
        self.points= int(points)
        self.pointsArray=PoseArray()
        self.size= len(pedestrians)
        self.predict = predict # To predict or not
        self.pedestrians= pedestrians
        self.counters=[0,0]
        self.markers = [MarkerArray() for size in range(self.size)]
        self.markers_pred = [MarkerArray() for size in range(self.size)]
        self.marker_collision = MarkerArray()
        self.InitMarkers()
        self.all_pedestrians = []
        self.local_plan = []  # Initialize the local plan list
        self.vehicle_trajectory = []
        self.points_pred = 12
        self.enough_history = False
    
    def ModelStates_callback(self,data):
        self.data = data


    def local_plan_callback(self, data):
        # Update the local plan with the new trajectory
        self.local_plan = data.poses
        self.vehicle_trajectory = []
        for posestamp in self.local_plan:
           self.vehicle_trajectory.append((posestamp.pose.position.x ,posestamp.pose.position.y)) 


    def InitMarkers(self):
        if self.data:
            idx_car = self.data.name.index("catvehicle")
            position_car = self.data.pose[idx_car].position
            orientation_car = self.data.pose[idx_car].orientation   
            roll,pitch,yaw= tf.transformations.euler_from_quaternion([orientation_car.x, orientation_car.y, orientation_car.z, orientation_car.w])
            self.all_pedestrians = []
            for p in range(0, self.size):

                idx_actor = self.data.name.index(self.pedestrians[p].name) # Get the ID of pedestrian in gazebo
                position_actor = self.data.pose[idx_actor].position
                #print("Actor : ",position_actor,"   values: ",position_actor)
                position_xy = [position_actor.x, position_actor.y]
                
                

                self.marker = Marker()
                self.marker.header.frame_id = "map"
                self.marker.type = self.marker.SPHERE
                self.marker.action = self.marker.ADD # DELETE



                self.marker.scale.x = 0.1
                self.marker.scale.y = 0.1
                self.marker.scale.z = 0.1
                self.marker.color.a = 1.0
                self.marker.color.r = 0.0
                self.marker.color.g = 1.0
                self.marker.color.b = 0.0

                self.marker.pose.position.x = position_actor.x
                self.marker.pose.position.y = position_actor.y
                self.marker.pose.position.z = position_actor.z
                self.counters[p] = self.counters[p] + 1
                #print("Counter: ",self.counters[p]," Points: ", self.points)
                self.marker.id = self.counters[p]%self.points + (p *self.points)
                #print("P: ",p," Marker ID: ", self.marker.id)
                #self.marker.id = self.counter + (p * int(self.points))
                
                angle = math.atan2(position_actor.y- position_car.y, position_actor.x - position_car.x)


                quat = tf.transformations.quaternion_from_euler(0, 0, angle)
                self.marker.pose.orientation.x = quat[0]
                self.marker.pose.orientation.y = quat[1]
                self.marker.pose.orientation.z = quat[2]
                self.marker.pose.orientation.w = quat[3]
                self.markers[p].markers.append(deepcopy(self.marker))
                #print("Marker size: for ", p, " pedestrian: ", len(self.markers[p].markers))
                if len(self.markers[p].markers) >  int(self.points):
                    # remove the first element
                    self.enough_history = True
                    self.markers[p].markers.pop(0)
                    if self.counters[p] == int(self.points):
                        self.counters[p] = 0
                     #print(" AFTER  Marker size: ",len(self.markers[p].markers))
                self.pedestrians[p].add(position_xy)
                self.all_pedestrians.append(self.pedestrians[p].history())

            if (self.predict and self.enough_history):
                input_dataset = torch.Tensor(np.array(self.all_pedestrians))
                with torch.no_grad():
                    self.attn_mdn.eval()
                    if self.normalized:
                        inp_val=(input_dataset[:,1:,2:4].to(self.device)-self.mean.to(self.device))/self.std.to(self.device)
                        
                    
                    else :
                        inp_val = input_dataset[:,1:,2:4].to(self.device)

                    if (self.add_features):                
                        input_valc = torch.sqrt(torch.square(input_dataset[:,1:,2].to(self.device)) + torch.square(input_dataset[:,1:,3].to(self.device))).unsqueeze(-1)
                        input_val = torch.cat((inp_val,input_valc),-1)
                    else:
                        input_val = inp_val
                    tgt_val = torch.Tensor([0, 0]).unsqueeze(0).unsqueeze(1).repeat(input_val.shape[0],12,1).to(self.device)
                    tgt_val_mask = generate_square_mask(dim_trg = self.dec_seq ,dim_src = self.enc_seq, mask_type="tgt").to(self.device)
                    pi, sigma_x,sigma_y, mu_x , mu_y,decoder_out = self.attn_mdn(input_val,tgt_val,tgt_mask = tgt_val_mask)
                    mus = torch.cat((mu_x.unsqueeze(-1),mu_y.unsqueeze(-1)),-1)
                    #sigmas = torch.cat((sigma_x.unsqueeze(-1),sigma_y.unsqueeze(-1)),-1)

                    src_value = input_dataset[:, -1:,0:2].detach().cpu().numpy()
                    src_value = src_value[:,np.newaxis,:,:]
                    cluster_mus = (mus[:, :,:] * self.std.to(self.device) + self.mean.to(self.device)).detach().cpu().numpy().cumsum(1) + src_value

                    # cluster_real = val_batch['trg'][:, :, 0:2]
                    cluster_real = input_dataset[:, :, 0:2]
                    cluster_src = input_dataset[:,:,0:2]
                    batch_trajs,batch_weights,best_trajs,best_weights = run_cluster(cluster_mus,pi,cluster_real,cluster_src)
                    #print("best_candiates: ",best_trajs,best_weights)
                    
                    self.start = 20
                    self.pred_size = len(best_trajs)
                    #print("Size: ",self.pred_size)
                    #trajectory_vehicle = [(0, 2), (0.05, 2.15875), (0.1, 2.314), (0.15, 2.46575), (0.2, 2.614), (0.25, 2.75875), (0.3, 2.9), (0.35, 3.03775), (0.4, 3.172), (0.45, 3.30275), (0.5, 3.43), (0.55, 3.55375), (0.6, 3.674), (0.65, 3.79075), (0.7, 3.904), (0.75, 4.01375), (0.8, 4.12), (0.85, 4.22275), (0.9, 4.322), (0.95, 4.41775), (1, 4.51), (1.05, 4.59875), (1.1, 4.684), (1.15, 4.76575), (1.2, 4.844), (1.25, 4.91875), (1.3, 4.99), (1.35, 5.05775), (1.4, 5.122), (1.45, 5.18275), (1.5, 5.24), (1.55, 5.29375), (1.6, 5.344), (1.65, 5.39075), (1.7, 5.434), (1.75, 5.47375), (1.8, 5.51), (1.85, 5.54275), (1.9, 5.572), (1.95, 5.59775), (2, 5.62), (2.05, 5.63875), (2.1, 5.654), (2.15, 5.66575), (2.2, 5.674), (2.25, 5.67875), (2.3, 5.68), (2.35, 5.67775), (2.4, 5.672), (2.45, 5.66275), (2.5, 5.65), (2.55, 5.63375), (2.6, 5.614), (2.65, 5.59075), (2.7, 5.564), (2.75, 5.53375), (2.8, 5.5), (2.85, 5.46275), (2.9, 5.422)]
                    #print("length of trajectory: " ,len(self.vehicle_trajectory))
                    #print("pred size: ",self.pred_size)
                    #print("trajectory: ",self.vehicle_trajectory)
                    collisions = check_intersection(self.vehicle_trajectory,batch_trajs,batch_weights)
                    #print("Collisions: ",collisions)
                            
                    for i,prediction in enumerate(best_trajs):
                        for j,positions in enumerate(prediction):                             
                            self.marker.id= self.start + j + (i *self.points_pred)
                            self.marker.color.r = 0.0
                            self.marker.color.g = 0.0
                            self.marker.color.b = 1.0
                            self.marker.pose.position.x = positions[0]
                            self.marker.pose.position.y = positions[1]
                            self.marker.pose.position.z = position_actor.z
                            #print("X: ",positions[0]," Y: ",positions[1])
                            self.markers_pred[i].markers.append(deepcopy(self.marker))
                            #self.pointsArray.poses.append(deepcopy(points))
                    # if len(collisions[0]):
                    #     jj=0
                    #     for term in collisions:
                    #         jj =j+1
                    #         print("term: ",term)
                    #         for i,point in enumerate(term):
                    #             self.markerss = Marker()
                    #             self.markerss.header.frame_id = "map"
                    #             self.markerss.id= self.start + jj + (i *self.points_pred)
                    #             self.markerss.ns = "collision"
                    #             self.markerss.type = self.marker.SPHERE
                    #             self.markerss.action = self.marker.ADD # DELETE
                                                
                    #             angle = math.atan2(position_car.y, position_car.x)


                    #             quat = tf.transformations.quaternion_from_euler(0, 0, angle)
                    #             self.markerss.pose.orientation.x = quat[0]
                    #             self.markerss.pose.orientation.y = quat[1]
                    #             self.markerss.pose.orientation.z = quat[2]
                    #             self.markerss.pose.orientation.w = quat[3]
                    #             self.markerss.color.r = 1.0
                    #             self.markerss.color.g = 0.0
                    #             self.markerss.color.b = 0.0
                    #             self.markerss.color.a = 1.0
                    #             self.markerss.scale.x = 0.3
                    #             self.markerss.scale.y = 0.3
                    #             self.markerss.scale.z = 0.3
                    #             self.markerss.pose.position.x = point[0]
                    #             self.markerss.pose.position.y = point[1]
                    #             self.markerss.pose.position.z = 1.2
                    #             self.marker_collision.markers.append(deepcopy(self.markerss))

                    colors = [[0.0,0.0,1.0],[1.0,0.0,0.0],[1.0,1.0,0.0],[1.0,0.0,1.0],[0.8,0.2,0.2],[0.2,0.8,0.2],[0.2,0.2,0.8],[0.8,0.8,0.2],[0.8,0.2,0.8],[0.2,0.8,0.8],[0.8,0.8,0.8],[0.2,0.2,0.2]]
                    for i,(batch_pred,batch_wgt) in enumerate(zip(batch_trajs,batch_weights)):
                        #print("size: ",len(batch_pred))
                        for temp in range (5):#,(prediction,wgt) in enumerate(zip(batch_pred,batch_wgt)):  
                            j = temp % len(batch_pred)
                            for k,positions in enumerate(batch_pred[j]):                             
                                self.marker.id= temp*self.start + k + (i *self.points_pred)
                                self.marker.color.r = colors[j][0]
                                self.marker.color.g = colors[j][1]
                                self.marker.color.b = colors[j][2]
                                self.marker.pose.position.x = positions[0]
                                self.marker.pose.position.y = positions[1]
                                self.marker.pose.position.z = position_actor.z
                                #print("X: ",positions[0]," Y: ",positions[1])
                                self.markers_pred[i].markers.append(deepcopy(self.marker))


    def predictor(self,attn_mdn,device,add_features,mean,std,pf,data_rate,normalized=True,enc_seq = 8,dec_seq=12):
        self.attn_mdn = attn_mdn
        self.device = device
        self.add_features = add_features
        self.mean = mean
        self.std = std
        self.pf = pf
        self.data_rate = data_rate
        self.normalized = normalized
        self.enc_seq = enc_seq
        self.dec_seq = dec_seq

    def start(self):
        while not rospy.is_shutdown():
            self.InitMarkers()
            #print("Pedestrian:",self.all_pedestrians)
            for marker_array,pred_marker in zip(self.markers,self.markers_pred):
                self.marker_publisher.publish(marker_array)
                self.predicted_publisher.publish(pred_marker)
            #self.publish_collision.publish(self.marker_collision)
            self.rate.sleep()

class PedestrianPast(MarkerBase):
    def __init__(self, name, points):
        self.past_data =  []
        self.name = name
        self.points = points

    def add(self, position):
        
        if len(self.past_data)==0:
            info = position[0],position[1],0,0
            self.past_data.append(info)
        else:
            info = position[0],position[1],position[0] - self.past_data[-1][0],position[1]-self.past_data[-1][1]
            self.past_data.append(info)
        if len(self.past_data) > int(self.points):
            self.past_data.pop(0)
    
    def history(self):
        return self.past_data
    



if __name__ == '__main__':
    rospy.init_node('pedestrian_marker')
    #rospy.init_node('visualization_marker')
    # pedestrians_names = ['actor1','actor2']#sys.argv[1].split(",")
    # frequency = 5#  sys.argv[2]
    # points = 8 #sys.argv[3]
    # print(pedestrians_names)
    # PedestrianPastPoints(pedestrians_names, frequency, points)

    points = ROS_PARAMS.ped_points
    frequency = ROS_PARAMS.ped_points_frequency
    pedestrians = []
    pedestrians.append(PedestrianPast("actor1", points))
    pedestrians.append(PedestrianPast("actor2", points))

    Base = MarkerBase(pedestrians, frequency, points)
    Base.start()
    
