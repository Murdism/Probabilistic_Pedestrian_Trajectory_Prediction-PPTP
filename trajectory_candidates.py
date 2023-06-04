from sklearn.cluster import DBSCAN
import numpy as np
import math
import torch

from sklearn.cluster import DBSCAN
import numpy as np
import math
class TreeNode:
    def __init__(self, value, weight, max,level=0):
        self.value = value
        self.weight = weight
        self.children = []
        self.level = level
        self.is_max = max # means it is the max value
        self.max_connected = False

    def add_child(self, child, prev_level):
        if self.level == prev_level:
            # print("True")
            # print("closest_node: ",self.level)
            self.children.append(child)
        else:
            min_distance = math.inf
            closest_node = None
            not_connected = True
            stack = [self]
            #print("For node : ",child.value," at level ",child.level)
            while stack:
                node = stack.pop()
                if node.level==prev_level:
                    # print("node at level ",node.level, " does not have children")
                    distance = math.sqrt((node.value[0] - child.value[0]) ** 2 + (node.value[1] - child.value[1]) ** 2)
                    if distance < min_distance:
                        min_distance = distance
                        closest_node = node
                        #print("For node : ",child.value," at level ",child.level," Node: ",closest_node.value," at level ",closest_node.level)
                    # Check if both are max
                    if (node.is_max and child.is_max):
                        closest_node.children.append(child)
                        not_connected = False
                        closest_node.max_connected = True

                elif(node.children):
                    # print("node at level ",node.level, "has children")
                    for node_child in node.children:
                        stack.append(node_child)
                
            if(not_connected):   # connect if its not max
                closest_node.children.append(child)
            # closest_node.children.append(child)

    
    def get_trajectories(self,depth = 12,full=True):
        # branches = self.get_all_branches()
        branches,weights = self.get_all_branches()
        # full_branches = [branch for branch in branches if len(branch[0]) >= depth]
        full_branches = [branch for branch in branches if len(branch) >= depth]
        full_weights = [weights[i] for i in range(len(weights)) if len(branches[i]) >= depth]
        return full_branches,full_weights
           

    def get_all_branches(self):
        if not self.children:
            # return [[[self.value.tolist()],self.weight]]
            return [[self.value.tolist()]],self.weight
        branches = []
        weights = []
        full_branches = []
        full_weights = []

        for child in self.children:
                
                brncs, wgts = child.get_all_branches()
                for branch,weight in zip(brncs, wgts) :
                    if self.level == 0:
                        # branches.append([branch, weight])  
                        branches.append(branch) 
                        weights.append(weight) 
                    else:
                        branches.append([self.value.tolist()] + branch)
                        weights.append(self.weight + weight)
        return branches,weights    


def run_cluster(mus,pi,cluster_real,cluster_src,eps=0.08, min_samples=1,dbscan=False,threshold=0.1,pred_len=12):
    mue_data = mus#.detach().cpu().numpy()
    #pi_expanded = pi.unsqueeze(-1).repeat(1,1,1,2)
    data_weight = pi.detach().cpu().numpy()
    batch_cetriods = []
    max = 1
    trajs,best_trajs,weights,best_weights = [],[],[],[]
    if (dbscan):
        for batch,pis,gt,src in zip(mue_data,data_weight,cluster_real,cluster_src):
            timestamp_cetriods = []
            root = TreeNode([0,0],0,0)
            level = 0
            for timestamp,pie in zip(batch,pis):
                level +=1
                index = np.argmax(pie, axis=None, out=None)
                res = timestamp[pie > threshold]

                pie = pie[pie > threshold]
                # print("timestamp[index]: ",timestamp[index])
                labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(res)
                pie = pie.reshape((pie.shape[0], 1), order='F')
                # labels contain lebel -1 which is not used in numpy so add 1 to all labels
                # print(labels)
                labels = labels + 1
                # Remove outliers [class -1] since we added 1 now they are class 0
                unique_labels = set(labels) - {0}

                #change set to numpy
                unique_labels = np.array(list(unique_labels))
                
                not_nill = len(unique_labels)>0
                weights_sum = np.array([np.sum(pie[labels == i], axis=0) for i in (labels)]) if (len(unique_labels)>0) else np.array(pis[index])
                weights_unique = np.array([np.sum(pie[labels == i], axis=0) for i in (unique_labels)]) if (len(unique_labels)>0) else np.array(pis[index])
                # total_weight = sum(weights_unique)
                # centroid_weights = weights_unique/total_weight
                # Do not distribute probabilities
                centroid_weights = weights_unique
                max_weight = np.argmax(centroid_weights)
                res = res*(pie/weights_sum) if (len(unique_labels)>0) else timestamp
                centroids = np.array([np.sum(res[labels == i], axis=0) for i in np.unique(unique_labels)]) if (len(unique_labels)>0) else np.array(res[index])
                
                #print("centroid: ",centroids)
                max_index = np.argmax(weights_unique)
                # max_weight = np.argmax(max_index)
                for i in range(len(centroids)):
                    node = centroids[i]
                    weight = weights_unique[i]
                    if max_index ==i:
                        max = True
                    else:
                        max = False
                    tree_node = TreeNode(node,weight,max,level)
                    #print("tree_node: ",tree_node.value,level)
                    root.add_child(tree_node,level-1)
                if (level == pred_len):
                    pos_trajs,pos_weights = root.get_trajectories(level)
                    
                    trajs.append(pos_trajs)
                    # Change trajectory weights into probabilies
                    # print("OLD pos_weights: ",pos_weights)
                    pos_weights = np.round(pos_weights/np.sum(pos_weights), decimals=1, out=None)
                    # print("New pos_weights: ",pos_weights)
                    weights.append(pos_weights)
                    best_weight = pos_weights[0]
                    best = pos_trajs[0]
                    for traj,weight in zip(pos_trajs,pos_weights):
                        if weight>best_weight:
                            best = traj
                            best_weight = weight
                    best_trajs.append(best)
                    best_weights.append(best_weight)
    else:
        for batch,pis,gt,src in zip(mue_data,data_weight,cluster_real,cluster_src):#batch,pis,gt,src in zip(mue_data,data_weight,cluster_real,cluster_src):
            timestamp_cetriods = []
            root = TreeNode([0,0],0,0)
            level = 0
            for timestamp,pie in zip(batch,pis):
                # use index if no centroids are selected
                # print("timestamp: ",timestamp)
                # print("pie: ",pie)
                # print("pied shape: ",type(pie),type(timestamp))
                level +=1
                index = np.argmax(pie, axis=None, out=None)
                centroids = np.array(timestamp[pie > threshold])
                
                pied = pie[pie > threshold]
                pied = pied.reshape(pied.shape[0], 1)
                max_index = np.argmax(pied)
                for i in range(len(centroids)):
                    node = np.array(centroids[i])
                    weight = pied[i]
                    if max_index==i:
                        max=True
                    else:
                        max=False
                    tree_node = TreeNode(node,weight,max,level)
                    root.add_child(tree_node,level-1)
                if (level == pred_len):
                    pos_trajs,pos_weights = root.get_trajectories(level)
                    trajs.append(pos_trajs)
                    # Change trajectory weights into probabilies
                    # print("OLD pos_weights: ",pos_weights)
                    pos_weights = np.round(pos_weights/np.sum(pos_weights), decimals=1, out=None)
                    # print("New pos_weights: ",pos_weights)
                    weights.append(pos_weights)
                    best_weight = pos_weights[0]
                    best = pos_trajs[0]
                    for traj,weight in zip(pos_trajs,pos_weights):
                        if weight>best_weight:
                            best = traj
                            best_weight = weight
                    best_trajs.append(best)
                    best_weights.append(best_weight)
    return trajs,weights,best_trajs,best_weights    



# if __name__ == "__main__":
#     #device = "cuda:0" if torch.cuda.is_available() else "cpu"
#     device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
#     # Create an array of 12 x,y values
#     pis = torch.tensor([[[4.6214e-02, 7.1607e-03, 2.8492e-06, 7.0503e-02, 1.6083e-04, 1.4708e-04,
#          7.3081e-04, 8.7508e-01],
#         [4.6214e-02, 7.1607e-03, 2.8492e-06, 7.0503e-02, 1.6082e-04, 1.4708e-04,
#          7.3081e-04, 8.7508e-01],
#         [4.6214e-02, 7.1607e-03, 2.8491e-06, 7.0503e-02, 1.6082e-04, 1.4708e-04,
#          7.3081e-04, 8.7508e-01],
#         [4.6213e-02, 7.1606e-03, 2.8491e-06, 7.0503e-02, 1.6082e-04, 1.4708e-04,
#          7.3080e-04, 8.7508e-01],
#         [4.6213e-02, 7.1606e-03, 2.8491e-06, 7.0503e-02, 1.6082e-04, 1.4708e-04,
#          7.3080e-04, 8.7508e-01],
#         [4.6213e-02, 7.1606e-03, 2.8491e-06, 7.0502e-02, 1.6082e-04, 1.4708e-04,
#          7.3080e-04, 8.7508e-01],
#         [4.6213e-02, 7.1606e-03, 2.8491e-06, 7.0502e-02, 1.6082e-04, 1.4708e-04,
#          7.3080e-04, 8.7508e-01],
#         [4.6213e-02, 7.1606e-03, 2.8491e-06, 7.0502e-02, 1.6082e-04, 1.4708e-04,
#          7.3080e-04, 8.7508e-01],
#         [4.6213e-02, 7.1606e-03, 2.8491e-06, 7.0502e-02, 1.6082e-04, 1.4708e-04,
#          7.3079e-04, 8.7508e-01],
#         [4.6213e-02, 7.1606e-03, 2.8490e-06, 7.0502e-02, 1.6082e-04, 1.4707e-04,
#          7.3079e-04, 8.7508e-01],
#         [4.6212e-02, 7.1606e-03, 2.8490e-06, 7.0502e-02, 1.6082e-04, 1.4707e-04,
#          7.3079e-04, 8.7508e-01],
#         [4.6212e-02, 7.1606e-03, 2.8490e-06, 7.0502e-02, 1.6082e-04, 1.4707e-04,
#          7.3079e-04, 8.7508e-01]]]).to(device)
#     mus = np.array([[[[11.4288225 , 5.496643 ],
#                     [11.598162  , 5.3759165],
#                     [11.875911  , 5.357697 ],
#                     [11.409382  , 5.3286457],
#                     [11.8946295 , 5.389827 ],
#                     [12.090345  , 5.5648637],
#                     [11.640764  , 5.4818273],
#                     [11.4072485 , 5.399251 ]],
#                     [[11.023613  , 5.599574 ],
#                     [11.362291  , 5.358121 ],
#                     [11.917789  , 5.321682 ],
#                     [10.984732  , 5.26358  ],
#                     [11.955227  , 5.3859415],
#                     [12.34666   , 5.736016 ],
#                     [11.447497  , 5.569943 ],
#                     [10.980464  , 5.4047904]],
#                     [[10.618404  , 5.7025056],
#                     [11.126421  , 5.340326 ],
#                     [11.959669  , 5.2856674],
#                     [10.5600815 , 5.198514 ],
#                     [12.015824  , 5.3820567],
#                     [12.602973  , 5.907168 ],
#                     [11.25423   , 5.6580586],
#                     [10.55368   , 5.41033  ]],
#                     [[10.213196  , 5.8054366],
#                     [10.890552  , 5.3225303],
#                     [12.001548  , 5.2496524],
#                     [10.135431  , 5.1334476],
#                     [12.076422  , 5.3781714],
#                     [12.859286  , 6.0783205],
#                     [11.060963  , 5.7461743],
#                     [10.126897  , 5.415869 ]],
#                     [[ 9.807986  , 5.9083676],
#                     [10.654681  , 5.304735 ],
#                     [12.043427  , 5.2136374],
#                     [ 9.710781  , 5.068382 ],
#                     [12.137019  , 5.3742867],
#                     [13.1156    , 6.2494726],
#                     [10.867696  , 5.83429  ],
#                     [ 9.700112  , 5.421408 ]],
#                     [[ 9.402778  , 6.011299 ],
#                     [10.418811  , 5.2869396],
#                     [12.085306  , 5.1776223],
#                     [ 9.286131  , 5.003316 ],
#                     [12.197616  ,  5.370402 ],
#                     [13.371913  , 6.4206247],
#                     [10.674429  , 5.9224057],
#                     [ 9.273329  , 5.4269476]],
#                     [[ 8.997569  , 6.11423  ],
#                     [10.182941  , 5.269144 ],
#                     [12.127186  , 5.1416078],
#                     [ 8.861481  , 4.9382496],
#                     [12.258213  , 5.3665166],
#                     [13.628226 ,  6.591777 ],
#                     [10.481161 ,  6.0105214],
#                     [ 8.846544  , 5.432487 ]],
#                     [[ 8.59236  ,  6.217161 ],
#                     [ 9.947071 ,  5.251349 ],
#                     [12.169065 ,  5.1055927],
#                     [ 8.4368305,  4.8731837],
#                     [12.31881  ,  5.362632 ],
#                     [13.88454  ,  6.762929 ],
#                     [10.287894 ,  6.098637 ],
#                     [ 8.419761 ,  5.4380264]],
#                     [[ 8.187151 ,  6.3200927],
#                     [ 9.711201 ,  5.2335534],
#                     [12.210945 ,  5.0695777],
#                     [ 8.01218  ,  4.8081174],
#                     [12.379408 ,  5.3587465],
#                     [14.140853 ,  6.934081 ],
#                     [10.094627 , 6.186753 ],
#                     [ 7.992976 , 5.443566 ]],
#                     [[ 7.781943  , 6.4230237],
#                     [ 9.475331 ,  5.215758 ],
#                     [12.252825 ,  5.0335627],
#                     [ 7.5875297 , 4.743051 ],
#                     [12.440004  , 5.3548617],
#                     [14.397166  , 7.105233 ],
#                     [ 9.901361  , 6.2748685],
#                     [ 7.566192  , 5.449105 ]],
#                     [[ 7.3767343 , 6.5259547],
#                     [ 9.239462 ,  5.1979628],
#                     [12.294704 ,  4.9975476],
#                     [ 7.1628795 , 4.677985 ],
#                     [12.500602  , 5.350977 ],
#                     [14.653479 ,  7.2763853],
#                     [ 9.708094 ,  6.362984 ],
#                     [ 7.139408 ,  5.454644 ]],
#                     [[ 6.9715257,  6.6288857],
#                     [ 9.003592  , 5.180167 ],
#                     [12.336584 ,  4.9615326],
#                     [ 6.738229  , 4.612919 ],
#                     [12.561199 ,  5.3470917],
#                     [14.909792 ,  7.4475374],
#                     [ 9.514828  , 6.4511003],
#                     [ 6.712624 ,  5.4601836]]]])
#     trajs,weights,best_trajs,best_weights = run_cluster(mus,pis)
#     print(mus[0][:,-1] ==best_trajs)
    # print("trajs: ",trajs)
    # print("best_trajs: ",best_trajs)