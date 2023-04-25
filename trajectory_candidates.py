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


def run_cluster(mus,pi,cluster_real,cluster_src,eps=0.08, min_samples=1,dbscan =False,threshold=0.1,pred_len=12):
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
                # print("threshold: ",threshold)
                # print("res: ",res)
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
        for batch,pis,gt,src in zip(mue_data,data_weight,cluster_real,cluster_src):
            timestamp_cetriods = []
            root = TreeNode([0,0],0,0)
            level = 0
            for timestamp,pie in zip(batch,pis):
                # use index if no centroids are selected
                level +=1
                index = np.argmax(pie, axis=None, out=None)
                centroids = np.array(timestamp[pie > threshold])
                
                pie = pie[pie > threshold]
                pie = pie.reshape((pie.shape[0], 1), order='F')
                max_index = np.argmax(pie)
                for i in range(len(centroids)):
                    node = np.array(centroids[i])
                    weight = pie[i]
                    if max_index ==i:
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