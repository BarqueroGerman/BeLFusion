
import numpy as np


# https://colorbrewer2.org/#type=qualitative&scheme=Dark2&n=3
CBS1 = "#000000"
CBS2_hist = "#4dac26"  # green
CBS3_hist = "#d01c8b"  # pink


CBS1 = "#000000" # black
CBS2 = "#3d65a5" # blue
CBS3 = "#e57a77" # orange

class Skeleton:
    def __init__(self, parents, joints, colors, dim='3d', fps=25, colors_hist=None):
        #assert len(joints_left) == len(joints_right)
        
        self._parents = np.array(parents)
        self._joints = joints
        self._colors = colors
        self._colors_hist = colors_hist if colors_hist is not None else colors
        self.dim = dim
        self.fps = fps
        self._compute_metadata()
    
    def num_joints(self):
        return len(self._parents)
    
    def parents(self):
        return self._parents
    
    def has_children(self):
        return self._has_children
    
    def children(self):
        return self._children

    def get_color(self, idx):
        return self._colors[idx]

    def get_color_hist(self, idx):
        return self._colors_hist[idx]

    def get_3d_radius(self):
        raise NotImplementedError()

    def get_camera_params(self):
        # returns elev, azim, roll degrees
        return 15, 0, 0 # best for h36m
    
    def remove_joints(self, joints_to_remove):
        """
        Remove the joints specified in 'joints_to_remove'.
        """
        self._colors = [col for i, col in enumerate(self._colors) if i not in joints_to_remove]
        self._colors_hist = [col for i, col in enumerate(self._colors_hist) if i not in joints_to_remove]
        
        valid_joints = []
        for joint in range(len(self._parents)):
            if joint not in joints_to_remove:
                valid_joints.append(joint)

        for i in range(len(self._parents)):
            while self._parents[i] in joints_to_remove:
                self._parents[i] = self._parents[self._parents[i]]
                
        index_offsets = np.zeros(len(self._parents), dtype=int)
        new_parents = []
        for i, parent in enumerate(self._parents):
            if i not in joints_to_remove:
                new_parents.append(parent - index_offsets[parent])
            else:
                index_offsets[i:] += 1
        self._parents = np.array(new_parents)
        
        
        if self._joints is not None:
            new_joints = []
            for joint in self._joints:
                if joint in valid_joints:
                    new_joints.append(joint - index_offsets[joint])
            self._joints = new_joints

        self._compute_metadata()
        
        return valid_joints
    
    def joints_left(self):
        return self._joints
        
    def _compute_metadata(self):
        self._has_children = np.zeros(len(self._parents)).astype(bool)
        for i, parent in enumerate(self._parents):
            if parent != -1:
                self._has_children[parent] = True

        self._children = []
        for i, parent in enumerate(self._parents):
            self._children.append([])
        for i, parent in enumerate(self._parents):
            if parent != -1:
                self._children[parent].append(i)



class SkeletonH36M(Skeleton):
    def __init__(self, parents, joints_left, joints_right):
        assert len(joints_left) == len(joints_right)

        self.joints_left = joints_left
        self.joints_right = joints_right
        
        colors = [CBS1 for i in range(len(parents))]
        colors_hist = [CBS1 for i in range(len(parents))]
        for i in self.joints_left:
            colors[i] = CBS2
            colors_hist[i] = CBS2_hist
        for i in self.joints_right:
            colors[i] = CBS3
            colors_hist[i] = CBS3_hist

        super().__init__(parents=parents,
                                 joints=list(sorted(joints_left + joints_right)),
                                 colors=colors,
                                 dim='3d',
                                 fps=50,
                                 colors_hist=colors_hist
                                 )
        self.xlim = (-0.5, 0.5)
        self.ylim = (-0.5, 1.5)

    def get_3d_radius(self):
        r = 5
        x = [-r/2, r/2]
        y = [-r/2, r/2]
        z = [-r/2, r/2]
        return r, x, y, z



class SkeletonAMASS(Skeleton):
    """
    - Hips # 0
    - LeftUpLeg # 1
    - RightUpLeg # 2
    - Spine0 # 3
    - LeftLeg # 4
    - RightLeg # 5
    - Spine1 # 6
    - LeftFoot # 7
    - RightFoot # 8
    - Spine2 # 9
    - LeftToeBase # 10
    - RightToeBase # 11
    - Neck # 12
    - LeftShoulder # 13
    - RightShoulder # 14
    - Head # 15
    - LeftArm # 16
    - RightArm # 17
    - LeftElbow # 18
    - RightElbow # 19
    - LeftHand # 20
    - RightHand # 21
    """
    def __init__(self, parents, joints_left, joints_right):
        assert len(joints_left) == len(joints_right)

        self.joints_left = joints_left
        self.joints_right = joints_right
        
        colors = [CBS1 for i in range(len(parents))]
        colors_hist = [CBS1 for i in range(len(parents))]
        for i in self.joints_left:
            colors[i] = CBS2
            colors_hist[i] = CBS2_hist
        for i in self.joints_right:
            colors[i] = CBS3
            colors_hist[i] = CBS3_hist

        super().__init__(parents=parents,
                                 joints=list(sorted(joints_left + joints_right)),
                                 colors=colors,
                                 dim='3d',
                                 fps=50,
                                 colors_hist=colors_hist
                                 )
        self.xlim = (-0.5, 0.5)
        self.ylim = (-0.5, 1.5)

    def get_3d_radius(self):
        r = 5
        x = [-r/2, r/2]
        y = [-r/2, r/2]
        z = [-r/2, r/2]
        return r, x, y, z
