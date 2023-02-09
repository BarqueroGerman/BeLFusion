# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import matplotlib.pyplot as plt
import os
from matplotlib.cbook import sanitize_sequence
from matplotlib.animation import FuncAnimation, writers, PillowWriter
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import time
from tqdm import tqdm

class AnimationRenderer:

    def __init__(self, skeleton, poses_generator, algos, t_hist, t_pred, baselines=['gt', 'context'], 
                        fix_0=True, output_dir="./out", size=6, ncol=5, bitrate=-1, gif_dpi=60,
                        type="3d"):
        """
        Render an animation. The supported output modes are:
        -- 'interactive': display an interactive figure -> 'n' for next sample, 'i' to export last predicted frame, ' ' to pause/resume
                        (also works on notebooks if associated with %matplotlib inline)
        -- [NOT IMPLEMENTED] 'html': render the animation as HTML5 video. Can be displayed in a notebook using HTML(...).
        -- 'export to .mp4': render and export the animation as an h264 video (requires ffmpeg) -> press 'v'
        -- 'export to .gif': render and export the animation a gif file (requires imagemagick) -> press 'g'

        """
        os.makedirs(output_dir, exist_ok=True)

        self.skeleton = skeleton
        self.poses_generator = poses_generator
        self.algos = algos
        self.t_hist = t_hist
        self.t_pred = t_pred
        self.baselines = baselines
        self.fix_0 = fix_0
        self.output_dir = output_dir
        self.ncol = ncol
        self.size=size
        self.bitrate = bitrate
        self.gif_dpi = gif_dpi
        self.t0_digit_pressed = time.time()
        self.digits_pressed = []
        assert type.lower() in ["3d", "2d"], f"'{type}' is not a supported visualization type"
        self.fig_3d = type.lower() == "3d"
        self.show_hist = True

        self.fps = self.skeleton.fps
        self.elev, self.azim, self.roll = self.skeleton.get_camera_params()
        self.all_poses, self.sample_idx = next(poses_generator)
        
        # all_poses is a dictionary with results for each model/gt/baseline in the form of (SeqLength, Landmarks, Dimensions)
        self.algo = algos[-1] if len(algos) > 0 else next(iter(self.all_poses.keys()))
        self.t_total = next(iter(self.all_poses.values())).shape[0]

        # filter poses so that match 'gt', 'context' or the name of the first algorithm selected
        self.poses_dict = dict(filter(lambda x: x[0] in baselines or self.algo == x[0].split('_')[0], self.all_poses.items()))
        self.poses = list(self.poses_dict.values()) # so that we can access as 'poses[n][i]' the pose from algorithm 'n', frame 'i'

        self.anim = None
        self.initialized = False
        self.animating = False
        self.gif_writer = PillowWriter(fps=self.fps, bitrate=-1, codec="libx264")
        self.mp4_writer = writers['ffmpeg'](fps=self.fps, metadata={}, bitrate=self.bitrate, codec='libx264', extra_args=['-pix_fmt', 'yuv420p'])

    def run_animation(self, show=True):
        plt.ioff()
        nrow = int(np.ceil(len(self.poses_dict) / self.ncol))
        self.fig = plt.figure(figsize=(self.size*self.ncol, self.size*nrow))
        self.axes = []
        self.lines = []
        if self.fig_3d:
            self.radius, x_r, y_r, z_r = self.skeleton.get_3d_radius()
            for index, (title, data) in enumerate(self.poses_dict.items()):
                ax = self.fig.add_subplot(nrow, self.ncol, index+1, projection='3d')
                ax.view_init(elev=self.elev, azim=self.azim, roll=self.roll)
                ax.set_xlim3d(x_r)
                ax.set_zlim3d(y_r)
                ax.set_ylim3d(z_r)
                ax.set_aspect('auto')
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_zticklabels([])
                ax.dist = 5.0
                ax.set_title(title, y=1.2)
                ax.set_axis_off()
                ax.patch.set_alpha(0.0)
                self.axes.append(ax)
                self.lines.append([])
            self.fig.tight_layout()
        else:
            for index, (title, data) in enumerate(self.poses_dict.items()):
                ax = self.fig.add_subplot(nrow, self.ncol, index+1)
                ax.set_xlim(self.skeleton.xlim)
                ax.set_ylim(self.skeleton.ylim)
                ax.set_aspect('auto')
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_axis_off()
                ax.set_title(title)#, y=1.2)
                self.axes.append(ax)
                self.lines.append([])
            self.fig.tight_layout()
            self.fig.subplots_adjust(top=0.75)

        self.fig.canvas.mpl_connect('key_press_event', self._on_key)
        if self.fig_3d:
            self.fig.canvas.mpl_connect('motion_notify_event', self._on_move)
        self.animating = True
        self._show_animation()
        if show:
            plt.show()
        return self.anim

    def store_all(self, type="gif", idx=-1):
        assert type.lower() in ["gif", "mp4"], f"'{type}' is not a supported storage output type"
        self.algo = self.algos[idx]

        writer = self.gif_writer if type == "gif" else self.mp4_writer
        ext = ".gif" if type == "gif" else ".mp4"
        kwargs = {"dpi": self.gif_dpi} if type == "gif" else {}

        stored_paths = []
        os.makedirs(self.output_dir, exist_ok=True)

        self.run_animation(show=False)
        self._reload_poses()
        output_path = os.path.join(self.output_dir, f"%s_%s{ext}" % (self.algo, self.sample_idx))
        
        try:
            self.anim.save(output_path, writer=writer, progress_callback=None, **kwargs)
        except:
            # ffmpeg does not have the appropriate encoding ffmped algorithm installed => try libopenh264
            self.mp4_writer = writers['ffmpeg'](fps=self.fps, metadata={}, bitrate=self.bitrate, codec='libopenh264', extra_args=['-pix_fmt', 'yuv420p'])
            writer = self.mp4_writer
            self.anim.save(output_path, writer=writer, progress_callback=None, **kwargs)
        stored_paths.append(output_path)

        for all_poses, sample_idx in tqdm(self.poses_generator):
            self.all_poses, self.sample_idx = all_poses, sample_idx
            self._reload_poses()
            self._show_animation()

            output_path = os.path.join(self.output_dir, f"%s_%s{ext}" % (self.algo, self.sample_idx))
            self.anim.save(output_path, writer=writer, progress_callback=None, **kwargs)
            stored_paths.append(output_path)

        return stored_paths

    def _update_video(self, i):
        if self.show_hist:
            i = i % (self.t_hist + self.t_pred)
        else: # do not show history
            i = self.t_hist + i % (self.t_pred)
        self.fig.suptitle(f'Sample {self.sample_idx} - [{"H" if i < self.t_hist else "P"}] Frame {i}', fontsize=12, y=0.97)

        if i < self.t_hist:
            linewidth = 1
        else:
            linewidth = 2

        parents = self.skeleton.parents()

        if self.fig_3d:
            max_zorder, min_zorder = float('-inf'), float('inf')
            # we compute the z_orders to improve the 3d visualization
            z_orders = []

            for n, ax in enumerate(self.axes):
                n_orders = []
                for k in range(self.poses[n].shape[1]):
                    k_orders = []
                    for j, j_parent in enumerate(parents):
                        if j_parent == -1:
                            k_orders.append(2) # we don't care, it will not be used
                            continue

                        pos = self.poses[n][i, k]
                        mid = (pos[j, 0] + pos[j_parent, 0]) / 2
                        zorder = 2 + mid
                        k_orders.append(zorder) # higher zorders are drawn on top

                        max_zorder = max(max_zorder, zorder)
                        min_zorder = min(min_zorder, zorder)
                    n_orders.append(k_orders)
                z_orders.append(n_orders)
                
        if not self.initialized:

            for j, j_parent in enumerate(parents):
                if j_parent == -1:
                    continue

                col = self.skeleton.get_color(j)
                for n, ax in enumerate(self.axes):
                    to_be_appended = []
                    for k in range(self.poses[n].shape[1]):
                        pos = self.poses[n][i, k]
                        if self.fig_3d:

                            to_be_appended.append(ax.plot([pos[j, 0], pos[j_parent, 0]],
                                                    [pos[j, 1], pos[j_parent, 1]],
                                                    [pos[j, 2], pos[j_parent, 2]], zdir='z', c=col, 
                                                    zorder=z_orders[n][k][j]))
                        else:
                            to_be_appended.append(ax.plot([pos[j, 0], pos[j_parent, 0]],
                                                    [pos[j, 1], pos[j_parent, 1]], c=col))
                    self.lines[n].append(to_be_appended)
            self.initialized = True
        else:

            counter = 0 # this is a fix w.r.t. old DLow code.
            for j, j_parent in enumerate(parents):
                if j_parent == -1:
                    continue

                col = self.skeleton.get_color(j)
                for n, ax in enumerate(self.axes):
                    if self.fix_0 and n == 0 and i >= self.t_hist:
                        continue
                    for k in range(self.poses[n].shape[1]):
                        pos = self.poses[n][i, k] # pose from algorithm 'n', pose in frame 'i', participant 'k'. Result -> [N_LANDMARKS, DIMENSIONS]
                        
                        if self.fig_3d:
                            z_order = z_orders[n][k][j]
                            #alpha = (z_order - min_zorder) / (max_zorder - min_zorder)

                            self.lines[n][counter][k][0].set_data_3d([pos[j, 0], pos[j_parent, 0]], 
                                                                    [pos[j, 1], pos[j_parent, 1]], 
                                                                    [pos[j, 2], pos[j_parent, 2]])
                            self.lines[n][counter][k][0].set_zorder(z_order)              
                            #self.lines[n][counter][k][0].set_alpha(alpha)                              
                            self.lines[n][counter][k][0].set_linewidth(linewidth)#*2)
                            self.lines[n][counter][k][0].set_color(col)
                        else:
                            self.lines[n][counter][k][0].set_xdata([pos[j, 0], pos[j_parent, 0]])
                            self.lines[n][counter][k][0].set_ydata([pos[j, 1], pos[j_parent, 1]])
                            self.lines[n][counter][k][0].set_linewidth(linewidth)
                            self.lines[n][counter][k][0].set_color(col)
                counter += 1



    def _reload_poses(self):
        #print(self.algo)
        self.poses_dict = dict(filter(lambda x: x[0] in self.baselines or self.algo == x[0].split('_')[0], self.all_poses.items()))
        for ax, title in zip(self.axes, self.poses_dict.keys()):
            ax.set_title(title)#, y=1.2)
        self.poses = list(self.poses_dict.values())
        #print(list(self.poses_dict.keys()))
        #print(self.poses)

        if self.fig_3d:
            # world is modified only at the beginning
            for n, ax in enumerate(self.axes):
                # poses[n] has shape: (seq_length, participants, landmarks, dims)
                #print(n, len(self.poses), self.poses[n].shape)
                trajectory = self.poses[n][0, 0, 0, [0, 1, 2]] # Limits computed according to first frame - first trajectory - first landmark
                ax.set_xlim3d([-self.radius/2 + trajectory[0], self.radius/2 + trajectory[0]])
                ax.set_ylim3d([-self.radius/2 + trajectory[1], self.radius/2 + trajectory[1]])
                ax.set_zlim3d([-self.radius/2 + trajectory[2], self.radius/2 + trajectory[2]])

    def _show_animation(self):
        if self.anim is not None:
            self.anim.event_source.stop()
        self.anim = FuncAnimation(self.fig, self._update_video, frames=np.arange(0, (self.t_hist + self.t_pred)), interval=1000 / self.fps, repeat=True)
        plt.draw()
        self.animating = True

    def _save_figs(self):
        old_algo = algo
        for algo in self.algos:
            self._reload_poses()
            self._update_video(self.t_total - 1)
            self.fig.savefig(os.path.join(self.output_dir, '%s_%d.png' % (algo, self.sample_idx)), dpi=self.gif_dpi, transparent=True)
        algo = old_algo

    def _pause_animation(self, finish = False):
        if self.animating:
            self.animating = False
            self.anim.event_source.stop()
            if finish:
                self.anim = None

    def _resume_animation(self):
        if self.animating:
            self.animating = True
            self.anim.event_source.start()

    def _switch_animating(self):
        self.animating = not self.animating
        if self.animating:
            self.anim.event_source.stop()
        else:
            self.anim.event_source.start()

    def _on_key(self, event):
        if event.key == 'n':
            self.all_poses, self.sample_idx = next(self.poses_generator)
            self._reload_poses()
            self._show_animation()
        elif event.key == 'h': # do not show history
            self.show_hist = not self.show_hist
            self._reload_poses()
            self._show_animation()
        elif event.key == 'p':
            if self.animating:
                self._pause_animation()
                self.animating = False
            else:
                self._show_animation()
                self.animating = True
        elif event.key == ' ':
            self._switch_animating()
        elif event.key in ['g', 'v']:
            self._pause_animation(finish=True)
            mode = "gif" if event.key == 'g' else "mp4"
            self._save(mode)
            self._show_animation()
        elif event.key == 'i':  # save images
            self._pause_animation(finish=True)
            self._save_figs()
            self._reload_poses()
            self._show_animation()
        elif event.key == 'q' or event.key == "escape":  # save images
            exit()
        elif event.key.isdigit() and len(self.algos) > 1 and int(event.key) < len(self.algos): # change algorithm (or diffusion step) shown
            current_time = time.time()
            idx = int(np.sum([self.digits_pressed[i] * 10 ** (len(self.digits_pressed) - i) for i in range(len(self.digits_pressed))])) + int(event.key)
            continue_seq = (current_time - self.t0_digit_pressed < 1) # max 1 sec between digits pressed
            if idx >= len(self.algos) or not continue_seq: # can't surpass max of algorithm, of course
                # start new digits sequence
                idx = int(event.key)
                self.digits_pressed = [idx]
            else:
                self.digits_pressed.append(int(event.key))

            #print(self.digits_pressed, idx, current_time - self.t0_digit_pressed)
            self.algo = self.algos[idx]
            self._reload_poses()
            self._show_animation()
            self.t0_digit_pressed = current_time
            

    def _save(self, mode):
        self.anim = FuncAnimation(self.fig, self._update_video, frames=np.arange(0, (self.t_hist + self.t_pred)), interval=1000 / self.fps, repeat=False)
        if mode == 'mp4':
            output_path = os.path.join(self.output_dir, "%s_%d.mp4" % (self.algo, self.sample_idx))
            print(f"Saving to '{output_path}'...")
            self.anim.save(output_path, writer=self.mp4_writer)
        elif mode == 'gif':
            output_path = os.path.join(self.output_dir, "%s_%d.gif" % (self.algo, self.sample_idx))
            print(f"Saving to '{output_path}'...")
            self.anim.save(output_path, dpi=self.gif_dpi, writer=self.gif_writer)
        else:
            raise ValueError('Unsupported output format (only .mp4 and .gif are supported)')
        print(f'video saved to {self.output_dir}!')

    def _on_move(self, event):
        # ONLY ON 3D MODE
        ax_triggered = event.inaxes
        if ax_triggered is None:
            return # no motion in any axis
        for ax in self.axes:
            if ax_triggered == ax:
                continue # nothing to be done
            # update other axis
            elif ax_triggered.button_pressed in ax_triggered._rotate_btn:
                ax.view_init(elev=ax_triggered.elev, azim=ax_triggered.azim, roll=ax_triggered.roll)
                ax.dist = ax_triggered.dist
            elif ax_triggered.button_pressed in ax_triggered._zoom_btn:
                ax.set_xlim3d(ax_triggered.get_xlim3d())
                ax.set_ylim3d(ax_triggered.get_ylim3d())
                ax.set_zlim3d(ax_triggered.get_zlim3d())
        self.fig.canvas.draw_idle()




