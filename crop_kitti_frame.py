import yaml
import os
import vispy
from vispy.scene import visuals, SceneCanvas
import numpy as np
from  matplotlib import pyplot as plt
from laserscan import LaserScan, SemLaserScan


def draw(self, event):
    if self.canvas.events.key_press.blocked():
        self.canvas.events.key_press.unblock()
    if self.img_canvas.events.key_press.blocked():
        self.img_canvas.events.key_press.unblock()


def destroy(self):
    # destroy the visualization
    self.canvas.close()
    self.img_canvas.close()
    vispy.app.quit()


def run(self):
    vispy.app.run()


def get_mpl_colormap(cmap_name):
    cmap = plt.get_cmap(cmap_name)
    sm = plt.cm.ScalarMappable(cmap=cmap)
    color_range = sm.to_rgba(np.linspace(0, 1, 256), bytes=True)[:, 2::-1]
    return color_range.reshape(256, 3).astype(np.float32) / 255.0

def cart2polar(input_xyz):
    rho = np.sqrt(input_xyz[:, 0] ** 2 + input_xyz[:, 1] ** 2)
    phi = np.arctan2(input_xyz[:, 1], input_xyz[:, 0])
    return np.stack((rho, phi, input_xyz[:, 2]), axis=1)



if __name__ == '__main__':
    dataset = "/media/ammar/HDD/LIDAR_datasets/KITTISem/dataset/"
    config = "config/labels/semantic-kitti.yaml"
    sequence = "00"
    offset = 4

    try:
        print("Opening Configuration file")
        CFG = yaml.safe_load(open(config,'r'))
    except Exception as e:
        print(e)
        print("Error loading Yaml file")
        quit()

    scan_paths = os.path.join(dataset, "sequences", sequence, "velodyne")
    print(scan_paths)
    scan_names = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(scan_paths)) for f in fn]
    print(len(scan_names))
    scan_names.sort()
    label_paths = os.path.join(dataset, "sequences", sequence, "labels")
    print(label_paths)
    label_names = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(label_paths)) for f in fn]
    print(len(label_names))
    label_names.sort()
    color_dict = CFG["color_map"]
    scan = SemLaserScan(color_dict, project=True)

    canvas = SceneCanvas(keys='interactive', show=True)
    canvas.events.draw.connect(draw)
    grid = canvas.central_widget.add_grid()

    scan_view = vispy.scene.widgets.ViewBox(border_color='white', parent=canvas.scene)
    grid.add_widget(scan_view, 0, 0)
    scan_vis = visuals.Markers()
    scan_view.camera = 'turntable'
    scan_view.add(scan_vis)
    visuals.XYZAxis(parent=scan_view.scene)

    multiplier = 1
    canvas_W = 1024 + 1
    canvas_H = 64 + 1
    img_canvas = SceneCanvas(keys='interactive', show=True, size=(canvas_W, canvas_H * multiplier))
    img_grid = img_canvas.central_widget.add_grid()
    img_canvas.events.draw.connect(draw)

    img_view = vispy.scene.widgets.ViewBox(border_color='white', parent=img_canvas.scene)
    img_grid.add_widget(img_view, 0, 0)
    img_vis = visuals.Image(cmap='viridis')
    img_view.add(img_vis)

    scan.open_scan(scan_names[offset])
    scan.open_label(label_names[offset])
    scan.colorize()
    title = "scan " + str(offset) + " of " + str(len(scan_names))
    canvas.title = title
    img_canvas.title = title
    power = 16
    range_data = np.copy(scan.unproj_range)
    range_data = range_data ** (1 / power)
    viridis_range = ((range_data - range_data.min()) /
                     (range_data.max() - range_data.min()) *
                     255).astype(np.uint8)
    viridis_map = get_mpl_colormap("viridis")
    viridis_colors = viridis_map[viridis_range]

########## Cropping
    xyz_polar = cart2polar(scan.points)
    xyz_polar_cropped = np.where((xyz_polar[:,1]<1.04)&(xyz_polar[:,1]>-1.04) )
##########

    final_points = scan.points[xyz_polar_cropped,:][0]
    final_colors = viridis_colors[..., ::-1][xyz_polar_cropped,:][0]
    scan_vis.set_data(final_points, face_color=final_colors,
                      edge_color=final_colors,
                      size=1)

    data = np.copy(scan.proj_range)
    data[data > 0] = data[data > 0] ** (1 / power)
    data[data < 0] = data[data > 0].min()
    data = (data - data[data > 0].min()) / \
           (data.max() - data[data > 0].min())

    img_vis.set_data(data)
    img_vis.update()


    pass