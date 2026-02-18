from mujoco import viewer

DEFAULT_FREE_CAM = {
    "lookat": [0, 0, 0.85],    # Look at table surface
    "distance": 1.0,            # Closer distance
    "azimuth": 140,             # Slightly from side-front
    "elevation": -30,           # Look down at good angle
}


class MjviewerRenderer:
    def __init__(self, env, camera_id=None, cam_config=None):
        if cam_config is None:
            cam_config = DEFAULT_FREE_CAM
        self.env = env
        self.camera_id = camera_id
        self.viewer = None
        self.camera_config = cam_config

    def render(self):
        pass

    def set_camera(self, camera_id):
        self.camera_id = camera_id

    def update(self):
        if self.viewer is None:
            self.viewer = viewer.launch_passive(
                self.env.sim.model._model,
                self.env.sim.data._data,
                show_left_ui=False,
                show_right_ui=False,
            )
            self.viewer.opt.geomgroup[0] = 0
            # Disable contact force visualization
            self.viewer.opt.flags[10] = 0  # mjVIS_CONTACTFORCE
            self.viewer.opt.flags[11] = 0  # mjVIS_CONTACTPOINT
        
        # Set camera parameters before sync
        if self.camera_config is not None:
            self.viewer.cam.type = 0
            self.viewer.cam.lookat[0] = self.camera_config["lookat"][0]
            self.viewer.cam.lookat[1] = self.camera_config["lookat"][1]
            self.viewer.cam.lookat[2] = self.camera_config["lookat"][2]
            self.viewer.cam.distance = self.camera_config["distance"]
            self.viewer.cam.azimuth = self.camera_config["azimuth"]
            self.viewer.cam.elevation = self.camera_config["elevation"]
        elif self.camera_id is not None:
            if self.camera_id >= 0:
                self.viewer.cam.type = 2
                self.viewer.cam.fixedcamid = self.camera_id
            else:
                self.viewer.cam.type = 0

        self.viewer.sync()

    def reset(self):
        pass

    def close(self):

        self.sim = None
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    def add_keypress_callback(self, keypress_callback):
        self.keypress_callback = keypress_callback
