from viser import ViserServer
import torch
from matplotlib.pyplot import get_cmap
import numpy as np
from loguru import logger as guru
class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]

class VisManager(metaclass=Singleton):
    _servers = {}


def vis_tracks_3d(
    server: ViserServer,
    vis_tracks: np.ndarray,
    vis_label: np.ndarray,
    name: str = "tracks",
):
    """
    :param vis_tracks (np.ndarray): (N, T, 3)
    :param vis_label (np.ndarray): (N)
    """
    cmap = get_cmap("gist_rainbow")
    if vis_label is None:
        vis_label = np.linspace(0, 1, len(vis_tracks))
    colors = cmap(np.asarray(vis_label))[:, :3]
    # guru.info(f"{colors.shape}, {vis_tracks.shape}")
    N, T = vis_tracks.shape[:2]
    vis_tracks = np.asarray(vis_tracks)
    for i in range(N):
        server.scene.add_spline_catmull_rom(
            f"/{name}/{i}/spline", vis_tracks[i], color=colors[i], segments=T - 1
        )
        server.scene.add_point_cloud(
            f"/{name}/{i}/start",
            vis_tracks[i, [0]],
            colors=colors[i : i + 1],
            point_size=0.05,
            point_shape="circle",
        )
        server.scene.add_point_cloud(
            f"/{name}/{i}/end",
            vis_tracks[i, [-1]],
            colors=colors[i : i + 1],
            point_size=0.05,
            point_shape="diamond",
        )


def get_server(port):
    manager = VisManager()
    if port is None:
        avail_ports = list(manager._servers.keys())
        port = avail_ports[0] if len(avail_ports) > 0 else 8890
    if port not in manager._servers:
        manager._servers[port] = ViserServer(port=port, verbose=False)
    return manager._servers[port]

def vis_init_params(
    server,
    gs,
    pose,
    name="init_params",
    num_vis: int = 100,
):
    idcs = np.random.choice(len(gs.get_xyz), num_vis)
    labels = np.linspace(0, 1, num_vis)
    ts = torch.arange(pose.num_cams)
    with torch.no_grad():
        pred_means = compute_means(ts, fg, bases)
        vis_means = pred_means[idcs].detach().cpu().numpy()
    vis_tracks_3d(server, vis_means, labels, name=name)
