"""pytorch3d Phong renderer for HandFlow demo outputs (HaWoR-style smooth shading).

Replaces the old cv2 painter's-algorithm renderer (flat triangles -> faceted flicker).
Both outputs are Phong-shaded (per-vertex normals + point lights):

  render_overlay      : perspective camera (OpenCV intrinsics) -> mesh composited on RGB
  render_ortho_video  : orthographic world/cam view (topdown | side) -> per-frame mesh
                        + camera trajectory polyline; the mesh and the trajectory are
                        projected through the SAME pytorch3d camera so they stay aligned

Coordinate system: HandFlow CV convention (X right, Y down, Z forward), meters. Verts are
passed to pytorch3d in their native CV space (no axis flip), so right-hand MANO face winding
keeps surface normals pointing outward. The ortho look-at is defined directly in CV world.
"""

from typing import List, Sequence

import cv2
import numpy as np
import torch
from pytorch3d.renderer import (
    FoVOrthographicCameras,
    Materials,
    MeshRasterizer,
    MeshRenderer,
    PointLights,
    RasterizationSettings,
    SoftPhongShader,
    TexturesVertex,
)
from pytorch3d.renderer.camera_conversions import _cameras_from_opencv_projection
from pytorch3d.renderer.cameras import look_at_rotation
from pytorch3d.structures import Meshes
from pytorch3d.structures.meshes import join_meshes_as_scene


def _bgr_to_unit_rgb(color_bgr: Sequence[float]) -> torch.Tensor:
    """BGR (uint8 tuple) -> normalized RGB tensor (3,)."""
    c = np.asarray(color_bgr, dtype=np.float32)
    if c.max() > 1.0:
        c = c / 255.0
    return torch.tensor(np.ascontiguousarray(c[::-1]), dtype=torch.float32)  # BGR -> RGB


# Camera marker: a square-base pyramid (pentahedron) drawn as a black hollow wireframe — the
# standard camera-symbol used for camera-pose viz (COLMAP / SLAM). Apex = optical center (camera
# position); rectangular base in front (+Z) = image plane / field of view.
_CAM_PYRAMID_VERTS = np.array([
    [0.0, 0.0, 0.0],          # apex = camera center (focal point)
    [0.50, 0.38, 0.80],       # base corners (forward image plane, ~4:3)
    [-0.50, 0.38, 0.80],
    [-0.50, -0.38, 0.80],
    [0.50, -0.38, 0.80],
], dtype=np.float32)
_CAM_PYRAMID_EDGES = [(0, 1), (0, 2), (0, 3), (0, 4),   # apex -> base (FOV rays)
                      (1, 2), (2, 3), (3, 4), (4, 1)]   # base loop (image plane)



class PhongRenderer:
    """pytorch3d SoftPhongShader renderer for overlay + orthographic views."""

    def __init__(self, device: str | torch.device):
        self.device = torch.device(device)
        self._raster_cache: dict[tuple[int, int], MeshRenderer] = {}

    # ── internals ───────────────────────────────────────────────────────────

    def _get_renderer(self, H: int, W: int) -> MeshRenderer:
        key = (int(H), int(W))
        if key not in self._raster_cache:
            # blur_radius=0 keeps each face's coverage hard and full-opacity. A larger blur
            # erodes the alpha of small faces (e.g. the hand rendered small in the ortho view)
            # below the composite threshold, making the mesh vanish; and soft alpha at shared
            # edges lets the background bleed through as triangle "division lines". The binary
            # mask in _composite then yields a solid, seam-free surface.
            raster = MeshRasterizer(raster_settings=RasterizationSettings(
                image_size=(H, W), blur_radius=0.0, faces_per_pixel=1,
            ))
            self._raster_cache[key] = MeshRenderer(raster, SoftPhongShader(device=self.device))
        return self._raster_cache[key]

    @staticmethod
    def _faces(faces: np.ndarray, side: str, device) -> torch.Tensor:
        f = torch.as_tensor(faces, dtype=torch.int64, device=device)
        if side == "left":           # mirror winding so normals stay outward for the left hand
            f = f[:, [0, 2, 1]]
        return f

    def _mesh(self, verts: torch.Tensor, faces_t: torch.Tensor, color_bgr) -> Meshes:
        rgb = _bgr_to_unit_rgb(color_bgr).to(self.device).view(1, 1, 3).expand(1, verts.shape[0], 3)
        return Meshes(verts=verts.unsqueeze(0), faces=faces_t.unsqueeze(0),
                      textures=TexturesVertex(verts_features=rgb))

    def _matte_lights(self, location) -> PointLights:
        """Diffuse-only point light (ambient + diffuse, no specular) -> matte surface.
        Colors must be (N_lights, 3) i.e. ((r,g,b),) — a flat (3,) is misread as 3 lights."""
        return PointLights(
            device=self.device,
            ambient_color=((0.45, 0.45, 0.45),),
            diffuse_color=((0.65, 0.65, 0.65),),
            specular_color=((0.0, 0.0, 0.0),),
            location=location,
        )

    def _render_rgba(self, mesh: Meshes, cameras, lights, H: int, W: int) -> np.ndarray:
        """Render -> (H, W, 4) uint8 RGBA."""
        renderer = self._get_renderer(H, W)
        materials = Materials(device=self.device, shininess=0.0)
        out = renderer(mesh, cameras=cameras, lights=lights, materials=materials)[0].detach().cpu().numpy()
        rgba = np.zeros((H, W, 4), dtype=np.uint8)
        rgba[..., :3] = np.clip(out[..., :3] * 255.0, 0, 255).astype(np.uint8)
        rgba[..., 3] = np.clip(out[..., 3] * 255.0, 0, 255).astype(np.uint8)
        return rgba

    @staticmethod
    def _composite(background_bgr: np.ndarray, rgba: np.ndarray, opacity: float = 0.8) -> np.ndarray:
        """Composite the rendered mesh at a fixed opacity over the background.

        A binary coverage mask (from the alpha channel) is used so the mesh is uniformly
        opaque across its surface — no per-edge transparency, hence no visible triangle seams.
        """
        bgr = np.ascontiguousarray(rgba[..., :3][:, :, ::-1])           # RGB -> BGR
        # Low threshold (any real face coverage) so small meshes still show; combined with
        # blur_radius=0 the surface is solid (no per-edge transparency -> no division lines).
        a = (rgba[..., 3] > 30).astype(np.float32)[..., None] * opacity
        out = background_bgr.astype(np.float32) * (1 - a) + bgr.astype(np.float32) * a
        return np.clip(out, 0, 255).astype(np.uint8)

    # ── overlay: perspective (OpenCV intrinsics) ────────────────────────────

    def render_overlay(
        self,
        background_bgr: np.ndarray,
        verts_cam_m: np.ndarray,          # (V,3) camera-space meters (CV convention)
        faces: np.ndarray,                # (F,3)
        intrinsics: Sequence[float],      # [fx, fy, cx, cy]
        side: str = "right",
        color_bgr: Sequence[float] = (235, 206, 135),   # sky blue #87CEEB (BGR)
    ) -> np.ndarray:
        """Phong-shaded MANO mesh alpha-composited onto the RGB background."""
        H, W = background_bgr.shape[:2]
        fx, fy, cx, cy = intrinsics
        K = torch.tensor([[[fx, 0, cx], [0, fy, cy], [0, 0, 1]]], dtype=torch.float32, device=self.device)
        R = torch.eye(3, device=self.device).unsqueeze(0)
        T = torch.zeros((1, 3), device=self.device)
        cameras = _cameras_from_opencv_projection(R, T, K, torch.tensor([[H, W]], device=self.device))

        verts = torch.as_tensor(verts_cam_m, dtype=torch.float32, device=self.device)
        mesh = self._mesh(verts, self._faces(faces, side, self.device), color_bgr)
        # Light from the camera, slightly up/right, so the hand reads as 3D (CV: -Y is up).
        lights = self._matte_lights([[0.10, -0.12, -0.05]])
        rgba = self._render_rgba(mesh, cameras, lights, H, W)
        return self._composite(background_bgr, rgba)

    # ── orthographic world / camera view ────────────────────────────────────

    @staticmethod
    def _checkerboard_mesh(center, floor_y, half, n_tiles, color_a_bgr, color_b_bgr, device):
        """Checkerboard floor on the world X-Z plane at Y=floor_y (normal points up, -Y in CV)."""
        cx, _, cz = float(center[0]), 0.0, float(center[2])
        step = 2 * half / n_tiles
        verts, feats = [], []
        faces = []
        a = _bgr_to_unit_rgb(color_a_bgr).to(device)
        b = _bgr_to_unit_rgb(color_b_bgr).to(device)
        for i in range(n_tiles):
            for j in range(n_tiles):
                x0, x1 = cx - half + i * step, cx - half + (i + 1) * step
                z0, z1 = cz - half + j * step, cz - half + (j + 1) * step
                base = len(verts)
                verts += [[x0, floor_y, z0], [x1, floor_y, z0], [x1, floor_y, z1], [x0, floor_y, z1]]
                faces += [[base, base + 1, base + 2], [base, base + 2, base + 3]]
                c = a if (i + j) % 2 == 0 else b
                feats += [c, c, c, c]
        verts_t = torch.tensor(np.asarray(verts, dtype=np.float32), device=device)
        faces_t = torch.tensor(np.asarray(faces, dtype=np.int64), device=device)
        feats_t = torch.stack(feats, dim=0)
        return Meshes(verts=verts_t.unsqueeze(0), faces=faces_t.unsqueeze(0),
                      textures=TexturesVertex(verts_features=feats_t.unsqueeze(0)))

    @staticmethod
    def _camera_wireframe_px(cam_pos, cam_rot, scale, cameras, S) -> list:
        """Camera marker -> 8 edge pixel segments of a square-pyramid (black hollow wireframe).

        Apex at the camera optical center, rectangular base (image plane) in the +Z viewing
        direction. World verts = rotate by cam_rot, scale, translate to cam_pos; projected through
        the SAME ortho camera so it stays aligned with the hand/floor/trajectory.
        """
        pts_world = (cam_rot @ (_CAM_PYRAMID_VERTS * scale).T).T + cam_pos      # (5,3)
        ndc = cameras.transform_points(
            torch.as_tensor(pts_world, dtype=torch.float32, device=cameras.device)
        ).cpu().numpy()[:, :2]
        # pytorch3d NDC has +X pointing LEFT, so screen px = (0.5 - ndc_x*0.5)*S (and +Y up -> top).
        px = np.stack([(0.5 - ndc[:, 0] * 0.5) * S, (0.5 - ndc[:, 1] * 0.5) * S], axis=-1).astype(int)
        return [(tuple(px[a]), tuple(px[b])) for a, b in _CAM_PYRAMID_EDGES]

    def render_ortho_video(
        self,
        verts_world_seq: np.ndarray,   # (T, V, 3) meters (CV convention)
        faces: np.ndarray,             # (F,3)
        c2w: np.ndarray,               # (T, 4, 4) camera-to-world (identity frames for --fix_camera)
        view: str = "third_person",    # "third_person" (oblique) | "topdown" | "side"
        side: str = "right",
        img_size: int = 720,
        mesh_color_bgr: Sequence[float] = (235, 206, 135),   # sky blue #87CEEB (BGR)
        trail_color_bgr: Sequence[float] = (35, 35, 35),
        cam_wire_bgr: Sequence[float] = (0, 0, 0),            # black hollow tetrahedron wireframe
        floor_a_bgr: Sequence[float] = (236, 236, 236),      # checkerboard light tile
        floor_b_bgr: Sequence[float] = (208, 208, 208),      # checkerboard dark tile
        bg_color_bgr: Sequence[float] = (250, 250, 250),
    ) -> List[np.ndarray]:
        """Orthographic 3D scene video: checkerboard floor (third_person view) + sky-blue hand
        mesh + camera trajectory polyline + a pyramid marker showing the camera orientation.
        The SLAM world is gravity-aligned (avg camera-down -> +Y) so the scene renders upright."""
        S = int(img_size)
        verts = np.asarray(verts_world_seq, dtype=np.float32)
        c2w = np.asarray(c2w, dtype=np.float32)
        n = min(verts.shape[0], c2w.shape[0])
        verts = verts[:n]
        camp = c2w[:n, :3, 3]
        camrot = c2w[:n, :3, :3]

        # Gravity-align the SLAM world: ViPE's world is camera-frame-0 aligned, not gravity aligned,
        # so for an ego clip the whole scene is rolled about the forward axis. Rotate so the average
        # camera-down axis (≈ gravity, since the headset is upright on average) -> world +Y. This
        # unrolls the scene (floor horizontal, camera/hand upright). No-op for --fix_camera (c2w=I).
        avg_down = camrot[:, :, 1].mean(axis=0)
        nd = float(np.linalg.norm(avg_down))
        if nd > 1e-6:
            avg_down = avg_down / nd
            tgt = np.array([0.0, 1.0, 0.0], dtype=np.float32)
            c = float(np.dot(avg_down, tgt))
            if abs(c - 1.0) > 1e-3:
                axis = np.cross(avg_down, tgt)
                an = float(np.linalg.norm(axis))
                if an > 1e-6:
                    axis = axis / an
                    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]], dtype=np.float32)
                    R_align = np.eye(3, dtype=np.float32) + an * K + (1 - c) * (K @ K)
                else:  # anti-parallel: 180° about X
                    R_align = np.diag([1.0, -1.0, -1.0]).astype(np.float32)
                verts = np.einsum('ij,tvj->tvi', R_align, verts).astype(np.float32)
                camp = (R_align @ camp.T).T.astype(np.float32)
                camrot = np.einsum('ij,tjk->tik', R_align, camrot).astype(np.float32)

        all_pts = np.concatenate([verts.reshape(-1, 3), camp.reshape(-1, 3)], axis=0)
        pmin, pmax = all_pts.min(axis=0), all_pts.max(axis=0)
        center = (pmin + pmax) / 2.0
        extent = max(float(np.max((pmax - pmin) / 2.0) * 1.3), 1e-3)
        floor_y = float(pmax[1]) + 0.02 * extent       # CV +Y is down: ground = max Y (just below scene)

        # Look-at in CV world (distance is irrelevant for orthographic projection).
        if view == "topdown":
            eye = center + np.array([0.0, -1.0, 0.0], dtype=np.float32); up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        elif view == "side":
            eye = center + np.array([1.0, 0.0, 0.0], dtype=np.float32); up = np.array([0.0, -1.0, 0.0], dtype=np.float32)
        else:  # third_person: over-the-shoulder oblique — behind/above the camera looking forward
            # at the hand (CV: -Z is behind the camera, -Y is up), like the project-page perspective.
            eye = center + np.array([-0.45, -0.80, -0.40], dtype=np.float32); up = np.array([0.0, -1.0, 0.0], dtype=np.float32)
        eye_t = torch.tensor(eye, dtype=torch.float32, device=self.device).view(1, 3)
        at_t = torch.tensor(center, dtype=torch.float32, device=self.device).view(1, 3)
        up_t = torch.tensor(up, dtype=torch.float32, device=self.device).view(1, 3)
        R = look_at_rotation(eye_t, at=at_t, up=up_t, device=self.device)
        # pytorch3d look_at_rotation returns R (cam<-world uses R^T); T must place the camera
        # at `eye`: x_cam = R^T (x - eye), so T = -(R^T @ eye). (Using -(R @ eye) mis-aims it.)
        T_cam = -(R.mT @ eye_t.unsqueeze(-1)).squeeze(-1)
        cameras = FoVOrthographicCameras(
            R=R, T=T_cam, min_x=-extent, max_x=extent, min_y=-extent, max_y=extent,
            znear=-extent * 50.0, zfar=extent * 50.0, scale_xyz=((1.0, 1.0, 1.0),), device=self.device,
        )
        lights = self._matte_lights([eye.tolist()])

        # Static checkerboard floor (only for the oblique third-person scene).
        floor_mesh = None
        if view == "third_person":
            floor_mesh = self._checkerboard_mesh(center, floor_y, extent * 1.4, 10, floor_a_bgr, floor_b_bgr, self.device)

        # Camera trajectory -> 2D pixels via the SAME camera (transform_points returns NDC).
        camp_ndc = cameras.transform_points(
            torch.as_tensor(camp, dtype=torch.float32, device=self.device)
        ).cpu().numpy()[:, :2]
        traj_px = np.stack([(0.5 - camp_ndc[:, 0] * 0.5) * S, (0.5 - camp_ndc[:, 1] * 0.5) * S], axis=-1)

        faces_t = self._faces(faces, side, self.device)
        cam_scale = extent * 0.08
        bg = np.asarray(bg_color_bgr, dtype=np.uint8)
        wire = tuple(int(c) for c in cam_wire_bgr)
        frames: List[np.ndarray] = []
        for t in range(n):
            # Scene = floor + hand only (rendered together for correct occlusion).
            verts_t = torch.as_tensor(verts[t], dtype=torch.float32, device=self.device)
            hand_mesh = self._mesh(verts_t, faces_t, mesh_color_bgr)
            scene = join_meshes_as_scene([floor_mesh, hand_mesh]) if floor_mesh is not None else hand_mesh
            rgba = self._render_rgba(scene, cameras, lights, S, S)
            canvas = self._composite(np.full((S, S, 3), bg, dtype=np.uint8), rgba, opacity=1.0)

            # Camera trajectory history polyline + current camera as a hollow tetrahedron wireframe
            # (drawn on top, always visible, apex = camera forward direction).
            for i in range(1, t + 1):
                cv2.line(canvas, tuple(traj_px[i - 1].astype(int)), tuple(traj_px[i].astype(int)),
                         trail_color_bgr, 2, cv2.LINE_AA)
            for p1, p2 in self._camera_wireframe_px(camp[t], camrot[t], cam_scale, cameras, S):
                cv2.line(canvas, p1, p2, wire, 2, cv2.LINE_AA)
            frames.append(canvas)

        return frames

