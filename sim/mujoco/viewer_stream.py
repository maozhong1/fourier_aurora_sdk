"""Headless MuJoCo simulator that streams the rendered view as MJPEG over HTTP.

Same physics / LCM behavior as viewer.py, but renders offscreen (EGL) on this
machine at a fixed low frame rate and serves JPEG frames, so a remote viewer
only needs a browser and ~0.3-1 MB/s instead of X11/GLX forwarding.

Open http://<host>:<port>/ in a browser. Keys in the page:
  arrows: rotate camera   +/-: zoom   r: reset camera
  Backspace (or the button): reset the robot, like Reset in the MuJoCo viewer
The robot can also be reset with `curl http://<host>:<port>/reset`.
"""
import os
os.environ.setdefault('MUJOCO_GL', 'egl')

import io
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import lcm
import mujoco
import numpy as np
from PIL import Image

import simulator_sub_msg
import viewer as V

_PAGE = b"""<!doctype html><html><head><title>MuJoCo stream</title>
<style>body{margin:0;background:#222;color:#ccc;font:13px sans-serif}
img{display:block;margin:auto;max-width:100vw;max-height:95vh}</style></head>
<body><img src="/stream"><div style="text-align:center">
arrows: rotate &nbsp; +/-: zoom &nbsp; r: reset camera &nbsp; Backspace:
<button onclick="reset()">reset robot</button> &nbsp; <span id="s"></span></div>
<script>
function show(t){document.getElementById('s').textContent=t;}
function reset(){fetch('/reset').then(r=>r.text()).then(show);}
const k={ArrowLeft:'az=-10',ArrowRight:'az=10',ArrowUp:'el=5',ArrowDown:'el=-5',
'+':'dist=-0.3','=':'dist=-0.3','-':'dist=0.3',r:'reset=1'};
document.onkeydown=e=>{if(e.key==='Backspace'){reset();e.preventDefault();return;}
if(k[e.key]){fetch('/cam?'+k[e.key]).then(r=>r.text())
.then(t=>document.getElementById('s').textContent=t);e.preventDefault();}};
</script></body></html>"""


class FrameHub:
  """Latest JPEG frame shared by all HTTP clients."""

  def __init__(self):
    self.cond = threading.Condition()
    self.jpeg = None
    self.seq = 0

  def publish(self, jpeg: bytes):
    with self.cond:
      self.jpeg = jpeg
      self.seq += 1
      self.cond.notify_all()

  def wait_next(self, last_seq: int, timeout: float = 2.0):
    with self.cond:
      self.cond.wait_for(lambda: self.seq != last_seq, timeout=timeout)
      return self.seq, self.jpeg


def _physics_loop(m, d, lock, stop: threading.Event, max_lag: float = 0.1):
  """Real-time stepping; same pub/control cadence as viewer._physics_loop."""
  sync_cpu = time.perf_counter()
  sync_sim = d.time
  while not stop.is_set():
    time.sleep(0.001)
    with lock:
      V.simulator_state_pub(m, d)
      V.apply_control(m, d)
      now = time.perf_counter()
      if abs((now - sync_cpu) - (d.time - sync_sim)) > max_lag:
        # Fell behind (or first step): re-sync instead of fast-forwarding.
        sync_cpu, sync_sim = now, d.time
        mujoco.mj_step(m, d)
        continue
      while (d.time - sync_sim) < (time.perf_counter() - sync_cpu):
        mujoco.mj_step(m, d)


def _render_loop(m, d, lock, cam, hub, stop, width, height, fps, quality):
  renderer = mujoco.Renderer(m, height=height, width=width)
  period = 1.0 / fps
  next_t = time.perf_counter()
  while not stop.is_set():
    with lock:
      renderer.update_scene(d, camera=cam)
    rgb = renderer.render()
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format='JPEG', quality=quality)
    hub.publish(buf.getvalue())
    next_t += period
    delay = next_t - time.perf_counter()
    if delay > 0:
      time.sleep(delay)
    else:
      next_t = time.perf_counter()


def _reset_robot(m, d, lock):
  """Same as the Reset button of the MuJoCo viewer: back to qpos0 (or keyframe 0)."""
  with lock:
    if m.nkey > 0:
      mujoco.mj_resetDataKeyframe(m, d, 0)
    else:
      mujoco.mj_resetData(m, d)
    mujoco.mj_forward(m, d)


def _make_handler(hub: FrameHub, cam, cam_default, reset_robot):

  class Handler(BaseHTTPRequestHandler):

    def log_message(self, *args):
      pass

    def do_GET(self):
      url = urlparse(self.path)
      if url.path == '/':
        self._send(200, 'text/html', _PAGE)
      elif url.path == '/reset':
        reset_robot()
        self._send(200, 'text/plain', b'robot reset')
      elif url.path == '/cam':
        q = {k: float(v[0]) for k, v in parse_qs(url.query).items()}
        if q.get('reset'):
          cam.azimuth, cam.elevation, cam.distance = cam_default
        cam.azimuth += q.get('az', 0.0)
        cam.elevation = float(np.clip(cam.elevation + q.get('el', 0.0), -89, 89))
        cam.distance = max(0.3, cam.distance + q.get('dist', 0.0))
        msg = f'az={cam.azimuth:.0f} el={cam.elevation:.0f} dist={cam.distance:.1f}'
        self._send(200, 'text/plain', msg.encode())
      elif url.path == '/stream':
        self.send_response(200)
        self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
        self.send_header('Cache-Control', 'no-cache')
        self.end_headers()
        seq = -1
        try:
          while True:
            seq, jpeg = hub.wait_next(seq)
            if jpeg is None:
              continue
            self.wfile.write(b'--frame\r\nContent-Type: image/jpeg\r\n'
                             b'Content-Length: ' + str(len(jpeg)).encode() +
                             b'\r\n\r\n' + jpeg + b'\r\n')
        except (BrokenPipeError, ConnectionResetError):
          pass
      else:
        self._send(404, 'text/plain', b'not found')

    def _send(self, code, ctype, body):
      self.send_response(code)
      self.send_header('Content-Type', ctype)
      self.send_header('Content-Length', str(len(body)))
      self.end_headers()
      self.wfile.write(body)

  return Handler


def main(argv) -> None:
  del argv
  m, d, _ = V._file_loader(  # pylint: disable=protected-access
      os.path.expanduser(_MJCF_PATH.value),
      os.path.expanduser(_TERRAIN_PATH.value))()
  mujoco.mj_forward(m, d)
  m.vis.global_.offwidth = max(m.vis.global_.offwidth, _WIDTH.value)
  m.vis.global_.offheight = max(m.vis.global_.offheight, _HEIGHT.value)

  cam = mujoco.MjvCamera()
  mujoco.mjv_defaultFreeCamera(m, cam)
  if _TRACK_CAMERA.value:
    body_id = V._tracking_body_id(m)  # pylint: disable=protected-access
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = body_id
    cam.lookat[:] = d.xpos[body_id]
  cam.azimuth, cam.elevation, cam.distance = _AZIMUTH.value, _ELEVATION.value, _DISTANCE.value
  cam_default = (cam.azimuth, cam.elevation, cam.distance)

  # LCM wiring identical to viewer.py's __main__ block.
  V.lc = lcm.LCM()
  V.lcm_sub_msg_decoder = simulator_sub_msg.simulator_sub_msg()
  V.lcm_sub_msg = None
  threading.Thread(target=V.lcm_message_handler_loop, daemon=True).start()

  lock = threading.Lock()
  stop = threading.Event()
  hub = FrameHub()
  threads = [
      threading.Thread(target=_physics_loop, args=(m, d, lock, stop), daemon=True),
      threading.Thread(
          target=_render_loop,
          args=(m, d, lock, cam, hub, stop, _WIDTH.value, _HEIGHT.value,
                _FPS.value, _QUALITY.value),
          daemon=True),
  ]
  for t in threads:
    t.start()

  server = ThreadingHTTPServer(('0.0.0.0', _PORT.value), _make_handler(hub, cam, cam_default, lambda: _reset_robot(m, d, lock)))
  server.daemon_threads = True
  print(f'Streaming {_WIDTH.value}x{_HEIGHT.value} @ {_FPS.value} fps, '
        f'MUJOCO_GL={os.environ["MUJOCO_GL"]}: http://<this-host>:{_PORT.value}/', flush=True)
  try:
    server.serve_forever()
  except KeyboardInterrupt:
    pass
  finally:
    stop.set()
    os._exit(0)  # lcm handler thread blocks in lc.handle()


if __name__ == '__main__':
  from absl import app
  from absl import flags

  _MJCF_PATH = flags.DEFINE_string('mjcf', '/usr/local/resources/GR2/mjcf/main.xml', 'Path to MJCF file.')
  _TERRAIN_PATH = flags.DEFINE_string(
      'terrain', '/usr/local/terrains/flat.xml', 'Path to TERRAIN file.')
  _TRACK_CAMERA = flags.DEFINE_boolean('track_camera', True, 'Track the robot root body.')
  _PORT = flags.DEFINE_integer('port', 8080, 'HTTP port.')
  _FPS = flags.DEFINE_float('fps', 10.0, 'Stream frame rate.')
  _WIDTH = flags.DEFINE_integer('width', 640, 'Frame width.')
  _HEIGHT = flags.DEFINE_integer('height', 480, 'Frame height.')
  _QUALITY = flags.DEFINE_integer('quality', 70, 'JPEG quality (1-95).')
  _AZIMUTH = flags.DEFINE_float('azimuth', 90.0, 'Camera azimuth (deg).')
  _ELEVATION = flags.DEFINE_float('elevation', -15.0, 'Camera elevation (deg).')
  _DISTANCE = flags.DEFINE_float('distance', 3.0, 'Camera distance (m).')
  app.run(main)
