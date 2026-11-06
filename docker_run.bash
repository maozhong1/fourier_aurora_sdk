#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${DISPLAY:-}" ]]; then
  echo "DISPLAY is unset. Run this from an SSH session with X11 forwarding enabled." >&2
  exit 1
fi

if [[ -n "${SUDO_USER:-}" ]]; then
  CALLER_HOME="$(getent passwd "$SUDO_USER" | cut -d: -f6)"
  XAUTH_FILE="${XAUTHORITY:-$CALLER_HOME/.Xauthority}"
  if [[ ! -f "$XAUTH_FILE" && -f "$CALLER_HOME/.Xauthority" ]]; then
    XAUTH_FILE="$CALLER_HOME/.Xauthority"
  fi
else
  XAUTH_FILE="${XAUTHORITY:-$HOME/.Xauthority}"
fi

if [[ ! -f "$XAUTH_FILE" ]]; then
  echo "Xauthority file does not exist: $XAUTH_FILE" >&2
  exit 1
fi

XSOCK="${XSOCK:-/tmp/.X11-unix}"

# GR2 foot contact. "aurora" (default) keeps the image's original model (real-robot
# sole, foot mesh contact on; matches UniLab scene_sonic_27dof_realfoot). "sonic" mounts
# the older Sonic/UniLab training sole (scene_sonic_27dof: wider box, foot mesh contact
# off) over the image's gr2t2v2.xml.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GR2_FEET="${GR2_FEET:-aurora}"
EXTRA_MOUNTS=()
if [[ "$GR2_FEET" == "sonic" ]]; then
  FEET_XML="$SCRIPT_DIR/aurora_mjcf/resources/GR2/mjcf/gr2t2v2_sonic_feet.xml"
  if [[ ! -f "$FEET_XML" ]]; then
    echo "Sonic-feet GR2 model not found: $FEET_XML" >&2
    exit 1
  fi
  EXTRA_MOUNTS+=(-v "$FEET_XML:/usr/local/resources/GR2/mjcf/gr2t2v2.xml:ro")
  echo "GR2 feet: sonic ($FEET_XML)"
elif [[ "$GR2_FEET" == "aurora" ]]; then
  echo "GR2 feet: aurora (image default)"
else
  echo "GR2_FEET must be sonic or aurora, got: $GR2_FEET" >&2
  exit 1
fi

docker run -it \
 -e DISPLAY="$DISPLAY" \
 -e XAUTHORITY=/tmp/.Xauthority \
 -v "$(pwd)":/workspace \
 -w /workspace \
 -v "$XSOCK:$XSOCK" \
 -v "$XAUTH_FILE:/tmp/.Xauthority:ro" \
 ${EXTRA_MOUNTS[@]+"${EXTRA_MOUNTS[@]}"} \
 --privileged \
 --net=host \
 --ipc=host \
 --cap-add=SYS_NICE \
 --name fourier_aurora_server \
 fourier_aurora_sdk_gr2:v1.3.0 bash
